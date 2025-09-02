from utils.data_loader import EntDataset
from utils.data_utils import load_data
from utils.config import Config
from transformers import AutoTokenizer, BertModel, RobertaModel
from torch.utils.data import DataLoader
import torch
import json
from models.model import CNNNer
from utils.metrics import MetricsCalculator
from tqdm import tqdm
from utils.logger import logger
from transformers import set_seed
import argparse
from transformers import get_linear_schedule_with_warmup
import gc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_cache():
    """Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801"""
    print(f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}")

def main(args, seed, max_len = 512):
    train_path = f"data/{args.task}/train.txt"
    dev_path = f"data/{args.task}/dev.txt"
    test_path = f"data/{args.task}/test.txt"

    id2ent = {}
    for k, v in args.ent2id.items(): id2ent[v] = k

    weight_decay = 1e-2
    ent_thres = 0.5

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, add_prefix_space=True, use_fast=True)

    ner_train = EntDataset(train_path, tokenizer=tokenizer, deplabel2id=args.deplabel2id, ent2id=args.ent2id, 
                           model_name=args.bert_name, max_len=max_len, json=args.json_flag)
    ner_dev = EntDataset(dev_path, tokenizer=tokenizer, deplabel2id=args.deplabel2id, ent2id=args.ent2id, 
                         model_name=args.bert_name, max_len=max_len, is_train=False, json=args.json_flag)
    ner_test = EntDataset(test_path, tokenizer=tokenizer, deplabel2id=args.deplabel2id, ent2id=args.ent2id, 
                          model_name=args.bert_name, max_len=max_len, is_train=False, json=args.json_flag)

    # 使用标准DataLoader
    ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, collate_fn=ner_train.collate, shuffle=True, num_workers=2)
    ner_loader_dev = DataLoader(ner_dev, batch_size=args.batch_size, collate_fn=ner_dev.collate, shuffle=False, num_workers=2)
    ner_loader_test = DataLoader(ner_test, batch_size=args.batch_size, collate_fn=ner_test.collate, shuffle=False, num_workers=2)
    dev_example = load_data(dev_path, args.ent2id, args.json_flag)
    test_example = load_data(test_path, args.ent2id, args.json_flag)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and int(args.device)>=0 else 'cpu')
    print(f"Using device: {device}")

    if 'roberta' in args.bert_name:
        encoder = RobertaModel.from_pretrained(args.bert_name)
    elif 'bert' in args.bert_name:
        encoder = BertModel.from_pretrained(args.bert_name)

    model = CNNNer(args.bert_name, num_rel_tag=len(args.deplabel2id), num_ner_tag=args.ENT_CLS_NUM, cnn_dim=args.cnn_dim, 
                   biaffine_size=args.biaffine_size, size_embed_dim=args.size_embed_dim, logit_drop=args.logit_drop,
                   n_head=config.n_head, cnn_depth=args.cnn_depth, attn_dropout=0.2).to(device) # cuda()
    
    for n, p in model.named_parameters():
        # if 'pretrain_model' not in n:
        p.requires_grad_()

    # optimizer
    import collections
    counter = collections.Counter()
    for name, param in model.named_parameters():
        counter[name.split('.')[0]] += torch.numel(param)
    print(counter)
    print("Total param ", sum(counter.values()))
    logger.info(json.dumps(counter, indent=2))
    logger.info(sum(counter.values()))

    # 优化器设置 - 全量微调版本
    def set_optimizer(model):
        ln_params = []
        non_ln_params = []
        non_pretrain_params = []
        non_pretrain_ln_params = []
        
        for name, param in model.named_parameters():
            name = name.lower()
            if not param.requires_grad:
                continue
            
            # 区分预训练部分和任务特定部分
            if 'encoder' in name:  # BERT编码器部分
                if 'norm' in name or 'bias' in name:
                    ln_params.append(param)
                else:
                    non_ln_params.append(param)
            else:  # CNN和分类器部分
                if 'norm' in name or 'bias' in name:
                    non_pretrain_ln_params.append(param)
                else:
                    non_pretrain_params.append(param)
        
        # 不同部分使用不同学习率
        optimizer_grouped_parameters = [
            {'params': non_ln_params, 'lr': args.lr, 'weight_decay': weight_decay},  # BERT权重
            {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},                # BERT LayerNorm/bias
            {'params': non_pretrain_ln_params, 'lr': args.lr, 'weight_decay': 0},  # 任务层LayerNorm/bias
            {'params': non_pretrain_params, 'lr': args.lr, 'weight_decay': weight_decay},  # 任务层权重
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    optimizer = set_optimizer(model)
    total_steps = (int(len(ner_train) / args.batch_size) + 1) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup * total_steps, num_training_steps = total_steps)

    metrics = MetricsCalculator(ent_thres=ent_thres, id2ent=id2ent, allow_nested=True)
    best_dev_f1, best_test_f1 = 0.0, 0.0
    best_epoch, patience_counter = 0, 0

    for eo in range(args.n_epochs):
        loss_total = 0
        n_item = 0
        model.train()
        for idx, batch in tqdm(enumerate(ner_loader_train), desc="Training", total=len(ner_loader_train)):
            input_ids, attention_mask, orig_to_tok_index, heads, rels, matrix = batch
            input_ids, attention_mask, orig_to_tok_index, matrix, heads, rels = input_ids.to(device), attention_mask.to(device), orig_to_tok_index.to(device), matrix.to(device), heads.to(device), rels.to(device)
            # 标准PyTorch训练流程
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, orig_to_tok_index, rels, matrix)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()
            cur_n_item = input_ids.shape[0]
            n_item += cur_n_item
        avg_loss = loss_total / n_item
        logger.info(f'*** Epoch {eo} loss: {avg_loss} ***')
        with torch.no_grad():
            model.eval()
            # 评估验证集
            dev_f1, dev_eval_info, dev_entity_info = evaluate_dataset(
                model, ner_loader_dev, dev_example, "Dev", eo, device, metrics, logger)
            
            # 评估测试集
            test_f1, test_eval_info, test_entity_info = evaluate_dataset(
                model, ner_loader_test, test_example, "Test", eo, device, metrics, logger)
        
            if dev_f1 > best_dev_f1:
                logger.info("Find best dev f1 at epoch {} (Dev F1: {:5.2f}, Test F1: {:5.2f})".format(eo, dev_f1, test_f1))
                best_epoch = eo
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_dev_results = dev_eval_info.copy()    # 保存最佳验证集详细结果
                best_test_results = test_eval_info.copy()  # 保存对应测试集详细结果
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= 10:
            break
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info("Total training epochs: {}".format(eo))
    logger.info("Best epoch: {}".format(best_epoch if 'best_epoch' in locals() else 'N/A'))
    logger.info("Early stopping patience: {}".format(patience_counter))

    logger.info("\n--- BEST VALIDATION RESULTS ---")
    logger.info("Best Dev F1: {:5.2f}".format(best_dev_f1))
    if 'best_dev_results' in locals():
        logger.info("Best Dev Precision: {:5.2f}".format(best_dev_results['acc'] * 100))
        logger.info("Best Dev Recall: {:5.2f}".format(best_dev_results['recall'] * 100))
        logger.info("Best Dev Origin: {}".format(best_dev_results['origin']))
        logger.info("Best Dev Found: {}".format(best_dev_results['found']))
        logger.info("Best Dev Right: {}".format(best_dev_results['right']))

    logger.info("\n--- CORRESPONDING TEST RESULTS ---")
    if 'best_test_f1' in locals():
        logger.info("Best Test F1: {:5.2f}".format(best_test_f1))
        if 'best_test_results' in locals():
            logger.info("Test Precision: {:5.2f}".format(best_test_results['acc'] * 100))
            logger.info("Test Recall: {:5.2f}".format(best_test_results['recall'] * 100))
            logger.info("Test Origin: {}".format(best_test_results['origin']))
            logger.info("Test Found: {}".format(best_test_results['found']))
            logger.info("Test Right: {}".format(best_test_results['right']))

    logger.info("\n--- MODEL INFORMATION ---")
    logger.info("Task: {}".format(args.task if 'args' in locals() else 'N/A'))
    logger.info("Max length: {}".format(max_len if 'max_len' in locals() else 'N/A'))
    logger.info("Seed: {}".format(seed if 'seed' in locals() else 'N/A'))
    logger.info("Device: {}".format(device))

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

def evaluate_dataset(model, data_loader, examples, dataset_name, epoch, device, metrics, logger):
    total_X, total_Y, total_Z = [], [], []
    pre, word_lens = [], []
    
    for batch in tqdm(data_loader, desc=dataset_name):
        input_ids, attention_mask, orig_to_tok_index, heads, rels = batch
        input_ids, attention_mask, orig_to_tok_index, heads, rels = input_ids.to(device), attention_mask.to(device), orig_to_tok_index.to(device), heads.to(device), rels.to(device)
        logits = model(input_ids, attention_mask, orig_to_tok_index, rels)
        pre += logits
        actual_lengths = (heads != -2).sum(dim=-1)  # [batch_size]
        word_lens += actual_lengths.cpu().tolist()


    total_X, total_Y, total_Z = metrics.get_evaluate_fpr(examples, pre, word_lens)
    eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
    f1_score = eval_info['f1'] * 100
    
    logger.info('\n{} Eval Epoch: {}  p.:{:5.2f}  r.:{:5.2f}  f1:{:5.2f}  origin:{}  found:{}  right:{}'.format(
        dataset_name, epoch, eval_info['acc'] * 100, eval_info['recall'] * 100, 
        f1_score, eval_info['origin'], eval_info['found'], eval_info['right']))
   
    # for item in entity_info.keys():
    #     logger.info('-- {} item:  {}  p.:{:5.2f}  r.:{:5.2f}  f1:{:5.2f} origin:{}  found:{}  right:{}'.format(
    #         dataset_name, item, entity_info[item].get('precision', 0) * 100, entity_info[item].get('recall', 0) * 100, entity_info[item]['f1'] * 100, 
    #         entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))
    
    return f1_score, eval_info, entity_info


if __name__ == '__main__':
    
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="conll03")
    parser.add_argument('--warmup', default=0.1, type=float)
    parser.add_argument('--cnn_depth', default=3, type=int)
    parser.add_argument('--cnn_dim', default=200, type=int)
    parser.add_argument('--size_embed_dim', default=25, type=int)
    parser.add_argument('--logit_drop', default=0.1, type=float)
    parser.add_argument('--biaffine_size', default=200, type=int)


    args = parser.parse_args()
    args.config = f'./configs/{args.task}.json'
    config = Config(args)

    seed = random.sample(range(10,20),3)
    seed = [13, 42, 43]
    logger.info('seed: {}'.format(seed))
    max_len = 512
    for idx in range(len(seed)): # 函数会执行 3 次
        main(config, int(seed[idx]), max_len)
        clean_cache()
