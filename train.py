# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DataPrecessForSentence
from utils import train, validate, eval_object, my_plot
from model import BertModel
from transformers.optimization import AdamW
from config import *

Tokenizer = eval_object(model_dict[MODEL][0])
bert_path_or_name = model_dict[MODEL][-1]


def main():
    tokenizer = Tokenizer.from_pretrained(bert_path_or_name)                                # 加载分词器
    device = torch.device("cuda")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 对训练集和验证集进行预处理
    print("\t* Loading training data...")
    processed_datasets_train = DataPrecessForSentence(tokenizer, train_file)
    processed_datasets_dev = DataPrecessForSentence(tokenizer, dev_file)
    print("\t* Loading validation data...")
    
    # 创建模型
    print("\t* Building model...")
    model = BertModel().to(device)
    
    # 加载训练集和验证集数据
    train_loader = DataLoader(processed_datasets_train, shuffle=True, batch_size=batch_size)
    dev_loader = DataLoader(processed_datasets_dev, shuffle=True, batch_size=batch_size)

    # 训练参数分组
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    lstm_param_optimizer = list(model.lstm.named_parameters())
    
    # 对不同的参数组设置不同的学习率
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': bert_lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': bert_lr},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': crf_lr},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_lr},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': linear_lr},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': linear_lr},
        
        {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': lstm_lr},
        {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lstm_lr}
        ]

    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=bert_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_f1_list, dev_f1_list = [], []
    
    # 模型继续训练设置
    if checkpoint:
        print(f'载入checkpoint文件{checkpoint}')
        checkpoint_save = torch.load(checkpoint)
        
        start_epoch = checkpoint_save["epoch"] + 1
        best_score = checkpoint_save["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint_save["model"])                             # 获取已有模型参数
        
        epochs_count = checkpoint_save["epochs_count"]
        train_losses = checkpoint_save["train_losses"]
        valid_losses = checkpoint_save["valid_losses"]
    
    # 对模型进行训练前，先评估一下它在验证集上的表现
    epoch_time, epoch_loss, f1_score = validate(model, dev_loader)
    print("before train-> Valid. time: {:.4f}s, loss: {:.4f}, f1_score: {:.4f}% \n"
          .format(epoch_time, epoch_loss, (f1_score * 100)))

    # 模型训练过程
    print("\n", 20 * "=", "Training model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss = train(model, train_loader, optimizer, max_grad_norm)                   
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: "
              .format(epoch_time, epoch_loss))
        print("* Validation for epoch {}:".format(epoch))

        # 在验证集上评估模型
        epoch_time, epoch_loss, f1_score = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, f1_score: {:.4f}% \n"
              .format(epoch_time, epoch_loss, (f1_score * 100)))
        
        # 根据验证集上的F1值来动态调整优化器的学习率
        scheduler.step(f1_score)
        # 保存验证集上F1值最高的模型，用于测试；若经多次迭代后，验证集F1值不再提升，则停止训练
        if f1_score < best_score:
            patience_counter += 1
        else:
            print('save model')
            best_score = f1_score
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       target_file
                       )
        dev_f1_list.append(f1_score)
        my_plot(dev_f1_list, train_losses)
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    main()
