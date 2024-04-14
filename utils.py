# -*- coding: utf-8 -*-
import numpy as np
import os

import logging
from collections import Counter
from importlib import import_module

import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from config import *


def eval_object(object_):
    """
    方便导入相关模块
    """
    if '.' in object_:
        module_, class_ = object_.rsplit('.', 1)
        module_ = import_module(module_)
        return getattr(module_, class_)
    else:
        module_ = import_module(object_)
        return module_


def validate(model, dataloader):
    """
    在验证集上评估模型性能
    参数:
        model: 待评估的模型
        dataloader: torch.utils.data提供的DataLoader对象
    """
    # 将模型切换到评估模式，关闭Dropout等
    print('正在对验证集进行测试')
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    eval_loss = 0.0
    nb_eval_steps = 0
    
    # 禁用梯度
    with torch.no_grad():
        # 对每个batch进行预测
        for step, tokened_data_dict in enumerate(dataloader):
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
            labels = tokened_data_dict['labels']
            tmp_eval_loss, logits = model(**tokened_data_dict)
            tags = model.crf.decode(logits, tokened_data_dict['attention_mask'])                # 维特比算法求解最优标签序列
            running_loss += tmp_eval_loss.item()

            out_classes_raw = tags[0]
            out_classes = [(torch.masked_select(i, mask.type(torch.bool)))[1:-1] for i, mask in zip(out_classes_raw, tokened_data_dict['attention_mask'])]
            labels = [torch.masked_select(i, mask.type(torch.bool))[1:-1] for i, mask in zip(labels, tokened_data_dict['attention_mask'])]
            all_prob.extend(out_classes)
            all_labels.extend(labels)
    
    epoch_time = time.time() - epoch_start                              # 计算评估时间
    epoch_loss = running_loss / len(dataloader)                         # 计算平均损失

    all_prob = [[j.item() for j in i.cpu()] for i in all_prob]
    all_labels = [[j.item() for j in i.cpu()] for i in all_labels]
    
    # 评估验证集预测结果
    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}
    decode_labels = [[id2label[j] for j in i] for i in all_labels]
    decode_preds = [[id2label[j] for j in i] for i in all_prob]
    dict_all, dict_every_type = my_metrics(decode_labels, decode_preds)
    print(dict_all)
    print(dict_every_type)
    
    # 返回验证集的评估结果，包括对验证集数据进行预测的时间、损失均值和F1值
    return epoch_time, epoch_loss, dict_all['f1']


import json


def load_json(file_name, value_is_array=False):
    """
    加载标签ID与标签的对应关系
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        dict_ = json.load(f)
        return dict_


def compute(origin, found, right):
    """
    计算召回率、精确率和F1值
    参数：
        origin: 原始实体数量
        found: 预测实体数量
        right: 预测正确的实体数量
    """
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


def my_metrics(decode_labels, decode_preds, id2label=None):
    """
    计算实体识别的各类别评估指标和总体评估指标，包括精确率、召回率和F1值
    参数：
        decode_labels: 真实实体标签序列
        decode_preds: 预测实体标签序列
        id2label: 标签ID与标签的对应关系
    """
    markup = 'bio'                                                                      # 标注方式
    origins, founds, rights = [], [], []
    # 根据标签序列进行实体抽取，包含实体类型、起始位置和结束位置
    for label_path, pre_path in zip(decode_labels, decode_preds):
        label_entities = get_entities(label_path, id2label, markup)
        pre_entities = get_entities(pre_path, id2label, markup)
        origins.extend(label_entities)
        founds.extend(pre_entities)
        rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    # 计算各类别的精确率、召回率和F1值
    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(origin, found, right)
        class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'num': count}
    
    # 计算总体的精确率、召回率和F1值，这里采用微平均的方式
    micro_precision = 0
    micro_recall = 0
    total_num = 0

    for _, metrics_dict in class_info.items():
        micro_precision += metrics_dict['precision'] * metrics_dict['num']
        micro_recall += metrics_dict['recall'] * metrics_dict['num']
        total_num += metrics_dict['num']

    micro_precision =  micro_precision / total_num
    micro_recall = micro_recall / total_num
    if micro_precision + micro_recall == 0.0:                               # 模型未训练时，可能出现分母为0的情况，需单独处理
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    
    origin = len(origins)
    return {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1, 'num': origin}, class_info


def ner_decode(data):
    """
    将模型输出的标签ID序列转换为标签序列，data为模型输出的标签ID序列
    """
    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}
    id2label[-100] = '-'
    d = [[id2label[j.item()] for j in i] for i in data]
    d = [','.join(i) for i in d]
    return d


def test(model, dataloader):
    """
    在测试集上评估模型性能
    参数:
        model: 待评估的模型
        dataloader: torch.utils.data提供的DataLoader对象
    """
    # 将模型切换到评估模式，关闭Dropout等
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    all_prob = []
    all_labels = []
    all_pred = []
    # 禁用梯度
    with torch.no_grad():
        # 对每个batch进行预测
        for tokened_data_dict in tqdm(dataloader):
            batch_start = time.time()
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
            labels = tokened_data_dict.get('labels')
            if labels is not None:
                loss, logits = model(**tokened_data_dict)
            else:
                logits = model(**tokened_data_dict)[0]
            tags = model.crf.decode(logits, tokened_data_dict['attention_mask'])            # 维特比算法求解最优标签序列
            batch_time += time.time() - batch_start
            out_classes = tags[0]

            out_classes = [(torch.masked_select(i, mask.type(torch.bool)))[1:-1] for i, mask in zip(out_classes, tokened_data_dict['attention_mask'])]
            if labels is not None:
                labels = [torch.masked_select(i, mask.type(torch.bool))[1:-1] for i, mask in zip(labels, tokened_data_dict['attention_mask'])]
                all_labels.extend(labels)
            all_pred.extend(out_classes)
    
    batch_time /= len(dataloader)                                           # 计算每个batch的平均预测时间
    total_time = time.time() - time_start                                   # 计算测试集总的预测时间
    
    # 返回测试集的预测结果，包括对测试集数据进行预测的时间信息、真实实体标签和预测实体标签
    return batch_time, total_time, all_labels, all_pred


def train(model, dataloader, optimizer, max_gradient_norm):
    """
    在训练集上训练模型（只训练一轮，在train.py中每次迭代都会调用）
    参数:
        model: 待训练的模型
        dataloader: torch.utils.data提供的DataLoader对象
        optimizer: torch.optim提供的优化器对象
        max_gradient_norm: 梯度裁剪所用的阈值
    """
    # 将模型切换到训练模式，开启Dropout等
    model.train()
    device = model.device
    model.to(device)
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    tqdm_batch_iterator = tqdm(dataloader)
    
    # 以batch为单位进行训练
    for batch_index, (tokened_data_dict) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
        optimizer.zero_grad()
        loss, logits = model(**tokened_data_dict)
        tqdm_batch_iterator.set_description(f'loss:{loss.cpu().item()}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)                         # 进行梯度裁剪
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

    epoch_time = time.time() - epoch_start                                      # 计算本次迭代的总训练时间
    epoch_loss = running_loss / len(dataloader)                                 # 计算本次迭代在每个batch上的平均损失
    
    # 返回本轮迭代的总训练时间和平均损失
    return epoch_time, epoch_loss


def get_entity_bios(seq, id2label):
    """
    从BIOS标签序列中获取实体，包括实体类型、起始位置和结束位置
    参数:
        seq: BIOS标签序列
        id2label: 标签ID与标签的对应关系
    调用效果:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """
    从BIO标签序列中获取实体，包括实体类型、起始位置和结束位置
    参数:
        seq: BIO标签序列
        id2label: 标签ID与标签的对应关系
    示例:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.rsplit('-', 1)[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.rsplit('-', 1)[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    """
    从标签序列中获取实体，包括实体类型、起始位置和结束位置
    seq: 标签序列
    id2label: 标签ID与标签的对应关系
    markup: 标注方式，支持bio和bios两种
    """
    assert markup in ['bio', 'bios']                            # 仅支持bio和bios两种标注方式
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


import sys
import time


class ProgressBar(object):
    '''
    用于显示训练或测试过程中的进度条
    '''

    def __init__(self, n_total, width=30, desc='Training', num_epochs=None):

        self.width = width
        self.n_total = n_total
        self.desc = desc
        self.start_time = time.time()
        self.num_epochs = num_epochs

    def reset(self):
        self.start_time = time.time()

    def _time_info(self, now, current):
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'
        return time_info

    def _bar(self, now, current):
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1: recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        return bar

    def epoch_start(self, current_epoch):
        sys.stdout.write("\n")
        if (current_epoch is not None) and (self.num_epochs is not None):
            sys.stdout.write(f"Epoch: {current_epoch}/{self.num_epochs}")
            sys.stdout.write("\n")

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        bar = self._bar(now, current)
        show_bar = f"\r{bar}" + self._time_info(now, current)
        if len(info) != 0:
            show_bar = f'{show_bar} ' + " [" + "-".join(
                [f' {key}={value:.4f} ' for key, value in info.items()]) + "]"
        if current >= self.n_total:
            show_bar += '\n'
        sys.stdout.write(show_bar)
        sys.stdout.flush()


def save_json(file_name, dict_):
    """
    将字典dict_保存为json文件，路径为file_name
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict_, f, ensure_ascii=False)


def get_max(x, y):
    """
    获取y中的最大值以及对应的x值
    """
    max_x_index = np.argmax(y)
    max_x = x[max_x_index]
    max_y = y[max_x_index]
    return max_x, max_y


def my_plot(train_acc_list, losses):
    """
    绘制训练过程中，验证集上的F1值和训练集上的损失随迭代次数的变化曲线
    参数:
        train_acc_list: 验证集上的F1值列表
        losses: 训练集上的损失列表
    """
    
    plt.figure()
    plt.plot(train_acc_list, color='r', label='dev_f1')
    x = [i for i in range(len(train_acc_list))]
    for add, list_ in enumerate([train_acc_list, ]):
        max_x, max_y = get_max(x, list_)
        plt.text(max_x, max_y, f'{(max_x, max_y)}')
        plt.vlines(max_x, min(train_acc_list), max_y, colors='r' if add==0 else 'b', linestyles='dashed')
        plt.hlines(max_y, 0, max_x, colors='r' if add==0 else 'b', linestyles='dashed')
    
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(train_file), f'{MODEL}_dev_f1.png'), dpi=800)
    plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(os.path.dirname(train_file), f'{MODEL}_loss.png'), dpi=800)
