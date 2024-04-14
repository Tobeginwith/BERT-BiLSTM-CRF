# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd
import torch
from config import *
from utils import load_json

label_2_id = load_json(json_dict)


class DataPrecessForSentence(Dataset):
    """
    对文本进行预处理
    """

    def __init__(self, bert_tokenizer, LCQMC_file):
        """
        bert_tokenizer :BERT分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.data = self.get_input(LCQMC_file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def get_input(self, file):
        """
        通对输入文本进行处理，得到可用于BERT输入的序列。
        参数:
            file: 训练集/验证集/测试集csv文件路径
        """
        
        # 读取数据集
        if n_nums:
            df = pd.read_csv(file, engine='python', encoding=csv_encoding, error_bad_lines=False, nrows=n_nums)
        else:
            df = pd.read_csv(file, engine='python', encoding=csv_encoding, error_bad_lines=False)
        self.length = len(df)
        self.bert_tokenizer.model_max_length = max_seq_len                              # 设置文本最大长度
        sentences = df[csv_rows[0]].tolist()
        for ii, i in enumerate(sentences):
            assert isinstance(i, str), f"{i}, {ii}"

        data = self.my_bert_tokenizer(sentences)        # 将原始文本转为BERT模型输入
        seq_len = data['input_ids'].size(1)
        
        # 对标签序列也要进行截断和填充
        if csv_rows[-1] in df.columns:
            labels = df[csv_rows[-1]].tolist()
            self.labels = labels.copy()
            labels = [eval(i) for i in labels]
            labels = [(i + ([label_2_id['pad'], ] * (seq_len - len(i)))) if seq_len > len(i) else i[:seq_len]
                      for i in labels]
            labels = torch.Tensor(labels).type(torch.long)
            data['labels'] = labels
        print('输入例子')
        print(sentences[0] if isinstance(sentences[0], str) else sentences[0][0])
        for k, v in data.items():
            print(k)
            print(v[0])
        print(f"实际序列转换后的长度为{len(data['input_ids'][0])}, 设置最长为{max_seq_len}")
        return data

    def my_bert_tokenizer(self, sentences):
        """
        对文本进行分词、token ID化、截断和填充
        """
        sen_max_len = max([len(i) for i in sentences])
        max_len = min(max_seq_len, sen_max_len)
        sentences = [i[:max_len-2] for i in sentences]
        sentences_list = [['[CLS]'] + [i for i in sen] + ['[SEP]'] for sen in sentences]            # 在句子前后加上[CLS]和[SEP]
        input_ids = [self.bert_tokenizer.convert_tokens_to_ids(sen) for sen in sentences_list]      # 将句子分词并转换为token ID序列
        # 构建attention mask，只在句子的有效长度上使用attention机制
        attention_mask = [[1, ] * len(i) + [0, ] * (max_len - len(i)) for i in input_ids]
        input_ids = [i + [0, ] * (max_len - len(i)) for i in input_ids]                             # 将句子填充到最大长度
        data_dict = {'attention_mask': attention_mask, 'input_ids': input_ids}
        data_dict = {k: torch.Tensor(v).type(torch.long) for k, v in data_dict.items()}             # 转换为张量
        
        # 返回结果是一个字典，包括input_ids和attention_mask
        return data_dict


