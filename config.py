# -*- coding: utf-8 -*-
import json

# 可用模型种类，我们尝试了bert、ernie、roberta、albert，其中bert效果最好
model_dict = {
    'bert': ('transformers.BertTokenizer',
             'transformers.BertModel',
             'transformers.BertConfig',
             './bbc'                                        # 预训练模型路径
             ),
    'ernie': (
        'transformers.AutoTokenizer',
        'transformers.BertModel',
        'transformers.AutoConfig',
        "nghuyong/ernie-1.0",
    ),
    'roberta': (
        'transformers.BertTokenizer',
        'transformers.RobertaModel',
        'transformers.RobertaConfig',
        'hfl/chinese-roberta-wwm-ext',
    ),
    'albert': ('transformers.AutoTokenizer',
               'transformers.AlbertModel',
               'transformers.AutoConfig',
               "voidful/albert_chinese_tiny",
               ),
}
MODEL = 'bert'

epochs = 200                                                # 最大训练轮数              
batch_size = 64                                             # 每个批次的大小          
bert_lr = 3e-5                                              # BERT层学习率
crf_lr = 1e-3                                               # CRF层学习率
linear_lr = 1e-3                                            # 全连接层学习率
lstm_lr = 1e-3                                              # BiLSTM层学习率
hidden_size = 256                                           # BiLSTM隐藏层维度
n_layers = 2                                                # BiLSTM层数
patience = 30                                               # 早停轮数
max_grad_norm = 10.0                                        # 梯度修剪阈值
target_file = 'models/best.pth.tar'                         # 模型存储路径
checkpoint = None                                           # 设置需要继续训练的模型路径
n_nums = None                                               # 读取csv行数，None表示读取所有


csv_rows = ['raw_sen', 'label']

# 数据集路径
dir_name = 'cner'
train_file = f"data/{dir_name}/train.csv"
dev_file = f"data/{dir_name}/dev.csv"
test_file = f"data/{dir_name}/test.csv"
csv_encoding = 'utf-8'
json_dict = f'data/{dir_name}/label_2_id.json'
test_pred_out = f"data/{dir_name}/test_data_predict.csv"


PREFIX = ''
max_seq_len = 150                                           # 序列最大长度
ignore_pad_token_for_loss = True
overwrite_cache = None


# 读取标签ID与标签的对应关系
with open(json_dict, 'r', encoding='utf-8') as f:
    dict_ = json.load(f)
num_labels = len(dict_)
