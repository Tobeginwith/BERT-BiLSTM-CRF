# -*- coding: utf-8 -*-
import pandas as pd
from sys import platform
from torch.utils.data import DataLoader
from model import BertModel
from utils import *
from dataset import DataPrecessForSentence
from config import *
bert_path_or_name = model_dict[MODEL][-1]


def main():
    device = torch.device("cuda")
    Tokenizer = eval_object(model_dict[MODEL][0])                                       # 加载分词器
    tokenizer = Tokenizer.from_pretrained(bert_path_or_name)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    print(target_file)
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(target_file)
    else:
        checkpoint = torch.load(target_file, map_location=device)
    print(f"\t* Loading test data {test_file}...")

    test_dataset = DataPrecessForSentence(tokenizer, test_file)                         # 对测试集进行预处理
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)        # 加载测试集数据
    print("\t* Building model...")
    model = BertModel().to(device)                                                      # 创建模型
    model.load_state_dict(checkpoint["model"])

    print(20 * "=", " Testing model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, all_labels, all_pred = test(model, test_loader)             # 在测试集上评估模型
    decoded_preds = ner_decode(all_pred)                                                # 将模型预测的标签ID序列转为标签序列
    all_pred = [[j.item() for j in i.cpu()] for i in all_pred]
    all_labels = [[j.item() for j in i.cpu()] for i in all_labels]
    print(
        "\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s\n".format(batch_time, total_time))
    
    # 将预测结果写入test_data_predict.csv文件
    df = pd.read_csv(test_file, engine='python', encoding=csv_encoding, on_bad_lines='skip')
    df['pred'] = all_pred
    df['pred_decode'] = decoded_preds

    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}

    # 对测试集预测结果进行评估，包括计算整体的评估指标和每个实体类别的评估指标
    if all_labels:
        decode_labels = [[id2label[j] for j in i] for i in all_labels]
        decode_preds = [[id2label[j] for j in i] for i in all_pred]

        dict_all, dict_every_type = my_metrics(decode_labels, decode_preds)
        print(dict_all)
        print(dict_every_type)

    df['label_entities'] = [get_entities(i, id2label, 'bio') for i in all_pred]
    df.to_csv(test_pred_out, index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
