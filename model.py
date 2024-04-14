# -*- coding: utf-8 -*-
from torch import nn
from config import *
from utils import eval_object

modelClass = eval_object(model_dict[MODEL][1])
modelConfig = eval_object(model_dict[MODEL][2])
bert_path_or_name = model_dict[MODEL][-1]

import torch
from typing import List, Optional


class CRF(nn.Module):
    """
    条件随机场的实现
    参数:
        num_tags: 标签的数量
        batch_first: 数据的第一个维度是否为batch_size
    属性:
        start_transitions: 起始概率，即从<start>转移到各标签的概率，形状为(num_tags,)
        end_transitions: 终止概率，即从各标签转移到<end>的概率，形状为(num_tags,)
        transitions：转移概率矩阵，形状为(num_tags, num_tags)
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化参数
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        """
        给定发射分数，计算ground-truth标签序列y的对数条件概率ln(p(x|y))。发射分数由BiLSTM层给出
        参数:
            emissions: 发射分数张量，当batch_first参数为False时，形状为(seq_length, batch_size, num_tags)，否则为(batch_size, seq_length, num_tags)
            tags: 标签张量，当batch_first参数为False时，形状为(seq_length, batch_size)，否则为(batch_size, seq_length)
            mask: 掩码张量，当batch_first参数为False时，形状为(seq_length, batch_size)，否则为(batch_size, seq_length)
            reduction: 缩减方式，可选值为'none', 'sum', 'mean', 'token_mean'，默认为'mean'
        """
        
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """
        采用维特比算法求解最优标签序列
        参数:
            emissions: 发射分数张量，当batch_first参数为False时，形状为(seq_length, batch_size, num_tags)，否则为(batch_size, seq_length, num_tags)
            mask: 掩码张量，当batch_first参数为False时，形状为(seq_length, batch_size)，否则为(batch_size, seq_length)
            nbest: 最优标签序列的数量，默认为1，即返回最优的标签序列
            pad_tag: 填充标签的索引
        """
        
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor,
                  tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        """
        检查输入的维度是否正确
        """
        
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        """
        计算给定的标签序列tags的分数score(x, y)，计算公式可见PPT
        emissions: (seq_length, batch_size, num_tags)
        tags: (seq_length, batch_size)
        mask: (seq_length, batch_size)
        """
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # 计算起始转移概率与起始发射概率的和
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # 从标签i-1转移到标签i的转移概率加上标签i的发射概率
            # 注意这里只计算了非填充标签序列的得分
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # 最后还要加上终止转移概率
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        """
        计算所有可能的标签序列的分数的指数对数和ln(e^score(x, y))，计算公式可见PPT
        emissions: (seq_length, batch_size, num_tags)
        mask: (seq_length, batch_size)
        """
        seq_length = emissions.size(0)

        # 计算起始转移概率与起始发射概率的和，此时的score是一个(batch_size, num_tags)的矩阵，其中每个batch的第j列存储了第一个时间步的标签为j的得分
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # 将score的形状广播为(batch_size, num_tags, 1)，emissions[i]的形状广播为(batch_size, 1, num_tags)，以便后续计算
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # 此时的score是一个(batch_size, num_tags, num_tags)的矩阵，其中每个batch的第i行第j列存储了前一个时间步的标签为i，当前时间步的标签为j的得分
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # 对当前所有可能的标签分数的指数求对数和
            next_score = torch.logsumexp(next_score, dim=1)

            # 如果当前时间步是填充时间步，则保持上一个时间步的得分不变，否则更新得分
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # 最后还要加上终止转移概率
        score += self.end_transitions

        # 对最后一步所有可能的标签分数的指数求对数和
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        """
        维特比算法解码唯一最优标签序列
        emissions: (seq_length, batch_size, num_tags)
        mask: (seq_length, batch_size)
        """
        if pad_tag is None:                                                     # 默认用0填充
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # 计算起始转移概率与起始发射概率的和
        score = self.start_transitions + emissions[0]
        
        # 保存下一个时间步的标签为i时，当前时间步的标签，便于回溯
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        
        # 超出语句有效长度的标签，用pad_tag填充
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)

        # 维特比算法前向过程：在每一时间步t，计算截止到下一时间步的最大分数，并保存下一时刻标签为i时当前最有可能的标签
        for i in range(1, seq_length):
            # 将得分广播为(batch_size, num_tags, 1)，发射分数广播为(batch_size, 1, num_tags)，以便后续计算
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)

            # 计算由所有可能的当前时间步标签转移到所有可能的下一步时间步标签的分数
            next_score = broadcast_score + self.transitions + broadcast_emission

            # 找到最大分数和对应的当前步标签
            next_score, indices = next_score.max(dim=1)

            # 如果下一步时间步是填充时间步，则保持当前时间步的得分不变，否则更新得分，并保存当前时间步的标签
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # 最后还要加上终止转移概率
        end_score = score + self.end_transitions
        # 维特比算法回溯过程：根据保存的上一个时间步的标签，回溯得到最优标签序列
        # 首先找到每个序列最后一个时间步最有可能转移到<end>的标签
        _, end_tag = end_score.max(dim=1)
        seq_ends = mask.long().sum(dim=0) - 1

        # best_tags初始化为0的原因是我们默认最后一个时间步的下一个时间步都是<end>，这样我们可以得到最后一个时间步的最优标签
        # 每个序列最后一个时间步的最优标签得以确定后，可以确定最有可能的倒数第二个时间步的标签，以此类推，从而回溯得到整个序列的最优标签
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor,
                              mask: torch.ByteTensor,
                              nbest: int,
                              pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """
        维特比算法解码前n_best个最佳的最优标签序列
        与_viterbi_decode的区别仅在于每个时间步都会计算最大的n个分数，保存对应的n个标签
        emissions: (seq_length, batch_size, num_tags)
        mask: (seq_length, batch_size)
        """
        
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        seq_ends = mask.long().sum(dim=0) - 1

        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
            .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = modelClass.from_pretrained(bert_path_or_name)                       # BERT-Base-Chinese预训练模型
        self.device = torch.device("cuda")                                              # 模型所在的设备
        self.hidden = hidden_size                                                       # BiLSTM隐藏层维度
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden, num_layers=n_layers, batch_first=True, bidirectional=True)        # 双向LSTM层
        self.dropout = nn.Dropout(0.5)                                                   # Dropout层              
        self.classifier = nn.Linear(self.hidden*2, num_labels)                           # 全连接层（输入维度为2倍的BiLSTM隐藏层维度）
        self.crf = CRF(num_tags=num_labels, batch_first=True)                            # 条件随机场层，用于序列标注

    def forward(self, **input_):
        if 'labels' in input_:
            labels = input_.pop('labels', '')
        else:
            labels = None
        
        # input为字典，包含input_ids和attention_mask
        bert_output = self.bert(**input_)
        sequence_output = bert_output[0]                                                # [0]表示只取BERT的Embedding输出，为768维
        sequence_output, _ = self.lstm(sequence_output)                                 # 依次通过BiLSTM和Dropout层
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)             # 经过全连接层得到输出，为序列每个位置各标签的预测分数，这里没有softmax，因为CRF层用的就是对数概率
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=input_['attention_mask'])         # 若有标签，则计算损失（公式详见PPT）
            outputs = (-1 * loss,) + outputs
        return outputs


