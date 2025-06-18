import math

import torch
import torch.nn.functional as F
from torch import nn

from models.common import FeatureExtractor, js_div


class JSScore(nn.Module):
    def __init__(self):
        super(JSScore, self).__init__()

    def forward(self, ref, dist):
        return 1 - js_div(ref, dist, get_softmax=True)


class CosScore(nn.Module):
    def __init__(self):
        super(CosScore, self).__init__()

    def forward(self, ref, dist):
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(ref, dist, dim=1).view(-1, 1)
        # 计算模长相似度
        ref_norm = torch.norm(ref, p=2, dim=1, keepdim=True)
        dist_norm = torch.norm(dist, p=2, dim=1, keepdim=True)
        norm_sim = 1 - torch.abs(ref_norm - dist_norm) / torch.max(ref_norm, dist_norm)
        # 结合余弦相似度和模长相似度
        return cos_sim * norm_sim


class MemoryModel(nn.Module):
    def __init__(self, num_words=1024):
        super().__init__()
        self.encoder = FeatureExtractor()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vocab = nn.Parameter(torch.Tensor(num_words, 2048), requires_grad=True)
        self.cos_score = CosScore()
        # 权重生成器
        self.weight_net = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        # 初始化词汇表
        nn.init.kaiming_normal_(self.vocab, a=math.sqrt(5))

    def forward(self, dist, ref=None, train_mode='hybrid'):
        # 提取特征
        dist_feat = self.encoder(dist)[-1]
        ref_score = 0
        alpha = 0
        if ref is not None and train_mode != 'memory_only':
            ref_feat = self.encoder(ref)[-1]
            # 对特征进行归一化和池化
            dist_feat_norm = F.normalize(dist_feat, p=2, dim=1)
            ref_feat_norm = F.normalize(ref_feat, p=2, dim=1)
            dist_feat_pool = self.avgpool(dist_feat_norm).view(-1, dist_feat.shape[1])
            ref_feat_pool = self.avgpool(ref_feat_norm).view(-1, dist_feat.shape[1])
            # 动态生成alpha
            alpha_input = torch.cat([dist_feat_pool, ref_feat_pool], dim=1)  # 拼接特征
            alpha = self.weight_net(alpha_input)  # [B,1]
            ref_score = self.cos_score(dist_feat_pool, ref_feat_pool)

        # 单失真图像 匹配失真模式
        if train_mode != 'no_memory':
            vocab_norm = F.normalize(self.vocab, p=2, dim=1)
            vocab_norm = vocab_norm.unsqueeze(2).unsqueeze(3)
            feat_norm = F.normalize(dist_feat, p=2, dim=1)
            feat_match = F.conv2d(feat_norm, weight=vocab_norm)
            feat_vector = self.avgpool(feat_match).flatten(1)
            # 单失真图像 匹配失真模式 计算二范数
            dist_score = torch.norm(feat_vector, p=2, dim=1, keepdim=True)
        else:
            dist_score = 0

        if train_mode == 'hybrid':
            score = alpha * ref_score + (1 - alpha) * dist_score
        elif train_mode == 'memory_only':
            score = dist_score
        else:  # no_memory
            score = ref_score
        if self.training:
            return score, ref_score, dist_score, alpha
        else:
            return score


if __name__ == '__main__':
    model = MemoryModel()
    model.cuda()

    ref = torch.randn(4, 3, 224, 224, device='cuda')
    dist = torch.randn(4, 3, 224, 224, device='cuda')
    (score, ref_light_img, ref_struct_img, ref_contrast_img, dist_light_img,
     dist_struct_img, dist_contrast_img) = model(dist, ref)
    print(score)
    print(ref_light_img.shape)
