import torch
import torch.nn as nn
import random

class MixStyle(nn.Module):
    """
    MixStyle 模块 (用于无监督域自适应的跨域混合)
    根据 MixStyle 论文，通常插在 ResNet 的 layer1 和 layer2 之后
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='crossdomain'):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix

    def forward(self, x):
        # 测试阶段或按概率跳过时，不进行混合
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        # 计算每个实例在空间维度上的均值和标准差 (类似 InstanceNorm)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        
        # 归一化特征
        x_normed = (x - mu) / sig

        # 生成混合索引
        if self.mix == 'crossdomain':
            # 假设输入是 [Source_Batch, Target_Batch] 拼接的
            half = B // 2
            idx = torch.arange(B)
            # 源域特征打乱并应用目标域统计量，目标域特征打乱并应用源域统计量
            idx[:half] = torch.randperm(half) + half 
            idx[half:] = torch.randperm(half)
        else:
            # 随机混合 (Domain Generalization 常用)
            idx = torch.randperm(B)

        mu2, sig2 = mu[idx], sig[idx]
        
        # 采样混合权重 lambda
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)

        # 混合统计量
        mu_mix = mu * lam + mu2 * (1 - lam)
        sig_mix = sig * lam + sig2 * (1 - lam)

        # 重新应用混合后的风格
        return x_normed * sig_mix + mu_mix