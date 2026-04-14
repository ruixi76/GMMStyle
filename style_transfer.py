import torch
import torch.nn as nn

class PixelStyleTransfer(nn.Module):
    """
    像素级风格迁移模块
    核心公式: I' = (I - μ_s)/σ_s * σ_t + μ_t
    其中μ和σ是通道级统计量，非GMM协方差矩阵
    """
    def __init__(self, num_components=5, eps=1e-6, alpha=0.5):
        super().__init__()
        self.num_components = num_components
        self.eps = eps # 避免分母为零
        self.alpha = alpha  # 保存为成员变量 0.5
    
    def forward(self, source_images, responsibilities,
                source_stats, target_stats, alpha=None):
        """
        批次级风格迁移 (按GMM分量分组)
        
        Args:
            source_images: 源域图像 [B, C, H, W]
            responsibilities: 责任矩阵 [B, H, W, K]
            source_stats: 源域分量统计 {'means': [k, C], 'stds': [k, C]}
            target_stats: 目标域分量统计 {'means': [k, C], 'stds': [k, C]}
        
        Returns:
            styled_images: 风格迁移后的图像 [B, C, H, W]
        """
        # 如果调用时没有指定 alpha，就使用初始化时的默认值
        if alpha is None:
            alpha = self.alpha

        bsz, channels, h, w = source_images.shape
        num_components = responsibilities.shape[-1]
        if num_components != self.num_components:
            raise ValueError(
                f"responsibilities components ({num_components}) != num_components ({self.num_components})"
            )

        # [B, H, W, C]
        source_nhwc = source_images.permute(0, 2, 3, 1)

        # 广播成 [1, 1, 1, K, C]
        mu_s = source_stats['means'].view(1, 1, 1, self.num_components, channels)
        sigma_s = source_stats['stds'].view(1, 1, 1, self.num_components, channels)
        mu_t = target_stats['means'].view(1, 1, 1, self.num_components, channels)
        sigma_t = target_stats['stds'].view(1, 1, 1, self.num_components, channels)

        # 扩展像素到 [B, H, W, 1, C]，对每个 k 计算候选风格化结果
        source_expand = source_nhwc.unsqueeze(3)
        candidates = ((source_expand - mu_s) / (sigma_s + self.eps)) * sigma_t + mu_t

        # 软分配融合: x_style = Σ_k γ_nk * x_hat_nk
        gamma = responsibilities.unsqueeze(-1)  # [B, H, W, K, 1]
        styled_nhwc = torch.sum(gamma * candidates, dim=3) # [B, H, W, C]

        # 内容保留混合: x_final = alpha*x_style + (1-alpha)*x_source
        styled_nhwc = alpha * styled_nhwc + (1.0 - alpha) * source_nhwc

        styled_images = styled_nhwc.permute(0, 3, 1, 2)
        
        # 确保像素值在[0,1]范围内
        styled_images = torch.clamp(styled_images, 0.0, 1.0)
        
        return styled_images