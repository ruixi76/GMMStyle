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
        self.eps = eps
        self.alpha = alpha  # 保存为成员变量
    
    def forward(self, source_images, source_assignments, 
                source_stats, target_stats, alpha=None):
        """
        批次级风格迁移 (按GMM分量分组)
        
        Args:
            source_images: 源域图像 [B, C, H, W]
            source_assignments: 像素分量分配 [B, H, W]
            source_stats: 源域分量统计 {'means': [k, C], 'stds': [k, C]}
            target_stats: 目标域分量统计 {'means': [k, C], 'stds': [k, C]}
        
        Returns:
            styled_images: 风格迁移后的图像 [B, C, H, W]
        """
        # 如果调用时没有指定 alpha，就使用初始化时的默认值
        if alpha is None:
            alpha = self.alpha

        B, C, H, W = source_images.shape
        device = source_images.device
        
        # 创建输出张量
        styled_images = torch.zeros_like(source_images)
        
        # 按分量处理
        for k in range(self.num_components):
            # 创建分量掩码 [B, H, W]
            mask_2d = (source_assignments == k)
            if torch.sum(mask_2d) == 0:
                continue
            
            # 扩展为3通道掩码 [B, C, H, W]
            mask_3d = mask_2d.unsqueeze(1).expand(-1, C, -1, -1)
            
            # 提取该分量的源域像素
            source_pixels = source_images[mask_3d].reshape(-1, C)  # [N_k, C]
            
            # 获取统计量
            mu_s = source_stats['means'][k].view(1, C)  # [1, C]
            sigma_s = source_stats['stds'][k].view(1, C)  # [1, C]
            mu_t = target_stats['means'][k].view(1, C)  # [1, C]
            sigma_t = target_stats['stds'][k].view(1, C)  # [1, C]
            
            # 应用风格迁移公式: (x - μ_s)/σ_s * σ_t + μ_t
            normalized = (source_pixels - mu_s) / (sigma_s + self.eps)
            # styled_pixels = normalized * sigma_t + mu_t
            # alpha = 0.5 
            styled_pixels = (normalized * sigma_t + mu_t) * alpha + source_pixels * (1 - alpha)
            
            # 写回输出张量
            styled_images[mask_3d] = styled_pixels.reshape(-1)
        
        # 确保像素值在[0,1]范围内
        styled_images = torch.clamp(styled_images, 0.0, 1.0)
        
        return styled_images