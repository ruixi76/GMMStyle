import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
from PIL import Image

from pixel_gmm import PixelGaussianMixture

def visualize_gmm_components(gmm, num_samples=5, save_path='gmm_components.png'):
    """
    可视化GMM组件的像素分布
    
    Args:
        gmm: 训练好的PixelGaussianMixture模型
        num_samples: 每个分量生成的样本数
        save_path: 保存路径
    """
    if not gmm.initialized:
        print("GMM not initialized, skipping visualization")
        return
    
    components = []
    for k in range(gmm.num_components):
        components.append({
            'mean': gmm.means[k].cpu().numpy(),
            'std': torch.sqrt(torch.diagonal(gmm.covariances[k])).cpu().numpy(),
            'weight': gmm.weights[k].cpu().item()
        })
    
    # 为每个分量生成样本
    samples = []
    labels = []
    
    for i, comp in enumerate(components):
        mean = comp['mean']
        std = comp['std']
        
        # 生成样本（从高斯分布采样）
        for _ in range(num_samples):
            sample = np.random.normal(mean, std)
            sample = np.clip(sample, 0, 1)  # 确保在[0,1]范围内
            samples.append(sample)
            labels.append(f'Comp {i+1}\n(w={comp["weight"]:.2f})')
    
    # 可视化
    num_show = min(15, len(samples))
    cols = min(5, num_show)
    rows = (num_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for i in range(num_show):
        # 创建颜色块
        color_block = np.ones((100, 100, 3))
        color_block[:, :, 0] = samples[i][0]  # R
        color_block[:, :, 1] = samples[i][1]  # G
        color_block[:, :, 2] = samples[i][2]  # B
        
        axes[i].imshow(color_block)
        axes[i].set_title(labels[i], fontsize=9)
        axes[i].axis('off')
    
    # 隐藏多余子图
    for i in range(num_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"GMM components visualization saved to {save_path}")

def visualize_style_transfer(original_images, styled_images, assignments, 
                           save_path='style_transfer_results.png', num_show=8):
    """
    可视化像素级风格转换结果
    
    Args:
        original_images: 原始图像 [B, C, H, W]
        styled_images: 风格转换后的图像 [B, C, H, W]
        assignments: 像素分配 [B, H, W]
        save_path: 保存路径
        num_show: 显示的样本数量
    """
    B, C, H, W = original_images.shape
    num_show = min(num_show, B)
    
    fig = plt.figure(figsize=(20, 4 * num_show))
    gs = plt.GridSpec(num_show, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    for i in range(num_show):
        # 原始图像
        orig_img = original_images[i].permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        ax_orig = fig.add_subplot(gs[i, 0])
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f'Original {i+1}', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # 风格转换图像
        styled_img = styled_images[i].permute(1, 2, 0).numpy()
        styled_img = np.clip(styled_img, 0, 1)
        
        ax_styled = fig.add_subplot(gs[i, 1])
        ax_styled.imshow(styled_img)
        ax_styled.set_title(f'Styled {i+1}', fontsize=12, fontweight='bold')
        ax_styled.axis('off')
        
        # 像素分配可视化
        assign_img = assignments[i].cpu().numpy() if torch.is_tensor(assignments) else assignments[i]
        
        ax_assign = fig.add_subplot(gs[i, 2])
        im = ax_assign.imshow(assign_img, cmap='viridis', interpolation='nearest')
        ax_assign.set_title(f'Pixel Assignments\n(Component IDs)', fontsize=11)
        ax_assign.axis('off')
        plt.colorbar(im, ax=ax_assign, fraction=0.046, pad=0.04)
        
        # 覆盖在原图上的分配
        overlay_img = orig_img.copy()
        unique_assigns = np.unique(assign_img)
        
        for k in unique_assigns:
            mask = (assign_img == k)
            if np.any(mask):
                # 为不同分量使用不同颜色
                color = plt.cm.tab20(int(k) % 20)[:3]
                overlay_img[mask] = overlay_img[mask] * 0.7 + np.array(color) * 0.3
        
        ax_overlay = fig.add_subplot(gs[i, 3])
        ax_overlay.imshow(overlay_img)
        ax_overlay.set_title('Assignment Overlay', fontsize=11)
        ax_overlay.axis('off')
    
    plt.suptitle('Pixel-Level Style Transfer Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Style transfer visualization saved to {save_path}")

def visualize_gmm_training_stats(stats_history, save_path='gmm_training_stats.png'):
    """
    可视化GMM训练过程中的统计信息
    
    Args:
        stats_history: 训练历史记录字典
        save_path: 保存路径
    """
    if 'log_likelihoods' not in stats_history:
        print("No training stats available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 对数似然
    axes[0, 0].plot(stats_history['log_likelihoods'], marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Log-Likelihood During EM Training', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Log-Likelihood')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 分量权重
    if 'weights_history' in stats_history and len(stats_history['weights_history']) > 0:
        weights_history = np.array(stats_history['weights_history'])
        for k in range(weights_history.shape[1]):
            axes[0, 1].plot(weights_history[:, k], label=f'Component {k+1}', linewidth=2)
        axes[0, 1].set_title('Component Weights Evolution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 分量均值（仅显示前3个通道）
    if 'means_history' in stats_history and len(stats_history['means_history']) > 0:
        means_history = np.array(stats_history['means_history'])
        for k in range(min(3, means_history.shape[1])):
            for c in range(min(3, means_history.shape[2])):
                axes[1, 0].plot(means_history[:, k, c], 
                               label=f'Comp {k+1}, Ch {c}', 
                               linewidth=1.5, 
                               alpha=0.7)
        axes[1, 0].set_title('Component Means Evolution (RGB Channels)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 分量标准差
    if 'stds_history' in stats_history and len(stats_history['stds_history']) > 0:
        stds_history = np.array(stats_history['stds_history'])
        for k in range(min(3, stds_history.shape[1])):
            for c in range(min(3, stds_history.shape[2])):
                axes[1, 1].plot(stds_history[:, k, c], 
                               label=f'Comp {k+1}, Ch {c}', 
                               linewidth=1.5, 
                               alpha=0.7)
        axes[1, 1].set_title('Component Std Dev Evolution (RGB Channels)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Std Dev')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"GMM training statistics saved to {save_path}")

def save_sample_images(images, labels, predictions, save_dir='./samples', prefix='sample'):
    """
    保存样本图像及其预测结果
    
    Args:
        images: 图像张量 [B, C, H, W]
        labels: 真实标签
        predictions: 预测标签
        save_dir: 保存目录
        prefix: 文件前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    
    B = images.shape[0]
    for i in range(min(B, 20)):  # 保存最多20个样本
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        is_correct = (labels[i] == predictions[i])
        status = 'correct' if is_correct else 'incorrect'
        filename = f"{prefix}_{i:03d}_{status}_true{labels[i]}_pred{predictions[i]}.png"
        
        Image.fromarray(img).save(os.path.join(save_dir, filename))
    
    print(f"Saved {min(B, 20)} sample images to {save_dir}")

def compute_per_class_accuracy(preds, labels, num_classes):
    """
    计算每个类别的准确率
    
    Args:
        preds: 预测标签列表
        labels: 真实标签列表
        num_classes: 类别数量
    
    Returns:
        per_class_acc: 每个类别的准确率列表
        class_counts: 每个类别的样本数量
    """
    per_class_acc = []
    class_counts = []
    
    for c in range(num_classes):
        mask = (np.array(labels) == c)
        if np.sum(mask) == 0:
            per_class_acc.append(0.0)
            class_counts.append(0)
        else:
            correct = np.sum(np.array(preds)[mask] == c)
            total = np.sum(mask)
            per_class_acc.append(correct / total * 100.0)
            class_counts.append(total)
    
    return per_class_acc, class_counts

def plot_per_class_accuracy(per_class_acc, class_counts, class_names=None, save_path='per_class_accuracy.png'):
    """
    绘制每个类别的准确率
    
    Args:
        per_class_acc: 每个类别的准确率列表
        class_counts: 每个类别的样本数量
        class_names: 类别名称列表
        save_path: 保存路径
    """
    num_classes = len(per_class_acc)
    indices = np.arange(num_classes)
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # 准确率柱状图
    colors = ['green' if acc >= 70 else 'orange' if acc >= 50 else 'red' for acc in per_class_acc]
    bars = ax1.bar(indices, per_class_acc, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Class ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices if class_names is None else class_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 样本数量折线图
    ax2 = ax1.twinx()
    ax2.plot(indices, class_counts, color='blue', marker='o', linewidth=2, markersize=6, label='Sample Count')
    ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # 添加图例
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class accuracy plot saved to {save_path}")

def create_training_curves(train_losses, train_accs, val_accs, target_accs, save_path='training_curves.png'):
    """
    创建训练曲线图
    
    Args:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        target_accs: 目标域准确率列表（每5个epoch）
        save_path: 保存路径
    """
    epochs = len(train_losses)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 损失曲线
    ax1.plot(range(1, epochs+1), train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 准确率曲线
    ax2.plot(range(1, epochs+1), train_accs, 'g-', linewidth=2, label='Source Train Acc')
    ax2.plot(range(1, epochs+1), val_accs, 'c-', linewidth=2, label='Source Val Acc')
    
    # 目标域准确率（每5个epoch）
    target_epochs = [i*5+1 for i in range(len(target_accs))]
    ax2.plot(target_epochs, target_accs, 'r-o', linewidth=2, markersize=6, label='Target Test Acc')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def visualize_pixel_assignments(assignments, image=None, save_path='pixel_assignments.png'):
    """
    可视化单个图像的像素分配
    
    Args:
        assignments: 像素分配 [H, W]
        image: 可选的原始图像 [C, H, W] 用于叠加显示
        save_path: 保存路径
    """
    H, W = assignments.shape
    
    fig, axes = plt.subplots(1, 2 if image is not None else 1, figsize=(12, 5))
    axes = [axes] if image is None else axes
    
    # 仅显示分配
    im1 = axes[0].imshow(assignments, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Pixel Component Assignments', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 叠加显示（如果有原始图像）
    if image is not None:
        img = image.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 创建叠加图像
        overlay = img.copy()
        unique_assigns = np.unique(assignments)
        
        for k in unique_assigns:
            mask = (assignments == k)
            if np.any(mask):
                color = plt.cm.tab20(int(k) % 20)[:3]
                overlay[mask] = overlay[mask] * 0.6 + np.array(color) * 0.4
        
        axes[1].imshow(overlay)
        axes[1].set_title('Assignments Overlay on Original Image', fontsize=13, fontweight='bold')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Pixel assignments visualization saved to {save_path}")

def rgb_to_lab(rgb):
    """RGB [B,3,H,W] (0~1) -> CIELAB [B,3,H,W]"""
    if rgb.max() > 1.0: rgb = rgb / 255.0  # 兼容 0-255 输入
    
    # 1. sRGB -> Linear RGB
    mask = rgb > 0.04045
    linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # 2. Linear RGB -> XYZ (D65)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    B, C, H, W = rgb.shape
    rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz = torch.matmul(rgb_flat, M.T).reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    # 3. XYZ -> CIELAB
    ref = torch.tensor([0.95047, 1.00000, 1.08883], device=rgb.device, dtype=rgb.dtype)
    xyz_norm = xyz / ref.view(1, 3, 1, 1)
    f = torch.where(xyz_norm > 0.008856, xyz_norm ** (1/3), (903.3 * xyz_norm + 16) / 116)
    
    L = 116 * f[:, 0:1, :, :] - 16
    a = 500 * (f[:, 0:1, :, :] - f[:, 1:2, :, :])
    b = 200 * (f[:, 1:2, :, :] - f[:, 2:3, :, :])
    return torch.cat([L, a, b], dim=1)

def lab_to_rgb(lab):
    """CIELAB [B,3,H,W] -> RGB [B,3,H,W]，训练时允许越界，不默认 clamp 到 [0,1]"""
    L, a, b = lab[:, 0:1, :, :], lab[:, 1:2, :, :], lab[:, 2:3, :, :]
    f_y = (L + 16) / 116
    f_x = a / 500 + f_y
    f_z = f_y - b / 200
    
    delta = 6/29
    f_inv = torch.where(f_x > delta, f_x**3, 3*delta**2*(f_x - 4/29))
    f_inv = torch.cat([f_inv, torch.where(f_y > delta, f_y**3, 3*delta**2*(f_y - 4/29)), 
                        torch.where(f_z > delta, f_z**3, 3*delta**2*(f_z - 4/29))], dim=1)
    
    ref = torch.tensor([0.95047, 1.00000, 1.08883], device=lab.device, dtype=lab.dtype)
    xyz = f_inv * ref.view(1, 3, 1, 1)
    
    M_inv = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], device=lab.device, dtype=lab.dtype)
    
    B, C, H, W = lab.shape
    xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    rgb_linear = torch.matmul(xyz_flat, M_inv.T).reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    # 防止负数直接做幂运算造成 NaN
    rgb_linear_pos = rgb_linear.clamp_min(0.0)

    # Linear RGB -> sRGB
    mask = rgb_linear > 0.0031308
    rgb = torch.where(mask, 1.055 * (rgb_linear ** (1/2.4)) - 0.055, 12.92 * rgb_linear)
    # 改变正常的lab到rgb转换流程，增加平滑压制，避免 clamp 导致的死板边界问题
    # return torch.clamp(rgb, 0.0, 1.0)

    # rgb 是计算出来的可能越界的张量
    # 方案：用平滑曲线替代粗暴 clamp
    # 将中心点 0.5 保持不变，两端用 Tanh 平滑压制
    # rgb_shifted = (rgb - 0.5) * 2.0  # 映射到大概 [-1, 1] 及其以外
    # rgb_squashed = torch.tanh(rgb_shifted) # 平滑压缩到 [-1, 1] 绝对范围内
    # rgb_final = rgb_squashed * 0.5 + 0.5 # 映射回 [0, 1]

    return rgb
    
def soft_gamut_compress(rgb, delta=0.05):
    y = rgb.clone()

    low = rgb < delta
    y[low] = delta * torch.exp((rgb[low] - delta) / delta)

    high = rgb > 1.0 - delta
    y[high] = 1.0 - delta * torch.exp(-(rgb[high] - (1.0 - delta)) / delta)

    return y