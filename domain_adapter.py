import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pixel_gmm import PixelGaussianMixture
from style_transfer import PixelStyleTransfer
# 导入 transform
from torchvision import transforms

class GMMStyleDomainAdapter:
    """
    基于像素级GMM的域自适应训练器
    工作流程:
    1. 用目标域无标签数据初始化PixelGMM
    2. 每个epoch:
       a. 用源域有标签数据训练分类器
       b. 对源域图像进行像素级风格迁移
       c. 用风格迁移后的图像增强训练
       d. 定期用新目标域数据更新GMM
    """
    def __init__(self, model, config, target_gmm=None):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        self.conf_threshold = 0.95  # 置信度阈值，建议 0.95
        self.lambda_target = 0.1    # 目标域损失的权重
        self.style_mode = getattr(config, 'style_mode', 'pixel') # 风格迁移模式
        self.lambda_pixel = getattr(config, 'lambda_pixel', config.lambda_div) # 像素级风格损失权重 lambda_div 默认为1.0

        # 🆕 新增：全局 Batch 计数器（用于动态 Alpha 的倒余弦调度）
        self.global_step = 0

        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.SGD( # 建议改用 SGD
            self.model.parameters(),
            lr=config.lr, momentum=0.9, weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.lr_step_size if hasattr(config, 'lr_step_size') else 10,
            gamma=0.1
        )
        
        if target_gmm is None:
                # 默认先构建一个目标域 GMM（未初始化，由训练首批触发 fit）
            target_gmm = PixelGaussianMixture(
                num_components=config.num_gaussians,
                feature_dim=3,
                device=config.device,
                covariance_type=config.covariance_type,
                fit_iters=config.gmm_iters,
                tol=config.gmm_convergence_threshold,
            )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # GMM模型 (目标域)
        self.target_gmm = target_gmm
        
        # 风格迁移模块
        self.style_transfer = PixelStyleTransfer(
            num_components=self.target_gmm.num_components,
            eps=1e-6,
            alpha=config.style_alpha # 从配置读取 0.5
        ).to(self.device)
        
        # 目标域统计量 (用于风格迁移)
        self.target_stats = None
        
        # 训练状态
        self.best_accuracy = 0.0
        
        # 1. 定义 ImageNet 标准化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _current_style_alpha(self, global_step, warmup_batches):
        """
        Warm-up & Plateau 策略：平移缩放后的倒余弦曲线
        t <= T_warmup:  S型平滑过渡 (初期慢 -> 中期快 -> 后期平缓触顶)
        t > T_warmup:   锁定为 end_alpha (平台期，让网络在纯正目标域风格下充分训练)
        """
        start_alpha = getattr(self.config, 'style_alpha_start', 0.3)
        end_alpha   = getattr(self.config, 'style_alpha_end', 0.8)
        
        if global_step >= warmup_batches:
            return end_alpha  # ✅ 进入平台期，保持高强度风格注入
            
        import math
        t_ratio = global_step / warmup_batches
        # α(t) = α_start + 0.5*(α_end - α_start) * (1 - cos(π * t/T))
        return start_alpha + 0.5 * (end_alpha - start_alpha) * (1 - math.cos(t_ratio * math.pi))

    def _prepare_data(self, images):
        """
        统一数据预处理：将 4D 图像/特征张量展平为 GMM 要求的 2D 矩阵 [N, D]
        Args:
            images: [B, C, H, W] 的输入张量（源域或目标域通用）
        Returns:
            data_2d: [N, D] 的展平矩阵，N = B×H×W（或特征图的空间乘积）
        """
        if self.style_mode == 'pixel':
            # 像素级：直接展平空间维度，保留通道
            return images.permute(0, 2, 3, 1).reshape(-1, images.shape[1])
        else:
            # 特征级：通过 Backbone 提取特征后展平
            with torch.no_grad():
                feats = self.backbone(images)
                return feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
    
    def train_epoch(self, source_loader, target_loader, epoch):
        """像素级训练策略: CIELAB 空间 + AdaIN + 软分配"""
        self.model.train() # 设置为训练模式
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        # 目标域迭代器 (用于定期更新GMM)
        target_iter = iter(target_loader) # target_train_loader没有标签
        
        # 训练进度
        # 如果 source_loader 原本产出的是 (图片，标签，...)，加上 enumerate 后，产出的数据变成了 (索引，(图片，标签，...))。
        # total=len(source_loader):告诉 tqdm 总共需要迭代多少次。
        # desc 是 "description" 的缩写，设置进度条左侧的描述文字。
        total_batches = len(source_loader)
        pbar = tqdm(enumerate(source_loader), total=total_batches, desc=f'Epoch {epoch+1}')
        
        # 从 enumerate(source_loader) 中取出一个元组，结构是：(batch_idx, batch_data)。
        for batch_idx, (source_images, labels, _, _) in pbar:
            self.global_step += 1
            source_images, labels = source_images.to(self.device), labels.to(self.device)

            try:
                # 获取一个 batch 的目标域数据
                target_images, _, _, _ = next(target_iter) # target_train_loader没有标签
            except StopIteration:
                # 如果遍历完了，重新开始
                target_iter = iter(target_loader)
                target_images, _, _, _ = next(target_iter)
            target_images = target_images.to(self.device)

            # ── 1. 🎨 RGB -> CIELAB (确保通道独立，完美适配 diag 协方差) ──
            source_lab_4d = self._rgb_to_lab(source_images)
            target_lab_4d = self._rgb_to_lab(target_images)

            # ── 1. 统一数据展平 [B,C,H,W] -> [N,D] ──
            source_lab = self._prepare_data(source_lab_4d) # [N, 3] Lab空间像素
            target_lab = self._prepare_data(target_lab_4d)

            # ── 2. 🎯 动态 Alpha 调度 (课程学习核心) ──
            # 随着 batch 推进，GMM 逐渐稳定，alpha 线性增加
            # 早期小 alpha 保护语义结构，后期大 alpha 充分注入目标域风格
            style_alpha = self._current_style_alpha(self.global_step, total_batches)

            if not self.target_gmm.initialized:
                # 【Batch 0】首次初始化：完整 EM 拟合，建立目标域分布先验
                print("\n[Train] Warmup EM on first target batch...")
                self.target_gmm.fit(target_lab, fit_iters=getattr(self.config, 'gmm_iters', 20))
                print("[Train] GMM initialized successfully.")
            
            # 3. 🎯 直接获取最终 GMM 参数（零计算开销）
            self.target_stats = self.target_gmm.get_target_parameters()

            # ✅ 【关键修复】从对角协方差中提取标准差 σ = sqrt(diag(Σ))
            cov_t = self.target_stats['covariances']
            self.target_stats['stds'] = torch.sqrt(torch.diagonal(cov_t, dim1=1, dim2=2).clamp(min=1e-6))
        
            # ── 4. 源域责任矩阵计算 (使用当前最新的目标域 GMM) ──
            source_resp, _ = self.target_gmm.e_step(source_lab)  # [N, K]

            # 源域成分级统计量 (M步计算，update_params=False 绝不污染目标域GMM)
            source_means, source_covs, source_weights = self.target_gmm.m_step(
                source_lab, source_resp, update_params=False
            )
            source_stds = torch.sqrt(torch.diagonal(source_covs, dim1=1, dim2=2).clamp(min=1e-6))
            
            source_stats = {
                'means': source_means,
                'covariances': source_covs,
                'weights': source_weights,
                'stds': source_stds  # ✅ 补全 style_transfer 期望的键
            }

            B, C, H, W = source_lab_4d.shape
            K = self.target_gmm.num_components
            source_resp_4d = source_resp.reshape(B, H, W, K)

            # 4. 风格迁移（直接传入计算好的统计量）
            styled_lab = self.style_transfer(
                source_lab_4d,
                source_resp_4d,
                source_stats,
                self.target_stats,
                alpha=style_alpha
            )

            # ── 6. 🎨 CIELAB -> RGB (恢复原始空间供分类网络使用) ──
            styled_images = self._lab_to_rgb(styled_lab)

            # ── 5. 梯度清空 ──
            self.optimizer.zero_grad()
            loss_pixel = torch.tensor(0.0, device=self.device)

            # ── 6. 原始源域分类损失 ──
            source_inputs = self.normalize(source_images)
            source_features = self.model.extract_features(source_inputs)
            logits_src = self.model.classify_features(source_features)
            loss_src = self.criterion(logits_src, labels)

            styled_inputs = self.normalize(styled_images)
            logits_pixel, _ = self.model(styled_inputs)
            loss_pixel = self.criterion(logits_pixel, labels)
            
            # ── 8. 目标域伪标签损失 ──
            target_inputs = self.normalize(target_images)
            # 1. 计算目标域 logits
            logits_tgt, _ = self.model(target_inputs)
            # 2. 计算概率
            probs_tgt = torch.softmax(logits_tgt, dim=1)
            # 3. 获取最大概率和对应的预测类别
            max_probs, tgt_preds = torch.max(probs_tgt, dim=1)
            # 4. 生成掩码：只选择置信度 > 0.95 的样本
            mask = max_probs.ge(self.conf_threshold).float()
            # 5. 计算损失 (只计算高置信度样本)
            # reduction='none' 是为了保留每个样本的损失，以便乘以 mask
            # loss_target_raw = nn.CrossEntropyLoss(reduction='none')(logits_target, target_preds)
            # loss_target = (loss_target_raw * mask).mean()
            loss_tgt_raw = nn.CrossEntropyLoss(reduction='none')(logits_tgt, tgt_preds)
            loss_tgt = (loss_tgt_raw * mask).sum() / (mask.sum() + 1e-8)

            # ── 9. 总损失合成 ──
            loss = loss_src + self.lambda_target * loss_tgt + self.lambda_pixel * loss_pixel

            # ── 10. 反向传播 & 优化 ──
            loss.backward()
            self.optimizer.step()
            
            # ── 11. 统计 & 进度条 ──
            total_loss += loss.item()
            _, predicted = torch.max(logits_src.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            pbar.set_postfix({
                'Loss': f"{total_loss/(batch_idx+1):.4f}",
                'Acc': f"{100.*total_correct/total_samples:.2f}%",
                'a': f"{style_alpha:.2f}",
                'Lp': f"{loss_pixel.item():.3f}"
            })

            # 【Batch ≥ 1】在线 EMA 更新：逐步追踪目标域分布漂移
            # E步：获取当前目标批次的责任矩阵
            resp_t, _ = self.target_gmm.e_step(target_lab)
            # M步：仅计算新参数，绝不覆盖 self.*
            new_means, new_covs, new_weights = self.target_gmm.m_step(target_lab, resp_t, update_params=False)
            
            # EMA 平滑融合 (tau=0.99 保证历史分布占主导，抗单批次噪声)
            tau = getattr(self.config, 'gmm_ema_tau', 0.99)
            self.target_gmm.means       = tau * self.target_gmm.means       + (1 - tau) * new_means
            self.target_gmm.covariances = tau * self.target_gmm.covariances + (1 - tau) * new_covs
            self.target_gmm.weights     = tau * self.target_gmm.weights     + (1 - tau) * new_weights

        return total_loss / len(source_loader), 100. * total_correct / total_samples
    
    def validate(self, loader, domain_name='target'):
        """
        在指定域上验证模型
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, _, _ in tqdm(loader, desc=f'Validating {domain_name}'):
                images = images.to(self.device)
                images = self.normalize(images)
                labels = labels.to(self.device)
                
                outputs, _ = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = 100. * total_correct / total_samples
        print(f"{domain_name.capitalize()} domain accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def train(self, source_train_loader, source_val_loader, 
             target_train_loader, target_test_loader):
        """
        完整训练流程
        """
        # 2. 训练循环
        for epoch in range(self.config.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(
                source_train_loader, 
                target_train_loader, 
                epoch
            )
            
            # 源域验证 (仅用于监控)
            source_acc = self.validate(source_val_loader, 'source')
            
            # 目标域测试 (关键指标)
            if (epoch + 1) % 5 == 0 or epoch == self.config.epochs - 1:
                target_acc = self.validate(target_test_loader, 'target')
                
                # 保存最佳模型
                if target_acc > self.best_accuracy:
                    self.best_accuracy = target_acc
                    self._save_checkpoint(epoch, target_acc, is_best=True)
                    print(f"✓ New best model saved: {target_acc:.2f}%")
            
            # 学习率调度
            self.scheduler.step()
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, self.best_accuracy)
        
        print(f"\n{'='*50}")
        print(f"Training completed. Best target accuracy: {self.best_accuracy:.2f}%")
        print(f"{'='*50}")
        
        return self.best_accuracy
    
    def _save_checkpoint(self, epoch, accuracy, is_best=False):
        """保存模型检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'gmm_means': self.target_gmm.means,
            'gmm_covariances': self.target_gmm.covariances,
            'gmm_weights': self.target_gmm.weights,
            'target_stats': self.target_stats
        }
        
        # 保存最新模型
        filename = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(state, filename)
        
        # 保存最佳模型
        if is_best:
            best_filename = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_filename)

    def _rgb_to_lab(self, rgb):
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

    def _lab_to_rgb(self, lab):
        """CIELAB [B,3,H,W] -> RGB [B,3,H,W] (0~1)"""
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
        
        # Linear RGB -> sRGB
        mask = rgb_linear > 0.0031308
        rgb = torch.where(mask, 1.055 * (rgb_linear ** (1/2.4)) - 0.055, 12.92 * rgb_linear)
        return torch.clamp(rgb, 0.0, 1.0)
    