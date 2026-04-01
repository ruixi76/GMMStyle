import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import tqdm
# 引入 sklearn 进行 K-Means 初始化
from sklearn.cluster import KMeans

class PixelGaussianMixture:
    """
    像素级高斯混合模型 - 纯PyTorch实现，避免CPU/GPU切换
    理论基础: p(x) = Σ_k π_k * N(x|μ_k, Σ_k)
    """
    def __init__(self, num_components=5, feature_dim=3, device='cuda:0', 
                 covariance_type='diag', max_iters=100, tol=1e-4):
        """
        Args:
            num_components: 高斯分量数量k
            feature_dim: 特征维度 (RGB图像为3)
            device: 计算设备
            covariance_type: 'full' (完整协方差) 或 'diag' (对角协方差，推荐)
            max_iters: EM算法最大迭代次数
            tol: 收敛阈值
        """
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.covariance_type = covariance_type
        self.max_iters = max_iters
        self.tol = tol # 收敛阈值
        
        # GMM参数 (全部在GPU上)
        self.means = None          # [k, C] - 每个分量的均值向量
        self.covariances = None    # [k, C, C] - 每个分量的协方差矩阵
        self.weights = None        # [k] - 混合权重
        
        # 初始化为None，等待数据初始化
        self.initialized = False
    
    def _initialize_parameters(self, pixels):
        """
        使用 K-Means 初始化 GMM 参数 (更稳定)
        Args:
            pixels: 像素数据 [N, C]
        """
        print(f"[PixelGMM] Initializing parameters using K-Means (k={self.num_components})...")
        N = pixels.shape[0]
        
        # 1. 转为 CPU numpy 数组以使用 sklearn
        pixels_cpu = pixels.detach().cpu().numpy()
        
        # 2. 运行 K-Means
        # n_init=10 意味着 KMeans 算法内部会尝试 10 次不同的随机初始质心，最终只保留效果最好（误差最小）的那一次结果。
        kmeans = KMeans(n_clusters=self.num_components, n_init=10, random_state=42)
        # fit_predict(...): 这是 K-Means 的核心方法，它同时做了两件事：
        # 1.fit (拟合)：在传入的像素数据 pixels_cpu 上进行迭代，找出最佳的那些聚类中心。
        # 2.predict (预测)：为你传入的每一个像素点分配一个它所属的类别序号（从 0 到 n_clusters - 1）。
        # labels: 这一步返回一个一维数组，里面记录了每个像素点被分配到了哪个聚类簇中。
        labels = kmeans.fit_predict(pixels_cpu) 
        # cluster_centers_: 这是拟合好的 K-Means 模型的一个属性，里面保存了最终计算出的所有簇的中心点坐标。
        # 特征空间的维度是 C（通常是 3），所以 centers 的形状是 (n_clusters, C)，每一行对应一个簇的中心坐标。
        centers = kmeans.cluster_centers_ # centers: 返回一个形状为 (n_clusters, 特征维度) 的数组。
        
        # 3. 初始化均值 (Means)
        self.means = torch.tensor(centers, device=self.device, dtype=pixels.dtype)
        
        # 4. 初始化权重 (Weights)
        # np.bincount 是 NumPy 中用来统计非负整数出现频次的快捷函数。它就像是在做“点名统计”。
        # 输出 counts：bincount 会返回一个新的数组。结果数组的第 0 个位置存放 labels 里 0 出现的次数，第 1 个位置存放 1 出现的次数，依此类推。
        counts = np.bincount(labels, minlength=self.num_components)
        weights = counts / counts.sum()
        self.weights = torch.tensor(weights, device=self.device, dtype=pixels.dtype)
        
        # 5. 初始化协方差 (Covariances)
        # 计算每个簇的经验协方差
        # 既然 'full' 能直接建模相关性，为什么工程上偏偏选 'diag'？
        # 1. 计算量：'full' 需要计算和存储 k 个 CxC 的矩阵，计算协方差时需要 O(N*C^2) 的复杂度；而 'diag' 只需要 O(N*C)。
        # 2. 稳定性：'full' 的协方差矩阵可能会因为样本不足或特征相关性过高而变得奇异（不可逆），导致 EM 算法失败；'diag' 由于假设特征独立，更稳定。
        self.covariances = torch.zeros(self.num_components, self.feature_dim, 
                                        self.feature_dim, device=self.device)

        # 将 labels 转回 GPU 以便计算
        # labels: 这一步返回一个一维数组，里面记录了每个像素点被分配到了哪个聚类簇中。
        labels_torch = torch.tensor(labels, device=self.device)

        for k in range(self.num_components):
            # 选出属于第 k 个簇的像素
            mask = (labels_torch == k) # mask 是一个与 labels_torch 形状相同的布尔张量
            # mask.sum()：在 PyTorch/NumPy 中，布尔值参与数学运算时，True 被视为 1，False 被视为 0。
            if mask.sum() > 1: # 至少有两个点才能算方差
                # 程序将 mask 覆盖在 pixels 的第一维度（行）上。只保留 mask 为 True 的那些行。丢弃 mask 为 False 的那些行。
                # cluster_pixels 是一个新的张量，只包含属于类别 k 的像素数据。
                cluster_pixels = pixels[mask] 
                # 计算方差/协方差
                # 在高斯分布中，协方差矩阵 描述了特征之间的相关性。
                # 'full'：完整矩阵，考虑特征间的相关性（计算量大，参数多）。
                # 'diag'：对角矩阵，假设特征间互不相关（计算快，参数少，本题走这个分支）。
                # 'spherical'：球形，假设所有维度方差相同（更简化）。
                if self.covariance_type == 'diag': # 假设协方差矩阵的类型是对角矩阵（Diagonal），这意味着假设各个特征维度之间是相互独立的。
                    var = torch.var(cluster_pixels, dim=0, unbiased=True) # var 是一个长度为 pixel_dim 的向量，包含了每个特征维度的波动程度。
                    # var.clamp方法：将 var 中所有小于 1e-4（0.0001）的值强制设为 1e-4
                    # 将一维向量转换为二维对角矩阵：torch.diag(var) 会创建一个对角矩阵，其中对角线上的元素是 var 中的值，非对角线上的元素为 0。
                    self.covariances[k] = torch.diag(var.clamp(min=1e-4))  
                else:
                    centered = cluster_pixels - self.means[k]
                    cov = torch.matmul(centered.T, centered) / (mask.sum() - 1)
                    self.covariances[k] = cov + 1e-6 * torch.eye(self.feature_dim, device=self.device)
            else:
                # 兜底：如果簇内样本太少，使用全局方差或单位阵
                self.covariances[k] = torch.eye(self.feature_dim, device=self.device) * 0.01

        self.initialized = True
        print(f"[PixelGMM] Initialization complete on {self.device}")
    
    def e_step(self, pixels):
        """
        E步: 计算每个像素属于各高斯分量的后验概率
        Args:
            pixels: 像素数据 [N, C]
        Returns:
            responsibilities: 责任矩阵 [N, k]
            log_likelihood: 对数似然 (标量)
        """
        N = pixels.shape[0]
        log_probs = torch.zeros(N, self.num_components, device=self.device) # 计算每个像素属于各高斯分量的后验概率
        
        # 计算每个像素到各高斯分量的对数概率
        for k in range(self.num_components):
            # 创建多变量正态分布
            # 注意: PyTorch的MultivariateNormal需要协方差矩阵可逆
            try:
                dist = MultivariateNormal(self.means[k], self.covariances[k] + 1e-6 * torch.eye(self.feature_dim, device=self.device))
                # log_prob：N维向量，表示每个像素在第 k 个高斯分量下的对数概率密度值。这个值越大（越接近0），说明像素越可能属于这个分量；反之，值越小（越负），说明像素越不可能属于这个分量。
                log_prob = dist.log_prob(pixels)  # [N, C] - 每个像素的对数概率，每个像素其实是3维的，返回的是每个像素（rgb整个）的对数概率值
            except:
                # 协方差矩阵奇异时的回退方案
                diff = pixels - self.means[k]
                if self.covariance_type == 'diag':
                    inv_cov = 1.0 / (torch.diagonal(self.covariances[k]) + 1e-8)
                    log_det = torch.sum(torch.log(torch.diagonal(self.covariances[k]) + 1e-8))
                    mahalanobis = torch.sum(diff * diff * inv_cov, dim=1)
                else:
                    # 使用Cholesky分解计算马氏距离
                    try:
                        L = torch.linalg.cholesky(self.covariances[k] + 1e-6 * torch.eye(self.feature_dim, device=self.device))
                        y = torch.linalg.solve_triangular(L, diff.T, upper=False)
                        mahalanobis = torch.sum(y.T ** 2, dim=1)
                        log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
                    except:
                        # 最后的回退: 使用欧氏距离
                        mahalanobis = torch.sum(diff ** 2, dim=1)
                        log_det = self.feature_dim * np.log(2 * np.pi)
                
                log_prob = -0.5 * (self.feature_dim * np.log(2 * np.pi) + log_det + mahalanobis)
            
            # 加上混合权重的对数
            log_probs[:, k] = log_prob + torch.log(self.weights[k] + 1e-10) # [N,k]
        
        # 数值稳定的归一化 (log-sum-exp trick)
        # torch.max 带 dim 参数时会返回一个元组 (values, indices)，其中 values 是沿指定维度的最大值，indices 是最大值所在的位置索引。
        max_log_prob = torch.max(log_probs, dim=1, keepdim=True)[0] # 取的是其中的 values 部分，其形状是 [N, 1]
        probs = torch.exp(log_probs - max_log_prob) # [N,k]
        #  (torch.sum(probs, dim=1, keepdim=True)得到维度为[N,1]的张量，表示每个像素在所有高斯分量下的概率总和。
        # 对于第n个特征像素，它到底有百分之几的概率是由第 k 个高斯分布（风格成分）生成的？
        responsibilities = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-10) # [N,k] - 每个像素属于每个分量的责任值（后验概率）
        
        # 计算总对数似然
        log_likelihood = torch.sum(max_log_prob.squeeze() + 
                                  torch.log(torch.sum(probs, dim=1) + 1e-10))
        # 使用 .item() 后打印输出会是干净的 12.3456，而不是带有张量标记的 tensor(12.3456, device='cuda:0')
        return responsibilities, log_likelihood.item() 
    
    def m_step(self, pixels, responsibilities):
        """
        M步: 基于责任矩阵更新GMM参数
        Args:
            pixels: 像素数据 [N, C]
            responsibilities: 责任矩阵 [N, k]
        """
        N = pixels.shape[0]
        new_means = torch.zeros_like(self.means)
        new_covariances = torch.zeros_like(self.covariances)
        new_weights = torch.zeros_like(self.weights)
        
        for k in range(self.num_components):
            resp_k = responsibilities[:, k]  # [N]
            Nk = torch.sum(resp_k) + 1e-10
            
            # 更新均值: μ_k = Σ_n γ_k(x_n) * x_n / Σ_n γ_k(x_n)
            # resp_k.unsqueeze(1) 会在第 1 个维度（从 0 开始数）插进一个新维度。
            # 结果的维度会从 [N] 变成 [N, 1]（变成了一个由 N 行、每行 1 个元素组成的列向量）。
            new_means[k] = torch.sum(resp_k.unsqueeze(1) * pixels, dim=0) / Nk # new_means[k]为C维的向量，表示第 k 个分量的均值。它是通过对所有像素进行加权平均计算得到的，其中权重是每个像素属于第 k 个分量的责任值 resp_k。
            
            # 更新协方差矩阵: Σ_k = Σ_n γ_k(x_n) * (x_n-μ_k)(x_n-μ_k)^T / Σ_n γ_k(x_n)
            diff = pixels - new_means[k] # diff 是一个形状为 [N, C] 的张量，表示每个像素与第 k 个分量的均值之间的差异。对于第 n 个像素，diff[n] 就是该像素与均值 new_means[k] 之间的差向量。
            
            if self.covariance_type == 'diag':
                # 仅更新对角元素 (通道独立假设)
                # diff ** 2 是 PyTorch 的逐元素平方
                weighted_var = torch.sum(resp_k.unsqueeze(1) * (diff ** 2), dim=0) / Nk
                new_covariances[k] = torch.diag(weighted_var.clamp(min=1e-4))
            else:
                # 完整协方差矩阵
                weighted_outer = torch.zeros(self.feature_dim, self.feature_dim, device=self.device)
                for i in range(min(N, 1000)):  # 限制计算量
                    if resp_k[i] > 1e-8:
                        outer = torch.outer(diff[i], diff[i])
                        weighted_outer += resp_k[i] * outer
                new_covariances[k] = weighted_outer / Nk + 1e-6 * torch.eye(self.feature_dim, device=self.device)
            
            # 更新权重: π_k = Σ_n γ_k(x_n) / N
            new_weights[k] = Nk / N
        
        # 更新参数
        self.means = new_means.detach()
        self.covariances = new_covariances.detach()
        self.weights = new_weights.detach()
    
    def fit(self, pixels, max_iters=None):
        """
        使用EM算法拟合GMM
        Args:
            pixels: 像素数据 [N, C]
            max_iters: 最大迭代次数 (覆盖初始化时的设置)
        Returns:
            assignments: 每个像素的分量分配 [N]
        """
        if not self.initialized:
            self._initialize_parameters(pixels)
        
        max_iters = max_iters or self.max_iters
        prev_log_likelihood = float('-inf') # 负无穷大
        log_likelihoods = []
        
        print(f"[PixelGMM] Running EM algorithm for {max_iters} iterations...")

        pbar= tqdm(range(max_iters), desc='EM Iterations')
        for i in pbar:
            # E步
            responsibilities, log_likelihood = self.e_step(pixels)
            log_likelihoods.append(log_likelihood)
            
            # 检查收敛
            if i > 0 and abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"EM converged at iteration {i+1}")
                break
            
            prev_log_likelihood = log_likelihood
            
            # M步：根据E步计算的责任矩阵更新GMM参数（均值，协方差，权重）
            self.m_step(pixels, responsibilities)
        
        # 最终分配: 每个像素分配到概率最高的分量
        # 硬分配：尝试着改一下试试
        assignments = torch.argmax(responsibilities, dim=1)  # [N]
        print(f"[PixelGMM] EM completed. Final log-likelihood: {log_likelihood:.4f}")
        # np.bincount 是 NumPy 中用来统计非负整数出现频次的快捷函数。它就像是在做“点名统计”。
        # torch.bincount 是 PyTorch 中的一个函数，用于统计非负整数在一个一维张量中出现的频次。它的用法和 NumPy 的 np.bincount 类似，但适用于 PyTorch 张量。
        # 例如，如果你有一个张量 assignments，其中包含了每个像素被分配到的分量索引（从 0 到 k-1），你可以使用 torch.bincount(assignments) 来统计每个分量被分配到的像素数量。
        # minlength=self.num_components 参数确保输出的计数数组至少有 num_components 个元素，即使某些分量没有被分配到任何像素，也会在对应位置返回 0。
        print(f"Component assignments: {torch.bincount(assignments, minlength=self.num_components).cpu().numpy()}")
        
        return assignments
    
    def predict(self, pixels):
        """
        预测像素的分量分配
        Args:
            pixels: 像素数据 [N, C] 或 [B, C, H, W]
        Returns:
            assignments: 分量分配 [N] 或 [B, H, W]
            probabilities: 概率分布 [N, k] 或 [B, H, W, k]
        """
        # 处理图像格式输入
        if len(pixels.shape) == 4:  # [B, C, H, W]
            B, C, H, W = pixels.shape
            pixels_flat = pixels.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            is_image = True
        else:
            pixels_flat = pixels
            is_image = False
        
        # E步计算责任矩阵
        responsibilities, _ = self.e_step(pixels_flat)
        assignments_flat = torch.argmax(responsibilities, dim=1) # N = B * H * W
        
        # 恢复原始形状
        if is_image:
            assignments = assignments_flat.reshape(B, H, W)
            probabilities = responsibilities.reshape(B, H, W, self.num_components)
            return assignments, probabilities
        else:
            return assignments_flat, responsibilities
    
    def get_component_statistics(self, pixels, assignments):
        """
        计算每个GMM分量的风格统计量 (用于风格迁移)
        注意: 这些统计量不同于GMM参数!
        Args:
            pixels: 像素数据 [N, C]
            assignments: 像素分配 [N]
        Returns:
            component_stats: {
                'means': [k, C] - 每个分量的通道均值,
                'stds': [k, C] - 每个分量的通道标准差
            }
        """
        component_stats = {
            'means': torch.zeros(self.num_components, self.feature_dim, device=self.device),
            'stds': torch.zeros(self.num_components, self.feature_dim, device=self.device),
            'counts': torch.zeros(self.num_components, device=self.device)
        }
        
        for k in range(self.num_components):
            mask = (assignments == k)
            count_k = torch.sum(mask.float())
            component_stats['counts'][k] = count_k
            
            if count_k > 0:
                pixels_k = pixels[mask]
                # 通道级均值 (风格统计量)
                component_stats['means'][k] = torch.mean(pixels_k, dim=0)
                # 通道级标准差 (风格统计量，从数据计算，非从协方差矩阵提取)
                component_stats['stds'][k] = torch.std(pixels_k, dim=0) + 1e-8
        
        return component_stats
    
    def update_incremental(self, new_pixels, alpha=0.9):
        """
        增量更新GMM参数 (适用于在线学习)
        公式: new_param = alpha * old_param + (1-alpha) * new_param
        Args:
            new_pixels: 新像素数据 [N, C]
            alpha: 混合系数 (0.9表示90%保留历史)
        """
        if not self.initialized:
            self.fit(new_pixels, max_iters=20)
            return
        
        # 1. 为新像素分配分量
        assignments, _ = self.predict(new_pixels)
        
        # 2. 计算新统计量
        new_stats = self.get_component_statistics(new_pixels, assignments)

        # 1. 计算当前批次的总样本数 (所有簇的计数之和)
        # 注意: new_stats['counts'] 是一个 tensor 或 array，包含每个簇在这个 batch 里的像素数
        batch_total_count = torch.sum(new_stats['counts'])
        
        # 防止除以 0 (极少数情况)
        if batch_total_count < 1e-6:
            return  # 本次不做更新
        
        # 3. 指数移动平均更新
        for k in range(self.num_components):
            if new_stats['counts'][k] > 0:
                # 更新均值
                self.means[k] = alpha * self.means[k] + (1 - alpha) * new_stats['means'][k]
                
                # 更新协方差 (简化: 仅更新对角元素)
                if self.covariance_type == 'diag':
                    new_var = new_stats['stds'][k] ** 2
                    # torch.diagonal 的作用是提取一个矩阵的主对角线元素，并将其返回成一个一维（1D）张量。
                    old_var = torch.diagonal(self.covariances[k])
                    updated_var = alpha * old_var + (1 - alpha) * new_var
                    self.covariances[k] = torch.diag(updated_var)
                
                # 更新权重
                current_batch_weight = new_stats['counts'][k] / batch_total_count
                self.weights[k] = self.weights[k] * alpha + current_batch_weight * (1 - alpha)
        
        # 3. 归一化 (Normalization) - 非常重要！
        # 经过多次独立更新后，权重之和可能略微偏离 1.0，必须重新归一化
        self.weights = self.weights / self.weights.sum()
        print(f"[PixelGMM] Incremental update completed with alpha={alpha}")