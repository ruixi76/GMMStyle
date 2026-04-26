import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import tqdm
# 引入 sklearn 进行 K-Means 初始化
from sklearn.cluster import KMeans

class PixelGaussianMixture(nn.Module):
    """
    像素级高斯混合模型 - 纯PyTorch实现，避免CPU/GPU切换
    理论基础: p(x) = Σ_k π_k * N(x|μ_k, Σ_k)
    """
    def __init__(self, num_components=5, feature_dim=3, device='cuda:0', 
                 covariance_type='diag', fit_iters=50, tol=1e-4):
        """
        Args:
            num_components: 高斯分量数量k
            feature_dim: 特征维度 (RGB图像为3)
            device: 计算设备
            covariance_type: 'full' (完整协方差) 或 'diag' (对角协方差，推荐)
            fit_iters: EM算法最大迭代次数
            tol: 收敛阈值
        """
        super().__init__()
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.covariance_type = covariance_type
        self.fit_iters = fit_iters # 50
        self.tol = tol # 收敛阈值
        
        # GMM参数 (全部在GPU上)
        self.means = None          # [k, C] - 每个分量的均值向量
        self.covariances = None    # [k, C, C] - 每个分量的协方差矩阵
        self.weights = None        # [k] - 混合权重
        self.last_log_likelihood = None
        
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
    
    def m_step(self, pixels, responsibilities, update_params=True):
        """
        M步: 基于责任矩阵计算新的GMM参数
        
        【核心设计】: 
        - 默认不直接更新 self.means/covariances/weights
        - 通过 update_params 参数控制是否更新内部状态
        - 返回新参数供外部使用（如风格迁移时的统计量提取）
        
        Args:
            pixels: 像素数据 [N, C]
            responsibilities: 责任矩阵 [N, k]
            update_params: 是否更新模型内部参数 (默认True保持fit兼容性)
        
        Returns:
            new_means: [k, C] 新均值
            new_covariances: [k, C, C] 或 [k, C](diag) 新协方差
            new_weights: [k] 新混合权重
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
                # 完整协方差: 向量化计算替代循环 (修复原代码截断问题)
                resp_k_clamped = torch.clamp(resp_k, min=0)  # 确保权重非负
                diff_weighted = diff * resp_k_clamped.unsqueeze(1)  # [N, C]
                cov_k = (diff_weighted.T @ diff) / Nk  # [C, C]
                new_covariances[k] = cov_k + 1e-6 * torch.eye(self.feature_dim, device=self.device)
            
            # 更新权重: π_k = Σ_n γ_k(x_n) / N
            new_weights[k] = Nk / N
            
        # 【关键】: 仅当 update_params=True 时才更新内部状态
        if update_params:
            self.means = new_means.detach()
            self.covariances = new_covariances.detach()
            self.weights = new_weights.detach()

        return new_means, new_covariances, new_weights
    
    def fit(self, pixels, fit_iters=None, return_details=False, compute_hard_assignments=False):
        """
        使用EM算法拟合GMM

        Args:
            pixels: 像素数据 [N, C]
            fit_iters: 模型拟合迭代次数 (覆盖初始化设置)
            return_details: 是否返回责任矩阵等详细信息
            compute_hard_assignments: 是否计算硬分配(argmax)
        
        Returns:
            assignments/responsibilities + 可选的details
        """
        # 初始化gmm参数（如果还没有初始化的话）
        self._initialize_parameters(pixels)
        
        assignments = None
        hard_counts = None  # ✅ 关键修复：提前初始化为 None
        
        # Python 里 or 的返回规则是：左边为真，返回左边；左边为假，返回右边
        # 若fit_iters参数为None，则使用self.fit_iters的值（即初始化时设置的默认值）。如果 fit_iters 参数被显式传入了一个整数值，那么就使用这个值。
        fit_iters = fit_iters or self.fit_iters 
        log_likelihood_old = float('-inf')
        
        print(f"[PixelGMM] Running EM algorithm for {fit_iters} iterations...")

        pbar = tqdm(range(fit_iters), desc='EM Iterations')
        for i in pbar:
            # ── E步: 计算责任矩阵 ──
            responsibilities, log_likelihood_new = self.e_step(pixels)
            
            # ── M步: 计算并更新参数 (fit模式下update_params=True) ──
            self.m_step(pixels, responsibilities, update_params=True)
            
            # 检查收敛
            if i > 0 and abs(log_likelihood_new - log_likelihood_old) < self.tol:
                print(f"EM converged at iteration {i+1}")
                break
            log_likelihood_old = log_likelihood_new

        # ✅ 【关键新增】: 缓存最后一次迭代的责任矩阵
        # 后续计算 target_stats 时可直接使用，避免重复 E 步
        self._cached_responsibilities = responsibilities.detach()
        self._cached_pixels_shape = pixels.shape  # 用于形状校验

        self.last_log_likelihood = log_likelihood_new
        print(f"[PixelGMM] EM completed. Final log-likelihood: {log_likelihood_new:.4f}")

        assignments = None
        if compute_hard_assignments:
            # 仅在显式需要硬分配时才执行 argmax
            # 最终分配: 每个像素分配到概率最高的分量
            assignments = torch.argmax(responsibilities, dim=1)  # [N]
            # np.bincount 是 NumPy 中用来统计非负整数出现频次的快捷函数。它就像是在做“点名统计”。
            # torch.bincount 是 PyTorch 中的一个函数，用于统计非负整数在一个一维张量中出现的频次。它的用法和 NumPy 的 np.bincount 类似，但适用于 PyTorch 张量。
            # 例如，如果你有一个张量 assignments，其中包含了每个像素被分配到的分量索引（从 0 到 k-1），你可以使用 torch.bincount(assignments) 来统计每个分量被分配到的像素数量。
            # minlength=self.num_components 参数确保输出的计数数组至少有 num_components 个元素，即使某些分量没有被分配到任何像素，也会在对应位置返回 0。
            hard_counts = torch.bincount(assignments, minlength=self.num_components)
            print(f"Component assignments: {hard_counts.cpu().numpy()}")
        else:
            # 软分配统计更符合 soft-style transfer 的主流程
            soft_counts = torch.sum(responsibilities, dim=0)
            print(f"Soft component mass: {soft_counts.detach().cpu().numpy()}")
        
        if return_details:
            return assignments, {
                'responsibilities': responsibilities,
                'log_likelihood': log_likelihood_new,
                'hard_counts': hard_counts
            }

        return assignments if compute_hard_assignments else responsibilities
    
    def get_target_parameters(self):
        """
        【核心接口】直接返回当前 GMM 的最终参数（拟合后或更新后）
        无需任何额外计算，直接读取 self.* 即可
        """
        if not self.initialized:
            raise RuntimeError("GMM not initialized. Call fit() first.")
            
        return {
            'means': self.means,          # [K, D]
            'covariances': self.covariances, # [K, D, D] 或 [K, D]
            'weights': self.weights       # [K]
        }

    def get_bic(self, pixels, fit_iters=None): # 15
        """计算当前数据上的 BIC（内部会临时拟合，但不会污染当前模型状态）。"""
        n_samples = pixels.shape[0]

        if n_samples <= 1:
            raise ValueError("BIC requires at least 2 samples.")

        # ── 临时拟合计算似然 ──
        _, details = self.fit(
            pixels,
            fit_iters=fit_iters,
            return_details=True,
            compute_hard_assignments=False
        )
        log_l = details['log_likelihood']

        # ── 计算自由参数数量 m ──
        if self.covariance_type == 'diag':
            cov_params = self.feature_dim  # 对角协方差: D个方差
        else:
            cov_params = self.feature_dim * (self.feature_dim + 1) // 2  # 全协方差: D(D+1)/2
        
        # m = (K-1)权重 + K×(D均值 + 协方差参数)
        m = (self.num_components - 1) + self.num_components * (self.feature_dim + cov_params)

        # ── 计算BIC ──
        return m * np.log(n_samples) - 2.0 * log_l
