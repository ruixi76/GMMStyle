# Pixel-level GMM Domain Adaptation

本项目实现了基于像素级高斯混合模型（GMM）的域适应方法，支持多种数据集（Office-31, PACS, VisDA, Digits）和骨干网络（ResNet18/50, VGG16）。

## 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA (可选，用于 GPU 加速)

## 快速开始

### 1. 安装依赖

使用 `requirements.txt` 一键安装所有依赖：

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将数据集放置在 `data_root` 指定的目录下（默认为 `/home/amax/paperProject/GMM/data`），或修改 `--data_root` 参数指向你的数据集路径。

支持的数据集：
- **Office-31**: amazon, dslr, webcam
- **PACS**: art_painting, cartoon, sketch, photo
- **VisDA** 和 **Digits** 数据集

### 3. 训练模型

#### 基本训练命令

```bash
python main.py --source_domain amazon --target_domain dslr --num_gaussians 5 --batch_size 32 --epochs 50 --device cuda:0 --checkpoint_dir ./checkpoints --log_dir ./logs
```

#### 典型实验配置

```bash
# 实验1: Amazon → DSLR (标准配置)
python main.py --source_domain amazon --target_domain dslr --num_gaussians 5 --device cuda:0

# 实验2: Amazon → Webcam (更具挑战性)
python main.py --source_domain amazon --target_domain webcam --num_gaussians 7 --device cuda:0

# 实验3: 小批量快速验证 (调试用)
python main.py --source_domain amazon --target_domain dslr --batch_size 16 --epochs 10 --device cuda:0

# 实验4: PACS 数据集 (art_painting, cartoon, sketch → photo)
python main.py --dataset pacs --source_domain art_painting --target_domain photo --num_gaussians 5 --device cuda:0
```

### 4. 模型评估

训练完成后，使用以下命令评估最佳模型：

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth --source_domain amazon --target_domain dslr --device cuda:0
```

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source_domain` | amazon | 源域名称 |
| `--target_domain` | dslr | 目标域名称 |
| `--data_root` | /home/amax/paperProject/GMM/data | 数据集根目录 |
| `--dataset` | office31 | 数据集名称 (office31/pacs/visda/digits) |
| `--backbone` | resnet50 | 骨干网络 (resnet18/resnet50/vgg16) |
| `--num_gaussians` | 5 | GMM 高斯分量数量 |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 50 | 训练轮数 |
| `--device` | cuda:0 | 计算设备 (cuda:0/cpu) |
| `--checkpoint_dir` | ./checkpoints | 模型保存目录 |
| `--log_dir` | ./logs | 日志保存目录 |

## 项目结构

```
.
├── main.py              # 主训练入口
├── config.py            # 配置解析
├── model.py             # 模型定义
├── domain_adapter.py    # 域适配器核心逻辑
├── pixel_gmm.py         # 像素级 GMM 实现
├── mixstyle.py          # MixStyle 风格迁移
├── style_transfer.py    # 风格迁移模块
├── datasets.py          # 数据加载
├── evaluate.py          # 模型评估
├── utils.py             # 工具函数
├── requirements.txt     # 依赖列表
└── README.md            # 本文件
```

## 引用

如果本项目对您的研究有帮助，请考虑引用相关论文。
