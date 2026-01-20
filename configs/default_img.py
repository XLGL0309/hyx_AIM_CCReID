import os
import yaml
from yacs.config import CfgNode as CN
import time


_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = 'E:/PythonProjects/paper/AIM-CCReID/data/datasets'
# Dataset for evaluation
_C.DATA.DATASET = 'ltcc'
# Workers for dataloader
# 适配Windows环境+减少显存占用
_C.DATA.NUM_WORKERS = 0
# Height of input image
_C.DATA.HEIGHT = 256
# Width of input image
_C.DATA.WIDTH = 128
# Batch size for training
_C.DATA.TRAIN_BATCH = 32
# Batch size for testing
_C.DATA.TEST_BATCH = 64
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 8
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random crop prob (提升至0.5，增强数据泛化能力)
_C.AUG.RC_PROB = 0.5
# Random erase prob (提升至0.5，增强数据泛化能力)
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'resnet50'
# The stride for layer4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim (保持4096，保证模型特征表达能力)
_C.MODEL.FEATURE_DIM = 4096
# Model path for resuming
_C.MODEL.RESUME = ''
# Global pooling after the backbone
_C.MODEL.POOLING = CN()
# Choose in ['avg', 'max', 'gem', 'maxavg']
_C.MODEL.POOLING.NAME = 'maxavg'
# Initialized power for GeM pooling
_C.MODEL.POOLING.P = 3
# -----------------------------------------------------------------------------
# Losses for training
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropylabelsmooth'
# Clothes classification loss
_C.LOSS.CLOTHES_CLA_LOSS = 'cosface'
# Scale for classification loss
_C.LOSS.CLA_S = 16.
# Margin for classification loss
_C.LOSS.CLA_M = 0.
# Clothes-based adversarial loss
_C.LOSS.CAL = 'cal'
# Epsilon for clothes-based adversarial loss
_C.LOSS.EPSILON = 0.05
# Momentum for clothes-based adversarial loss with memory bank
_C.LOSS.MOMENTUM = 0.
# 补充缺失的 PAIR_LOSS 配置项
_C.LOSS.PAIR_LOSS = 'triplet'
# 补充缺失的 PAIR_M 配置项
_C.LOSS.PAIR_M = 0.3
# 补充缺失的 PAIR_LOSS_WEIGHT 参数
_C.LOSS.PAIR_LOSS_WEIGHT = 0.1
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
# 最大训练轮数，兼顾效果与耗时
_C.TRAIN.MAX_EPOCH = 80
# Start epoch for clothes classification
_C.TRAIN.START_EPOCH_CC = 25
# Start epoch for adversarial training
_C.TRAIN.START_EPOCH_ADV = 30
# Start epoch for debias
_C.TRAIN.START_EPOCH_GENERAL = 25
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 8e-5
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Stepsize to decay learning rate (延后至40、60，匹配80Epoch训练周期)
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [40, 60]
# LR decay rate， used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 5
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 1
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0'
# Path to output folder (修正路径拼写，匹配实际项目)
_C.OUTPUT = 'E:/PythonProjects/paper/AIM-CCReID/results'
# Tag of experiment
_C.TAG = 'eval_single_gpu_3060'
# -----------------------------------------------------------------------------
# Hyperparameters
# 降低超参数权重，减少计算量和显存占用
_C.k_cal = 0.5
_C.k_kl = 0.5

# -----------------------------------------------------------------------------
def update_config(config, args):
    config.defrost()

    # ==========  优化：复用yacs原生merge_from_file，规范yaml加载逻辑 ==========
    # 移除自定义yaml读取，使用原生方法（自带文件校验、编码处理）
    config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.root:
        config.DATA.ROOT = args.root
    if args.output:
        config.OUTPUT = args.output
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    if args.tag:
        config.TAG = args.tag
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.gpu:
        # 强制转换GPU参数类型为字符串，避免yacs类型冲突
        config.GPU = str(args.gpu)

    # ==========  核心修改：按数据集分类生成独立编号 ==========
    # 1. 基础路径（results根目录）
    base_output = config.OUTPUT
    os.makedirs(base_output, exist_ok=True)  # 确保基础路径存在

    # 2. 先创建数据集根目录（results/prcc 或 results/ltcc）
    dataset_dir = os.path.join(base_output, config.DATA.DATASET)
    os.makedirs(dataset_dir, exist_ok=True)

    # 3. 扫描该数据集目录下的数字文件夹，获取最大编号（仅统计当前数据集的）
    num_folders = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        # 只筛选纯数字的文件夹（1、2、3...）
        if os.path.isdir(item_path) and item.isdigit():
            num_folders.append(int(item))

    # 4. 确定新文件夹编号（无数字文件夹则从1开始）
    if num_folders:
        new_folder_num = max(num_folders) + 1
    else:
        new_folder_num = 1

    # 5. 构建最终输出路径：results/数据集/数字编号/tag
    final_output = os.path.join(dataset_dir, str(new_folder_num), config.TAG)
    config.OUTPUT = final_output

    config.freeze()

# -----------------------------------------------------------------------------
def get_img_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config