import os
import time
import datetime
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
import torchvision
from torchvision import datasets, models, transforms
# 替换为你主训练用的配置文件（避免多一份配置）
from configs.default_img import get_img_config
from models.img_resnet import ResNet50
from PIL import Image


def parse_option():
    parser = argparse.ArgumentParser(
        description='Extract ReID feature for single image')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--img_path', type=str, required=True, help='path to the image')  # 改为必填
    parser.add_argument('--weights', type=str, required=True, help='path to the weights')  # 改为必填
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')

    args, unparsed = parser.parse_known_args()
    config = get_img_config(args)
    return config, args


@torch.no_grad()
def extract_img_feature(model, img):
    flip_img = torch.flip(img, [3])
    img, flip_img = img.cuda(), flip_img.cuda()
    _, batch_features = model(img)
    _, batch_features_flip = model(flip_img)
    batch_features += batch_features_flip
    batch_features = F.normalize(batch_features, p=2, dim=1)
    features = batch_features.cpu()
    return features


# 主逻辑
if __name__ == '__main__':  # 添加main保护，避免导入时执行
    config, args = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载权重和模型
    checkpoint = torch.load(args.weights, map_location=f'cuda:{args.gpu}', weights_only=True)
    model = ResNet50(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    # 图片预处理
    data_transforms = transforms.Compose([
        transforms.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载并处理图片
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"图片文件不存在：{args.img_path}")
    image = Image.open(args.img_path).convert('RGB')  # 强制转RGB，避免灰度图报错
    image_tensor = data_transforms(image)
    input_batch = image_tensor.unsqueeze(0)

    # 提取特征
    feature = extract_img_feature(model, input_batch)

    # 优化输出：打印特征维度+前10个值（更直观）
    print("=" * 50)
    print(f"输入图片路径：{args.img_path}")
    print(f"特征维度：{feature.shape}")
    print(f"特征前10个值：{feature[0, :10].numpy()}")
    print("=" * 50)