import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from configs.default_img import get_img_config
from models.img_resnet import ResNet50
from PIL import Image
from scipy.spatial.distance import cosine  # 导入余弦相似度计算


def parse_option():
    parser = argparse.ArgumentParser(
        description='Match two images to determine if they are the same person (ReID)')
    # 1. 补充default_img.py需要的核心参数（避免属性缺失）
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--root', type=str, default='',
                        help='dataset root path (unused, just for config compatibility)')
    parser.add_argument('--output', type=str, default='', help='output path (unused, just for config compatibility)')
    parser.add_argument('--resume', type=str, default='', help='resume path (unused, just for config compatibility)')
    parser.add_argument('--eval', action='store_true', help='eval mode (unused, just for config compatibility)')
    parser.add_argument('--tag', type=str, default='', help='experiment tag (unused, just for config compatibility)')
    parser.add_argument('--dataset', type=str, default='', help='dataset name (unused, just for config compatibility)')

    # 2. 原有业务参数
    parser.add_argument('--img1_path', type=str, required=True, help='path to the first image')
    parser.add_argument('--img2_path', type=str, required=True, help='path to the second image')
    parser.add_argument('--weights', type=str, required=True, help='path to the model weights')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='similarity threshold for same person (default: 0.7)')

    args, unparsed = parser.parse_known_args()
    config = get_img_config(args)
    return config, args


@torch.no_grad()
def extract_img_feature(model, img, config):
    """提取单张图片的ReID特征（含水平翻转增强）"""
    # 图片预处理（移到函数内，避免重复定义）
    data_transforms = transforms.Compose([
        transforms.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 处理图片
    if not os.path.exists(img):
        raise FileNotFoundError(f"图片文件不存在：{img}")
    image = Image.open(img).convert('RGB')  # 强制转RGB，避免灰度图/透明通道报错
    image_tensor = data_transforms(image)
    input_batch = image_tensor.unsqueeze(0)  # 增加batch维度

    # 水平翻转增强提取特征（和训练逻辑一致）
    flip_img = torch.flip(input_batch, [3])
    input_batch, flip_img = input_batch.cuda(), flip_img.cuda()

    _, batch_features = model(input_batch)
    _, batch_features_flip = model(flip_img)

    # 融合翻转特征并归一化
    batch_features = (batch_features + batch_features_flip) / 2  # 平均更合理
    batch_features = F.normalize(batch_features, p=2, dim=1)
    features = batch_features.cpu().numpy().flatten()  # 转为一维数组，方便计算相似度
    return features


def calculate_similarity(f1, f2):
    """计算两个特征向量的余弦相似度（取值0~1，越高越相似）"""
    # 余弦相似度 = 1 - 余弦距离
    similarity = 1 - cosine(f1, f2)
    # 确保相似度在0~1范围内（避免数值误差导致的微小越界）
    similarity = np.clip(similarity, 0.0, 1.0)
    return similarity


# 主逻辑
if __name__ == '__main__':
    # 解析参数
    config, args = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    try:
        # 1. 加载模型和权重（修复PyTorch 2.6权重加载问题）
        print(f"[INFO] 加载模型权重：{args.weights}")
        # 方案1：关闭weights_only（兼容旧权重，信任权重来源时使用）
        checkpoint = torch.load(
            args.weights,
            map_location=f'cuda:{args.gpu}',
            weights_only=False  # 关键修改：改为False，兼容包含numpy的权重
        )
        model = ResNet50(config)

        # 权重加载容错处理
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except KeyError:
            # 兼容直接保存model.state_dict()的情况
            model.load_state_dict(checkpoint)
        model = model.cuda()
        model.eval()  # 评估模式，关闭dropout/bn更新
        print("[INFO] 模型加载成功！")

        # 2. 提取两张图片的特征
        print(f"\n[INFO] 提取第一张图片特征：{args.img1_path}")
        f1 = extract_img_feature(model, args.img1_path, config)
        print(f"[INFO] 提取第二张图片特征：{args.img2_path}")
        f2 = extract_img_feature(model, args.img2_path, config)

        # 3. 计算相似度并判断
        similarity = calculate_similarity(f1, f2)
        is_same_person = similarity >= args.threshold

        # 4. 输出结果（直观易读）
        print("\n" + "=" * 60)
        print("📊 ReID双图匹配结果")
        print("=" * 60)
        print(f"图片1路径：{args.img1_path}")
        print(f"图片2路径：{args.img2_path}")
        print(f"特征维度：{len(f1)} (和配置中FEATURE_DIM一致)")
        print(f"余弦相似度：{similarity:.4f} (0~1，越高越相似)")
        print(f"匹配阈值：{args.threshold}")
        print(f"是否为同一人：{'✅ 是' if is_same_person else '❌ 否'}")
        print(f"匹配概率：{similarity * 100:.2f}%")
        print("=" * 60)

        # 补充解读（帮助新手理解）
        print("\n📝 结果解读：")
        if similarity >= 0.9:
            print("   相似度极高，几乎可以确定是同一个人")
        elif similarity >= 0.7:
            print("   相似度较高，大概率是同一个人")
        elif similarity >= 0.5:
            print("   相似度中等，可能是同一个人")
        else:
            print("   相似度较低，基本可以确定不是同一个人")

    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)}")
        exit(1)