import os
import torch
import argparse
import GPUtil
from configs.default_img import get_img_config

# ---------------------- 导入项目核心模块 ----------------------
from data import build_dataloader
from models import build_model
from losses import build_losses


# ---------------------- 显存监控函数 ----------------------
def get_gpu_memory_usage(gpu_id=0):
    """获取GPU显存使用情况，返回：已用(MB)、总(MB)、峰值(MB)"""
    torch.cuda.synchronize()
    gpus = GPUtil.getGPUs()
    gpu = gpus[gpu_id]
    used = gpu.memoryUsed
    total = gpu.memoryTotal
    peak = torch.cuda.max_memory_allocated(gpu_id) / 1024 / 1024
    return {
        'used': round(used, 2),
        'total': round(total, 2),
        'peak': round(peak, 2),
        'usage_rate': round(used / total * 100, 2)
    }


# ---------------------- 核心测试逻辑 ----------------------
def test_peak_memory():
    # 1. 构造参数解析器（匹配main.py）
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/res50_cels_cal.yaml')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--tag', type=str, default='test_memory')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', type=str, default=None)
    parser.add_argument('--single_shot', action='store_true')
    parser.add_argument('--k_cal', type=str, default=None)
    parser.add_argument('--k_kl', type=str, default=None)

    args, _ = parser.parse_known_args()
    config = get_img_config(args)

    # 2. 强制启用所有损失（模拟训练后期）
    config.defrost()
    config.TRAIN.START_EPOCH_ADV = 0
    config.TRAIN.START_EPOCH_CC = 0
    config.TRAIN.START_EPOCH_GENERAL = 0
    config.LOSS.CAL = 'cal'
    config.LOSS.PAIR_LOSS_WEIGHT = 0.1
    config.freeze()

    # 3. 初始化GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    device = torch.device(f"cuda:{config.GPU}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # 4. 加载数据集/模型/损失/优化器
    trainloader, _, _, dataset, _ = build_dataloader(config)
    pid2clothes = torch.from_numpy(dataset.pid2clothes).to(device)

    model, model2, fuse, classifier, clothes_classifier, clothes_classifier2 = build_model(
        config, dataset.num_train_pids, dataset.num_train_clothes
    )
    model = model.to(device)
    model2 = model2.to(device)
    fuse = fuse.to(device)
    classifier = classifier.to(device)
    clothes_classifier = clothes_classifier.to(device)
    clothes_classifier2 = clothes_classifier2.to(device)

    criterion_cla, criterion_pair, criterion_clothes, criterion_adv, kl = build_losses(
        config, dataset.num_train_clothes
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(fuse.parameters()) + list(classifier.parameters()),
        lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )
    optimizer2 = torch.optim.Adam(
        list(model2.parameters()) + list(clothes_classifier2.parameters()),
        lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )
    optimizer_cc = torch.optim.Adam(
        clothes_classifier.parameters(),
        lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )

    # 5. 打印初始显存
    print("===== 初始显存状态 =====")
    mem_init = get_gpu_memory_usage(gpu_id=int(config.GPU))
    print(f"GPU {config.GPU} 总显存: {mem_init['total']} MB")
    print(f"初始已用显存: {mem_init['used']} MB (占用率 {mem_init['usage_rate']}%)\n")

    # 6. 模拟前向+反向传播（测峰值）
    print("===== 开始模拟训练后期（启用所有损失） =====")
    print(f"当前参数：TRAIN_BATCH={config.DATA.TRAIN_BATCH}, FEATURE_DIM={config.MODEL.FEATURE_DIM}")
    model.train()
    model2.train()
    fuse.train()
    classifier.train()
    clothes_classifier.train()
    clothes_classifier2.train()

    for batch_idx, (imgs, pids, camids, clothes_ids, _) in enumerate(trainloader):
        if batch_idx > 0: break

        # 数据移到GPU
        imgs = imgs.to(device)
        pids = pids.to(device)
        clothes_ids = clothes_ids.to(device)
        pos_mask = pid2clothes[pids].float().to(device)

        # 前向传播
        optimizer.zero_grad()
        optimizer2.zero_grad()
        optimizer_cc.zero_grad()

        pri_feat, features = model(imgs)
        pri_feat2, features2 = model2(imgs)
        pri_feat2 = pri_feat2.clone().detach()
        features_fuse = fuse(pri_feat, pri_feat2)
        outputs = classifier(features)
        outputs2 = clothes_classifier2(features2)
        outputs3 = classifier(features_fuse)
        pred_clothes = clothes_classifier(features.detach())

        # 计算所有损失并反向传播
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        clothes_loss.backward()
        optimizer_cc.step()

        new_pred_clothes = clothes_classifier(features)
        Q = torch.nn.functional.softmax(new_pred_clothes.clone().detach(), dim=-1)
        P = torch.nn.functional.softmax(outputs2.clone(), dim=-1)
        kl_loss = kl(torch.log(Q), P, reduction='sum') + kl(torch.log(P), Q, reduction='sum')
        clothes_loss2 = criterion_clothes(outputs2, clothes_ids)
        loss2 = clothes_loss2 + config.k_kl * kl_loss
        loss2.backward()
        optimizer2.step()

        cla_loss = criterion_cla(outputs, pids) + config.k_cal * criterion_cla(outputs - outputs3, pids)
        pair_loss = criterion_pair(features, pids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        loss.backward()
        optimizer.step()

        # 7. 打印显存结果
        print("\n===== 显存峰值结果 =====")
        mem_peak = get_gpu_memory_usage(gpu_id=int(config.GPU))
        print(f"GPU {config.GPU} 总显存: {mem_peak['total']} MB")
        print(f"训练峰值显存: {mem_peak['peak']} MB (占用率 {mem_peak['peak'] / mem_peak['total'] * 100:.2f}%)")
        print(f"当前已用显存: {mem_peak['used']} MB (占用率 {mem_peak['usage_rate']}%)")
        print(f"剩余显存: {mem_peak['total'] - mem_peak['used']:.2f} MB")

        # 8. 参数建议
        print("\n===== 参数调整建议 =====")
        if mem_peak['peak'] > mem_peak['total'] * 0.9:
            print("⚠️  显存不足，建议降低TRAIN_BATCH至{}".format(config.DATA.TRAIN_BATCH // 2))
        else:
            print("✅  显存充足，当前参数可直接使用")
        break

    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_peak_memory()