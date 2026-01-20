import data.img_transforms as T
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset
# ==========  修正：导入原版RandomIdentitySampler，删除冗余的SingleGPU类 ==========
from data.samplers import (
    DistributedRandomIdentitySampler,
    DistributedInferenceSampler,
    RandomIdentitySampler  # 替换：使用原版无分布式依赖的采样器，已存在于samplers.py
)
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC

__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
}


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError(
            "Invalid dataset, got '{}', but expected to be one of {}".format(config.DATA.DATASET, __factory.keys()))

    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)

    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    transform_train, transform_test = build_img_transforms(config)

    # ==========  修正：单卡环境调用原版RandomIdentitySampler，删除SingleGPU类调用 ==========
    try:
        # 尝试初始化分布式采样器（单卡环境会报错，进入except）
        from torch import distributed as dist
        if dist.is_available() and dist.is_initialized():
            # 分布式环境：使用原版分布式身份采样器
            train_sampler = DistributedRandomIdentitySampler(
                dataset.train,
                num_instances=config.DATA.NUM_INSTANCES,
                seed=config.SEED
            )
        else:
            # 单卡环境：使用原版RandomIdentitySampler（无分布式依赖，逻辑与原SingleGPU一致）
            train_sampler = RandomIdentitySampler(
                dataset.train,
                num_instances=config.DATA.NUM_INSTANCES
            )
    except (ImportError, RuntimeError):
        # 捕获分布式依赖错误：强制使用原版RandomIdentitySampler
        train_sampler = RandomIdentitySampler(
            dataset.train,
            num_instances=config.DATA.NUM_INSTANCES
        )

    trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train),
                              sampler=train_sampler,
                              batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                              pin_memory=True, drop_last=True)

    # ==========  保留：单卡推理采样器（简化版，无分布式依赖），无需修改 ==========
    # 定义单卡推理采样器（无需分布式依赖，直接返回完整索引）
    class SingleGPUInferenceSampler:
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            return iter(range(len(self.dataset)))

        def __len__(self):
            return len(self.dataset)

    # 自适应选择推理采样器（单卡/分布式）
    try:
        from torch import distributed as dist
        if dist.is_available() and dist.is_initialized():
            # 分布式环境：使用原版分布式推理采样器
            gallery_sampler = DistributedInferenceSampler(dataset.gallery)
        else:
            # 单卡环境：使用自定义单卡推理采样器
            gallery_sampler = SingleGPUInferenceSampler(dataset.gallery)
    except (ImportError, RuntimeError):
        # 捕获分布式依赖错误：强制使用单卡推理采样器
        gallery_sampler = SingleGPUInferenceSampler(dataset.gallery)

    galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                                sampler=gallery_sampler,  # 使用自适应推理采样器
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=False, shuffle=False)

    if config.DATA.DATASET == 'prcc':
        # ==========  保留：prcc数据集的query采样器自适应逻辑，无需修改 ==========
        try:
            from torch import distributed as dist
            if dist.is_available() and dist.is_initialized():
                query_same_sampler = DistributedInferenceSampler(dataset.query_same)
                query_diff_sampler = DistributedInferenceSampler(dataset.query_diff)
            else:
                query_same_sampler = SingleGPUInferenceSampler(dataset.query_same)
                query_diff_sampler = SingleGPUInferenceSampler(dataset.query_diff)
        except (ImportError, RuntimeError):
            query_same_sampler = SingleGPUInferenceSampler(dataset.query_same)
            query_diff_sampler = SingleGPUInferenceSampler(dataset.query_diff)

        queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test),
                                       sampler=query_same_sampler,
                                       batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=False, shuffle=False)
        queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test),
                                       sampler=query_diff_sampler,
                                       batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler
    else:
        # ==========  保留：普通数据集的query采样器自适应逻辑，无需修改 ==========
        try:
            from torch import distributed as dist
            if dist.is_available() and dist.is_initialized():
                query_sampler = DistributedInferenceSampler(dataset.query)
            else:
                query_sampler = SingleGPUInferenceSampler(dataset.query)
        except (ImportError, RuntimeError):
            query_sampler = SingleGPUInferenceSampler(dataset.query)

        queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test),
                                  sampler=query_sampler,  # 使用自适应推理采样器
                                  batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                  pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader, galleryloader, dataset, train_sampler