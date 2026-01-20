import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation. (Compatible with single GPU)"""

    @staticmethod
    def forward(ctx, input):
        # ==========  新增：单卡/分布式自适应判断  ==========
        try:
            # 检查分布式是否可用且已初始化
            is_dist_available = dist.is_available()
            is_dist_initialized = dist.is_initialized()
        except AttributeError:
            # 部分环境下 dist 模块属性缺失，直接判定为非分布式
            is_dist_available = False
            is_dist_initialized = False

        if is_dist_available and is_dist_initialized:
            # 分布式环境：保留原有逻辑
            ctx.save_for_backward(input)
            world_size = dist.get_world_size()
            output = [torch.zeros_like(input) for _ in range(world_size)]
            dist.all_gather(output, input)
            return tuple(output)
        else:
            # 单卡环境：无需收集，直接返回输入（保持元组格式，兼容后续代码）
            ctx.save_for_backward(torch.tensor([1]))  # 保存占位符，避免反向传播报错
            return (input,)  # 返回元组，与分布式环境的返回格式一致

    @staticmethod
    def backward(ctx, *grads):
        # ==========  新增：单卡/分布式自适应反向传播  ==========
        try:
            is_dist_available = dist.is_available()
            is_dist_initialized = dist.is_initialized()
        except AttributeError:
            is_dist_available = False
            is_dist_initialized = False

        if is_dist_available and is_dist_initialized:
            # 分布式环境：保留原有逻辑
            (input,) = ctx.saved_tensors
            grad_out = torch.zeros_like(input)
            grad_out[:] = grads[dist.get_rank()]
            return grad_out
        else:
            # 单卡环境：直接返回第一个梯度（与输入格式匹配）
            return grads[0]