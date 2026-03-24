"""
单卡预训练脚本（无 DDP）
用法: python pretrain_without_ddp.py [args]
"""
import os
import sys

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from dataset.pretrain_dataset import PretrainDataset
from utils import get_lr, Logger, SkipBatchSampler
from benchmark.evaluator import run_benchmark

warnings.filterwarnings('ignore')

# iters是每个epoch的步数,计算公式为：len(train_ds) // args.batch_size
# start_step是每个epoch的开始步数
# swanlab是swanlab的实例
# total_steps是总步数
# warmup_steps是warmup步数
# full_save_dir是模型保存目录
# args是参数
# optimizer是优化器
# scaler是scaler
# model是模型
# device是设备
# dtype是数据类型
# autocast_ctx是自动混合精度上下文
# res是结果
# loss是损失
# scaler是scaler
# optimizer是优化器
# global_step是全局步数
# spend_time是花费时间
# current_loss是当前损失
# current_lr是当前学习率
# eta_min是eta时间
# Logger是日志打印
# swanlab是swanlab的实例
# swanlab_run是swanlab的实例
def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, total_steps=None, warmup_steps=None, full_save_dir=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        current_step = epoch * iters + step
        lr = get_lr(current_step, total_steps, args.learning_rate, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度上下文
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / args.accumulation_steps # 梯度累计：将多个step的梯度累加起来，再除以accumulation_steps，得到平均梯度
        # 防止下溢出，scale是缩放，backward是反向传播
        scaler.scale(loss).backward() # scale：对下溢出进行缩放，backward：反向传播

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer) # 取消缩放，防止下溢出
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪，防止梯度过大，发生剧烈波动
            scaler.step(optimizer) # 更新参数
            scaler.update() # 更新缩放因子
            optimizer.zero_grad(set_to_none=True) # 由于pytorch的优化器会自动累积梯度，所以需要手动清零

        global_step = epoch * iters + step

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60  # 计算剩余时间，仅仅是当前这一个 Epoch 跑完还剩下多少时间
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if swanlab:
                swanlab.log({"loss": current_loss, "learning_rate": current_lr, "eta_time": eta_min}, step=global_step)

        # 保存 checkpoint，除了要保存模型参数，还要保存优化器参数，scaler参数，epoch，step，global_step，swanlab_id
        if global_step % args.save_interval == 0 or step == iters - 1:
            model.eval() # 评估模式，不进行梯度计算
            ckp_dir = f'{full_save_dir}/global_step_{global_step}'
            os.makedirs(ckp_dir, exist_ok=True)
            raw_model = getattr(model, '_orig_mod', model)
            state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
            torch.save(state_dict, f'{ckp_dir}/{args.save_weight}_{lm_config.hidden_size}.pth')
            torch.save({
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'global_step': global_step,
                'swanlab_id': getattr(swanlab, 'id', None) if swanlab else None
            }, f'{ckp_dir}/resume.pth')
            Logger(f'Saved checkpoint: {ckp_dir}')
            model.train()

        # Benchmark 评测
        if args.eval_bench == 1 and tokenizer is not None and global_step % args.eval_interval == 0:
            model.eval()
            c3_path = 'D:\桌面\SpongeBobPro\dataset\clue_c3_eval_500.jsonl'
            xcopa_path = 'D:\桌面\SpongeBobPro\dataset\xcopa_zh_merged.jsonl'
            eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
            if swanlab_run:
                swanlab_run.log(eval_results, step=global_step)
            Logger(f'Benchmark results: {eval_results}')
            model.train()

        del input_ids, labels, res, loss

# 如果本文件是主文件，则执行以下代码，如果该文件是被别的文件调用，则不执行以下代码
# 在 Python 中，__name__ 是一个特殊的变量，它代表当前模块的名称。当你直接运行一个 Python 文件时，__name__ 会被自动设置为 "__main__"。
# 但是，如果你导入了这个文件，或者在另一个文件中使用 import 语句导入这个文件，__name__ 会被设置为文件的实际名称。
# 这个机制使得你可以在同一个文件中编写可重用的代码，同时又能够在需要时执行特定的代码块。
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="SpongeBob Pretraining (Single GPU)")
    parser.add_argument("--save_dir", type=str, default="../pretrain_out", help="模型保存根目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="序列长度")
    parser.add_argument("--data_path", type=str, default="{你的文件路径}", help="预处理后的.bin文件路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_swanlab", type=int, default=1, choices=[0, 1], help="是否使用swanlab（0=否，1=是）")
    parser.add_argument("--swanlab_project", type=str, default="SpongeBob-Pretrain", help="swanlab项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--eval_bench", default=1, type=int, choices=[0, 1], help="是否评测benchmark（0=否，1=是）")
    parser.add_argument("--eval_interval", type=int, default=100, help="评测间隔步数")
    args = parser.parse_args()

    # ========== 1. 配置目录、模型参数、检查 ckp ==========
    lm_config = SpongeBobConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    run_name = f"h{args.hidden_size}_l{args.num_hidden_layers}_bs{args.batch_size}_lr{args.learning_rate}" # 根据模型参数生成run_name
    full_save_dir = os.path.join(args.save_dir, run_name) # 根据run_name生成保存目录
    os.makedirs(full_save_dir, exist_ok=True)

    ckp_data = None
    if args.from_resume == 1: # 判断训练是否从断点开始续传
        ckp_dirs = [d for d in os.listdir(full_save_dir) if d.startswith('global_step_')] 
        if ckp_dirs:
            latest_ckp = max(ckp_dirs, key=lambda x: int(x.split('_')[-1])) # 取最新的断点位
            resume_path = f'{full_save_dir}/{latest_ckp}/resume.pth'
            if os.path.exists(resume_path):
                ckp_data = torch.load(resume_path, map_location='cpu')
                Logger(f'Found checkpoint: {full_save_dir}/{latest_ckp}')

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype) # 用于前向传播自动实现混合精度

    # ========== 4. SwanLab ==========
    swanlab_run = None
    if args.use_swanlab:
        import swanlab
        swanlab.login(api_key="rlwi7MJhp0sgBHtuatQud")
        swanlab_id = ckp_data.get('swanlab_id') if ckp_data else None
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=run_name,
            id=swanlab_id,
            config=vars(args)
        )
        Logger(f'SwanLab initialized: {run_name}')

    # ========== 5. 模型、数据、优化器 ==========
    if args.from_weight != 'none' and os.path.exists(args.from_weight): # 是否要续训
        Logger(f'Loading model from {args.from_weight}')
        model = SpongeBobForCausalLM.from_pretrained(args.from_weight)
    else:
        Logger(f'Creating new model: hidden_size={args.hidden_size}, num_layers={args.num_hidden_layers}')
        model = SpongeBobForCausalLM(lm_config)
    model = model.to(args.device)
    Logger(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M') # 记录模型的参数量

    # 跑前benchmark一下
    if args.eval_bench == 1:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob/tokenizer_15k')
        Logger('Tokenizer loaded for benchmark evaluation')
    else:
        tokenizer = None
    # 使用算子加速器
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    Logger('Loading dataset...')
    train_ds = PretrainDataset(args.data_path, seq_len=args.max_seq_len)
    Logger('Dataset ready')

    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    Logger('Optimizer ready')

    # ========== 6. 从 ckp 恢复 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        Logger('Loading checkpoint...')
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'Checkpoint loaded: epoch={start_epoch}, step={start_step}')

    # ========== 7. 总步数（单卡）==========
    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.03)
    Logger(f'Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}')

    # ========== 8. 初始评测 (step 0) ==========
    if args.eval_bench == 1 and tokenizer is not None and start_epoch == 0 and start_step == 0:
        Logger('Running initial benchmark evaluation (step 0)...')
        model.eval()
        c3_path = '测试集地址'
        xcopa_path = '测试集地址'
        eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
        if swanlab_run:
            swanlab_run.log(eval_results, step=0)
        Logger(f'Initial benchmark results (step 0): {eval_results}')
        model.train()

    # ========== 9. 训练循环 ==========
    Logger(f'Starting training: {args.epochs} epochs, batch_size={args.batch_size} (single GPU)')
    for epoch in range(start_epoch, args.epochs):
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, swanlab_run, total_steps, warmup_steps, full_save_dir)
        else:
            train_epoch(epoch, loader, len(loader), 0, swanlab_run, total_steps, warmup_steps, full_save_dir)

    Logger('Training done.')