"""Train the extracted PBR diffusion network; single device or torchrun DDP."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from gs_pi.runtime import make_dataset, make_model, model_batch, read_config
from gs_pi.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume", help="Public training checkpoint to resume (optimizer + EMA)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = read_config(args.config)
    cfg = config["training"]
    steps, batch_size = int(cfg["max_iterations"]), int(cfg["batch_size"])
    accumulation = int(cfg.get("gradient_accumulation_steps", 1))
    interval = int(cfg.get("save_interval", 1000))
    if min(steps, batch_size, accumulation, interval) < 1:
        raise ValueError(
            "Iterations, batch size, accumulation and save interval must be positive"
        )
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        f"cuda:{local_rank}"
        if world > 1 and args.device.startswith("cuda")
        else args.device
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.dtype == "float16" and device.type != "cuda":
        raise ValueError("float16 training requires CUDA")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    set_seed(args.seed)
    model = make_model(config).to(device)
    dataset = make_dataset(config, "train")
    if getattr(dataset, "full_pointcloud_training", False):
        if batch_size != 1 or not model.denoiser.supports_full_pointcloud_training:
            raise ValueError(
                "Full-object training is not supported by the released backbones; use sampled training"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
    start = 0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("format") != "gs_pi_public_training_v1":
            raise ValueError("--resume requires a checkpoint produced by this train.py")
        if resume["config"]["model"] != config["model"]:
            raise ValueError("Resume model config differs from checkpoint")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start = int(resume["iteration"])
    else:
        model.coords_normalization.set_from_all_data(dataset.get_all_coords())
        model.feats_normalization.set_from_all_data(dataset.get_all_feats())
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError(
            "Nonfinite normalization; check for degenerate/constant training data"
        )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if resume:
        ema.load_state_dict(resume["ema_model_state_dict"])
    decay = float(cfg.get("ema_decay", 0.999))
    if not 0 <= decay < 1:
        raise ValueError("ema_decay must be in [0,1)")
    sampler = DistributedSampler(dataset, seed=args.seed) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        drop_last=False,
    )
    set_seed(args.seed + rank)
    net = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if world > 1
        else model
    )
    net.train()
    epoch = 0
    iterator = iter(loader)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for iteration in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        average = torch.zeros((), device=device)
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            with torch.autocast(
                device_type=device.type,
                dtype=getattr(torch, args.dtype),
                enabled=args.dtype != "float32",
            ):
                loss, _, _ = net(**model_batch(batch, device))
                loss = loss.mean() / accumulation
            finite = torch.isfinite(loss).int()
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            average += loss.detach()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(cfg.get("grad_clip", 1.0)),
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), model.parameters()):
                e.lerp_(p, 1 - decay)
            for e, p in zip(ema.buffers(), model.buffers()):
                e.copy_(p)
        if world > 1:
            dist.all_reduce(average)
            average /= world
        if rank == 0:
            record = {"iteration": iteration + 1, "loss": average.item()}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if (
                iteration == start
                or (iteration + 1) % int(cfg.get("log_interval", 10)) == 0
            ):
                print(record, flush=True)
            if (iteration + 1) % interval == 0 or iteration + 1 == steps:
                checkpoint = dict(
                    format="gs_pi_public_training_v1",
                    iteration=iteration + 1,
                    config=config,
                    model_state_dict=model.state_dict(),
                    ema_model_state_dict=ema.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scaler_state_dict=scaler.state_dict(),
                )
                temporary = output / "checkpoint.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output / "checkpoint.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
