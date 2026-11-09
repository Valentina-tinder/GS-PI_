# GS-PI：论文配套代码

**GS-PI: An Optimization-Decoupled Appearance Decomposition Approach for Generating PBR Gaussian Assets**

Jieting Xu · Rengan Xie · Zijian Huang · Zehui Jin · Rui Wang · Yuchi Huo

State Key Laboratory of CAD&CG, Zhejiang University

[论文](https://arxiv.org/abs/2609.19907) · [PDF](https://arxiv.org/pdf/2609.19907) · [English](README.md)

⭐ 如果 GS-PI 对你有帮助，或者你喜欢我们的工作，欢迎给仓库点一颗 **Star**！这会让作者们很开心。谢谢你的支持呀！🥰✨

本仓库提供 GS-PI 的 **Stage II 条件点云材质扩散模型**训练与评估代码。模型结合点坐标、多视角图像特征和逐点条件，预测反照率、粗糙度与金属度。

支持双流 Read–Compute–Write（RCW）和标准 Point Transformer，提供条件特征预计算、训练与续训、DDIM 采样及 PLY/NPZ 导出。另有材质与世界空间法线的联合预测配置。

## 安装

建议 Python 3.10、PyTorch 2.4.1 / torchvision 0.19.1；先根据显卡安装匹配的 PyTorch，再安装本包：

```bash
python -m pip install -e '.[test,data]'
```

`data` 附加依赖供原始数据集的 EXR 检查及质量筛选使用。已有 NPZ 数据或普通 PLY 时可只安装 `.[test]`。xFormers 是可选加速依赖，需要自行选择与 PyTorch/CUDA 匹配的版本；默认配置使用普通注意力。已有缓存时，训练和测试均不需要下载 DINOv2。

## 先验证安装

下面用随机合成的小数据集检查安装和训练、评估流程，CPU 即可运行。

```bash
python examples/make_toy_data.py --output data/toy
python train.py --config data/toy/config.yaml --output outputs/toy --device cpu
python evaluate.py --config data/toy/config.yaml \
  --checkpoint outputs/toy/checkpoint.pt --state-key ema_model_state_dict \
  --output outputs/toy_eval --device cpu --steps 2
python -m pytest -q
```

## 准备真实数据

支持两种输入格式，详见 [DATA.md](docs/DATA.md)：

1. 原项目 `PBRConditionalDataset`：每个对象的 PLY、metadata、多视角图像，以及 DINO / Top-K 缓存。
2. 简洁 `NPZDataset`：每对象一个 `.npz`，包含坐标、材质 GT 和可选条件，便于适配自有数据。

使用原格式时，每个对象需提供 `Render_1.png` 至 `Render_32.png` 多视角 RGB 图像，图像编号须与 metadata 中的视角编号对应。准备好输入文件后，可用所附工具计算条件：

```bash
export NPCD_DINOV2_REPO=/path/to/dinov2
python scripts/tools/extract_features_multiview.py \
  --root-dir data/train --save-dir data/dino/train --num-views 32 --device cuda
python scripts/tools/precompute_topk_cond_point_features.py \
  --dataset-root data/train --output-dir data/topk1/train \
  --num-views 32 --top-k 1 --device cpu
```

对测试集另行生成缓存。DINOv2 需自行从其官方仓库安装并按其许可使用；首次提取可能下载其预训练权重。图像提取器默认固定投影 seed；已有历史缓存应与训练 checkpoint 使用相同的特征空间。

## 训练

先修改 YAML 中的训练/测试数据路径。所有相对路径相对于执行命令时的工作目录。

```bash
python train.py --config configs/gs_pi_pbr.yaml \
  --output outputs/gs_pi --device cuda --dtype bfloat16 --num-workers 4
```

多 GPU：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/gs_pi_pbr.yaml --output outputs/gs_pi_ddp \
  --dtype bfloat16 --num-workers 4
```

`batch_size` 为每进程批大小；有效批大小为 `batch_size × GPU 数 × gradient_accumulation_steps`。不同进程采用不同采样随机种子，日志 loss 做全进程平均。

训练保存 `config.yaml`、`metrics.jsonl`、`checkpoint.pt`。checkpoint 包含模型、EMA、AdamW、AMP scaler、迭代数和配置。续训时将 `max_iterations` 设为目标总步数：

```bash
python train.py --config configs/gs_pi_pbr.yaml --output outputs/gs_pi \
  --resume outputs/gs_pi/checkpoint.pt --dtype bfloat16
```

训练使用固定学习率 AdamW、梯度裁剪、EMA、AMP 和 DDP。续训恢复模型和优化器等状态；数据游标与随机数状态会重新初始化。

| 配置 | 用途 |
| --- | --- |
| `gs_pi_pbr.yaml` | 论文 Stage II 起始配置：双流 RCW、PBR5、512 点、训练 16/32 DINO 视角 |
| `transformer_pbr.yaml` | 标准 Transformer，5 通道材质 |
| `two_stream_pbr.yaml` | 双流 Transformer |
| `transformer_normal.yaml` | 标准 Transformer，5 通道材质 + 3 通道法线 |

可根据数据和显存调整配置。训练使用固定点数采样；默认评估使用完整点云。标准 Transformer 全局注意力的显存需求随序列长度平方增长。

## 推理与评估

```bash
python evaluate.py --config configs/gs_pi_pbr.yaml \
  --checkpoint outputs/gs_pi/checkpoint.pt --state-key ema_model_state_dict \
  --output outputs/test --device cuda --steps 50
```

默认对每个测试对象的完整点云做一次 DDIM 采样，保持原始点序。显存有限时，可用 `--sampled` 在配置指定的 `num_points` 个采样点上评估。

输出每对象的 `.npz`（坐标、预测、GT）、`.ply`（坐标及预测属性）和 `metrics.json`。材质指标为截断到 `[0,1]` 后的逐点 MAE/MSE；法线指标为有效 GT 上的角度误差，无效预测按 180° 计。汇总对对象等权平均。

运行评估需要材质 GT。对未知材质的点云做预测时，可直接调用 `gs_pi.runtime.sample_features`。

原项目权重可以选择其实际 state key 加载；加载器识别原始 diffusion 权重及 `diffusion.` / DDP / compile 前缀，忽略旧 PointNeRF 权重，并对 diffusion 严格校验。旧配置需迁移到本包格式，骨干、宽度、层数、特征 schema 和条件布局必须一致。

## 目录与许可

```text
train.py / evaluate.py        训练、DDIM 测试与导出
gs_pi/models/diffusion/       骨干、扩散过程、归一化与损失
gs_pi/data/                   原 PLY 数据与条件缓存读取
gs_pi/runtime.py              NPZ 适配、权重加载、采样和指标
scripts/tools/               DINO / projected / Top-K 条件预计算
configs/                     四份可迁移配置
examples/                    合成数据示例
tests/                       数值、骨干及命令行链路测试
```

沿用源项目 MIT [LICENSE](LICENSE)，保留原版权声明。

## 引用

请使用 [CITATION.bib](CITATION.bib) 引用论文；GitHub 可通过 [CITATION.cff](CITATION.cff) 显示引用信息。
