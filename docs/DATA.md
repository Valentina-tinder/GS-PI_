# Data specification

## Original PBR dataset

```text
data/train/<object_id>/
  metadata.json
  Render_1.png ... Render_32.png
  extract_output/merged_model_render_color.ply
data/dino/train/<object_id>.pt
data/topk1/train/<object_id>.pt
```

The sample configuration sets `prefer_extract_output: true` to select this PLY layout explicitly. Precomputation, training and evaluation must use the same point cloud and the same point order. Camera metadata and per-view PLYs may be needed by the condition-precomputation tools; see their `--help` options for depth sources and layouts.

Required PLY fields are `x,y,z`, `albedo_red,albedo_green,albedo_blue` (or legacy `red,green,blue`), `roughness`, and `metallic`. Color values whose maximum exceeds 1 are divided by 255. Roughness and metallic are in `[0,1]`. Keep coordinates in their original world frame; the model uses normalization buffers saved in its checkpoint.

The experimental normal configuration also requires `nx,ny,nz` and optionally `normal_valid`. These are Blender-world-space ground-truth normals, not camera-space normals. The loader does not generate geometric fallback normals.

Top-K caches contain `K*5` raw channels: RGB, confidence and a zero-based view ID for each slot. The precomputation script stores RGB/confidence using its documented convention. For `K=1`, slot 0 is anchored to the point's source view. With one-hot view conditioning and confidence disabled, the model expands the raw five channels to `3 + 32 = 35` effective channels. Do not remove the confidence field from the cache merely because the model does not consume it.

The cache includes an ordered-XYZ fingerprint; the full-object path checks its alignment with the PLY. Do not reuse a cache after reordering or replacing points.

The DINO tool writes `[V,H,W,512]` patch features. The recommended configuration pools each view to `5 × 5`, producing `[800,512]` tokens for 32 views at evaluation time. During training it samples 16 views, producing 400 tokens. Standard and two-stream networks require the condition dimension to match `width`.

Dataset initialization may write an index cache in the data root. Keep train/test object splits separate. No real object lists or data are included in this repository.

## NPZ interface

Each `.npz` file represents one object:

| Field | Shape / dtype | Meaning |
| --- | --- | --- |
| `coords` | `[3,N]`, float32 | Original coordinates |
| `feats` | `[5,N]` or `[8,N]`, float32 | Albedo RGB, roughness, metallic, optionally world normals |
| `cond_tokens` | `[L,D]`, float32 | Optional image tokens |
| `cond_point_colors` | `[C,N]`, float32 | Optional per-point conditions; match model configuration |
| `normal_valid_mask` | `[N]` or `[1,N]`, bool | Optional normal validity |
| `view_source` | `[N]`, int64 | Optional source-view indices |

Training samples `num_points` points at random, with replacement when the object has fewer points. Default evaluation uses every point. The NPZ adapter does not perform camera projection or compute conditions; all point-aligned fields must already be in the same order. Use only the fields above and avoid pickle/object arrays.

```yaml
train_dataset: NPZDataset
train_dataset_params:
  root_dir: data/my_npz/train
  num_points: 512
test_dataset: NPZDataset
test_dataset_params:
  root_dir: data/my_npz/test
  num_points: 512
```

For no local conditions, set `model.cond_point_dim: 0`. For no image conditions, omit `cond_tokens` and set `condition_on_images: false`. These settings change the learning task and do not reproduce the image-conditioned paper setup.

## Synthetic smoke data

`examples/make_toy_data.py` writes two training objects and two test objects, each with 24 random points, random material targets and random condition tensors. Its tiny model and two-step training schedule only test software execution.
