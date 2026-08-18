# Avatar-DiT: Parametric Multi-View Human Rendering for Interactive Humans using Multi-modal DiT

Official implementation of **Avatar-DiT** (Pacific Graphics 2026).

Avatar-DiT is a diffusion-transformer framework for parametric, identity-consistent, full-body multi-view human rendering. It decouples facial control (FLAME parameters) from body control (MHR mesh renderings) and injects explicit camera conditioning (Plücker-ray maps), trained with a three-stage curriculum on top of the WanAnimate backbone.

## Repository structure

```
Avatar_DiT/
├── train.py               # training entry (all three stages, selected by config)
├── inference.py           # multi-view / face-control inference & validation entry
├── options.py             # command-line options
├── ckpt_util.py           # config / checkpoint loading utilities
├── logger.py              # training logger (image/video previews)
├── configs/
│   ├── training/          # stage configs: flame.yaml (stage 1), multiview.yaml (stage 2),
│   │                      #   multiview-lora.yaml (stage 3, rank-64 LoRA), base.yaml
│   └── inference/         # inference configs: face-control.yaml, multiview.yaml, wan-control.yaml
├── wan/                   # model code
│   ├── modules/           # DiT backbone (model.py), FLAME adapter (face_blocks.py),
│   │   │                  #   motion encoder, VAE, CLIP/T5 embedders
│   │   └── embedder/
│   ├── wrapper/           # training/inference wrappers
│   │   └── trainer/       # base.py, face_trainer.py (stage 1), mv_trainer.py (stages 2/3)
│   ├── ray_sampler.py     # Plücker-ray camera embedding (extrinsics normalization + ray map)
│   ├── scheduler.py       # flow-matching schedulers
│   └── util.py
├── data/                  # dataloaders (videoloader.py: VideoLoader / MultiViewLoader)
│   └── scripts/           # mask / depth / caption / bbox preprocessing helpers
├── data_pipeline/         # annotation & alignment pipeline (see its own README):
│                          #   video download / tracking / splitting / SyncNet filtering,
│                          #   EMICA-based FLAME fitting, FLAME mesh rendering
├── scripts/               # body-mesh (MHR/SMPL) rendering from calibrated cameras,
│                          #   training-dataset assembly
├── splits/                # dataset split lists used in the paper (relative paths)
├── libs/                  # checkpoint conversion utilities (DeepSpeed zero_to_fp32 etc.)
└── backend/               # video/image io utilities
```

## Environment

```bash
conda env create -f environment.yml   # python 3.12, torch + accelerate + diffusers
conda activate hf
```

Flash-Attention 2 is recommended. The annotation pipeline has its own requirements (`data_pipeline/requirements.txt`, incl. pytorch3d).

## Pretrained backbones (not redistributed here)

Download and place under `pretrained_models/wan2.2/`:

- Wan2.2-Animate-14B (`Wan-AI/Wan2.2-Animate-14B` on Hugging Face; transformer weights are loaded via `transformer_config.ckpt_path`)
- `Wan2.1_VAE.pth`, `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, `models_t5_umt5-xxl-enc-bf16.pth` (from the Wan2.2 release)

The FLAME 2020 model (`data_pipeline/flame_model/flame_2020.pt`) must be obtained from https://flame.is.tue.mpg.de under its license and is not redistributed.

## Data preparation

1. **Face-control data (stage 1)**: talking videos with per-frame FLAME labels.
   Use `data_pipeline/` — download/track/split (`run_tracksplit.py`), SyncNet filtering (`run_syncnet.py`), EMICA FLAME fitting (`run_trackflame*.py`), and FLAME mesh rendering (`render_flame_mesh.py`).
2. **Multi-view data (stage 2)**: calibrated multi-camera datasets (DNA-Rendering, MVHumanNet, ZJU-MoCap).
   Render per-view body meshes with `scripts/render_smpl_from_cam.py` / `scripts/render_smpl_batch.py` and assemble the training layout with `scripts/build_training_dataset.py`.
3. **Joint data (stage 3)**: hybrid mixture of the multi-view set and monocular FLAME-labeled talking clips.

The exact split lists used in the paper are in `splits/`:

| file | clips | usage |
|---|---|---|
| `flame.json` | 899 source segments (≈2,099 81-frame clips) | stage 1 training |
| `flame_val.json` | 59 | face-control validation / five-way ablation |
| `camera-hybrid.json` | 12,761 | stage 3 (joint) training |
| `camera-hybrid_val.json` | 115 | multi-view validation |

Paths inside the split files are relative to your dataset root (`-d` argument).

## Training (three-stage curriculum)

All stages: AdamW, bf16, batch size 1/GPU, 81-frame clips; six epochs (one at 512×768, five at 720×1280).

```bash
# Stage 1 — face control (FLAME adapter + motion encoder), lr 2e-5
bash scripts/train_stage1_face.sh

# Stage 2 — multi-view control (camera modulation + patch embeddings), lr 1e-5
bash scripts/train_stage2_multiview.sh

# Stage 3 — joint fine-tuning (rank-64 LoRA on backbone, adapters frozen), lr 1e-5
bash scripts/train_stage3_joint.sh
```

Each script wraps `accelerate launch train.py -cfg configs/training/<stage>.yaml` — edit the dataset root (`-d`) and output paths inside.

## Inference

`inference.py` runs batch multi-view novel-view synthesis on a prepared validation set: for each sequence it takes one reference view, the per-view driving mesh renderings, and the calibrated camera parameters, and synthesizes the held-out target views.

```bash
python inference.py --dataset dna          # DNA-Rendering validation split

# shard across GPUs/processes
python inference.py --dataset dna --num-chunks 8 --chunk-id 0
```

Set the dataset root via the `DATASET_ROOT` environment variable (default `./datasets`); the expected layout is `<preset>/{video,mesh,camera,reference}/...`. The model config is `configs/inference/multiview.yaml`, and the checkpoint path is set by `PRETRAINED_CKPT_PATH` in `inference.py`.

Supported presets: `dna` (per-video camera json), `bili` (per-video camera npz), `mocap` (per-subject camera json). For face-control generation driven by FLAME parameters, use `configs/inference/face-control.yaml` with the same entry.

## Acknowledgements

Built upon [Wan2.2 / WanAnimate](https://github.com/Wan-Video/Wan2.2). Camera ray embedding adapted from [Uni3C](https://github.com/alibaba-damo-academy/Uni3C). FLAME fitting based on EMICA; body meshes estimated with SAM-3D-Body (MHR).
