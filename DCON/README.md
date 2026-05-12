# SAA Training Notes

## Requirements

The current codebase is set up around the package versions in `requirements.txt`, with PyTorch pinned to `torch==1.10.1+cu113` and `torchvision==0.11.2+cu113`.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt
```

Notes:

- The provided training scripts assume a CUDA-enabled PyTorch environment.
- If you prefer to install PyTorch separately for a different CUDA version, make sure the installed `torch` and `torchvision` remain compatible with the rest of `requirements.txt`.

## Dataset Root

Training data is resolved from the environment variable `SAA_DATA_ROOT`. If it is not set, the code falls back to the local `./data` directory under the repository root.

Example:

```bash
export SAA_DATA_ROOT=/path/to/your/data
```

Expected layout:

```text
<SAA_DATA_ROOT>/
  abdominal/
    CHAOST2/
      processed/
        image_*.nii.gz
        label_*.nii.gz
    SABSCT/
      processed/
        image_*.nii.gz
        label_*.nii.gz
  cardiac/
    processed/
      bSSFP/
        image_*.nii.gz
        label_*.nii.gz
      LGE/
        image_*.nii.gz
        label_*.nii.gz
```

## Training Scripts

Run all commands from the repository root.

- `bash scripts/saa_cardiac.sh`: runs the cardiac mainline setting for both source domains `bSSFP` and `LGE`.
- `bash scripts/saa_abdominal.sh`: runs the abdominal mainline setting for both source domains `CHAOST2` and `SABSCT`.

Both scripts call the root entry point `train.py` and save outputs under:

```text
ckpts/<tr_domain>/<expname>/
```

Each experiment directory contains checkpoints, TensorBoard logs, and text logs.

If you want to run a single experiment manually instead of looping over all domains in the shell scripts, use `train.py` directly. Typical dataset settings are:

- `CARDIAC`: `--nclass 4`
- `ABDOMINAL`: `--nclass 5`

Example:

```bash
export SAA_DATA_ROOT=/path/to/your/data

python train.py \
  --data_name CARDIAC \
  --nclass 4 \
  --tr_domain bSSFP \
  --use_sgf 1 \
  --use_rccs 1 \
  --use_saam 1 \
  --lambda_01 1.0 \
  --lambda_02 1.0
```

This codebase now follows the paper terminology:

- `GIP`: Global Intensity Perturbation, used to construct the base view `x^(1)`.
- `CLP`: Class-conditional Local Perturbation, used to construct the local strong-view candidate.
- `SGF`: Saliency-Guided Fusion, used to fuse `GIP` and `CLP` into the pre-RCCS strong view.
- `RCCS`: Random Convolution Candidate Selection, used to select the final strong view `x^(2)`.
- `CGSD`: Channel-Gated Structure-Style Decoupling.
- `SAAM`: Stability-Aware Alignment Module.

## View Construction

The current training path uses three views:

1. `x^(0)`: anchor view, the original image.
2. `x^(1)`: base view, always constructed as dataset-side `GIP`.
3. `x^(2)`: strong view, constructed as `SGF(GIP, CLP)` when `--use_sgf 1`, or as `CLP` directly when `--use_sgf 0`.

In code, the SGF fusion step is:

```python
strong_view = gip * saliency + clp * (1 - saliency)
```

This is the exact implementation of the paper's SGF equation.

## Main CLI Terms

- GIP/CLP construction is always enabled in the dataset pipeline.
- `--use_sgf 1`: enable SGF for strong-view construction. When disabled, the strong view is `CLP`.
- `--sgf_grid_size 8`: saliency map grid size used in SGF.
- `--use_rccs 1`: enable RCCS candidate selection.
- `--use_saam 1`: enable SAAM selective alignment.
- `--lambda_01`: SAAM weight for the anchor-base pair.
- `--lambda_02`: SAAM weight for the anchor-strong pair.

Older aliases such as `--use_sbf`, `--use_triview`, `--lambda_0w`, and `--lambda_0s` are still accepted for compatibility, but the code and logs now use the paper names by default.

## Example

```bash
python train.py \
  --data_name CARDIAC \
  --tr_domain bSSFP \
  --use_sgf 1 \
  --use_rccs 1 \
  --use_saam 1 \
  --lambda_01 1.0 \
  --lambda_02 1.0
```

## TENT Test-Time Adaptation

TENT is available only in test mode and follows the official BN-affine setting:

- freeze the full segmentation network;
- set the network to train mode so BN uses target-batch statistics;
- update only `BatchNorm2d.weight` and `BatchNorm2d.bias`;
- minimize pixel-wise prediction entropy on unlabeled target images.

Example for an existing source-only DCON checkpoint:

```bash
python train.py \
  --phase test \
  --data_name CARDIAC \
  --nclass 4 \
  --tr_domain bSSFP \
  --resume_path ../ckpts/dcon-bl-1200.pth \
  --tta tent \
  --tent_lr 1e-4 \
  --tent_steps 1 \
  --use_cgsd 0 \
  --use_projector 0 \
  --use_saam 0 \
  --use_rccs 0
```

To run the three bundled source-only checkpoints:

```bash
bash scripts/tent_sourceonly_ckpts.sh
```

## One-shot MedSeg-TTA Runs on DCON

The DCON test entry point now supports the portable MedSeg-TTA-style adapters
that can run on the existing DCON U-Net and NIfTI slice datasets:

- source-free/common: `none`, `norm_test`, `norm_alpha`, `norm_ema`, `tent`,
  `dg_tta`, `cotta`, `memo`, `gold`
- source-dependent: `asm`, `sm_ppm`, `gtta`

`dg_tta` is adapted from the MedSeg-TTA output-consistency implementation: it
keeps DCON's model/data/evaluation path and updates only BatchNorm affine
parameters from target-image spatial/intensity consistency.

Run all supported methods across the four DCON shifts:

```bash
export SAA_DATA_ROOT=/Users/RexRyder/PycharmProjects/Dataset
PYTHON_BIN=/path/to/your/env/bin/python bash scripts/run_medseg_tta_dcon.sh
```

Run a subset:

```bash
METHODS="none tent dg_tta gold" \
PYTHON_BIN=/path/to/your/env/bin/python \
bash scripts/run_medseg_tta_dcon.sh
```

The script writes results to `results_medseg_tta` by default and skips the
source-domain re-evaluation for speed. Override with:

```bash
RESULTS_DIR=results_full EVAL_SOURCE_DOMAIN=true SAVE_PREDICTION=true \
bash scripts/run_medseg_tta_dcon.sh
```

When `SAVE_PREDICTION=true`, each run now also writes full-volume
`<scan_id>_image.nii.gz`, `<scan_id>_gt.nii.gz`, and `<scan_id>_pred.nii.gz`
files under:

```text
<result_root>/<source>/<expname>/log/volumes/
```

You can then generate slice-wise GT/pred overlay boards, per-slice metrics, and
a global worst-slice error-analysis board with:

```bash
python3 scripts/visualize_volume_predictions.py \
  --volume-dir <result_root>/<source>/<expname>/log/volumes
```

Summarize existing results:

```bash
python3 scripts/summarize_medseg_tta_dcon.py --roots results_medseg_tta
```

## PASS on DCON

The PASS repository's prompt-based online TTA path is available as `--tta pass`
in the DCON `train.py` test entry point. The adaptation keeps the DCON U-Net,
NIfTI slice loaders, and Dice evaluation path, then wraps the source model with:

- an image-space PASS data adaptor;
- a bottleneck shape prompt injected before DCON's decoder;
- source-BN statistic matching loss on target images;
- an EMA target prompt network and optional source-prediction fallback.

Run PASS across the four bundled DCON shifts:

```bash
export SAA_DATA_ROOT=/Users/RexRyder/PycharmProjects/Dataset
PYTHON_BIN=/path/to/your/env/bin/python bash scripts/run_pass_dcon.sh
```

Useful overrides:

```bash
PASS_STEPS=2 PASS_LR=1e-3 PASS_BN_LAYERS=8 \
PYTHON_BIN=/path/to/your/env/bin/python \
bash scripts/run_pass_dcon.sh
```

The script writes to `results_pass_dcon` by default. Summarize with:

```bash
python3 scripts/summarize_medseg_tta_dcon.py --roots results_pass_dcon --methods pass
```

## SAM-TTA on DCON

SAM-TTA is available as `--tta samtta` in the DCON test entry point. This is a
DCON-native port of the method core rather than a replacement with the SAM
backbone: it keeps DCON's U-Net checkpoints, NIfTI loaders, slice aggregation,
and Dice evaluation, then adds:

- a SAM-TTA-style learnable cubic Bezier input transform;
- an EMA teacher copied from the source model;
- teacher-confidence-weighted high/low resolution prediction consistency;
- bottleneck feature consistency on the U-Net encoder feature;
- lightweight model-side updates, defaulting to BatchNorm affine tensors.

Run all bundled DCON source-only checkpoints:

```bash
export SAA_DATA_ROOT=/Users/RexRyder/PycharmProjects/Dataset
PYTHON_BIN=/path/to/your/env/bin/python bash scripts/run_samtta_dcon.sh
```

Useful overrides:

```bash
SAMTTA_STEPS=2 SAMTTA_TRANSFORM_LR=5e-3 SAMTTA_UPDATE_SCOPE=bn_affine \
SAVE_PREDICTION=false EVAL_SOURCE_DOMAIN=false \
bash scripts/run_samtta_dcon.sh
```

The script writes to `results_samtta_dcon` by default. Summarize with:

```bash
python3 scripts/summarize_medseg_tta_dcon.py --roots results_samtta_dcon --methods samtta
```

## Medical GTTA Test-Time Adaptation

The DCON GTTA adapter is a lightweight medical-image version of GTTA. It uses
source labels for supervised adaptation, creates filtered target pseudo-labels
from the current model, and applies class-aware tensor-space AdaIN to source
images without changing source geometry. Target labels are used only by the
evaluation code.

Run the bundled source-only checkpoints:

```bash
bash scripts/run_gtta_sourceonly_ckpts.sh
```

For TTA comparisons, report only the target-domain `Overall mean dice by sample`
metric printed by `train.py`. Source-only baseline:

| 方法 | SABSCT->CHAOST2 | CHAOST2->SABSCT | bSSFP->LGE | LGE->bSSFP | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.8147 | 0.7769 | 0.8571 | 0.8654 | 0.8285 |

After running GTTA, summarize against the same baseline:

```bash
python3 scripts/summarize_overall_dice.py --methods none gtta
```

The summary script reads each experiment's `log/out.csv`, ignores the later
source-domain evaluation block, and compares only target `Overall mean dice by
sample`.

## SicTTA Test-Time Adaptation

SicTTA is available in test mode as a source-free, single-image continual TTA
adapter. The DCON adapter keeps a frozen source-anchor model for CCD filtering,
stores reliable target bottleneck features in a prototype pool, and decodes the
current slice with the nearest reliable prototype features. Target labels are
used only by the evaluation code.

Run all four bundled source-only DCON checkpoints:

```bash
bash scripts/run_sictta_sourceonly_ckpts.sh
```

Useful overrides:

```bash
SICTTA_MAX_LENS=40 SICTTA_TOPK=5 SICTTA_THRESHOLD=0.9 \
SAVE_PREDICTION=false EVAL_SOURCE_DOMAIN=false \
bash scripts/run_sictta_sourceonly_ckpts.sh
```

## VPTTA Test-Time Adaptation

VPTTA is available as `--tta vptta` in the DCON test entry point. This adapter
keeps DCON's own U-Net and dataset loaders, converts the model BatchNorm layers
to VPTTA-style BN-stat loss layers, and updates only a low-frequency FFT prompt
for each target slice. Target labels are used only by the evaluation code.

Run all bundled source-only checkpoints:

```bash
bash scripts/run_vptta_sourceonly_ckpts.sh
```

Useful overrides:

```bash
GPU_IDS=0 NUM_WORKERS=4 VPTTA_LR=1e-2 VPTTA_STEPS=1 \
  bash scripts/run_vptta_sourceonly_ckpts.sh
```

For a single run:

```bash
python train.py \
  --phase test \
  --data_name CARDIAC \
  --nclass 4 \
  --tr_domain bSSFP \
  --target_domain LGE \
  --resume_path ../ckpts/dcon-bl-1200.pth \
  --tta vptta \
  --vptta_lr 1e-2 \
  --vptta_steps 1 \
  --use_cgsd 0 \
  --use_projector 0 \
  --use_saam 0 \
  --use_rccs 0
```

## SPMO-TTA on DCON

SPMO-TTA is available as `--tta spmo` in the DCON test entry point. The adapter
keeps DCON's U-Net, NIfTI slice loaders, and Dice evaluation path. Instead of
requiring the original SPMO `sizes/*.csv` files, it uses a frozen source model
to build per-slice source-prediction size and shape-moment priors online, then
updates the target model with weighted entropy, size-prior KL, and optional
centroid / distance-to-centroid moment losses. Target labels are used only by
the evaluator.

Run all bundled source-only DCON checkpoints:

```bash
export SAA_DATA_ROOT=/Users/RexRyder/PycharmProjects/Dataset
PYTHON_BIN=/path/to/your/env/bin/python bash scripts/run_spmo_sourceonly_ckpts.sh
```

Useful overrides:

```bash
SPMO_STEPS=2 SPMO_MOMENT_MODE=centroid SPMO_MOMENT_WEIGHT=0.02 \
  bash scripts/run_spmo_sourceonly_ckpts.sh
```

For a single run:

```bash
python train.py \
  --phase test \
  --data_name CARDIAC \
  --nclass 4 \
  --tr_domain bSSFP \
  --target_domain LGE \
  --resume_path ../ckpts/dcon-bl-1200.pth \
  --tta spmo \
  --spmo_lr 1e-4 \
  --spmo_steps 1 \
  --spmo_moment_mode all \
  --use_cgsd 0 \
  --use_projector 0 \
  --use_saam 0 \
  --use_rccs 0
```
