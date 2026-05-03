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
