# Medical Segmentation TTA Codebase Implementation Map

This document maps the current repository as implemented. It is intentionally descriptive: it does not propose or apply algorithm changes.

## 1. Overall Framework Architecture

### Repository Layout

The workspace is a collection of TTA codebases plus a DCON medical segmentation integration layer.

| Path | Role |
|---|---|
| `DCON/` | Main medical segmentation entry point currently tying the DCON U-Net, medical NIfTI loaders, SAA training, and DCON-adapted TTA methods together. |
| `DCON/train.py` | Primary CLI for both source training and test-time adaptation. |
| `DCON/models/exp_trainer.py` | Core trainer/evaluator wrapper. Owns U-Net creation, source training losses, TTA configuration, and dispatch to method-specific `te_func_*`. |
| `DCON/dataloaders/` | Cardiac/abdominal NIfTI loaders, GIP/CLP construction, geometric/intensity transforms, RCCS helpers. |
| `DCON/models/tta_*.py`, `DCON/tta_*.py` | DCON-adapted TTA implementations. |
| `DCON/scripts/` | Experiment runners for SAA training and TTA sweeps across domain shifts/checkpoints. |
| `TCA/segmentation/` | Non-medical CARLA segmentation TTA framework. Provides reference implementations for TENT, ASM, GTTA, MEMO, SM-PPM, CoTTA, normalization. |
| `GOLD/segmentation/` | Copy/variant of the TCA segmentation framework with GOLD added. |
| `A3-TTA/`, `SAM-TTA/`, `SicTTA/`, `VPTTA/`, `PASS/`, `SPMO-TTA/` | Standalone upstream-style medical or segmentation TTA repos. DCON ports selected ideas into `DCON/tta_*.py`. |
| `ckpts/` | Bundled DCON source-only checkpoints used by scripts. |

### Main Execution Entry Points

`DCON/train.py` is the main medical entry point.

Training:

```text
train.py --phase train
  -> parse CLI
  -> build source train/val/test and target test datasets
  -> Train_process(opt, istest=0)
  -> per epoch: model.tr_func(train_batch, epoch)
  -> validation with prediction_wrapper()
  -> checkpoint under ckpts/<source>/<expname>/snapshots/
```

Testing/TTA:

```text
train.py --phase test --tta <method> --resume_path <ckpt>
  -> build target test loader
  -> optionally build labeled source train loader for asm/gtta/sm_ppm
  -> Train_process(opt, istest=1, source_loader=...)
  -> configure_<method>()
  -> prediction_wrapper()
       -> for each target slice:
          test_input = {image: base_view, label, names, scan flags}
          gth, pred = model.te_func(test_input)
       -> aggregate slices into volumes
       -> eval_list_wrapper()
```

The test loop is slice-wise but metrics are volume-wise. `prediction_wrapper()` reconstructs each scan using `is_start`, `is_end`, `nframe`, and `scan_id`, then `eval_list_wrapper()` computes Dice with `Train_process.ScoreDiceEval`.

### Core Model

The DCON path uses `DCON/models/unet.py::Unet1`.

Important behavior:

- Input channels are usually `3` because one 2D slice is repeated along channel dimension.
- Forward returns `(logits, x5)` by default, where `x5` is the bottleneck feature.
- Optional CGSD channel gating returns `(logits, x5, f_str, f_sty)` when `return_feat=True`.
- Most TTA adapters assume a model output of `(logits, feature)` and use `x5` as the feature.

### Source Training Pipeline

Source training is handled in `Train_process.tr_func()`.

```text
batch
  anchor_view = original image
  base_view   = GIP image
  strong_view = CLP seed, or SGF(GIP, CLP)

forward base/strong
  -> Dice + CE supervised segmentation losses
  -> optional CGSD structure/style losses
  -> optional SAAM anchor-base/anchor-strong alignment
  -> optional RCCS candidate selection for base or strong view
  -> optimizer_seg step
  -> optional separate optimizer_cgsd step
```

The source training loss components are:

| Component | File | Purpose |
|---|---|---|
| Dice loss | `DCON/models/segloss.py::SoftDiceLoss` | Source supervised segmentation. |
| CE loss | `DCON/models/segloss.py::My_CE` | Source supervised segmentation. |
| CGSD | `DCON/models/unet.py`, `exp_trainer.py` | Channel-gated structure/style feature decoupling. |
| SAAM | `DCON/models/saam.py`, `exp_trainer.py::forward_saam` | Stability-aware alignment between anchor/base/strong features. |
| SGF | `DCON/models/sgf.py` | Saliency-guided fusion of GIP and CLP. |
| RCCS | `DCON/dataloaders/rccs.py` | Random convolution candidate selection by semantic distance. |

### Experiment Organization

Training scripts:

| Script | Scope |
|---|---|
| `DCON/scripts/saa_cardiac.sh` | Trains CARDIAC on `bSSFP` and `LGE` with SAA mainline options. |
| `DCON/scripts/saa_abdominal.sh` | Trains ABDOMINAL on `CHAOST2` and `SABSCT`. |

TTA scripts:

| Script | Scope |
|---|---|
| `run_medseg_tta_dcon.sh` | General TTA runner over the four DCON shifts. |
| `run_common_tta_sourceonly_ckpts.sh` | Common baselines: none, norm, TENT, CoTTA. |
| `run_smppm_sourceonly_ckpts.sh`, `run_smppm_ablation_sourceonly_ckpts.sh` | SM-PPM sweeps/ablations. |
| `run_pass_dcon.sh`, `run_samtta_dcon.sh`, `run_a3_tta_dcon.sh`, etc. | Thin wrappers setting `METHODS=<method>` and delegating to `run_medseg_tta_dcon.sh`. |
| `summarize_medseg_tta_dcon.py` | Summarizes completed `out.csv` results. |

Domain shifts used by `run_medseg_tta_dcon.sh`:

| Dataset | Source -> Target | Checkpoint |
|---|---|---|
| ABDOMINAL | `SABSCT -> CHAOST2` | `../ckpts/dcon-sc-300.pth` |
| ABDOMINAL | `CHAOST2 -> SABSCT` | `../ckpts/dcon-cs-200.pth` |
| CARDIAC | `bSSFP -> LGE` | `../ckpts/dcon-bl-1200.pth` |
| CARDIAC | `LGE -> bSSFP` | `../ckpts/dcon-lb-500.pth` |

## 2. Data Pipeline

### Dataset Roots and Layout

`DCON/dataloaders/CardiacDataset.py` and `AbdominalDataset.py` resolve data from `SAA_DATA_ROOT`; otherwise they fall back to `DCON/data`.

Expected roots:

```text
<SAA_DATA_ROOT>/
  abdominal/
    CHAOST2/processed/image_*.nii.gz
    CHAOST2/processed/label_*.nii.gz
    SABSCT/processed/image_*.nii.gz
    SABSCT/processed/label_*.nii.gz
  cardiac/
    processed/bSSFP/image_*.nii.gz
    processed/bSSFP/label_*.nii.gz
    processed/LGE/image_*.nii.gz
    processed/LGE/label_*.nii.gz
```

### Split Handling

Both cardiac and abdominal loaders split scan IDs by `idx_pct=[0.7, 0.1, 0.2]`:

| Phase | Scan IDs |
|---|---|
| `train` / `trainsup` | source training IDs |
| `trval` | source validation IDs |
| `trtest` | held-out source test IDs |
| `test` | all target IDs in train/test/val order |

Target domain defaults are inferred:

| Dataset | If source is | Target defaults to |
|---|---|---|
| ABDOMINAL | `SABSCT` | `CHAOST2` |
| ABDOMINAL | other/`CHAOST2` | `SABSCT` |
| CARDIAC | `LGE` | `bSSFP` |
| CARDIAC | other/`bSSFP` | `LGE` |

`--target_domain` can override the target.

### Preprocessing and Batch Fields

Images and labels are loaded as full NIfTI volumes, transposed into `[H, W, Z]`, then represented as slice samples. Evaluation loaders keep `batch_size=1` so scan reconstruction is deterministic.

Common evaluation batch fields:

| Field | Meaning |
|---|---|
| `base_view` | Test image tensor, repeated to 3 channels when `tile_z_dim=3`. |
| `anchor_view` | Same original image path as `base_view` in non-training phases. |
| `label` | Ground truth. Used only for evaluation in TTA. |
| `is_start`, `is_end`, `nframe`, `scan_id`, `z_id` | Volume reconstruction metadata. |

Training batch fields:

| Field | Meaning |
|---|---|
| `anchor_view` | Original normalized source slice. |
| `base_view` | GIP view after shared geometry. |
| `strong_view` | CLP view after shared geometry; later fused by SGF if enabled. |
| `label` | Source label after the shared geometric transform. |

### GIP, CLP, SGF

GIP/CLP are dataset-side training augmentations, implemented through `LocationScaleAugmentation`:

- GIP: `Global_Location_Scale_Augmentation()`, applies nonlinear intensity mapping plus global location/scale intensity perturbation.
- CLP: `Local_Location_Scale_Augmentation()`, applies class-conditional local perturbations using the source label.

Training construction:

```text
raw normalized image
  -> denormalize to [0,1]
  -> GIP(raw)
  -> CLP(raw, source_label)
  -> renormalize
  -> concatenate [original, GIP, CLP, label]
  -> shared geometric transform
  -> optional intensity transform for SGF path
  -> base_view=GIP, strong_view=CLP
```

If `--use_sgf 1`, `Train_process.tr_func()` computes a saliency map from the gradient of the base-view segmentation loss and builds:

```text
strong = GIP * saliency + CLP * (1 - saliency)
```

### Source-Target Interaction

Source-target interaction only happens in selected TTA methods during `--phase test`.

| Method family | Source loader at test? | Target labels used? | Target images used for adaptation? |
|---|---:|---:|---:|
| none / norm / TENT / CoTTA / MEMO / GOLD / VPTTA / PASS / SAM-TTA / SPMO / SicTTA / A3-TTA | No | No | Yes |
| ASM | Yes | No | Yes, only for target style/statistics |
| GTTA | Yes | No | Yes, for pseudo-labels and target moments |
| SM-PPM full/source CE/SM/PPM ablations | Yes except `source_free_proto` | No | Yes, for style statistics/prototypes |

## 3. Method-by-Method Implementation Analysis

### Implementation Table

| Method | DCON file(s) | Trainable parameters | Main loss / update | Source data at test | Teacher / EMA / memory / prototype |
|---|---|---|---|---:|---|
| none | `exp_trainer.py::te_func` | none | forward only | No | No |
| norm_test | `AlphaBatchNorm2d` in `exp_trainer.py` | none | batch-stat BN with `alpha=1` | No | Source BN stats retained in wrapper |
| norm_alpha | same | none | `(1-alpha)*source_stats + alpha*batch_stats` | No | Source BN stats |
| norm_ema | `configure_norm_ema`, `te_func_norm` | BN running stats only | forward in train mode updates BN stats | No | Running BN state |
| TENT | `configure_tent`, `te_func_tent` | `BatchNorm2d.weight/bias` | pixel entropy minimization | No | No |
| DG-TTA | `DCON/tta_dg_tta.py` | BN affine | symmetric KL between original/aug logits + entropy + BN L2 | No | Source BN affine snapshot |
| CoTTA | `exp_trainer.py` | all model params | student cross-entropy to EMA teacher prediction | No | EMA teacher, anchor model, stochastic restore |
| MEMO | `DCON/tta_memo.py`, `te_func_memo` | all params or BN affine | marginal entropy over augmented test views | No | No |
| ASM | `DCON/models/tta_asm.py` | all model params | supervised source Dice+CE on target-stylized source + feature L2 | Yes | Target style sampling variable |
| SM-PPM | `DCON/models/tta_smppm.py` | all model params | weighted source Dice+CE/CE, or target entropy+prototype compactness | Conditional | Target patch prototypes; optional reliable target mask |
| GTTA | `DCON/models/tta_gtta.py` | all model params | supervised source loss + optional target pseudo-label CE | Yes | Running pseudo confidence threshold; class moments |
| GOLD | `DCON/tta_gold.py` | all model params plus low-rank scale vector | CoTTA-like student loss + low-rank feature adapter loss | No | EMA teacher, anchor, AGOP matrix/subspace |
| VPTTA | `DCON/tta_vptta.py` | low-frequency prompt only | BN-stat matching loss from converted BN layers | No | Prompt memory bank |
| PASS | `DCON/tta_pass.py` | prompts, adaptor, optional BN affine | source-BN feature-stat matching, optional entropy | No | EMA target prompt model, frozen source model fallback |
| SAM-TTA | `DCON/tta_samtta.py` | Bezier transform + chosen model scope | teacher-student prediction consistency + feature KL + entropy + transform regularization | No | EMA teacher |
| SPMO / SpaceMoment | `DCON/tta_spmo.py` | BN affine or all params | entropy + source-predicted size prior KL + shape moment loss | No data loader; frozen source model only | Frozen source model supplies priors |
| SicTTA | `DCON/tta_sictta.py` | none by optimizer; BN uses target batch stats | feature prototype mixing gated by CCD reliability | No | Anchor model, bottleneck prototype pool |
| A3-TTA | `DCON/tta_a3_tta.py` | all model params | aligned-feature teacher CE + entropy matching + EMA consistency | No | EMA teacher, anchor, prototype pool |

### TENT

Core DCON path:

- `configure_model_for_tent()` freezes the full model, sets train mode, and sets every `BatchNorm2d` to use target-batch statistics (`track_running_stats=False`, `running_mean=None`, `running_var=None`).
- `collect_bn_affine_params()` returns only BN affine `weight` and `bias`.
- `te_func_tent()` runs `tent_steps` updates on the current target slice and minimizes `softmax_entropy_seg(logits)`.

Instability risk: pure entropy minimization can sharpen wrong predictions. In this code the update surface is constrained to BN affine tensors, which limits but does not remove confirmation bias.

### Normalization-Based Methods

`norm_test` and `norm_alpha` replace each `BatchNorm2d` with `AlphaBatchNorm2d`.

```text
running_mean = (1-alpha) * source_running_mean + alpha * current_batch_mean
running_var  = (1-alpha) * source_running_var  + alpha * current_batch_var
```

`norm_test` uses `alpha=1.0`; `norm_alpha` uses `--bn_alpha`. `norm_ema` puts the model in train mode once per target batch to update BN running stats, then evaluates.

Risk: single-slice batch statistics can be noisy. `norm_alpha` is more source-anchored than `norm_test`.

### MEMO

DCON MEMO builds conservative medical augmentations in `tta_memo.py`:

- optional identity view
- affine warp
- optional horizontal flip
- monotone curve, location-scale, gamma, brightness/contrast, noise in per-slice intensity range

`te_func_memo()` updates either all model parameters or BN affine params using marginal entropy over the augmented batch, then predicts the original slice.

Risk: if augmentations are not label-preserving for a modality/anatomy, marginal entropy can encourage invariance to invalid transformations.

### ASM

DCON ASM is source-dependent.

Flow:

```text
target image
  -> target channel mean/std
source batch + labels
  -> AdaIN-like tensor-space style transfer from target to source
  -> concatenate [stylized_source, original_source]
  -> supervised segmentation loss on duplicated source labels
  -> optional bottleneck feature L2 regularization
  -> update model
  -> predict target image
```

Important files:

- `DCON/models/tta_asm.py`
- `Train_process.configure_asm()`
- `Train_process.asm_segmentation_loss()`

The target image supplies style/statistics only. No target pseudo-label or target label is used. Instability can come from style transfer that changes medical contrast in a way that makes the source label less aligned with the resulting image appearance.

### GTTA

DCON GTTA is source-dependent and uses both source supervision and filtered target pseudo-labels.

Flow:

```text
target image
  -> model pseudo-label
  -> update running avg_conf
  -> threshold: confidence >= sqrt(avg_conf)
  -> target class-wise image moments
source batch + labels
  -> class-aware AdaIN from target moments
  -> supervised source Dice+CE update
target image + filtered pseudo labels
  -> optional target CE update weighted by lambda_ce_trg
```

Instability risk: target pseudo-label filtering depends on current model confidence. Wrong high-confidence target regions influence both target CE and class-wise moments.

### GOLD

DCON GOLD combines:

- CoTTA-like student/EMA/anchor update
- stochastic restore to source parameters
- low-rank pre-classifier feature adapter
- AGOP matrix estimated from confident teacher pixels

It hooks the classifier layer (`seg1` or equivalent class-output conv), captures pre-classifier features, updates an AGOP matrix from confident teacher pixels, refreshes an eigenspace periodically, and trains a low-rank scale vector `S`.

Risk: if teacher confidence is misplaced, the AGOP subspace and adapter scale can reinforce wrong structures. Stochastic restore and EMA reduce drift.

### VPTTA

DCON VPTTA:

- replaces each `BatchNorm2d` with `VPTTABatchNorm2d`;
- freezes the segmentation network;
- learns only a low-frequency Fourier amplitude prompt;
- uses a prompt memory bank to initialize the current prompt from nearest previous low-frequency keys;
- optimizes converted BN statistic matching loss.

This is source-free in the sense that no source data loader is used, but it relies on source BN running statistics inside the checkpoint.

### PASS

DCON PASS wraps the U-Net in `PASSPromptedUNet`:

- image-space data adaptor before the backbone;
- bottleneck shape prompt injected before decoding;
- prompt cross-attention;
- BN feature monitor comparing target BN inputs to source running means/vars;
- EMA target prompt model;
- optional fallback to frozen source prediction if adapted entropy is worse.

Trainable surface is prompt/adaptor plus optional BN affine tensors. It does not read source data at test time but uses source BN statistics and a frozen source model.

### SAM-TTA

DCON SAM-TTA is a U-Net adaptation of the SAM-TTA idea, not the full SAM architecture.

It adds:

- learnable cubic Bezier intensity transform;
- EMA teacher;
- teacher-student dual-scale prediction consistency;
- bottleneck feature spatial KL;
- entropy minimization;
- transform regularization toward the input.

Trainable model scope is controlled by `--samtta_update_scope`: `transform_only`, `bn_affine`, or `all`.

### SPMO / SpaceMoment

`DCON/tta_spmo.py` ports the shape-prior/moment idea without reading size CSVs. A frozen copy of the source model predicts the current target slice, then the online model is optimized with:

- class entropy with background down-weighting;
- KL to source-predicted class proportions;
- foreground centroid and spread moment matching.

It is source-free with respect to test-time data access, but source-anchored through the frozen source checkpoint prediction.

### SicTTA

SicTTA uses no optimizer in the DCON port. It:

- configures BN to use target-batch stats;
- maintains a bottleneck prototype pool;
- computes a CCD entropy score from anchor/source-model predictions;
- accepts reliable target features into the pool;
- mixes current bottleneck feature with nearest pool features before decoding.

Risk: early pool contents matter. The code lowers the threshold while the pool is small, then uses CCD acceptance once the pool has enough entries.

### A3-TTA

A3-TTA updates all model parameters online. It:

- computes CCD scores from the current model;
- stores low-CCD bottleneck features in a prototype pool;
- aligns current bottleneck features to nearest memory prototypes;
- optimizes aligned-feature prediction against current logits, entropy-map matching, and EMA teacher consistency;
- updates an EMA model with dynamic momentum capped by `--a3_mt`.

Risk: prototype memory and EMA teacher are both model-generated. Wrong early prototypes can bias later updates unless episodic/reset options are used.

## 4. SM-PPM Deep Analysis

### Current DCON SM-PPM Modes

`SMPPMAdapter` supports:

| Mode | Source loader | SM style mixing | PPM prototype weighting | Loss |
|---|---:|---:|---:|---|
| `full` | Yes | Yes | Yes | weighted DCON Dice+CE |
| `source_ce_only` | Yes | No | No | weighted CE with weights all ones |
| `sm_ce` | Yes | Yes | No | CE on stylized source |
| `ppm_ce` | Yes | No | Yes | CE weighted by PPM and entropy |
| `source_free_proto` | No | No | target-only | masked entropy + target feature compactness |

Note: `DCON/scripts/run_smppm_ablation_sourceonly_ckpts.sh` contains a stale comment saying `sm_ce` is excluded because there is no explicit SM implementation. The current `DCON/models/tta_smppm.py` does implement tensor-space AdaIN style mixing and allows `sm_ce`.

### Source Supervision Path

For source-dependent modes:

```text
target_img
  -> if PPM: _target_prototypes(target_img)

for each step:
  source_img, source_label = next(source_loader)
  if SM: adapt_img = medical_adain_style_mix(source_img, target_img)
  else:  adapt_img = source_img
  logits, source_feature = model(adapt_img)
  entropy = normalized_entropy(logits).detach()
  if PPM:
      confidence = max cosine similarity(source_feature, target prototypes)
      pixel_weight = confidence * (1 - entropy)
  else:
      pixel_weight = 1
  loss = source CE or weighted Dice+CE using source_label
  optimizer.step()

predict target_img with adapted model
```

The strong anchor is still the source label. Target information changes either image statistics (SM) or per-pixel source loss weights (PPM).

### Prototype Weighting

`_target_prototypes()`:

1. runs the current model on the target image;
2. extracts bottleneck feature `x5`;
3. resizes it to `feature_size x feature_size`;
4. splits into non-overlapping `patch_size x patch_size` patches;
5. averages each patch over batch and spatial patch pixels;
6. L2-normalizes prototypes.

`_similarity_confidence()`:

1. L2-normalizes source features;
2. computes cosine similarity to every target patch prototype;
3. keeps max similarity per source pixel;
4. maps `[-1, 1]` to `[0, 1]`;
5. upsamples to label resolution.

Final source pixel weight:

```text
pixel_weight = prototype_confidence * (1 - normalized_entropy(source_logits))
```

This weights supervised source pixels that look target-like and are currently predicted confidently.

### Style Mixing

`medical_adain_style_mix()` is tensor-space AdaIN:

```text
source_norm = (source - source_mean) / source_std
stylized = source_norm * target_std + target_mean
output = alpha * stylized + (1-alpha) * source
```

It transfers per-sample/channel target mean/std to source images. Labels remain source labels. This is safer than geometric target/source mixing, but it only captures low-order intensity moments.

### Target Feature Extraction

Target features are extracted with the current online model before the source update. Because the model is adapted over time, target prototypes are not fixed source-model prototypes; they drift with the online state.

This is useful when adaptation improves features, but it also creates a feedback loop:

```text
current model -> target prototypes -> source update weights -> updated model -> next prototypes
```

### Adaptation Update Logic

SM-PPM config in `Train_process.configure_smppm()`:

- sets the model to train mode;
- enables gradients on all segmentation parameters;
- uses SGD with momentum, weight decay, and Nesterov;
- stores optional source/optimizer state if episodic;
- delegates adaptation to `SMPPMAdapter.forward()`.

The target label loaded by the dataset is returned to the evaluator only. It is not passed to SM-PPM losses.

### Interaction Between Source CE and Target Information

In `full`, target information does not directly define labels. It only:

- changes source image intensity statistics through SM;
- weights source supervised pixels by target feature similarity and source prediction entropy through PPM.

Thus the adaptation objective remains source-supervised. The method is source-anchored and should resist target pseudo-label collapse better than entropy-only methods. The tradeoff is that it may under-adapt if source labels and target appearance remain poorly matched after moment-level SM.

### What Appears to Contribute

Based on implementation mechanics:

- Source CE/Dice is the strongest stabilizer because it is the only ground-truth supervision during adaptation.
- SM can help when target shift is primarily intensity/statistics and source geometry remains valid.
- PPM can focus updates on source pixels whose bottleneck representation resembles target patches.
- Entropy in the PPM weight suppresses uncertain source predictions, but it is not a target reliability estimate.

### Weak Points and Likely Bottlenecks

Likely bottlenecks in the current DCON SM-PPM path:

1. Target prototypes are class-agnostic patch prototypes. They do not know which organ/anatomical class they represent.
2. PPM weights source pixels by max similarity to any target patch. Background-like or ambiguous target patches can dominate.
3. Source entropy in `pixel_weight` measures confidence on adapted source images, not target reliability.
4. Style mixing uses only global per-channel mean/std; local organ-specific contrast shifts are not modeled in `full`.
5. Full model SGD updates can drift if a single target slice produces unrepresentative prototypes.
6. `feature_size`/`patch_size` compress target information heavily. With default `32/8`, only 16 target prototypes are built per target batch.

### Natural Insertion Points for Stability-Aware Weighting

Stable additions can be inserted without refactoring the whole adapter:

| Insertion point | File/function | Natural signal |
|---|---|---|
| After `_target_prototypes()` | `SMPPMAdapter._forward_source_dependent` | filter or weight target prototypes before source update |
| Inside `_similarity_confidence()` | `SMPPMAdapter._similarity_confidence` | combine similarity with prototype reliability |
| Before `pixel_weight` is formed | `_adapt_one_source_batch` | multiply by stable-region / reliable-region mask |
| In `source_free_proto` reliable mask | `_source_free_losses` | replace confidence-only mask with stability-aware mask |
| Before optimizer step | `_adapt_one_source_batch`, `_adapt_one_target_batch` | skip update when reliable mass is too small |
| Logging | `last_losses` | expose reliable fraction, prototype count, weight distribution |

## 5. Adaptation Taxonomy

| Category | Definition | Current methods |
|---|---|---|
| Target-only TTA | Uses only target images at test time, no source data loader, no frozen source prediction except checkpoint parameters. | TENT, MEMO, SAM-TTA transform-only/BN-affine, normalization methods. |
| Source-free but source-prior anchored | No source samples at test time, but uses source checkpoint statistics, source model copy, or source parameters as priors. | norm_alpha, norm_ema, CoTTA, GOLD, VPTTA, PASS, SPMO, SicTTA, A3-TTA. |
| Source-relaxed | Does not require raw source samples, but uses frozen source predictions or source BN statistics strongly enough that it is not purely target-only. | PASS, SPMO, VPTTA, CoTTA/GOLD via anchor/restore. |
| Source-access / source-anchored | Requires labeled source loader during test-time adaptation. | ASM, GTTA, SM-PPM source-dependent modes. |

Practical difference:

- Target-only/source-free methods are easier to deploy when raw source data is unavailable.
- Source-prior anchored methods usually have better drift control than entropy-only methods but still depend on checkpoint statistics being reliable.
- Source-access methods can use real supervised losses at test time, but they require retaining source data and are slower because each target slice triggers source minibatch updates.

## 6. Stable Extension Opportunities

### Stability Estimation

Already implemented training-side:

- SAAM computes three-view feature stability in `DCON/models/saam.py`.
- It uses pairwise cosine distances, top-k stable selection, and soft reliability `exp(-d/tau)`.

Candidate extension points:

- Reuse SAAM-like pairwise stability for test augmentations in TENT/MEMO/SAM-TTA.
- Compute target prototype stability in SM-PPM using multiple target views before `_target_prototypes()`.
- Add scan-temporal stability for online slice methods using `is_start/is_end/z_id`.

### Reliable-Region Masking

Existing masks:

- SM-PPM `source_free_proto` uses `max_prob > tau` and optional entropy threshold.
- GTTA filters pseudo-label pixels with `confidence >= sqrt(avg_conf)`.
- GOLD samples only teacher pixels above `gold_tau`.
- SicTTA and A3 use CCD scores to accept features into memory.

Easiest additions:

- Add target confidence/entropy filtering to SM-PPM target prototypes.
- Add minimum reliable mass checks before entropy-based methods update.
- Add organ foreground masks for source-anchored methods when reliable target foreground is available.

### Prototype Filtering

Prototype-heavy methods:

- SM-PPM target patch prototypes.
- SicTTA bottleneck prototype pool.
- A3-TTA bottleneck memory pool.
- PASS bottleneck prompt.

Natural filters:

- confidence-weighted prototype construction;
- class-aware prototype grouping from pseudo-labels;
- entropy or consistency threshold per prototype;
- reject prototypes with too few foreground pixels or excessive background dominance.

### Source Anchoring Already Present

Source anchoring exists through:

- source labels in ASM/GTTA/SM-PPM;
- frozen source model in PASS/SPMO/SicTTA/A3;
- source BN stats in normalization/VPTTA/PASS;
- stochastic restore in CoTTA/GOLD;
- supervised source CE/Dice in source-dependent methods.

### Easiest Methods to Extend Safely

| Method | Why |
|---|---|
| SM-PPM | Adapter is isolated; prototype/weight formation is explicit; source supervision already stabilizes updates. |
| GTTA | Pseudo-label threshold and target moments are explicit. |
| SAM-TTA | Loss terms and update scopes are cleanly separated. |
| SPMO | Priors are computed in one function and losses are scalar-weighted. |
| VPTTA | Only prompt is trainable; low blast radius. |

## 7. Risks and Confirmation Bias Hotspots

| Method | Main risk |
|---|---|
| TENT | Entropy minimization sharpens wrong predictions. |
| MEMO | Invalid augmentations can force bad invariance. |
| CoTTA/GOLD | EMA teacher can reinforce early mistakes; restore helps. |
| GTTA | Pseudo-labels and target moments come from the current model. |
| SM-PPM | Target prototypes are class-agnostic and can drift with the online model. |
| VPTTA | BN loss may overfit prompt to noisy single-slice stats. |
| PASS | BN-stat matching may not correlate with segmentation correctness; entropy fallback is coarse. |
| SAM-TTA | Teacher and student start from same model; consistency can preserve errors. |
| SPMO | Frozen source prediction supplies priors; wrong source prediction constrains the adapted model. |
| SicTTA/A3-TTA | Memory/prototype pool quality is crucial; early accepted errors can persist. |

## 8. Reference vs DCON Implementations

The TCA/GOLD segmentation directories provide the original/reference-style framework:

```text
TCA/segmentation/test_time.py
  -> yacs config
  -> load CARLA DeepLab model
  -> create CARLA loader
  -> setup_<method>()
  -> evaluate_sequence()
```

DCON does not directly run those CARLA loaders/models for medical experiments. Instead, DCON ports method cores into DCON-specific adapters while keeping:

- DCON U-Net;
- DCON NIfTI slice loaders;
- DCON Dice evaluation;
- DCON checkpoint format;
- DCON CLI/script organization.

Important implementation differences:

| Method | TCA/GOLD reference | DCON medical port |
|---|---|---|
| ASM | RAIN/VGG style transfer networks and CARLA source loader. | Tensor-space medical AdaIN, no external style net. |
| GTTA | AdaIN transfer net with class-wise feature moments. | Tensor-space class-aware image moments. |
| SM-PPM | CARLA DeepLab feature list and layer-4 prototypes. | DCON U-Net bottleneck `x5` prototypes. |
| MEMO | PIL/ImageNet-style augmentations. | Medical affine/intensity augmentations on normalized slices. |
| VPTTA/PASS/SAM/SicTTA/A3/SPMO | Standalone repos have their own data/model assumptions. | DCON adapters wrap or reconstruct equivalent behavior around `Unet1`. |

## 9. Minimal Execution Flow Diagrams

### Source Training

```text
NIfTI source volume
  -> source split
  -> slice
  -> source normalization
  -> anchor original
  -> GIP + CLP
  -> shared geometry
  -> optional SGF/RCCS
  -> Unet1
  -> Dice + CE
  -> optional CGSD + SAAM
  -> optimizer step
```

### Target Evaluation Without Source Loader

```text
NIfTI target volume
  -> slice loader batch_size=1
  -> Train_process.te_func()
  -> method-specific online update, if any
  -> predicted slice
  -> volume reconstruction
  -> Efficient_DiceScore
```

### Source-Dependent TTA

```text
target slice
  -> method extracts target stats/prototypes/pseudo labels
  -> fetch labeled source batch
  -> source-supervised update influenced by target information
  -> predict target slice
```

## 10. Ground Rules for Future Modifications

- Do not use target labels in adapter losses. The current DCON adapters consistently return labels only for evaluation.
- Preserve `prediction_wrapper()` scan reconstruction assumptions: test batch size is `1`, scan flags matter.
- Be explicit about whether a method needs `source_loader`; `train.py` only constructs it for ASM, GTTA, and source-dependent SM-PPM.
- If changing SM-PPM, keep ablation modes truthful and update scripts/comments together.
- If adding stability masks, log reliable mass/weight statistics so failed updates are diagnosable.
- If updating all model parameters online, provide an episodic/reset option or an anchoring mechanism.
