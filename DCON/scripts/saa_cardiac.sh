#!/bin/bash
# SAA mainline training script for the CARDIAC dataset.
#
# View construction:
# 1. View1 = GIP(x)
# 2. View2 = SGF(GIP, CLP)
# 3. Loss = Dice + CE + CGSD + SAAM + RCCS

domains=('bSSFP' 'LGE')

for dm in "${domains[@]}"
do
   echo "=========================================="
   echo "SAA mainline: training CARDIAC on domain $dm"
   echo "=========================================="

   python train.py \
   --use_sgf 1 \
   --sgf_grid_size 18 \
   --num_workers 8 \
   --expname     "saa_baseline_${dm}" \
   --phase       'train' \
   --ckpt_dir    './ckpts' \
   --gpu_ids      '0' \
   --f_seed       42 \
   --lr           0.0005 \
   --model        'unet' \
   --batchSize    20 \
   --all_epoch    1800 \
   --validation_freq 50 \
   --display_freq 5000 \
   --save_freq    100 \
   --data_name    'CARDIAC' \
   --nclass       4 \
   --tr_domain    $dm \
   --save_prediction True \
   --w_ce         1.0 \
   --w_dice       1.0 \
   --w_seg        1.0 \
   --use_cgsd      1 \
   --cgsd_layer    1 \
   --use_projector 1 \
   --use_separate_cgsd_optimizer 1 \
   --lambda_str    0.3 \
   --lambda_sty    0.3 \
   --use_saam      1 \
   --saam_tau      0.5 \
   --saam_topk     0.3 \
   --saam_stability_mode mean \
   --lambda_01     1.0 \
   --lambda_02     1.0 \
   --saam_warmup_epochs 50 \
   --saam_rampup_epochs 100 \
   --anchor_seg_alpha 0.0 \
   --strong_seg_alpha 1.0 \
   --use_rccs      1 \
   --p_rccs        0.3 \
   --rccs_candidates 4 \
   --rccs_metric   cos \
   --rccs_embed_dim 128

   echo ""
   echo "Finished training domain: $dm"
   echo ""
done

echo "=========================================="
echo "SAA mainline CARDIAC experiments completed!"
echo "=========================================="
