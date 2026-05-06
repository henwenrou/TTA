
整理一下昨天部署的4个脚本 里面各自方便对应的4个任务，整理一下跑出来的分数，效果如何，有哪些脚本没跑完？
(tpsdg) root@a24ecaf4e701:~/TTA/DCON# bash scripts/run_medseg_tta_dcon.sh && \
bash scripts/run_vptta_sourceonly_ckpts.sh && \
bash DCON/scripts/run_sictta_sourceonly_ckpts.sh && \
bash scripts/run_pass_dcon.sh
Project: /root/TTA/DCON
Python: python
SAA_DATA_ROOT: /root/TTA/DCON/data
Results: results_medseg_tta
Methods: none norm_test norm_alpha norm_ema tent dg_tta cotta memo asm sm_ppm gtta gold

==========================================
none: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: none_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/none_dcon_sabsct_to_chaost2
================================================================================

[22:32:07.244] data:ABDOMINAL
[22:32:07.244] tta:none
labmap: {'0': 0, '1': 63, '2': 126, '3': 189, '4': 255}
get_train abd: ['SABSCT']
Applying GIP/CLP location-scale augmentation on train split
train_SABSCT: Using fold data statistics for normalization
For train on ['SABSCT'] using scan ids {'SABSCT': ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']}
get_trval abd: ['SABSCT']
For trval on ['SABSCT'] using scan ids {'SABSCT': ['6', '7', '8']}
get_trtest abd: ['SABSCT']
For trtest on ['SABSCT'] using scan ids {'SABSCT': ['0', '1', '2', '3', '4', '5']}
get_test abd: ['CHAOST2']
test_CHAOST2: Using fold data statistics for normalization
For test on ['CHAOST2'] using scan ids {'CHAOST2': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '0', '1', '2', '3', '4', '5']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: none
================================================================================

Loading checkpoint: ../ckpts/dcon-sc-300.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-sc-300.pth
optimizer_seg initialized: 106 parameter groups (includes all)

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:08<00:00, 65.95it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9825768500566483 
, std: 0.0051070600697923334
Organ liver with dice: mean: 0.8547287076711655 
, std: 0.050440316957574824
Organ rk with dice: mean: 0.8474860429763794 
, std: 0.04579864552548221
Organ lk with dice: mean: 0.8327155232429504 
, std: 0.057534872737115106
Organ spleen with dice: mean: 0.7239555567502975 
, std: 0.07266962235215355
Overall mean dice by sample 0.8147214576601982
Overall mean dice by domain 0.8147214651107788

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/none_dcon_sabsct_to_chaost2/log

==========================================
none: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: none_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/none_dcon_chaost2_to_sabsct
===================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [00:25<00:00, 68.22it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9823474287986755 
, std: 0.006294773487884593
Organ liver with dice: mean: 0.8787498513857523 
, std: 0.04363842193868433
Organ rk with dice: mean: 0.8017500375707944 
, std: 0.16138532787994292
Organ lk with dice: mean: 0.7915036708116532 
, std: 0.11946161639593475
Organ spleen with dice: mean: 0.6354031572739284 
, std: 0.1418681897726255
Overall mean dice by sample 0.7768516792605321
Overall mean dice by domain 0.7768516540527344

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/none_dcon_chaost2_to_sabsct/log

==========================================
none: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: none_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/none_dcon_bssfp_to_lge
=========================================================
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [00:10<00:00, 66.88it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9944673127598233 
, std: 0.0017869722677870533
Organ LV with dice: mean: 0.7900222910775079 
, std: 0.05877936659421599
Organ Myo with dice: mean: 0.9030750486585829 
, std: 0.04723790584105998
Organ RV with dice: mean: 0.8781003475189209 
, std: 0.04943508607685293
Overall mean dice by sample 0.8570658957516706
Overall mean dice by domain 0.8570659160614014

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/none_dcon_bssfp_to_lge/log

==========================================
none: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: none_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/LGE/none_dcon_lge_to_bssfp
================================================================================
3', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45']}
get_trval cardiac: ['LGE']
trval_LGE: Using fold data statistics for normalization
For trval on ['LGE'] using scan ids {'LGE': ['11', '12', '13', '14']}
get_trtest cardiac: ['LGE']
trtest_LGE: Using fold data statistics for normalization
For trtest on ['LGE'] using scan ids {'LGE': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}
get_test cardiac: ['bSSFP']
test_bSSFP: Using fold data statistics for normalization
For test on ['bSSFP'] using scan ids {'bSSFP': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: none
================================================================================

Loading checkpoint: ../ckpts/dcon-lb-500.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-lb-500.pth
optimizer_seg initialized: 106 parameter groups (includes all)

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:08<00:00, 68.39it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9932652195294698 
, std: 0.0016873779733531707
Organ LV with dice: mean: 0.7953795472780864 
, std: 0.035416717208508426
Organ Myo with dice: mean: 0.9095634897549947 
, std: 0.027358264146046165
Organ RV with dice: mean: 0.8912999576992459 
, std: 0.036197308811298665
Overall mean dice by sample 0.8654143315774423
Overall mean dice by domain 0.8654143214225769

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/none_dcon_lge_to_bssfp/log

==========================================
norm_test: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: norm_test_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/norm_test_dcon_sabsct_to_chaost2
================================================================================

[22:34:52.581] 
labmap: {'0': 0, '1': 63, '2': 126, '3': 189, '4': 255}
get_train abd: ['SABSCT']
Applying GIP/CLP location-scale augmentation on train split
train_SABSCT: Using fold data statistics for normalization
For train on ['SABSCT'] using scan ids {'SABSCT': ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']}
get_trval abd: ['SABSCT']
For trval on ['SABSCT'] using scan ids {'SABSCT': ['6', '7', '8']}
get_trtest abd: ['SABSCT']
For trtest on ['SABSCT'] using scan ids {'SABSCT': ['0', '1', '2', '3', '4', '5']}
get_test abd: ['CHAOST2']
test_CHAOST2: Using fold data statistics for normalization
For test on ['CHAOST2'] using scan ids {'CHAOST2': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '0', '1', '2', '3', '4', '5']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_test
================================================================================

Loading checkpoint: ../ckpts/dcon-sc-300.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-sc-300.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_test enabled: replaced 26 BatchNorm2d layers with alpha=1.0

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:10<00:00, 50.21it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9769440025091172 
, std: 0.006040109441229794
Organ liver with dice: mean: 0.7910270690917969 
, std: 0.05388203360700964
Organ rk with dice: mean: 0.8260813027620315 
, std: 0.0526249505197338
Organ lk with dice: mean: 0.824672743678093 
, std: 0.050213839689762146
Organ spleen with dice: mean: 0.7977015972137451 
, std: 0.06987895476005226
Overall mean dice by sample 0.8098706781864167
Overall mean dice by domain 0.809870719909668

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/norm_test_dcon_sabsct_to_chaost2/log

==========================================
norm_test: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: norm_test_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/norm_test_dcon_chaost2_to_sabsct
================================================================================

================================================================================
TEST MODE
TTA: norm_test
================================================================================

Loading checkpoint: ../ckpts/dcon-cs-200.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-cs-200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_test enabled: replaced 26 BatchNorm2d layers with alpha=1.0

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [00:34<00:00, 50.11it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9720552245775859 
, std: 0.007365227074717524
Organ liver with dice: mean: 0.7972316245237986 
, std: 0.05964053931478433
Organ rk with dice: mean: 0.7509276300668717 
, std: 0.16204752871100767
Organ lk with dice: mean: 0.6940133839845657 
, std: 0.13131812486489858
Organ spleen with dice: mean: 0.5783436954021454 
, std: 0.14589773579023071
Overall mean dice by sample 0.7051290834943453
Overall mean dice by domain 0.7051289677619934

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/norm_test_dcon_chaost2_to_sabsct/log

==========================================
norm_test: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: norm_test_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/norm_test_dcon_bssfp_to_lge
================================================================================
'41', '42', '43', '44', '45', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_test
================================================================================

Loading checkpoint: ../ckpts/dcon-bl-1200.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-bl-1200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_test enabled: replaced 26 BatchNorm2d layers with alpha=1.0

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [00:14<00:00, 51.78it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.990351156393687 
, std: 0.0019028940871405762
Organ LV with dice: mean: 0.7719918171564738 
, std: 0.05499716031432146
Organ Myo with dice: mean: 0.8465773264567057 
, std: 0.05684167119996433
Organ RV with dice: mean: 0.8313772612147861 
, std: 0.0601381279448013
Overall mean dice by sample 0.8166488016093219
Overall mean dice by domain 0.8166487812995911

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/norm_test_dcon_bssfp_to_lge/log

==========================================
norm_test: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: norm_test_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/LGE/norm_test_dcon_lge_to_bssfp
================================================================================
 '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_test
================================================================================

Loading checkpoint: ../ckpts/dcon-lb-500.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-lb-500.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_test enabled: replaced 26 BatchNorm2d layers with alpha=1.0

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:11<00:00, 49.90it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9852653238508436 
, std: 0.0037861814356045156
Organ LV with dice: mean: 0.696858979596032 
, std: 0.05517959031445284
Organ Myo with dice: mean: 0.8364078813129001 
, std: 0.05137947685677142
Organ RV with dice: mean: 0.8146173622873094 
, std: 0.06045147413920346
Overall mean dice by sample 0.7826280743987472
Overall mean dice by domain 0.782628059387207

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/norm_test_dcon_lge_to_bssfp/log

==========================================
norm_alpha: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: norm_alpha_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/norm_alpha_dcon_sabsct_to_chaost2
================================================================================

[22:37:55.724] bn_alpha:0.1
labmap: {'0': 0, '1': 63, '2': 126, '3': 189, '4': 255}
get_train abd: ['SABSCT']
Applying GIP/CLP location-scale augmentation on train split
train_SABSCT: Using fold data statistics for normalization
For train on ['SABSCT'] using scan ids {'SABSCT': ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']}
get_trval abd: ['SABSCT']
For trval on ['SABSCT'] using scan ids {'SABSCT': ['6', '7', '8']}
get_trtest abd: ['SABSCT']
For trtest on ['SABSCT'] using scan ids {'SABSCT': ['0', '1', '2', '3', '4', '5']}
get_test abd: ['CHAOST2']
test_CHAOST2: Using fold data statistics for normalization
For test on ['CHAOST2'] using scan ids {'CHAOST2': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '0', '1', '2', '3', '4', '5']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_alpha
norm_alpha config: alpha=0.1
================================================================================

Loading checkpoint: ../ckpts/dcon-sc-300.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-sc-300.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_alpha enabled: replaced 26 BatchNorm2d layers with alpha=0.1

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:11<00:00, 45.36it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9830349415540696 
, std: 0.005329486860785811
Organ liver with dice: mean: 0.8556012451648712 
, std: 0.04767200501344606
Organ rk with dice: mean: 0.8433791369199752 
, std: 0.05015620194858921
Organ lk with dice: mean: 0.8341988503932953 
, std: 0.0561490309340548
Organ spleen with dice: mean: 0.7814699888229371 
, std: 0.07272396745530008
Overall mean dice by sample 0.8286623053252697
Overall mean dice by domain 0.8286622762680054

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/norm_alpha_dcon_sabsct_to_chaost2/log

==========================================
norm_alpha: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: norm_alpha_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/norm_alpha_dcon_chaost2_to_sabsct
================================================================================


================================================================================
TEST MODE
TTA: norm_alpha
norm_alpha config: alpha=0.1
================================================================================

Loading checkpoint: ../ckpts/dcon-cs-200.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-cs-200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_alpha enabled: replaced 26 BatchNorm2d layers with alpha=0.1

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [00:34<00:00, 49.89it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9814255793889364 
, std: 0.00641291789053432
Organ liver with dice: mean: 0.8714577257633209 
, std: 0.04396367062315027
Organ rk with dice: mean: 0.7940273649990559 
, std: 0.15752991511492348
Organ lk with dice: mean: 0.783774197101593 
, std: 0.1178842104861175
Organ spleen with dice: mean: 0.6303548336029052 
, std: 0.14187652140452137
Overall mean dice by sample 0.7699035303667188
Overall mean dice by domain 0.7699034810066223

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/norm_alpha_dcon_chaost2_to_sabsct/log

==========================================
norm_alpha: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: norm_alpha_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/norm_alpha_dcon_bssfp_to_lge
================================================================================

trtest_bSSFP: Using fold data statistics for normalization
For trtest on ['bSSFP'] using scan ids {'bSSFP': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}
get_test cardiac: ['LGE']
test_LGE: Using fold data statistics for normalization
For test on ['LGE'] using scan ids {'LGE': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_alpha
norm_alpha config: alpha=0.1
================================================================================

Loading checkpoint: ../ckpts/dcon-bl-1200.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-bl-1200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_alpha enabled: replaced 26 BatchNorm2d layers with alpha=0.1

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [00:13<00:00, 54.02it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9944197416305542 
, std: 0.0017335545940597781
Organ LV with dice: mean: 0.7904080576366849 
, std: 0.057682972726903835
Organ Myo with dice: mean: 0.9036331348949008 
, std: 0.04770985691038861
Organ RV with dice: mean: 0.8779757950041029 
, std: 0.05068885446320363
Overall mean dice by sample 0.8573389958452295
Overall mean dice by domain 0.8573390245437622

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/norm_alpha_dcon_bssfp_to_lge/log

==========================================
norm_alpha: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: norm_alpha_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/LGE/norm_alpha_dcon_lge_to_bssfp
================================================================================

================================================================================
TEST MODE
TTA: norm_alpha
norm_alpha config: alpha=0.1
================================================================================

Loading checkpoint: ../ckpts/dcon-lb-500.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-lb-500.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_alpha enabled: replaced 26 BatchNorm2d layers with alpha=0.1

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:11<00:00, 49.09it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9923083053694831 
, std: 0.002233801306450084
Organ LV with dice: mean: 0.7811220606168111 
, std: 0.0418148852725659
Organ Myo with dice: mean: 0.9014003025160895 
, std: 0.03641308515963322
Organ RV with dice: mean: 0.8796840111414591 
, std: 0.0428496840845674
Overall mean dice by sample 0.8540687914247866
Overall mean dice by domain 0.8540687561035156

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/norm_alpha_dcon_lge_to_bssfp/log

==========================================
norm_ema: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: norm_ema_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/norm_ema_dcon_sabsct_to_chaost2
================================================================================

   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_ema
================================================================================

Loading checkpoint: ../ckpts/dcon-sc-300.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-sc-300.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_ema enabled: updating BatchNorm running statistics online

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:14<00:00, 37.76it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9828083157539368 
, std: 0.004271703086468363
Organ liver with dice: mean: 0.8548518002033234 
, std: 0.03534671689164361
Organ rk with dice: mean: 0.859758871793747 
, std: 0.04217646201821916
Organ lk with dice: mean: 0.8426553905010223 
, std: 0.05436467288204521
Organ spleen with dice: mean: 0.8055207163095475 
, std: 0.06252117738404196
Overall mean dice by sample 0.84069669470191
Overall mean dice by domain 0.8406966924667358

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/norm_ema_dcon_sabsct_to_chaost2/log

==========================================
norm_ema: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: norm_ema_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/norm_ema_dcon_chaost2_to_sabsct
================================================================================
4': 255}
get_train abd: ['CHAOST2']
Applying GIP/CLP location-scale augmentation on train split
train_CHAOST2: Using fold data statistics for normalization
For train on ['CHAOST2'] using scan ids {'CHAOST2': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']}
get_trval abd: ['CHAOST2']
For trval on ['CHAOST2'] using scan ids {'CHAOST2': ['4', '5']}
get_trtest abd: ['CHAOST2']
For trtest on ['CHAOST2'] using scan ids {'CHAOST2': ['0', '1', '2', '3']}
get_test abd: ['SABSCT']
test_SABSCT: Using fold data statistics for normalization
For test on ['SABSCT'] using scan ids {'SABSCT': ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '0', '1', '2', '3', '4', '5', '6', '7', '8']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_ema
================================================================================

Loading checkpoint: ../ckpts/dcon-cs-200.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-cs-200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_ema enabled: updating BatchNorm running statistics online

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [00:45<00:00, 37.76it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9769887765248616 
, std: 0.007715665928735389
Organ liver with dice: mean: 0.8391427814960479 
, std: 0.060910864462696446
Organ rk with dice: mean: 0.7636132173240184 
, std: 0.1522783883678139
Organ lk with dice: mean: 0.7244245931506157 
, std: 0.14325527472611208
Organ spleen with dice: mean: 0.5957609742879868 
, std: 0.1529560058076754
Overall mean dice by sample 0.7307353915646673
Overall mean dice by domain 0.7307354211807251

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/norm_ema_dcon_chaost2_to_sabsct/log

==========================================
norm_ema: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: norm_ema_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/norm_ema_dcon_bssfp_to_lge
================================================================================
================================================================================
TEST MODE
TTA: norm_ema
================================================================================

Loading checkpoint: ../ckpts/dcon-bl-1200.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-bl-1200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_ema enabled: updating BatchNorm running statistics online

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [00:20<00:00, 36.06it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9943347149425082 
, std: 0.0017630755191136594
Organ LV with dice: mean: 0.7957286450597975 
, std: 0.05625976816579266
Organ Myo with dice: mean: 0.905156946182251 
, std: 0.046442074990430966
Organ RV with dice: mean: 0.8722863833109538 
, std: 0.05490587329599721
Overall mean dice by sample 0.8577239915176674
Overall mean dice by domain 0.8577240109443665

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/norm_ema_dcon_bssfp_to_lge/log

==========================================
norm_ema: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: norm_ema_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/LGE/norm_ema_dcon_lge_to_bssfp
================================================================================

   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: norm_ema
================================================================================

Loading checkpoint: ../ckpts/dcon-lb-500.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-lb-500.pth
optimizer_seg initialized: 106 parameter groups (includes all)
norm_ema enabled: updating BatchNorm running statistics online

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:14<00:00, 37.91it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9923427104949951 
, std: 0.002023731240884317
Organ LV with dice: mean: 0.7837107009357877 
, std: 0.040965363355487186
Organ Myo with dice: mean: 0.9039224518669976 
, std: 0.03274711239489595
Organ RV with dice: mean: 0.881940135690901 
, std: 0.037446022299551705
Overall mean dice by sample 0.8565244294978954
Overall mean dice by domain 0.8565243482589722

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/norm_ema_dcon_lge_to_bssfp/log

==========================================
tent: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: tent_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/tent_dcon_sabsct_to_chaost2
================================================================================

[22:44:27.204] 
================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:28<00:00, 18.74it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9774946391582489 
, std: 0.006148640224929439
Organ liver with dice: mean: 0.7894657403230667 
, std: 0.051620612771231186
Organ rk with dice: mean: 0.82153779566288 
, std: 0.05193421060259069
Organ lk with dice: mean: 0.8215144455432892 
, std: 0.049563742923604645
Organ spleen with dice: mean: 0.7967307001352311 
, std: 0.07291748249105948
Overall mean dice by sample 0.8073121704161167
Overall mean dice by domain 0.8073121905326843

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/tent_dcon_sabsct_to_chaost2/log

==========================================
tent: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: tent_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/tent_dcon_chaost2_to_sabsct
================================================================================

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [01:29<00:00, 19.37it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9752148409684499 
, std: 0.007108267624585696
Organ liver with dice: mean: 0.8011550744374593 
, std: 0.05880329793049365
Organ rk with dice: mean: 0.7564557862778505 
, std: 0.1566456688315527
Organ lk with dice: mean: 0.7197311232487361 
, std: 0.11309988887352446
Organ spleen with dice: mean: 0.6117537468671799 
, std: 0.14520622380369394
Overall mean dice by sample 0.7222739327078064
Overall mean dice by domain 0.7222739458084106

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/tent_dcon_chaost2_to_sabsct/log

==========================================
tent: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: tent_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/tent_dcon_bssfp_to_lge
================================================================================
================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [00:39<00:00, 18.37it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9910406086179945 
, std: 0.001911175854091104
Organ LV with dice: mean: 0.7720018148422241 
, std: 0.0556221062640506
Organ Myo with dice: mean: 0.8524334655867682 
, std: 0.056757304612273296
Organ RV with dice: mean: 0.8388889061080085 
, std: 0.059014639017140075
Overall mean dice by sample 0.8211080621790003
Overall mean dice by domain 0.8211079835891724

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/tent_dcon_bssfp_to_lge/log

==========================================
tent: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: tent_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/LGE/tent_dcon_lge_to_bssfp
================================================================================

Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:29<00:00, 19.04it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9864868296517266 
, std: 0.0034938157978804695
Organ LV with dice: mean: 0.716664097044203 
, std: 0.05187634505929556
Organ Myo with dice: mean: 0.8458209130499098 
, std: 0.04988397898822766
Organ RV with dice: mean: 0.815442497200436 
, std: 0.05788873402228908
Overall mean dice by sample 0.7926425024315163
Overall mean dice by domain 0.7926425337791443

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/tent_dcon_lge_to_bssfp/log

==========================================
dg_tta: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: dg_tta_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/dg_tta_dcon_sabsct_to_chaost2
================================================================================


================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:44<00:00, 12.09it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9770801275968551 
, std: 0.005975312078837457
Organ liver with dice: mean: 0.7922440201044083 
, std: 0.05367515995808895
Organ rk with dice: mean: 0.8292018085718155 
, std: 0.05278341245445044
Organ lk with dice: mean: 0.8252923101186752 
, std: 0.05101164363080493
Organ spleen with dice: mean: 0.8001030057668685 
, std: 0.06940675682470877
Overall mean dice by sample 0.8117102861404419
Overall mean dice by domain 0.8117102384567261

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/dg_tta_dcon_sabsct_to_chaost2/log

==========================================
dg_tta: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: dg_tta_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/dg_tta_dcon_chaost2_to_sabsct
================================================================================

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [02:26<00:00, 11.76it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9724342882633209 
, std: 0.0070381588827652004
Organ liver with dice: mean: 0.8003069837888082 
, std: 0.059298918299472245
Organ rk with dice: mean: 0.7519647705058257 
, std: 0.1580147760907667
Organ lk with dice: mean: 0.6994961043198903 
, std: 0.12712387359389254
Organ spleen with dice: mean: 0.5759776969750722 
, std: 0.14483785669496094
Overall mean dice by sample 0.7069363888973991
Overall mean dice by domain 0.7069364190101624

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/dg_tta_dcon_chaost2_to_sabsct/log

==========================================
dg_tta: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: dg_tta_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/dg_tta_dcon_bssfp_to_lge
================================================================================

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [01:02<00:00, 11.71it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.991457286145952 
, std: 0.001979118635359876
Organ LV with dice: mean: 0.7783921480178833 
, std: 0.0547871374403142
Organ Myo with dice: mean: 0.8567449437247382 
, std: 0.055275438638547836
Organ RV with dice: mean: 0.8483021815617879 
, std: 0.060305858512182405
Overall mean dice by sample 0.8278130911014698
Overall mean dice by domain 0.8278130292892456

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/dg_tta_dcon_bssfp_to_lge/log

==========================================
dg_tta: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: dg_tta_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:47<00:00, 11.73it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9860311269760131 
, std: 0.0037530912224033987
Organ LV with dice: mean: 0.7081943194071452 
, std: 0.055773642205851987
Organ Myo with dice: mean: 0.8378604226642185 
, std: 0.05096104852935271
Organ RV with dice: mean: 0.8225465138753255 
, std: 0.05903937512666149
Overall mean dice by sample 0.7895337519822297
Overall mean dice by domain 0.7895338535308838

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/dg_tta_dcon_lge_to_bssfp/log

==========================================
cotta: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: cotta_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/cotta_dcon_sabsct_to_chaost2
[22:56:21.853] cotta_mt:0.999
[22:56:21.854] cotta_rst:0.01
[22:56:21.854] cotta_ap:0.9
labmap: {'0': 0, '1': 63, '2': 126, '3': 189, '4': 255}
get_train abd: ['SABSCT']
Applying GIP/CLP location-scale augmentation on train split
train_SABSCT: Using fold data statistics for normalization
For train on ['SABSCT'] using scan ids {'SABSCT': ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']}
get_trval abd: ['SABSCT']
For trval on ['SABSCT'] using scan ids {'SABSCT': ['6', '7', '8']}
get_trtest abd: ['SABSCT']
For trtest on ['SABSCT'] using scan ids {'SABSCT': ['0', '1', '2', '3', '4', '5']}
get_test abd: ['CHAOST2']
test_CHAOST2: Using fold data statistics for normalization
For test on ['CHAOST2'] using scan ids {'CHAOST2': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '0', '1', '2', '3', '4', '5']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: cotta
CoTTA config: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9
================================================================================

Loading checkpoint: ../ckpts/dcon-sc-300.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-sc-300.pth
optimizer_seg initialized: 106 parameter groups (includes all)
CoTTA enabled: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:59<00:00,  9.09it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9816280215978622 
, std: 0.0052767591798589105
Organ liver with dice: mean: 0.8425022006034851 
, std: 0.052601959339863294
Organ rk with dice: mean: 0.8462543457746505 
, std: 0.04108408829296933
Organ lk with dice: mean: 0.8301074296236038 
, std: 0.054972776207195
Organ spleen with dice: mean: 0.7007659435272217 
, std: 0.06839885524164738
Overall mean dice by sample 0.8049074798822403
Overall mean dice by domain 0.804907500743866

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/cotta_dcon_sabsct_to_chaost2/log

==========================================
cotta: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: cotta_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/cotta_dcon_chaost2_to_sabsct
================================================================================

[22:57:52.554] cotta_ap:0.9
labmap: {'0': 0, '1': 63, '2': 126, '3': 189, '4': 255}
get_train abd: ['CHAOST2']
Applying GIP/CLP location-scale augmentation on train split
train_CHAOST2: Using fold data statistics for normalization
For train on ['CHAOST2'] using scan ids {'CHAOST2': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']}
get_trval abd: ['CHAOST2']
For trval on ['CHAOST2'] using scan ids {'CHAOST2': ['4', '5']}
get_trtest abd: ['CHAOST2']
For trtest on ['CHAOST2'] using scan ids {'CHAOST2': ['0', '1', '2', '3']}
get_test abd: ['SABSCT']
test_SABSCT: Using fold data statistics for normalization
For test on ['SABSCT'] using scan ids {'SABSCT': ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '0', '1', '2', '3', '4', '5', '6', '7', '8']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: cotta
CoTTA config: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9
================================================================================

Loading checkpoint: ../ckpts/dcon-cs-200.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-cs-200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
CoTTA enabled: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [03:08<00:00,  9.15it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9824418803056081 
, std: 0.0059010834925559445
Organ liver with dice: mean: 0.8723278443018595 
, std: 0.04975932322828017
Organ rk with dice: mean: 0.7972576498985291 
, std: 0.16211950607423406
Organ lk with dice: mean: 0.8018832415342331 
, std: 0.11416416475444012
Organ spleen with dice: mean: 0.6369250814119974 
, std: 0.14104098999091347
Overall mean dice by sample 0.7770984542866548
Overall mean dice by domain 0.7770984768867493

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/cotta_dcon_chaost2_to_sabsct/log

==========================================
cotta: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: cotta_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/cotta_dcon_bssfp_to_lge
================================================================================
[23:01:34.556] cotta_ap:0.9
labmap: {'0': 0, '1': 85, '2': 170, '3': 255}
get_train cardiac: ['bSSFP']
Applying GIP/CLP location-scale augmentation on train split
train_bSSFP: Using fold data statistics for normalization
For train on ['bSSFP'] using scan ids {'bSSFP': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45']}
get_trval cardiac: ['bSSFP']
trval_bSSFP: Using fold data statistics for normalization
For trval on ['bSSFP'] using scan ids {'bSSFP': ['11', '12', '13', '14']}
get_trtest cardiac: ['bSSFP']
trtest_bSSFP: Using fold data statistics for normalization
For trtest on ['bSSFP'] using scan ids {'bSSFP': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}
get_test cardiac: ['LGE']
test_LGE: Using fold data statistics for normalization
For test on ['LGE'] using scan ids {'LGE': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: cotta
CoTTA config: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9
================================================================================

Loading checkpoint: ../ckpts/dcon-bl-1200.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-bl-1200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
CoTTA enabled: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [01:20<00:00,  9.08it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9943927394019233 
, std: 0.0017954351878375883
Organ LV with dice: mean: 0.7845384889178806 
, std: 0.059695232390642984
Organ Myo with dice: mean: 0.9017906347910564 
, std: 0.046812701943182815
Organ RV with dice: mean: 0.8780648509661356 
, std: 0.04884937035158879
Overall mean dice by sample 0.8547979915583576
Overall mean dice by domain 0.8547979593276978

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/cotta_dcon_bssfp_to_lge/log

==========================================
cotta: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: cotta_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

[23:03:19.378] cotta_lr:0.0001
[23:03:19.378] cotta_steps:1
[23:03:19.378] cotta_mt:0.999
[23:03:19.378] cotta_rst:0.01
[23:03:19.379] cotta_ap:0.9
labmap: {'0': 0, '1': 85, '2': 170, '3': 255}
get_train cardiac: ['LGE']
Applying GIP/CLP location-scale augmentation on train split
train_LGE: Using fold data statistics for normalization
For train on ['LGE'] using scan ids {'LGE': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45']}
get_trval cardiac: ['LGE']
trval_LGE: Using fold data statistics for normalization
For trval on ['LGE'] using scan ids {'LGE': ['11', '12', '13', '14']}
get_trtest cardiac: ['LGE']
trtest_LGE: Using fold data statistics for normalization
For trtest on ['LGE'] using scan ids {'LGE': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}
get_test cardiac: ['bSSFP']
test_bSSFP: Using fold data statistics for normalization
For test on ['bSSFP'] using scan ids {'bSSFP': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: cotta
CoTTA config: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9
================================================================================

Loading checkpoint: ../ckpts/dcon-lb-500.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-lb-500.pth
optimizer_seg initialized: 106 parameter groups (includes all)
CoTTA enabled: lr=0.0001, steps=1, mt=0.999, rst=0.01, ap=0.9

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [01:01<00:00,  9.04it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9937229447894627 
, std: 0.001513218627183448
Organ LV with dice: mean: 0.8012837595409817 
, std: 0.03260525798222883
Organ Myo with dice: mean: 0.9126182701852587 
, std: 0.024411960115187783
Organ RV with dice: mean: 0.8966975702179802 
, std: 0.03234436724361425
Overall mean dice by sample 0.8701998666480736
Overall mean dice by domain 0.8701999187469482

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/cotta_dcon_lge_to_bssfp/log

==========================================
memo: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: memo_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/memo_dcon_sabsct_to_chaost2
================================================================================

TEST MODE
TTA: memo
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all
================================================================================

Loading checkpoint: ../ckpts/dcon-sc-300.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-sc-300.pth
optimizer_seg initialized: 106 parameter groups (includes all)
MEMO enabled: updating all segmentation parameters (106 tensors)
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:49<00:00, 10.90it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9523376017808914 
, std: 0.015570817402736109
Organ liver with dice: mean: 0.2553432039785548 
, std: 0.2719729413580656
Organ rk with dice: mean: 0.3250422307406552 
, std: 0.3334965222920417
Organ lk with dice: mean: 0.2753349285805598 
, std: 0.3462085469409464
Organ spleen with dice: mean: 0.13709437361249002 
, std: 0.2397750945085821
Overall mean dice by sample 0.24820368422806496
Overall mean dice by domain 0.24820367991924286

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/memo_dcon_sabsct_to_chaost2/log

==========================================
memo: ABDOMINAL CHAOST2->SABSCT
checkpoint: ../ckpts/dcon-cs-200.pth
expname: memo_dcon_chaost2_to_sabsct
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/CHAOST2/memo_dcon_chaost2_to_sabsct
================================================================================

⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: memo
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all
================================================================================

Loading checkpoint: ../ckpts/dcon-cs-200.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-cs-200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
MEMO enabled: updating all segmentation parameters (106 tensors)
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [02:33<00:00, 11.20it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9536400417486827 
, std: 0.01304647002592823
Organ liver with dice: mean: 0.13432600263040512 
, std: 0.26439494231847704
Organ rk with dice: mean: 0.16030036895923938 
, std: 0.28236627105954404
Organ lk with dice: mean: 0.0749125915269057 
, std: 0.21248421661291644
Organ spleen with dice: mean: 0.05636131824091232 
, std: 0.17270852989297655
Overall mean dice by sample 0.10647507033936562
Overall mean dice by domain 0.10647507011890411

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/memo_dcon_chaost2_to_sabsct/log

==========================================
memo: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: memo_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/memo_dcon_bssfp_to_lge
================================================================================

[23:09:13.728]  rccs_apply_to_base=0, rccs_embed_dim=128)
[23:09:13.728] name:memo_dcon_bssfp_to_lge
[23:09:13.729] use_cgsd:0
[23:09:13.729] view_pipeline:GIP->CLP
[23:09:13.729] opt.f_seed:1
[23:09:13.729] data:CARDIAC
[23:09:13.729] tta:memo
[23:09:13.729] memo_lr:1e-05
[23:09:13.729] memo_steps:1
[23:09:13.729] memo_n_augmentations:8
[23:09:13.729] memo_include_identity:1
[23:09:13.729] memo_hflip_p:0.0
[23:09:13.729] memo_update_scope:all
labmap: {'0': 0, '1': 85, '2': 170, '3': 255}
get_train cardiac: ['bSSFP']
Applying GIP/CLP location-scale augmentation on train split
train_bSSFP: Using fold data statistics for normalization
For train on ['bSSFP'] using scan ids {'bSSFP': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45']}
get_trval cardiac: ['bSSFP']
trval_bSSFP: Using fold data statistics for normalization
For trval on ['bSSFP'] using scan ids {'bSSFP': ['11', '12', '13', '14']}
get_trtest cardiac: ['bSSFP']
trtest_bSSFP: Using fold data statistics for normalization
For trtest on ['bSSFP'] using scan ids {'bSSFP': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}
get_test cardiac: ['LGE']
test_LGE: Using fold data statistics for normalization
For test on ['LGE'] using scan ids {'LGE': ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: memo
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all
================================================================================

Loading checkpoint: ../ckpts/dcon-bl-1200.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-bl-1200.pth
optimizer_seg initialized: 106 parameter groups (includes all)
MEMO enabled: updating all segmentation parameters (106 tensors)
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [01:07<00:00, 10.86it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9666494422488743 
, std: 0.01120671849130318
Organ LV with dice: mean: 0.11460008399379958 
, std: 0.24783603351627198
Organ Myo with dice: mean: 0.25619721619586927 
, std: 0.36461909446938584
Organ RV with dice: mean: 0.10704271673328347 
, std: 0.24938256738947492
Overall mean dice by sample 0.1592800056409841
Overall mean dice by domain 0.15928000211715698

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/memo_dcon_bssfp_to_lge/log

==========================================
memo: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: memo_dcon_lge_to_bssfp
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/LGE/memo_dcon_lge_to_bssfp
================================================================================2', '13', '14']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation

================================================================================
TEST MODE
TTA: memo
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all
================================================================================

Loading checkpoint: ../ckpts/dcon-lb-500.pth
Number of parameters (segmentation): 3185844
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-lb-500.pth
optimizer_seg initialized: 106 parameter groups (includes all)
MEMO enabled: updating all segmentation parameters (106 tensors)
MEMO config: lr=1e-05, steps=1, n_aug=8, include_identity=1, hflip_p=0.0, scope=all

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:50<00:00, 10.90it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9715028868781196 
, std: 0.010137787168228033
Organ LV with dice: mean: 0.200705062939475 
, std: 0.2921393030096692
Organ Myo with dice: mean: 0.3416289270389825 
, std: 0.37862087147924567
Organ RV with dice: mean: 0.14189404702822989 
, std: 0.29402972980855663
Overall mean dice by sample 0.22807601233556246
Overall mean dice by domain 0.2280759960412979

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/memo_dcon_lge_to_bssfp/log

==========================================
asm: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: asm_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/SABSCT/asm_dcon_sabsct_to_chaost2
================================================================================

[23:12:00.393] 
labmap: {'0': 0, '1': 63, '2': 126, '3': 189, '4': 255}
get_train abd: ['SABSCT']
Applying GIP/CLP location-scale augmentation on train split
train_SABSCT: Using fold data statistics for normalization
For train on ['SABSCT'] using scan ids {'SABSCT': ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']}
get_trval abd: ['SABSCT']
For trval on ['SABSCT'] using scan ids {'SABSCT': ['6', '7', '8']}
get_trtest abd: ['SABSCT']
For trtest on ['SABSCT'] using scan ids {'SABSCT': ['0', '1', '2', '3', '4', '5']}
get_test abd: ['CHAOST2']
test_CHAOST2: Using fold data statistics for normalization
For test on ['CHAOST2'] using scan ids {'CHAOST2': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '0', '1', '2', '3', '4', '5']}
⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor=4
   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation
ASM source loader: using labeled source-domain training split with batch_size=4, shuffle=True, drop_last=True. Target labels are not used for adaptation.

================================================================================
TEST MODE
TTA: asm
ASM config: source-dependent supervised TTA; target images provide style/statistics only, target labels are evaluation-only. lr=0.0001, steps=1, inner_steps=2, lambda_reg=0.0002, sampling_step=20.0, src_batch_size=4, style_backend=medical_adain, episodic=False
================================================================================

Loading checkpoint: ../ckpts/dcon-sc-300.pth
Number of parameters (segmentation): 3186133
CGSD disabled: model follows the old no-CGSD branch
reloaddir: ../ckpts/dcon-sc-300.pth
optimizer_seg initialized: 106 parameter groups (includes all)
[23:12:28.774] ASM initialized as source-dependent supervised TTA: target images provide style statistics only; source images and source labels provide the adaptation loss.
ASM enabled: source-dependent supervised TTA. Target images provide style/statistics only; target labels are used only for evaluation. lr=0.0001, steps=1, inner_steps=2, lambda_reg=0.0002, sampling_step=20.0, style_backend=medical_adain, episodic=False
[23:12:28.774] ASM enabled: source-dependent supervised TTA. Target images provide style/statistics only; target labels are used only for evaluation. lr=0.0001, steps=1, inner_steps=2, lambda_reg=0.0002, sampling_step=20.0, style_backend=medical_adain, episodic=False

================================================================================
Testing on target domain...
================================================================================
testfinal evaluation:   0%|                                                      | 0/538 [00:00<?, ?it/s][23:12:29.629] asm_loss_seg=0.353359 asm_loss_reg=0.116880 asm_loss_total=0.353383


=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9855970412492752 
, std: 0.005165210409658864
Organ liver with dice: mean: 0.8720343112945557 
, std: 0.05221062718536337
Organ rk with dice: mean: 0.8538014769554139 
, std: 0.04548150123130923
Organ lk with dice: mean: 0.8500885903835297 
, std: 0.07159460104306285
Organ spleen with dice: mean: 0.8415684327483177 
, std: 0.10633771015222429
Overall mean dice by sample 0.8543732028454543
Overall mean dice by domain 0.8543731570243835

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/asm_dcon_sabsct_to_chaost2/log

testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [04:48<00:00,  5.97it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.985914001862208 
, std: 0.005585982762701502
Organ liver with dice: mean: 0.8924695769945781 
, std: 0.03732605255066234
Organ rk with dice: mean: 0.7837470084739228 
, std: 0.17691050286657073
Organ lk with dice: mean: 0.7763104632496833 
, std: 0.17092252406124697
Organ spleen with dice: mean: 0.713198138276736 
, std: 0.15120632071972526
Overall mean dice by sample 0.79143129674873
Overall mean dice by domain 0.7914313077926636

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/asm_dcon_chaost2_to_sabsct/log

==========================================
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [01:59<00:00,  6.13it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9936107118924459 
, std: 0.00238822967195392
Organ LV with dice: mean: 0.7658037609524198 
, std: 0.06030531154401941
Organ Myo with dice: mean: 0.892869132094913 
, std: 0.05010534843567405
Organ RV with dice: mean: 0.868449232313368 
, std: 0.05680300168591105
Overall mean dice by sample 0.8423740417869002
Overall mean dice by domain 0.8423740267753601

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/asm_dcon_bssfp_to_lge/log

==========================================
asm: CARDIAC LGE->bSSFP
checkpoint: ../ckpts/dcon-lb-500.pth
expname: asm_dcon_lge_to_bssfp
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [01:25<00:00,  6.47it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9931235392888387 
, std: 0.0020736024021461383
Organ LV with dice: mean: 0.7962652391857571 
, std: 0.03594051383026917
Organ Myo with dice: mean: 0.9110480335023668 
, std: 0.028579308220140764
Organ RV with dice: mean: 0.8794776995976766 
, std: 0.04947984700604331
Overall mean dice by sample 0.8622636574286001
Overall mean dice by domain 0.8622636795043945

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/asm_dcon_lge_to_bssfp/log

==========================================
sm_ppm: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [00:50<00:00, 10.69it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9849790722131729 
, std: 0.004428823229994949
Organ liver with dice: mean: 0.8714988589286804 
, std: 0.038225912318541076
Organ rk with dice: mean: 0.8588436305522918 
, std: 0.04219890945686791
Organ lk with dice: mean: 0.8548527389764786 
, std: 0.05025940909224385
Organ spleen with dice: mean: 0.8216214835643768 
, std: 0.07904915303501879
Overall mean dice by sample 0.851704178005457
Overall mean dice by domain 0.8517042398452759

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/SABSCT/sm_ppm_dcon_sabsct_to_chaost2/log

testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [02:13<00:00, 12.94it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9831421136856079 
, std: 0.006370804732779318
Organ liver with dice: mean: 0.8780520061651865 
, std: 0.04427919619858329
Organ rk with dice: mean: 0.7903288286179304 
, std: 0.1599300727690252
Organ lk with dice: mean: 0.7793834686279297 
, std: 0.11823841371618825
Organ spleen with dice: mean: 0.6714556852976481 
, std: 0.15113897914670915
Overall mean dice by sample 0.7798049971771737
Overall mean dice by domain 0.7798050045967102

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/sm_ppm_dcon_chaost2_to_sabsct/log

==========================================
[23:29:17.767] smppm_loss_source=0.461771 smppm_weight_mean=0.846479 smppm_weight_max=0.935030
testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [01:08<00:00, 10.75it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9944385250409444 
, std: 0.0018374729730649482
Organ LV with dice: mean: 0.7943496147791544 
, std: 0.059432531872910914
Organ Myo with dice: mean: 0.9060995287365383 
, std: 0.04493653570943838
Organ RV with dice: mean: 0.8755055851406521 
, std: 0.050072129419469875
Overall mean dice by sample 0.8586515762187816
Overall mean dice by domain 0.8586516380310059

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/sm_ppm_dcon_bssfp_to_lge/log

30:31.232] smppm_loss_source=0.181240 smppm_weight_mean=0.913574 smppm_weight_max=0.977459
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [00:49<00:00, 11.30it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9941563341352675 
, std: 0.0013778566262980544
Organ LV with dice: mean: 0.8133519596523708 
, std: 0.03172650586968551
Organ Myo with dice: mean: 0.9194344706005521 
, std: 0.02332312093997834
Organ RV with dice: mean: 0.900235480732388 
, std: 0.02917970322299731
Overall mean dice by sample 0.877673970328437
Overall mean dice by domain 0.8776739239692688

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/sm_ppm_dcon_lge_to_bssfp/log

==========================================
gtta: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
s][23:32:18.106] gtta_loss_src=0.469424 gtta_loss_trg=0.000022 gtta_loss_total=0.469445 gtta_pseudo_valid_ratio=0.895589
testfinal evaluation: 100%|████████████████████████████████████████████| 538/538 [01:14<00:00,  7.18it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9835911214351654 
, std: 0.004703948927593478
Organ liver with dice: mean: 0.8585459113121032 
, std: 0.04396812365449386
Organ rk with dice: mean: 0.8618391484022141 
, std: 0.03832446362453147
Organ lk with dice: mean: 0.8443630546331405 
, std: 0.050324362531833346
Organ spleen with dice: mean: 0.8114173233509063 
, std: 0.0884140532913388
Overall mean dice by sample 0.844041359424591
Overall mean dice by domain 0.8440413475036621

Skipping source-domain evaluation (--eval_source_domain false).
testfinal evaluation: 100%|██████████████████████████████████████████| 1725/1725 [04:05<00:00,  7.02it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9821441352367402 
, std: 0.006574659244650437
Organ liver with dice: mean: 0.8687783022721608 
, std: 0.04925952127800926
Organ rk with dice: mean: 0.7871590359757344 
, std: 0.1560682681600629
Organ lk with dice: mean: 0.7683932781219482 
, std: 0.12142419909225372
Organ spleen with dice: mean: 0.6649176388978958 
, std: 0.14826309343163402
Overall mean dice by sample 0.7723120638169348
Overall mean dice by domain 0.7723120450973511

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/CHAOST2/gtta_dcon_chaost2_to_sabsct/log

==========================================
gtta: CARDIAC bSSFP->LGE
checkpoint: ../ckpts/dcon-bl-1200.pth
expname: gtta_dcon_bssfp_to_lge
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
SGF disabled: strong view uses CLP directly.

================================================================================
Checkpoint directory: /root/TTA/DCON/results_medseg_tta
Experiment directory: /root/TTA/DCON/results_medseg_tta/bSSFP/gtta_dcon_bssfp_to_lge
================================================================================

testfinal evaluation: 100%|████████████████████████████████████████████| 731/731 [01:47<00:00,  6.83it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9943238523271348 
, std: 0.0018058841477132926
Organ LV with dice: mean: 0.7995883067448933 
, std: 0.05846737633250712
Organ Myo with dice: mean: 0.9054127825631035 
, std: 0.04646888506502778
Organ RV with dice: mean: 0.8709112180603875 
, std: 0.05138814080387668
Overall mean dice by sample 0.8586374357894615
Overall mean dice by domain 0.8586374521255493

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/bSSFP/gtta_dcon_bssfp_to_lge/log

3:40:54.127] gtta_loss_src=0.260165 gtta_loss_trg=0.000019 gtta_loss_total=0.260184 gtta_pseudo_valid_ratio=0.836941
testfinal evaluation: 100%|████████████████████████████████████████████| 554/554 [01:20<00:00,  6.87it/s]
=======  Epoch 0 test result on mode testfinal seg:  =======
Organ bg with dice: mean: 0.9937377545568679 
, std: 0.0014910483848075639
Organ LV with dice: mean: 0.8039161059591505 
, std: 0.0337076317607516
Organ Myo with dice: mean: 0.9137492431534662 
, std: 0.02588034262237615
Organ RV with dice: mean: 0.8977672086821662 
, std: 0.027818043186208632
Overall mean dice by sample 0.871810852598261
Overall mean dice by domain 0.871810793876648

Skipping source-domain evaluation (--eval_source_domain false).

Test completed! Results saved in: /root/TTA/DCON/results_medseg_tta/LGE/gtta_dcon_lge_to_bssfp/log

==========================================
gold: ABDOMINAL SABSCT->CHAOST2
checkpoint: ../ckpts/dcon-sc-300.pth
expname: gold_dcon_sabsct_to_chaost2
==========================================
Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).
