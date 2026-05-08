import os
# ============================================================================
# Critical: Set environment variables BEFORE importing numpy/torch
# This prevents thread oversubscription in DataLoader workers
# ============================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Limit OpenMP threads
os.environ.setdefault("MKL_NUM_THREADS", "1")  # Limit Intel MKL threads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # Limit OpenBLAS threads
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")  # Limit NumExpr threads
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "23")

import random
import torch.backends.cudnn as cudnn
import time
import shutil
import torch
import numpy as np
import argparse
import os.path as osp
import glob
from PIL import Image
from models.exp_trainer import *

from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import logging
from datetime import datetime
import dataloaders.AbdominalDataset as ABD
import dataloaders.CardiacDataset as cardiac_cls
try:
    import dataloaders.ProstateDataset as PROS
except ImportError:
    PROS = None
te_metric,val_metric=[],[]

def pre_labmap():
    labmap={}
    tmp1={'0':0,'1':255} 
    labmap['PROSTATE']=tmp1
    tmp2={'0':0,'1':63,'2':126,'3':189,'4':255}
    labmap['ABDOMINAL']=tmp2
    tmp3={'0':0,'1':85,'2':170,'3':255}
    labmap['CARDIAC']=tmp3
    return labmap

def deal_wit_lbvis(tmp_mp,x,ncls):
    y=torch.zeros(size=x.shape).cuda()
    x=torch.from_numpy(x)
    for i in range(ncls):
        y[x==i]=tmp_mp[str(i)]
    return y

def prediction_wrapper(tb_writer,type1,savdir,model, test_loader,  epoch, label_name, save_prediction ):
    with torch.no_grad():
        out_prediction_list = {}
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader), desc=f'{type1} evaluation'):
            if batch['is_start']:
                slice_idx = 0
                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['base_view'].shape
              
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = torch.Tensor(np.zeros( [ nframe,nx, ny]  )).cuda()

            assert batch['label'].shape[0] == 1 # enforce a batchsize of 1
            test_input = {
                'image': batch['base_view'],
                'label': batch['label'],
                'names': [f'{scan_id_full}_{slice_idx:04d}'],
                'is_start': batch.get('is_start', False),
                'is_end': batch.get('is_end', False),
                'scan_id': batch.get('scan_id', None),
            }
            gth, pred = model.te_func(test_input)

            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0,0,...]
            curr_img[slice_idx,...] = batch['base_view'][0, 1,...]
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                out_prediction_list[scan_id_full]['img'] = curr_img

        print("=======  Epoch {} test result on mode {} seg:  =======".format(epoch, type1))

        eval_list_wrapper(tb_writer,epoch,type1,savdir,out_prediction_list,  model, label_name)

        if not save_prediction: 
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list

def eval_list_wrapper(tb_writer,epoch,type1,savdir,vol_list, model, label_name):
    nclass=len(label_name)
    out_count = len(vol_list)
    tables_by_domain = {} 
    dsc_table = np.ones([ out_count, nclass ] ) 

    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices = model.ScoreDiceEval(torch.unsqueeze(pred_, 1), gth_, dense_input = True).cpu().numpy() 
        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {} \n, std: {}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
        if type1=='testfinal' or type1=='tetrainfinal':
          with open(savdir+'/out.csv', 'a') as f:
            f.write("Organ"+label_name[organ] +"with dice: \n")
            f.write("mean:"+ str(mean_dc)+"\n")
            f.write("std:"+str(std_dc)+"\n")


    print("Overall mean dice by sample {}".format( dsc_table[:,1:].mean())) 
    error_dict['overall'] = dsc_table[:,1:].mean()

    if type1=='testfinal' or type1=='tetrainfinal':
        with open(savdir+'/out.csv', 'a') as f:
           f.write("Overall mean dice by sample:"+str(dsc_table[:,1:].mean())+" \n")

   
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)

    error_dict['overall_by_domain'] = np.mean(overall_by_domain)
    print("Overall mean dice by domain {}".format( error_dict['overall_by_domain'] ) )
    if type1=='testfinal' or type1=='tetrainfinal':
       with open(savdir+'/out.csv', 'a') as f:
           f.write("Overall mean dice by domain:"+str(error_dict['overall_by_domain'] )+" \n")
    # for prostate dataset, we use by-domain results to mitigate the differences in number of samples for each target domain

    if type1=='val':
       tmp=[str(epoch),str(error_dict['overall_by_domain'] )]
       val_metric.append(tmp)
       with open(osp.join(savdir+'/','valmetric.csv'), 'a') as f:
            f.write(str(epoch)+":"+str(error_dict['overall_by_domain'] )+" \n")
       tb_writer.add_scalar('val_metric',error_dict['overall_by_domain'], epoch)
    elif type1=='test' or type1=='testfinal':
        tmp=[str(epoch),str(error_dict['overall_by_domain'] )]
        te_metric.append(tmp)
        with open(osp.join(savdir+'/','temetric.csv'), 'a') as f:
            f.write(str(epoch)+":"+str(error_dict['overall_by_domain'] )+" \n")
        tb_writer.add_scalar('te_metric',error_dict['overall_by_domain'], epoch)


def convert_to_png(img,low_num,high_num):
    x = np.array([low_num*1.,high_num * 1.])
    newimg = (img-x[0])/(x[1]-x[0])  
    newimg = (newimg*255).astype('uint8')  
    return newimg

def save_teimgs(pred,img,gth,logdir,index,opt,tmp_mp):
    # Ensure directory exists
    os.makedirs(logdir, exist_ok=True)

    img1=img
    img1=convert_to_png(img1,img1.min(),img1.max())
    img1 = Image.fromarray(img1,mode='L')
    img1.save(os.path.join(logdir, f'{index}img.png'))

    img1=gth
    img1=deal_wit_lbvis(tmp_mp,img1,opt.nclass)
    img1=img1.cpu().numpy()
    img1=img1.astype('uint8')
    img1 = Image.fromarray(img1,mode='L')
    img1.save(os.path.join(logdir, f'{index}gth.png'))

    img1=pred
    img1=deal_wit_lbvis(tmp_mp,img1,opt.nclass)
    img1=img1.cpu().numpy()
    img1=img1.astype('uint8')
    img1 = Image.fromarray(img1,mode='L')
    img1.save(os.path.join(logdir, f'{index}pred.png'))
    

def save_trimgs(tr_viz,logdir,iternum,opt,tmp_mp):
    x=0
    size0=tr_viz['img_anchor'].shape[0]
    for i in range(size0):
        if len(np.unique(tr_viz['gth_tr'][i]))>1  and len(np.unique(tr_viz['seg_tr'][i]))>1:
            x=i
            break

    img_dir = os.path.join(logdir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    img=tr_viz['img_anchor']
    img=img[x]
    img=img[0]
    img=convert_to_png(img,img.min(),img.max())
    img = Image.fromarray(img,mode='L')
    img.save(os.path.join(img_dir, f'{int(iternum)}anchor.png'))

    img=tr_viz['img_base']
    img=img[x]
    img=np.mean(img, axis=0)
    img=convert_to_png(img,img.min(),img.max())
    img = Image.fromarray(img,mode='L')
    img.save(os.path.join(img_dir, f'{int(iternum)}base.png'))

    if 'img_strong' in tr_viz:
       img=tr_viz['img_strong']
       img=img[x]
       img=np.mean(img, axis=0)
       img=convert_to_png(img,img.min(),img.max())
       img = Image.fromarray(img,mode='L')
       img.save(os.path.join(img_dir, f'{int(iternum)}strong.png'))

    img=tr_viz['seg_tr']
    img=img[x]
    img=img[0]
    img=deal_wit_lbvis(tmp_mp,img,opt.nclass)
    img=img.cpu().numpy()
    img=img.astype('uint8')
    img = Image.fromarray(img,mode='L')
    img.save(os.path.join(img_dir, f'{int(iternum)}seg.png'))

    img=tr_viz['gth_tr']
    img=img[x]
    img=img[0]
    img=deal_wit_lbvis(tmp_mp,img,opt.nclass)
    img=img.cpu().numpy()
    img=img.astype('uint8')
    img = Image.fromarray(img,mode='L')
    img.save(os.path.join(img_dir, f'{int(iternum)}gt.png'))


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    if v in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_default_logging(name, save_path, level=logging.INFO, 
                          format="[%(asctime)s][%(levelname)s] - %(message)s"):
    tmp_timestr = time_str()
    logger = logging.getLogger(name)
    logging.basicConfig(
        filename=os.path.join(save_path, f'seg_{tmp_timestr}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level)
    return logger


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--expname', type=str, default='1', help='expname')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts', help='checkpoint directory')
    parser.add_argument('--resume_epoch', type=int, default=None, help='epoch to resume from')
    parser.add_argument('--resume_path', '--restore_from', dest='resume_path',
                        type=str, default=None, help='path to checkpoint file')

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu')
    parser.add_argument('--f_seed', type=int, default=1, help='seed')
    parser.add_argument('--lr', type=float, default=0.0005, help='lr')
    parser.add_argument('--model', type=str, default='unet', help='model')
    parser.add_argument('--batchSize', type=int, default=32, help='bs')
    parser.add_argument('--all_epoch', type=int, default=50, help='epochs')

    parser.add_argument('--data_name', '--dataset', dest='data_name',
                        type=str, default='CARDIAC', help='dataset')
    parser.add_argument('--nclass', type=int, default=4, help='nclass')
    parser.add_argument('--tr_domain', '--source', dest='tr_domain',
                        type=str, default='bSSFP', help='src_domain')
    parser.add_argument('--target_domain', '--target', dest='target_domain',
                        type=str, default=None, help='Optional explicit target domain.')
    parser.add_argument('--save_prediction', type=str2bool, nargs='?', const=True, default=True, help='save_pred')
    parser.add_argument('--eval_source_domain', type=str2bool, nargs='?', const=True, default=True,
                        help='Also evaluate the source-domain trtest split after target evaluation.')
    parser.add_argument('--tta', type=str, default='none',
                        choices=['none', 'norm_test', 'norm_alpha', 'norm_ema', 'tent', 'dg_tta', 'cotta', 'memo', 'asm', 'sm_ppm', 'gtta', 'gold', 'vptta', 'pass', 'samtta', 'spmo', 'sictta', 'a3_tta'],
                        help='Test-time adaptation method.')
    parser.add_argument('--bn_alpha', type=float, default=0.1,
                        help='Source/test BN-stat mixing coefficient for norm_alpha.')
    parser.add_argument('--tent_lr', type=float, default=1e-4,
                        help='Learning rate for TENT BN affine updates.')
    parser.add_argument('--tent_steps', type=int, default=1,
                        help='Number of TENT adaptation steps per test batch.')
    parser.add_argument('--dgtta_lr', type=float, default=1e-4,
                        help='Learning rate for DG-TTA BN affine updates.')
    parser.add_argument('--dgtta_steps', type=int, default=1,
                        help='Number of DG-TTA adaptation steps per test batch.')
    parser.add_argument('--dgtta_transform_strength', type=float, default=1.0,
                        help='Spatial/intensity augmentation strength for DG-TTA consistency.')
    parser.add_argument('--dgtta_entropy_weight', type=float, default=0.05,
                        help='Weight for DG-TTA target entropy term.')
    parser.add_argument('--dgtta_bn_l2_reg', type=float, default=1e-4,
                        help='Weight for DG-TTA BN affine L2 regularization to source values.')
    parser.add_argument('--dgtta_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset DG-TTA model and optimizer before each target batch.')
    parser.add_argument('--cotta_lr', type=float, default=1e-4,
                        help='Learning rate for CoTTA online updates.')
    parser.add_argument('--cotta_steps', type=int, default=1,
                        help='Number of CoTTA adaptation steps per test batch.')
    parser.add_argument('--cotta_mt', type=float, default=0.999,
                        help='EMA teacher momentum for CoTTA.')
    parser.add_argument('--cotta_rst', type=float, default=0.01,
                        help='Stochastic restore probability for CoTTA.')
    parser.add_argument('--cotta_ap', type=float, default=0.9,
                        help='Anchor confidence threshold for CoTTA augmentation ensembling.')
    parser.add_argument('--memo_lr', type=float, default=1e-5,
                        help='Learning rate for MEMO online updates.')
    parser.add_argument('--memo_steps', type=int, default=1,
                        help='Number of MEMO adaptation steps per test batch.')
    parser.add_argument('--memo_n_augmentations', type=int, default=8,
                        help='Number of medical MEMO views per test slice.')
    parser.add_argument('--memo_include_identity', type=int, default=1,
                        help='Include the unaugmented image as one MEMO view: 1=on, 0=off.')
    parser.add_argument('--memo_hflip_p', type=float, default=0.0,
                        help='Optional horizontal flip probability for MEMO. Default 0 for conservative medical TTA.')
    parser.add_argument('--memo_update_scope', type=str, default='all', choices=['all', 'bn_affine'],
                        help='Parameters updated by MEMO.')
    parser.add_argument('--asm_lr', type=float, default=1e-4,
                        help='Learning rate for ASM supervised source-batch updates.')
    parser.add_argument('--asm_steps', type=int, default=1,
                        help='Number of labeled source batches used per target batch.')
    parser.add_argument('--asm_inner_steps', type=int, default=2,
                        help='Number of ASM inner style-sampling/model updates per source batch.')
    parser.add_argument('--asm_lambda_reg', type=float, default=2e-4,
                        help='Weight for ASM feature mean-square regularization.')
    parser.add_argument('--asm_sampling_step', type=float, default=20.0,
                        help='Manual update scale for ASM style sampling state.')
    parser.add_argument('--asm_src_batch_size', type=int, default=4,
                        help='Source-domain labeled batch size for ASM.')
    parser.add_argument('--asm_style_backend', type=str, default='medical_adain',
                        choices=['medical_adain'],
                        help='ASM style transfer backend. medical_adain is tensor-space AdaIN statistics matching.')
    parser.add_argument('--asm_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset model and optimizer before each ASM target batch.')
    parser.add_argument('--smppm_lr', type=float, default=2.5e-4,
                        help='Learning rate for SM-PPM supervised source-batch updates.')
    parser.add_argument('--smppm_momentum', type=float, default=0.9,
                        help='SGD momentum for SM-PPM updates.')
    parser.add_argument('--smppm_wd', type=float, default=5e-4,
                        help='Weight decay for SM-PPM updates.')
    parser.add_argument('--smppm_steps', type=int, default=1,
                        help='Number of labeled source batches used per target batch by SM-PPM.')
    parser.add_argument('--smppm_src_batch_size', type=int, default=2,
                        help='Source-domain labeled batch size for SM-PPM.')
    parser.add_argument('--smppm_patch_size', type=int, default=8,
                        help='Patch size on the resized target feature map for SM-PPM prototypes.')
    parser.add_argument('--smppm_feature_size', type=int, default=32,
                        help='Spatial size used before target feature patching for SM-PPM.')
    parser.add_argument('--smppm_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset model and optimizer before each SM-PPM target batch.')
    parser.add_argument('--smppm_ablation_mode', type=str, default='full',
                        choices=['full', 'source_ce_only', 'sm_ce', 'ppm_ce', 'source_free_proto'],
                        help='SM-PPM ablation mode. sm_ce is unavailable unless an explicit SM style-mixing implementation exists.')
    parser.add_argument('--smppm_source_free_tau', type=float, default=0.7,
                        help='Confidence threshold for source_free_proto reliable target pixels.')
    parser.add_argument('--smppm_source_free_entropy_threshold', type=float, default=None,
                        help='Optional normalized entropy threshold for source_free_proto reliable target pixels.')
    parser.add_argument('--smppm_source_free_entropy_weight', type=float, default=1.0,
                        help='Weight for source_free_proto masked entropy minimization.')
    parser.add_argument('--smppm_source_free_lambda_proto', type=float, default=1.0,
                        help='Weight for source_free_proto target prototype compactness.')
    parser.add_argument('--smppm_plain_source_loader', type=str2bool, nargs='?', const=True, default=True,
                        help='Use non-augmented source-train slices for SM-PPM source CE ablations during test.')
    parser.add_argument('--gtta_lr', type=float, default=2.5e-4,
                        help='Learning rate for GTTA supervised source and target pseudo-label updates.')
    parser.add_argument('--gtta_momentum', type=float, default=0.9,
                        help='SGD momentum for GTTA updates.')
    parser.add_argument('--gtta_wd', type=float, default=5e-4,
                        help='Weight decay for GTTA updates.')
    parser.add_argument('--gtta_steps', type=int, default=1,
                        help='Number of GTTA adaptation steps per target batch.')
    parser.add_argument('--gtta_src_batch_size', type=int, default=2,
                        help='Source-domain labeled batch size for GTTA.')
    parser.add_argument('--gtta_lambda_ce_trg', type=float, default=0.1,
                        help='Weight for GTTA target pseudo-label cross-entropy.')
    parser.add_argument('--gtta_pseudo_momentum', type=float, default=0.9,
                        help='Momentum for GTTA running target pseudo-label confidence threshold.')
    parser.add_argument('--gtta_style_alpha', type=float, default=1.0,
                        help='Blend weight for GTTA class-aware tensor-space AdaIN style transfer.')
    parser.add_argument('--gtta_include_original', type=int, default=1,
                        help='Train GTTA source step on both stylized and original source images: 1=on, 0=off.')
    parser.add_argument('--gtta_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset model and optimizer before each GTTA target batch.')
    parser.add_argument('--gold_lr', type=float, default=2.5e-4,
                        help='Learning rate for GOLD student model updates.')
    parser.add_argument('--gold_momentum', type=float, default=0.9,
                        help='SGD momentum for GOLD student model updates.')
    parser.add_argument('--gold_wd', type=float, default=5e-4,
                        help='Weight decay for GOLD student model updates.')
    parser.add_argument('--gold_steps', type=int, default=1,
                        help='Number of GOLD adaptation steps per test batch.')
    parser.add_argument('--gold_rank', type=int, default=128,
                        help='Maximum rank for GOLD low-rank feature adapter.')
    parser.add_argument('--gold_tau', type=float, default=0.95,
                        help='Confidence threshold for GOLD AGOP pixel sampling.')
    parser.add_argument('--gold_alpha', type=float, default=0.02,
                        help='EMA coefficient for GOLD AGOP matrix updates.')
    parser.add_argument('--gold_t_eig', type=int, default=10,
                        help='Subspace eigendecomposition refresh interval for GOLD.')
    parser.add_argument('--gold_mt', type=float, default=0.999,
                        help='EMA teacher momentum for GOLD.')
    parser.add_argument('--gold_s_lr', type=float, default=5e-3,
                        help='Learning rate for GOLD low-rank adapter scale vector.')
    parser.add_argument('--gold_s_init_scale', type=float, default=0.0,
                        help='Initial random scale for GOLD adapter vector S.')
    parser.add_argument('--gold_s_clip', type=float, default=0.5,
                        help='Clamp range for GOLD adapter vector S.')
    parser.add_argument('--gold_adapter_scale', type=float, default=0.05,
                        help='Feature adapter residual scale for GOLD.')
    parser.add_argument('--gold_max_pixels_per_batch', type=int, default=512,
                        help='Maximum confident pixels sampled per GOLD batch.')
    parser.add_argument('--gold_min_pixels_per_batch', type=int, default=64,
                        help='Minimum confident pixels required for GOLD AGOP update.')
    parser.add_argument('--gold_n_augmentations', type=int, default=6,
                        help='Number of scale/flip EMA augmentations used by GOLD when anchor confidence is low.')
    parser.add_argument('--gold_rst', type=float, default=0.01,
                        help='Stochastic restore probability for GOLD.')
    parser.add_argument('--gold_ap', type=float, default=0.9,
                        help='Anchor confidence threshold for GOLD augmentation ensembling.')
    parser.add_argument('--gold_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset GOLD model and optimizer before each target batch.')
    parser.add_argument('--vptta_optimizer', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help='Optimizer for the VPTTA frequency prompt.')
    parser.add_argument('--vptta_lr', type=float, default=1e-2,
                        help='Learning rate for VPTTA prompt updates.')
    parser.add_argument('--vptta_momentum', type=float, default=0.99,
                        help='SGD momentum for VPTTA prompt updates.')
    parser.add_argument('--vptta_beta1', type=float, default=0.9,
                        help='Adam beta1 for VPTTA prompt updates.')
    parser.add_argument('--vptta_beta2', type=float, default=0.99,
                        help='Adam beta2 for VPTTA prompt updates.')
    parser.add_argument('--vptta_weight_decay', type=float, default=0.0,
                        help='Weight decay for VPTTA prompt updates.')
    parser.add_argument('--vptta_steps', type=int, default=1,
                        help='Number of VPTTA prompt adaptation steps per test batch.')
    parser.add_argument('--vptta_memory_size', type=int, default=40,
                        help='Number of previous prompts kept in the VPTTA memory bank.')
    parser.add_argument('--vptta_neighbor', type=int, default=16,
                        help='Number of nearest memory prompts used to initialize VPTTA.')
    parser.add_argument('--vptta_prompt_alpha', type=float, default=0.01,
                        help='Low-frequency prompt size ratio when --vptta_prompt_size is not set.')
    parser.add_argument('--vptta_prompt_size', type=int, default=None,
                        help='Explicit square low-frequency prompt size. Overrides --vptta_prompt_alpha.')
    parser.add_argument('--vptta_image_size', type=int, default=192,
                        help='Reference image size used to derive the default VPTTA prompt size.')
    parser.add_argument('--vptta_warm_n', type=int, default=5,
                        help='Warm-up constant for VPTTA source/target BN-stat interpolation.')
    parser.add_argument('--pass_optimizer', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help='Optimizer for PASS prompt/adaptor updates.')
    parser.add_argument('--pass_lr', type=float, default=5e-3,
                        help='Learning rate for PASS prompt/adaptor updates.')
    parser.add_argument('--pass_momentum', type=float, default=0.99,
                        help='SGD momentum for PASS.')
    parser.add_argument('--pass_beta1', type=float, default=0.9,
                        help='Adam beta1 for PASS.')
    parser.add_argument('--pass_beta2', type=float, default=0.999,
                        help='Adam beta2 for PASS.')
    parser.add_argument('--pass_weight_decay', type=float, default=0.0,
                        help='Weight decay for PASS optimizer.')
    parser.add_argument('--pass_steps', type=int, default=1,
                        help='Number of PASS adaptation steps per test batch.')
    parser.add_argument('--pass_bn_alpha', type=float, default=0.01,
                        help='Variance weight for PASS source-BN statistic matching loss.')
    parser.add_argument('--pass_bn_layers', type=int, default=0,
                        help='Number of BatchNorm layers used by PASS BN loss; 0 means all.')
    parser.add_argument('--pass_entropy_weight', type=float, default=0.0,
                        help='Optional softmax entropy weight added to PASS BN loss.')
    parser.add_argument('--pass_ema_decay', type=float, default=0.94,
                        help='EMA decay for PASS target prompt network.')
    parser.add_argument('--pass_min_momentum_constant', type=float, default=0.01,
                        help='Minimum EMA momentum constant for PASS.')
    parser.add_argument('--pass_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset PASS prompt/adaptor to source initialization for each test batch.')
    parser.add_argument('--pass_use_source_fallback', type=str2bool, nargs='?', const=True, default=True,
                        help='Use the source prediction when PASS adaptation increases prediction entropy.')
    parser.add_argument('--pass_image_size', type=int, default=192,
                        help='Reference image size used to derive the default PASS bottleneck prompt size.')
    parser.add_argument('--pass_prompt_size', type=int, default=None,
                        help='Explicit square bottleneck prompt size. Overrides --pass_image_size.')
    parser.add_argument('--pass_adaptor_hidden', type=int, default=64,
                        help='Hidden channels in the PASS image-space data adaptor.')
    parser.add_argument('--pass_perturb_scale', type=float, default=1.0,
                        help='Scale applied to PASS image-space adaptor residual.')
    parser.add_argument('--pass_prompt_scale', type=float, default=1.0,
                        help='Scale applied to PASS bottleneck prompt residual.')
    parser.add_argument('--pass_prompt_sparsity', type=float, default=0.1,
                        help='Sparse channel-attention keep ratio in PASS prompt matching.')
    parser.add_argument('--samtta_lr', type=float, default=1e-4,
                        help='Learning rate for SAM-TTA model-side updates.')
    parser.add_argument('--samtta_transform_lr', type=float, default=1e-2,
                        help='Learning rate for the SAM-TTA Bezier input transform.')
    parser.add_argument('--samtta_weight_decay', type=float, default=0.0,
                        help='Weight decay for SAM-TTA optimizer.')
    parser.add_argument('--samtta_steps', type=int, default=1,
                        help='Number of SAM-TTA adaptation steps per test batch.')
    parser.add_argument('--samtta_ema_momentum', type=float, default=0.95,
                        help='EMA teacher momentum for SAM-TTA.')
    parser.add_argument('--samtta_dpc_weight', type=float, default=1.0,
                        help='Weight for SAM-TTA dual-scale prediction consistency.')
    parser.add_argument('--samtta_feature_weight', type=float, default=0.1,
                        help='Weight for SAM-TTA bottleneck feature consistency.')
    parser.add_argument('--samtta_entropy_weight', type=float, default=0.05,
                        help='Weight for SAM-TTA prediction entropy minimization.')
    parser.add_argument('--samtta_transform_reg_weight', type=float, default=0.01,
                        help='Weight for preserving the source-normalized input after Bezier transform.')
    parser.add_argument('--samtta_feature_temp', type=float, default=2.0,
                        help='Temperature for SAM-TTA spatial feature KL.')
    parser.add_argument('--samtta_update_scope', type=str, default='bn_affine',
                        choices=['transform_only', 'bn_affine', 'all'],
                        help='Model parameters updated by SAM-TTA. Default updates BN affine plus Bezier transform.')
    parser.add_argument('--samtta_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset SAM-TTA model, transform and optimizer before each target batch.')
    parser.add_argument('--spmo_lr', type=float, default=1e-4,
                        help='Learning rate for SPMO BN-affine or model updates.')
    parser.add_argument('--spmo_weight_decay', type=float, default=0.0,
                        help='Weight decay for the SPMO optimizer.')
    parser.add_argument('--spmo_steps', type=int, default=1,
                        help='Number of SPMO adaptation steps per test batch.')
    parser.add_argument('--spmo_entropy_weight', type=float, default=1.0,
                        help='Weight for SPMO weighted entropy minimization.')
    parser.add_argument('--spmo_prior_weight', type=float, default=1.0,
                        help='Weight for SPMO source-prediction size prior KL loss.')
    parser.add_argument('--spmo_moment_weight', type=float, default=0.05,
                        help='Weight for SPMO source-prediction shape moment loss.')
    parser.add_argument('--spmo_moment_mode', type=str, default='all',
                        choices=['none', 'centroid', 'dist_centroid', 'all'],
                        help='SPMO shape moment constraint to use.')
    parser.add_argument('--spmo_softmax_temp', type=float, default=1.0,
                        help='Softmax temperature used by SPMO adaptation.')
    parser.add_argument('--spmo_size_power', type=float, default=1.0,
                        help='Power used in the SPMO norm_soft_size proportion estimate.')
    parser.add_argument('--spmo_bg_entropy_weight', type=float, default=0.1,
                        help='Background class weight for SPMO entropy minimization.')
    parser.add_argument('--spmo_prior_eps', type=float, default=1e-6,
                        help='Numerical floor for SPMO source size priors.')
    parser.add_argument('--spmo_min_pixels', type=int, default=10,
                        help='Minimum source-predicted pixels required for foreground moment constraints.')
    parser.add_argument('--spmo_source_pseudo', type=str, default='hard',
                        choices=['hard', 'soft'],
                        help='Use hard or soft frozen-source predictions as SPMO priors.')
    parser.add_argument('--spmo_update_scope', type=str, default='bn_affine',
                        choices=['bn_affine', 'all'],
                        help='Parameters updated by SPMO. Original-style default is BN affine only.')
    parser.add_argument('--spmo_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset SPMO model and optimizer before each target batch.')
    parser.add_argument('--sictta_max_lens', type=int, default=40,
                        help='Maximum number of reliable target bottleneck prototypes kept by SicTTA.')
    parser.add_argument('--sictta_topk', type=int, default=5,
                        help='Number of nearest prototype features mixed by SicTTA.')
    parser.add_argument('--sictta_threshold', type=float, default=0.9,
                        help='CCD reliability threshold used by SicTTA after the prototype pool is warm.')
    parser.add_argument('--sictta_select_points', type=int, default=200,
                        help='Number of random pixels sampled for SicTTA CCD entropy.')
    parser.add_argument('--sictta_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset SicTTA memory before every target slice. Default false for continual adaptation.')
    parser.add_argument('--a3_lr', type=float, default=1e-4,
                        help='Learning rate for A3-TTA online model updates.')
    parser.add_argument('--a3_steps', type=int, default=1,
                        help='Number of A3-TTA adaptation steps per test batch.')
    parser.add_argument('--a3_pool_size', type=int, default=40,
                        help='Maximum number of bottleneck prototypes kept by A3-TTA.')
    parser.add_argument('--a3_top_k', type=int, default=1,
                        help='Number of nearest A3-TTA prototypes used for feature alignment.')
    parser.add_argument('--a3_mt', type=float, default=0.99,
                        help='Maximum EMA teacher momentum for A3-TTA. Use a negative value for no cap.')
    parser.add_argument('--a3_feature_loss_weight', type=float, default=1.0,
                        help='Weight for A3-TTA aligned-feature prediction loss.')
    parser.add_argument('--a3_entropy_match_weight', type=float, default=5.0,
                        help='Weight for A3-TTA pixel-entropy matching loss.')
    parser.add_argument('--a3_ema_loss_weight', type=float, default=1.0,
                        help='Weight for A3-TTA EMA teacher consistency loss.')
    parser.add_argument('--a3_episodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset A3-TTA model and prototype pool before each target slice.')
    parser.add_argument('--a3_reset_on_scan_start', type=str2bool, nargs='?', const=True, default=False,
                        help='Reset A3-TTA model and prototype pool when a new target scan starts.')

    parser.add_argument('--validation_freq', type=int, default=10, help='valfreq')
    parser.add_argument('--display_freq', type=int, default=500, help='imgfreq')
    parser.add_argument('--save_freq', type=int, default=10, help='save model every N epochs')

    parser.add_argument('--w_ce', type=float, default=1.0, help='w_ce')
    parser.add_argument('--w_dice', type=float, default=1.0, help='w_dice')
    parser.add_argument('--w_seg', type=float, default=1.0, help='w_seg')
    # GIP/CLP is always enabled in the dataset pipeline; --use_sgf only controls
    # whether the strong view is fused as SGF(GIP, CLP) or uses CLP directly.
    parser.add_argument('--use_sgf', type=int, default=0,
                        help='Use saliency-guided fusion (SGF) for strong-view construction')
    parser.add_argument('--sgf_grid_size', type=int, default=8,
                        help='Grid size for the SGF saliency map')
    parser.add_argument('--sgf_view2_only', type=int, default=0,
                        help='SGF view2-only: skip base-view generation and base-view-dependent losses')

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='Number of batches to prefetch per worker')

    # Paper mainline: CGSD is an independent switch.
    parser.add_argument('--use_cgsd', type=int, default=1,
                        help='Enable CGSD: 1=on, 0=off')
    parser.add_argument('--lambda_str', type=float, default=0.3,
                        help='Weight for the structure consistency term in CGSD (default: 0.3)')
    parser.add_argument('--seg_alpha_view2', type=float, default=1.0,
                        help='Seg loss weight for the strong view when SAAM is disabled: L_seg = L1 + alpha*L2 (default 1.0)')
    parser.add_argument('--cgsd_layer', type=int, default=1,
                        help='Which encoder layer to apply CGSD: 1=layer1(16ch), 2=layer2(32ch), 3=layer3(64ch)')
    parser.add_argument('--lambda_sty', type=float, default=0.3,
                        help='Weight for the style diversity term in CGSD (default: 0.3)')

    # CGSD projector parameters
    parser.add_argument('--use_projector', type=int, default=1,
                        help='Enable projector for CGSD: 1=on (CCSDG-style abstract space), '
                             '0=off (direct feature space). Default: 1 (recommended for stability)')
    parser.add_argument('--proj_dim', type=int, default=1024,
                        help='Projection dimension for projector (default: 1024, CCSDG uses 1024)')
    parser.add_argument('--proj_hidden_channels', type=int, default=8,
                        help='Hidden channels in projector conv layer (default: 8, CCSDG uses 8)')
    parser.add_argument('--use_separate_cgsd_optimizer', type=int, default=1,
                        help='Use separate optimizer for CGSD (projector+gate) vs backbone: '
                             '1=two-step optimization (recommended), 0=single optimizer (may have gradient conflicts). '
                             'Default: 1')
    parser.add_argument('--cgsd_lr', type=float, default=None,
                        help='Learning rate for CGSD optimizer (projector+gate). If None, uses main LR.')
    parser.add_argument('--cgsd_momentum', type=float, default=0.99,
                        help='Momentum for CGSD optimizer (SGD with Nesterov). Default: 0.99')
    parser.add_argument('--use_temperature', type=int, default=0,
                        help='Optional ablation: use temperature-softmax gating instead of the default sigmoid gate')
    parser.add_argument('--gate_tau', type=float, default=0.1,
                        help='Temperature parameter for ChannelGate softmax (if use_temperature=1). '
                             'Lower = more decisive channel assignment. CCSDG uses 0.1')

    # SAAM parameters
    parser.add_argument('--use_saam', type=int, default=0,
                        help='Enable SAAM selective alignment: 1=on, 0=off (default: 0)')
    parser.add_argument('--saam_tau', type=float, default=0.5,
                        help='Soft stability temperature tau in SAAM (paper default: 0.5)')
    parser.add_argument('--saam_topk', type=float, default=0.3,
                        help='Top-k ratio rho for stable-region selection in SAAM (default: 0.3)')
    parser.add_argument('--saam_stability_mode', type=str, default='mean',
                        help='Stability aggregation mode used by SAAM: mean or max (default: mean)')
    parser.add_argument('--lambda_01', type=float, default=1.0,
                        help='Alignment weight for anchor-base pair (default: 1.0)')
    parser.add_argument('--lambda_02', type=float, default=1.0,
                        help='Alignment weight for anchor-strong pair (default: 1.0)')
    parser.add_argument('--saam_warmup_epochs', type=int, default=50,
                        help='Warmup epochs before enabling SAAM (default: 50)')
    parser.add_argument('--saam_rampup_epochs', type=int, default=100,
                        help='Rampup epochs to linearly increase SAAM weight from 0 to target (default: 100)')
    parser.add_argument('--anchor_seg_alpha', type=float, default=0.0,
                        help='Optional seg loss weight for the anchor view (default: 0.0)')
    parser.add_argument('--strong_seg_alpha', type=float, default=1.0,
                        help='Seg loss weight for the strong view when SAAM is enabled (paper default: 1.0)')

    # RCCS parameters
    parser.add_argument('--use_rccs', type=int, default=0,
                        help='Enable RCCS random-convolution candidate selection: 1=on, 0=off (default: 0)')
    parser.add_argument('--p_rccs', type=float, default=0.3,
                        help='Probability of applying RCCS to the strong view (default: 0.3)')
    parser.add_argument('--rccs_candidates', type=int, default=4,
                        help='Number of random-convolution candidates in RCCS (default: 4)')
    parser.add_argument('--rccs_metric', type=str, default='cos',
                        help='Semantic matching metric for RCCS: "cos" or "l2" (default: "cos")')
    parser.add_argument('--prefer_change', type=int, default=0,
                        help='Prefer candidates with more visual change (for exploration): 1=on, 0=off (default: 0)')
    parser.add_argument('--lambda_change', type=float, default=0.0,
                        help='Weight for change preference in RCCS selection (default: 0.0, try 0.1~0.2 if prefer_change=1)')
    parser.add_argument('--change_metric', type=str, default='l1',
                        help='Visual change metric: "l1" or "l2" (default: "l1")')
    parser.add_argument('--rccs_apply_to_saam', type=int, default=0,
                        help='Ablation: apply RCCS to SAAM features as well (default: 0=off, only affect seg loss)')
    parser.add_argument('--rccs_apply_to_base', type=int, default=0,
                        help='Ablation: apply RCCS to the base view instead of the strong view (default: 0=off)')
    parser.add_argument('--rccs_embed_dim', type=int, default=128,
                        help='Embedding dimension used by RCCS semantic matching (default: 128)')

    args = parser.parse_args()

    if args.phase == 'test' and args.tta == 'sm_ppm':
        mode_tag = f"smppm_{args.smppm_ablation_mode}"
        if mode_tag not in args.expname:
            args.expname = f"{mode_tag}_{args.expname}"

    # Validate RCCS parameters
    if hasattr(args, 'use_rccs') and args.use_rccs:
        # Validate p_rccs
        if not (0.0 <= args.p_rccs <= 1.0):
            raise ValueError(f"Invalid p_rccs={args.p_rccs}. Must be in [0, 1]")

        # Validate candidate count
        if args.rccs_candidates < 1:
            raise ValueError(f"Invalid rccs_candidates={args.rccs_candidates}. Must be >= 1")

        # Validate semantic metric
        if args.rccs_metric not in ['cos', 'l2']:
            raise ValueError(f"Invalid rccs_metric='{args.rccs_metric}'. Must be 'cos' or 'l2'")

        # Validate rccs_embed_dim
        if args.rccs_embed_dim <= 0:
            raise ValueError(f"Invalid rccs_embed_dim={args.rccs_embed_dim}. Must be > 0")

        # Validate change parameters
        if args.prefer_change and args.lambda_change < 0:
            raise ValueError(f"Invalid lambda_change={args.lambda_change}. Must be >= 0")

        if args.change_metric not in ['l1', 'l2']:
            raise ValueError(f"Invalid change_metric='{args.change_metric}'. Must be 'l1' or 'l2'")

    # Validate CGSD layer
    if hasattr(args, 'cgsd_layer'):
        if args.cgsd_layer not in [1, 2, 3]:
            raise ValueError(f"Invalid cgsd_layer={args.cgsd_layer}. Must be 1, 2, or 3")

    if args.tent_steps < 1:
        raise ValueError(f"Invalid tent_steps={args.tent_steps}. Must be >= 1")
    if args.tent_lr <= 0:
        raise ValueError(f"Invalid tent_lr={args.tent_lr}. Must be > 0")
    if args.dgtta_steps < 1:
        raise ValueError(f"Invalid dgtta_steps={args.dgtta_steps}. Must be >= 1")
    if args.dgtta_lr <= 0:
        raise ValueError(f"Invalid dgtta_lr={args.dgtta_lr}. Must be > 0")
    if args.dgtta_transform_strength < 0:
        raise ValueError(f"Invalid dgtta_transform_strength={args.dgtta_transform_strength}. Must be >= 0")
    if args.dgtta_entropy_weight < 0:
        raise ValueError(f"Invalid dgtta_entropy_weight={args.dgtta_entropy_weight}. Must be >= 0")
    if args.dgtta_bn_l2_reg < 0:
        raise ValueError(f"Invalid dgtta_bn_l2_reg={args.dgtta_bn_l2_reg}. Must be >= 0")
    if args.cotta_steps < 1:
        raise ValueError(f"Invalid cotta_steps={args.cotta_steps}. Must be >= 1")
    if args.cotta_lr <= 0:
        raise ValueError(f"Invalid cotta_lr={args.cotta_lr}. Must be > 0")
    if args.memo_steps < 1:
        raise ValueError(f"Invalid memo_steps={args.memo_steps}. Must be >= 1")
    if args.memo_lr <= 0:
        raise ValueError(f"Invalid memo_lr={args.memo_lr}. Must be > 0")
    if args.memo_n_augmentations < 1:
        raise ValueError(f"Invalid memo_n_augmentations={args.memo_n_augmentations}. Must be >= 1")
    if args.memo_include_identity not in [0, 1]:
        raise ValueError(f"Invalid memo_include_identity={args.memo_include_identity}. Must be 0 or 1")
    if not (0.0 <= args.memo_hflip_p <= 1.0):
        raise ValueError(f"Invalid memo_hflip_p={args.memo_hflip_p}. Must be in [0, 1]")
    if not (0.0 <= args.bn_alpha <= 1.0):
        raise ValueError(f"Invalid bn_alpha={args.bn_alpha}. Must be in [0, 1]")
    if args.asm_lr <= 0:
        raise ValueError(f"Invalid asm_lr={args.asm_lr}. Must be > 0")
    if args.asm_steps < 1:
        raise ValueError(f"Invalid asm_steps={args.asm_steps}. Must be >= 1")
    if args.asm_inner_steps < 1:
        raise ValueError(f"Invalid asm_inner_steps={args.asm_inner_steps}. Must be >= 1")
    if args.asm_lambda_reg < 0:
        raise ValueError(f"Invalid asm_lambda_reg={args.asm_lambda_reg}. Must be >= 0")
    if args.asm_sampling_step <= 0:
        raise ValueError(f"Invalid asm_sampling_step={args.asm_sampling_step}. Must be > 0")
    if args.asm_src_batch_size < 1:
        raise ValueError(f"Invalid asm_src_batch_size={args.asm_src_batch_size}. Must be >= 1")
    if args.smppm_lr <= 0:
        raise ValueError(f"Invalid smppm_lr={args.smppm_lr}. Must be > 0")
    if args.smppm_momentum < 0:
        raise ValueError(f"Invalid smppm_momentum={args.smppm_momentum}. Must be >= 0")
    if args.smppm_wd < 0:
        raise ValueError(f"Invalid smppm_wd={args.smppm_wd}. Must be >= 0")
    if args.smppm_steps < 1:
        raise ValueError(f"Invalid smppm_steps={args.smppm_steps}. Must be >= 1")
    if args.smppm_src_batch_size < 1:
        raise ValueError(f"Invalid smppm_src_batch_size={args.smppm_src_batch_size}. Must be >= 1")
    if args.smppm_patch_size < 1:
        raise ValueError(f"Invalid smppm_patch_size={args.smppm_patch_size}. Must be >= 1")
    if args.smppm_feature_size < args.smppm_patch_size:
        raise ValueError("smppm_feature_size must be >= smppm_patch_size")
    if args.smppm_feature_size % args.smppm_patch_size != 0:
        raise ValueError("smppm_feature_size must be divisible by smppm_patch_size")
    if not (0.0 <= args.smppm_source_free_tau <= 1.0):
        raise ValueError(f"Invalid smppm_source_free_tau={args.smppm_source_free_tau}. Must be in [0, 1]")
    if args.smppm_source_free_entropy_threshold is not None and args.smppm_source_free_entropy_threshold < 0:
        raise ValueError("smppm_source_free_entropy_threshold must be >= 0 when set")
    if args.smppm_source_free_entropy_weight < 0:
        raise ValueError("smppm_source_free_entropy_weight must be >= 0")
    if args.smppm_source_free_lambda_proto < 0:
        raise ValueError("smppm_source_free_lambda_proto must be >= 0")
    if args.gtta_lr <= 0:
        raise ValueError(f"Invalid gtta_lr={args.gtta_lr}. Must be > 0")
    if args.gtta_momentum < 0:
        raise ValueError(f"Invalid gtta_momentum={args.gtta_momentum}. Must be >= 0")
    if args.gtta_wd < 0:
        raise ValueError(f"Invalid gtta_wd={args.gtta_wd}. Must be >= 0")
    if args.gtta_steps < 1:
        raise ValueError(f"Invalid gtta_steps={args.gtta_steps}. Must be >= 1")
    if args.gtta_src_batch_size < 1:
        raise ValueError(f"Invalid gtta_src_batch_size={args.gtta_src_batch_size}. Must be >= 1")
    if args.gtta_lambda_ce_trg < 0:
        raise ValueError(f"Invalid gtta_lambda_ce_trg={args.gtta_lambda_ce_trg}. Must be >= 0")
    if not (0.0 <= args.gtta_pseudo_momentum <= 1.0):
        raise ValueError(f"Invalid gtta_pseudo_momentum={args.gtta_pseudo_momentum}. Must be in [0, 1]")
    if not (0.0 <= args.gtta_style_alpha <= 1.0):
        raise ValueError(f"Invalid gtta_style_alpha={args.gtta_style_alpha}. Must be in [0, 1]")
    if args.gtta_include_original not in [0, 1]:
        raise ValueError(f"Invalid gtta_include_original={args.gtta_include_original}. Must be 0 or 1")
    if args.gold_lr <= 0:
        raise ValueError(f"Invalid gold_lr={args.gold_lr}. Must be > 0")
    if args.gold_momentum < 0:
        raise ValueError(f"Invalid gold_momentum={args.gold_momentum}. Must be >= 0")
    if args.gold_wd < 0:
        raise ValueError(f"Invalid gold_wd={args.gold_wd}. Must be >= 0")
    if args.gold_steps < 1:
        raise ValueError(f"Invalid gold_steps={args.gold_steps}. Must be >= 1")
    if args.gold_rank < 1:
        raise ValueError(f"Invalid gold_rank={args.gold_rank}. Must be >= 1")
    if not (0.0 <= args.gold_tau <= 1.0):
        raise ValueError(f"Invalid gold_tau={args.gold_tau}. Must be in [0, 1]")
    if not (0.0 <= args.gold_alpha <= 1.0):
        raise ValueError(f"Invalid gold_alpha={args.gold_alpha}. Must be in [0, 1]")
    if args.gold_t_eig < 1:
        raise ValueError(f"Invalid gold_t_eig={args.gold_t_eig}. Must be >= 1")
    if not (0.0 <= args.gold_mt <= 1.0):
        raise ValueError(f"Invalid gold_mt={args.gold_mt}. Must be in [0, 1]")
    if args.gold_s_lr <= 0:
        raise ValueError(f"Invalid gold_s_lr={args.gold_s_lr}. Must be > 0")
    if args.gold_s_clip < 0:
        raise ValueError(f"Invalid gold_s_clip={args.gold_s_clip}. Must be >= 0")
    if args.gold_adapter_scale < 0:
        raise ValueError(f"Invalid gold_adapter_scale={args.gold_adapter_scale}. Must be >= 0")
    if args.gold_max_pixels_per_batch < 1:
        raise ValueError(f"Invalid gold_max_pixels_per_batch={args.gold_max_pixels_per_batch}. Must be >= 1")
    if args.gold_min_pixels_per_batch < 1:
        raise ValueError(f"Invalid gold_min_pixels_per_batch={args.gold_min_pixels_per_batch}. Must be >= 1")
    if args.gold_min_pixels_per_batch > args.gold_max_pixels_per_batch:
        raise ValueError("gold_min_pixels_per_batch must be <= gold_max_pixels_per_batch")
    if args.gold_n_augmentations < 0:
        raise ValueError(f"Invalid gold_n_augmentations={args.gold_n_augmentations}. Must be >= 0")
    if not (0.0 <= args.gold_rst <= 1.0):
        raise ValueError(f"Invalid gold_rst={args.gold_rst}. Must be in [0, 1]")
    if not (0.0 <= args.gold_ap <= 1.0):
        raise ValueError(f"Invalid gold_ap={args.gold_ap}. Must be in [0, 1]")
    if args.vptta_lr <= 0:
        raise ValueError(f"Invalid vptta_lr={args.vptta_lr}. Must be > 0")
    if args.vptta_momentum < 0:
        raise ValueError(f"Invalid vptta_momentum={args.vptta_momentum}. Must be >= 0")
    if not (0.0 <= args.vptta_beta1 < 1.0):
        raise ValueError(f"Invalid vptta_beta1={args.vptta_beta1}. Must be in [0, 1)")
    if not (0.0 <= args.vptta_beta2 < 1.0):
        raise ValueError(f"Invalid vptta_beta2={args.vptta_beta2}. Must be in [0, 1)")
    if args.vptta_weight_decay < 0:
        raise ValueError(f"Invalid vptta_weight_decay={args.vptta_weight_decay}. Must be >= 0")
    if args.vptta_steps < 1:
        raise ValueError(f"Invalid vptta_steps={args.vptta_steps}. Must be >= 1")
    if args.vptta_memory_size < 1:
        raise ValueError(f"Invalid vptta_memory_size={args.vptta_memory_size}. Must be >= 1")
    if args.vptta_neighbor < 1:
        raise ValueError(f"Invalid vptta_neighbor={args.vptta_neighbor}. Must be >= 1")
    if args.vptta_neighbor > args.vptta_memory_size:
        raise ValueError("vptta_neighbor must be <= vptta_memory_size")
    if not (0.0 < args.vptta_prompt_alpha <= 1.0):
        raise ValueError(f"Invalid vptta_prompt_alpha={args.vptta_prompt_alpha}. Must be in (0, 1]")
    if args.vptta_prompt_size is not None and args.vptta_prompt_size < 1:
        raise ValueError(f"Invalid vptta_prompt_size={args.vptta_prompt_size}. Must be >= 1")
    if args.vptta_image_size < 1:
        raise ValueError(f"Invalid vptta_image_size={args.vptta_image_size}. Must be >= 1")
    if args.vptta_warm_n < 1:
        raise ValueError(f"Invalid vptta_warm_n={args.vptta_warm_n}. Must be >= 1")
    if args.pass_lr <= 0:
        raise ValueError(f"Invalid pass_lr={args.pass_lr}. Must be > 0")
    if args.pass_momentum < 0:
        raise ValueError(f"Invalid pass_momentum={args.pass_momentum}. Must be >= 0")
    if not (0.0 <= args.pass_beta1 < 1.0):
        raise ValueError(f"Invalid pass_beta1={args.pass_beta1}. Must be in [0, 1)")
    if not (0.0 <= args.pass_beta2 < 1.0):
        raise ValueError(f"Invalid pass_beta2={args.pass_beta2}. Must be in [0, 1)")
    if args.pass_weight_decay < 0:
        raise ValueError(f"Invalid pass_weight_decay={args.pass_weight_decay}. Must be >= 0")
    if args.pass_steps < 1:
        raise ValueError(f"Invalid pass_steps={args.pass_steps}. Must be >= 1")
    if args.pass_bn_alpha < 0:
        raise ValueError(f"Invalid pass_bn_alpha={args.pass_bn_alpha}. Must be >= 0")
    if args.pass_bn_layers < 0:
        raise ValueError(f"Invalid pass_bn_layers={args.pass_bn_layers}. Must be >= 0")
    if args.pass_entropy_weight < 0:
        raise ValueError(f"Invalid pass_entropy_weight={args.pass_entropy_weight}. Must be >= 0")
    if not (0.0 <= args.pass_ema_decay <= 1.0):
        raise ValueError(f"Invalid pass_ema_decay={args.pass_ema_decay}. Must be in [0, 1]")
    if args.pass_min_momentum_constant < 0:
        raise ValueError("pass_min_momentum_constant must be >= 0")
    if args.pass_image_size < 1:
        raise ValueError(f"Invalid pass_image_size={args.pass_image_size}. Must be >= 1")
    if args.pass_prompt_size is not None and args.pass_prompt_size < 1:
        raise ValueError(f"Invalid pass_prompt_size={args.pass_prompt_size}. Must be >= 1")
    if args.pass_adaptor_hidden < 1:
        raise ValueError(f"Invalid pass_adaptor_hidden={args.pass_adaptor_hidden}. Must be >= 1")
    if args.pass_perturb_scale < 0:
        raise ValueError(f"Invalid pass_perturb_scale={args.pass_perturb_scale}. Must be >= 0")
    if args.pass_prompt_scale < 0:
        raise ValueError(f"Invalid pass_prompt_scale={args.pass_prompt_scale}. Must be >= 0")
    if not (0.0 < args.pass_prompt_sparsity <= 1.0):
        raise ValueError(f"Invalid pass_prompt_sparsity={args.pass_prompt_sparsity}. Must be in (0, 1]")
    if args.samtta_lr <= 0:
        raise ValueError(f"Invalid samtta_lr={args.samtta_lr}. Must be > 0")
    if args.samtta_transform_lr <= 0:
        raise ValueError(f"Invalid samtta_transform_lr={args.samtta_transform_lr}. Must be > 0")
    if args.samtta_weight_decay < 0:
        raise ValueError(f"Invalid samtta_weight_decay={args.samtta_weight_decay}. Must be >= 0")
    if args.samtta_steps < 1:
        raise ValueError(f"Invalid samtta_steps={args.samtta_steps}. Must be >= 1")
    if not (0.0 <= args.samtta_ema_momentum <= 1.0):
        raise ValueError(f"Invalid samtta_ema_momentum={args.samtta_ema_momentum}. Must be in [0, 1]")
    if args.samtta_dpc_weight < 0:
        raise ValueError("samtta_dpc_weight must be >= 0")
    if args.samtta_feature_weight < 0:
        raise ValueError("samtta_feature_weight must be >= 0")
    if args.samtta_entropy_weight < 0:
        raise ValueError("samtta_entropy_weight must be >= 0")
    if args.samtta_transform_reg_weight < 0:
        raise ValueError("samtta_transform_reg_weight must be >= 0")
    if args.samtta_feature_temp <= 0:
        raise ValueError(f"Invalid samtta_feature_temp={args.samtta_feature_temp}. Must be > 0")
    if args.spmo_lr <= 0:
        raise ValueError(f"Invalid spmo_lr={args.spmo_lr}. Must be > 0")
    if args.spmo_weight_decay < 0:
        raise ValueError(f"Invalid spmo_weight_decay={args.spmo_weight_decay}. Must be >= 0")
    if args.spmo_steps < 1:
        raise ValueError(f"Invalid spmo_steps={args.spmo_steps}. Must be >= 1")
    if args.spmo_entropy_weight < 0:
        raise ValueError("spmo_entropy_weight must be >= 0")
    if args.spmo_prior_weight < 0:
        raise ValueError("spmo_prior_weight must be >= 0")
    if args.spmo_moment_weight < 0:
        raise ValueError("spmo_moment_weight must be >= 0")
    if args.spmo_softmax_temp <= 0:
        raise ValueError(f"Invalid spmo_softmax_temp={args.spmo_softmax_temp}. Must be > 0")
    if args.spmo_size_power <= 0:
        raise ValueError(f"Invalid spmo_size_power={args.spmo_size_power}. Must be > 0")
    if args.spmo_bg_entropy_weight < 0:
        raise ValueError("spmo_bg_entropy_weight must be >= 0")
    if args.spmo_prior_eps <= 0:
        raise ValueError(f"Invalid spmo_prior_eps={args.spmo_prior_eps}. Must be > 0")
    if args.spmo_min_pixels < 0:
        raise ValueError("spmo_min_pixels must be >= 0")
    if args.sictta_max_lens < 1:
        raise ValueError(f"Invalid sictta_max_lens={args.sictta_max_lens}. Must be >= 1")
    if args.sictta_topk < 1:
        raise ValueError(f"Invalid sictta_topk={args.sictta_topk}. Must be >= 1")
    if not (0.0 <= args.sictta_threshold <= 1.0):
        raise ValueError(f"Invalid sictta_threshold={args.sictta_threshold}. Must be in [0, 1]")
    if args.sictta_select_points < 1:
        raise ValueError(f"Invalid sictta_select_points={args.sictta_select_points}. Must be >= 1")
    if args.a3_lr <= 0:
        raise ValueError(f"Invalid a3_lr={args.a3_lr}. Must be > 0")
    if args.a3_steps < 1:
        raise ValueError(f"Invalid a3_steps={args.a3_steps}. Must be >= 1")
    if args.a3_pool_size < 1:
        raise ValueError(f"Invalid a3_pool_size={args.a3_pool_size}. Must be >= 1")
    if args.a3_top_k < 1:
        raise ValueError(f"Invalid a3_top_k={args.a3_top_k}. Must be >= 1")
    if args.a3_mt >= 0 and not (0.0 <= args.a3_mt <= 1.0):
        raise ValueError(f"Invalid a3_mt={args.a3_mt}. Must be in [0, 1], or negative for no cap")
    if args.a3_feature_loss_weight < 0:
        raise ValueError("a3_feature_loss_weight must be >= 0")
    if args.a3_entropy_match_weight < 0:
        raise ValueError("a3_entropy_match_weight must be >= 0")
    if args.a3_ema_loss_weight < 0:
        raise ValueError("a3_ema_loss_weight must be >= 0")
    if args.phase != 'test' and args.tta != 'none':
        raise ValueError("TTA is only supported with --phase test in this DCON entry point.")

    return args
    
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    opt=get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    print("Dataset pipeline: base view uses GIP; strong view uses CLP or SGF(GIP, CLP).")
    if opt.use_sgf:
        print("SGF enabled: base view uses GIP, strong view uses SGF(GIP, CLP)")
        print("SGF inputs (GIP/CLP) use intensity transforms after shared geometry")
        if opt.sgf_view2_only:
            print("SGF view2-only: base view disabled; SAAM and pairwise CGSD losses are skipped")
    else:
        print("SGF disabled: strong view uses CLP directly.")

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(opt.f_seed)
    np.random.seed(opt.f_seed)
    torch.manual_seed(opt.f_seed)
    torch.cuda.manual_seed(opt.f_seed)

    # Use absolute path for checkpoint directory
    ckpt_dir = os.path.abspath(opt.ckpt_dir)
    dn_dir = os.path.join(ckpt_dir, opt.tr_domain)
    exp_dir = os.path.join(dn_dir, opt.expname)
    snap_dir = os.path.join(exp_dir, 'snapshots')
    tbfile_dir = os.path.join(exp_dir, 'tboard')
    logdir = os.path.join(exp_dir, 'log')

    print(f"\n{'='*80}")
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Experiment directory: {exp_dir}")
    print(f"{'='*80}\n")
    
    if opt.phase=='train':
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        code_dir = os.path.join(exp_dir, 'code')
        # Ignore patterns to prevent recursive copying and save space
        ignore_patterns = shutil.ignore_patterns('.git', '__pycache__', 'ckpts', 'data', '*.pyc', '.idea', '.ipynb_checkpoints')
        shutil.copytree('.', code_dir, ignore=ignore_patterns)
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(dn_dir):
        os.makedirs(dn_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(snap_dir):
        os.mkdir(snap_dir)
    if not os.path.exists(tbfile_dir):
        os.mkdir(tbfile_dir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        os.mkdir(os.path.join(logdir, 'train'))
        os.mkdir(os.path.join(logdir, 'img'))
        os.mkdir(os.path.join(logdir, 'pred'))

    finalfile = os.path.join(logdir, 'out.csv') 

    logger = logging.getLogger('log1')
    logging.basicConfig(filemode='a', level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.FileHandler(os.path.join(exp_dir, 'log.txt'), encoding='utf-8'))
    logging.info("config:"+str(opt))
    logging.info("name:"+opt.expname)
    logging.info("use_cgsd:"+str(opt.use_cgsd))
    logging.info("view_pipeline:"+("GIP->SGF(GIP,CLP)" if opt.use_sgf else "GIP->CLP"))
    if opt.use_cgsd:
        logging.info("lambda_str:"+str(opt.lambda_str))
        logging.info("lambda_sty:"+str(opt.lambda_sty))

    if opt.use_saam:
        logging.info("=== SAAM Configuration ===")
        logging.info("use_saam: "+str(opt.use_saam))
        logging.info("use_sgf: "+str(opt.use_sgf)+" (SAAM works with or without SGF)")
        logging.info("saam_tau: "+str(opt.saam_tau))
        logging.info("saam_topk: "+str(opt.saam_topk))
        logging.info("saam_stability_mode: "+str(opt.saam_stability_mode))
        logging.info("lambda_01: "+str(opt.lambda_01))
        logging.info("lambda_02: "+str(opt.lambda_02))
        logging.info("saam_warmup_epochs: "+str(opt.saam_warmup_epochs))
        logging.info("saam_rampup_epochs: "+str(opt.saam_rampup_epochs))
        logging.info("anchor_seg_alpha:"+str(opt.anchor_seg_alpha))
        logging.info("strong_seg_alpha:"+str(opt.strong_seg_alpha))

    # RCCS Configuration logging
    if hasattr(opt, 'use_rccs') and opt.use_rccs:
        logging.info("=== RCCS Configuration ===")
        logging.info("use_rccs: "+str(opt.use_rccs))
        logging.info("p_rccs: "+str(opt.p_rccs))
        logging.info("rccs_candidates: "+str(opt.rccs_candidates))
        logging.info("rccs_metric: "+str(opt.rccs_metric))
        logging.info("prefer_change: "+str(opt.prefer_change))
        logging.info("lambda_change: "+str(opt.lambda_change))
        logging.info("change_metric: "+str(opt.change_metric))
        logging.info("rccs_apply_to_saam: "+str(opt.rccs_apply_to_saam))
        logging.info("rccs_apply_to_base: "+str(opt.rccs_apply_to_base))
        logging.info("rccs_embed_dim: "+str(opt.rccs_embed_dim))

    logging.info("opt.f_seed:"+str(opt.f_seed))
    logging.info("data:"+opt.data_name)
    logging.info("tta:"+str(opt.tta))
    if opt.tta == 'tent':
        logging.info("tent_lr:"+str(opt.tent_lr))
        logging.info("tent_steps:"+str(opt.tent_steps))
    elif opt.tta == 'dg_tta':
        logging.info("dgtta_lr:"+str(opt.dgtta_lr))
        logging.info("dgtta_steps:"+str(opt.dgtta_steps))
        logging.info("dgtta_transform_strength:"+str(opt.dgtta_transform_strength))
        logging.info("dgtta_entropy_weight:"+str(opt.dgtta_entropy_weight))
        logging.info("dgtta_bn_l2_reg:"+str(opt.dgtta_bn_l2_reg))
        logging.info("dgtta_episodic:"+str(opt.dgtta_episodic))
    elif opt.tta == 'norm_alpha':
        logging.info("bn_alpha:"+str(opt.bn_alpha))
    elif opt.tta == 'cotta':
        logging.info("cotta_lr:"+str(opt.cotta_lr))
        logging.info("cotta_steps:"+str(opt.cotta_steps))
        logging.info("cotta_mt:"+str(opt.cotta_mt))
        logging.info("cotta_rst:"+str(opt.cotta_rst))
        logging.info("cotta_ap:"+str(opt.cotta_ap))
    elif opt.tta == 'memo':
        logging.info("memo_lr:"+str(opt.memo_lr))
        logging.info("memo_steps:"+str(opt.memo_steps))
        logging.info("memo_n_augmentations:"+str(opt.memo_n_augmentations))
        logging.info("memo_include_identity:"+str(opt.memo_include_identity))
        logging.info("memo_hflip_p:"+str(opt.memo_hflip_p))
        logging.info("memo_update_scope:"+str(opt.memo_update_scope))
    elif opt.tta == 'asm':
        logging.info("=== ASM Configuration ===")
        logging.info("ASM is source-dependent TTA: target images provide style/statistics only; source labels supervise adaptation.")
        logging.info("asm_lr:"+str(opt.asm_lr))
        logging.info("asm_steps:"+str(opt.asm_steps))
        logging.info("asm_inner_steps:"+str(opt.asm_inner_steps))
        logging.info("asm_lambda_reg:"+str(opt.asm_lambda_reg))
        logging.info("asm_sampling_step:"+str(opt.asm_sampling_step))
        logging.info("asm_src_batch_size:"+str(opt.asm_src_batch_size))
        logging.info("asm_style_backend:"+str(opt.asm_style_backend))
        logging.info("asm_episodic:"+str(opt.asm_episodic))
    elif opt.tta == 'sm_ppm':
        logging.info("=== SM-PPM Configuration ===")
        logging.info("[SM-PPM Ablation Mode] "+str(opt.smppm_ablation_mode))
        logging.info("Current DCON tta_smppm.py has no explicit SM style-mixing implementation; sm_ce is unavailable.")
        logging.info("smppm_lr:"+str(opt.smppm_lr))
        logging.info("smppm_momentum:"+str(opt.smppm_momentum))
        logging.info("smppm_wd:"+str(opt.smppm_wd))
        logging.info("smppm_steps:"+str(opt.smppm_steps))
        logging.info("smppm_src_batch_size:"+str(opt.smppm_src_batch_size))
        logging.info("smppm_patch_size:"+str(opt.smppm_patch_size))
        logging.info("smppm_feature_size:"+str(opt.smppm_feature_size))
        logging.info("smppm_episodic:"+str(opt.smppm_episodic))
        logging.info("smppm_source_free_tau:"+str(opt.smppm_source_free_tau))
        logging.info("smppm_source_free_entropy_threshold:"+str(opt.smppm_source_free_entropy_threshold))
        logging.info("smppm_source_free_entropy_weight:"+str(opt.smppm_source_free_entropy_weight))
        logging.info("smppm_source_free_lambda_proto:"+str(opt.smppm_source_free_lambda_proto))
        logging.info("smppm_plain_source_loader:"+str(opt.smppm_plain_source_loader))
    elif opt.tta == 'gtta':
        logging.info("=== GTTA Configuration ===")
        logging.info("GTTA is source-dependent TTA: source labels supervise adaptation; target labels are evaluation-only.")
        logging.info("gtta_lr:"+str(opt.gtta_lr))
        logging.info("gtta_momentum:"+str(opt.gtta_momentum))
        logging.info("gtta_wd:"+str(opt.gtta_wd))
        logging.info("gtta_steps:"+str(opt.gtta_steps))
        logging.info("gtta_src_batch_size:"+str(opt.gtta_src_batch_size))
        logging.info("gtta_lambda_ce_trg:"+str(opt.gtta_lambda_ce_trg))
        logging.info("gtta_pseudo_momentum:"+str(opt.gtta_pseudo_momentum))
        logging.info("gtta_style_alpha:"+str(opt.gtta_style_alpha))
        logging.info("gtta_include_original:"+str(opt.gtta_include_original))
        logging.info("gtta_episodic:"+str(opt.gtta_episodic))
    elif opt.tta == 'gold':
        logging.info("=== GOLD Configuration ===")
        logging.info("GOLD is source-free TTA: target labels are used only for evaluation.")
        logging.info("gold_lr:"+str(opt.gold_lr))
        logging.info("gold_momentum:"+str(opt.gold_momentum))
        logging.info("gold_wd:"+str(opt.gold_wd))
        logging.info("gold_steps:"+str(opt.gold_steps))
        logging.info("gold_rank:"+str(opt.gold_rank))
        logging.info("gold_tau:"+str(opt.gold_tau))
        logging.info("gold_alpha:"+str(opt.gold_alpha))
        logging.info("gold_t_eig:"+str(opt.gold_t_eig))
        logging.info("gold_mt:"+str(opt.gold_mt))
        logging.info("gold_s_lr:"+str(opt.gold_s_lr))
        logging.info("gold_s_init_scale:"+str(opt.gold_s_init_scale))
        logging.info("gold_s_clip:"+str(opt.gold_s_clip))
        logging.info("gold_adapter_scale:"+str(opt.gold_adapter_scale))
        logging.info("gold_max_pixels_per_batch:"+str(opt.gold_max_pixels_per_batch))
        logging.info("gold_min_pixels_per_batch:"+str(opt.gold_min_pixels_per_batch))
        logging.info("gold_n_augmentations:"+str(opt.gold_n_augmentations))
        logging.info("gold_rst:"+str(opt.gold_rst))
        logging.info("gold_ap:"+str(opt.gold_ap))
        logging.info("gold_episodic:"+str(opt.gold_episodic))
    elif opt.tta == 'vptta':
        logging.info("=== VPTTA Configuration ===")
        logging.info("VPTTA is source-free TTA: only the frequency prompt is updated; target labels are evaluation-only.")
        logging.info("vptta_optimizer:"+str(opt.vptta_optimizer))
        logging.info("vptta_lr:"+str(opt.vptta_lr))
        logging.info("vptta_momentum:"+str(opt.vptta_momentum))
        logging.info("vptta_beta1:"+str(opt.vptta_beta1))
        logging.info("vptta_beta2:"+str(opt.vptta_beta2))
        logging.info("vptta_weight_decay:"+str(opt.vptta_weight_decay))
        logging.info("vptta_steps:"+str(opt.vptta_steps))
        logging.info("vptta_memory_size:"+str(opt.vptta_memory_size))
        logging.info("vptta_neighbor:"+str(opt.vptta_neighbor))
        logging.info("vptta_prompt_alpha:"+str(opt.vptta_prompt_alpha))
        logging.info("vptta_prompt_size:"+str(opt.vptta_prompt_size))
        logging.info("vptta_image_size:"+str(opt.vptta_image_size))
        logging.info("vptta_warm_n:"+str(opt.vptta_warm_n))
    elif opt.tta == 'pass':
        logging.info("=== PASS Configuration ===")
        logging.info("PASS is source-free TTA: prompt/adaptor and optional BN affine tensors are updated; target labels are evaluation-only.")
        logging.info("pass_optimizer:"+str(opt.pass_optimizer))
        logging.info("pass_lr:"+str(opt.pass_lr))
        logging.info("pass_momentum:"+str(opt.pass_momentum))
        logging.info("pass_beta1:"+str(opt.pass_beta1))
        logging.info("pass_beta2:"+str(opt.pass_beta2))
        logging.info("pass_weight_decay:"+str(opt.pass_weight_decay))
        logging.info("pass_steps:"+str(opt.pass_steps))
        logging.info("pass_bn_alpha:"+str(opt.pass_bn_alpha))
        logging.info("pass_bn_layers:"+str(opt.pass_bn_layers))
        logging.info("pass_entropy_weight:"+str(opt.pass_entropy_weight))
        logging.info("pass_ema_decay:"+str(opt.pass_ema_decay))
        logging.info("pass_min_momentum_constant:"+str(opt.pass_min_momentum_constant))
        logging.info("pass_episodic:"+str(opt.pass_episodic))
        logging.info("pass_use_source_fallback:"+str(opt.pass_use_source_fallback))
        logging.info("pass_image_size:"+str(opt.pass_image_size))
        logging.info("pass_prompt_size:"+str(opt.pass_prompt_size))
        logging.info("pass_adaptor_hidden:"+str(opt.pass_adaptor_hidden))
        logging.info("pass_perturb_scale:"+str(opt.pass_perturb_scale))
        logging.info("pass_prompt_scale:"+str(opt.pass_prompt_scale))
        logging.info("pass_prompt_sparsity:"+str(opt.pass_prompt_sparsity))
    elif opt.tta == 'samtta':
        logging.info("=== SAM-TTA Configuration ===")
        logging.info("SAM-TTA is source-free TTA: Bezier input transform plus EMA teacher-student consistency; target labels are evaluation-only.")
        logging.info("samtta_lr:"+str(opt.samtta_lr))
        logging.info("samtta_transform_lr:"+str(opt.samtta_transform_lr))
        logging.info("samtta_weight_decay:"+str(opt.samtta_weight_decay))
        logging.info("samtta_steps:"+str(opt.samtta_steps))
        logging.info("samtta_ema_momentum:"+str(opt.samtta_ema_momentum))
        logging.info("samtta_dpc_weight:"+str(opt.samtta_dpc_weight))
        logging.info("samtta_feature_weight:"+str(opt.samtta_feature_weight))
        logging.info("samtta_entropy_weight:"+str(opt.samtta_entropy_weight))
        logging.info("samtta_transform_reg_weight:"+str(opt.samtta_transform_reg_weight))
        logging.info("samtta_feature_temp:"+str(opt.samtta_feature_temp))
        logging.info("samtta_update_scope:"+str(opt.samtta_update_scope))
        logging.info("samtta_episodic:"+str(opt.samtta_episodic))
    elif opt.tta == 'spmo':
        logging.info("=== SPMO Configuration ===")
        logging.info("SPMO is source-free shape-moment TTA: a frozen source model supplies priors; target labels are evaluation-only.")
        logging.info("spmo_lr:"+str(opt.spmo_lr))
        logging.info("spmo_weight_decay:"+str(opt.spmo_weight_decay))
        logging.info("spmo_steps:"+str(opt.spmo_steps))
        logging.info("spmo_entropy_weight:"+str(opt.spmo_entropy_weight))
        logging.info("spmo_prior_weight:"+str(opt.spmo_prior_weight))
        logging.info("spmo_moment_weight:"+str(opt.spmo_moment_weight))
        logging.info("spmo_moment_mode:"+str(opt.spmo_moment_mode))
        logging.info("spmo_softmax_temp:"+str(opt.spmo_softmax_temp))
        logging.info("spmo_size_power:"+str(opt.spmo_size_power))
        logging.info("spmo_bg_entropy_weight:"+str(opt.spmo_bg_entropy_weight))
        logging.info("spmo_prior_eps:"+str(opt.spmo_prior_eps))
        logging.info("spmo_min_pixels:"+str(opt.spmo_min_pixels))
        logging.info("spmo_source_pseudo:"+str(opt.spmo_source_pseudo))
        logging.info("spmo_update_scope:"+str(opt.spmo_update_scope))
        logging.info("spmo_episodic:"+str(opt.spmo_episodic))
    elif opt.tta == 'sictta':
        logging.info("=== SicTTA Configuration ===")
        logging.info("SicTTA is source-free single-image continual TTA: target labels are evaluation-only.")
        logging.info("sictta_max_lens:"+str(opt.sictta_max_lens))
        logging.info("sictta_topk:"+str(opt.sictta_topk))
        logging.info("sictta_threshold:"+str(opt.sictta_threshold))
        logging.info("sictta_select_points:"+str(opt.sictta_select_points))
        logging.info("sictta_episodic:"+str(opt.sictta_episodic))
    elif opt.tta == 'a3_tta':
        logging.info("=== A3-TTA Configuration ===")
        logging.info("A3-TTA is source-free online TTA: target labels are evaluation-only.")
        logging.info("a3_lr:"+str(opt.a3_lr))
        logging.info("a3_steps:"+str(opt.a3_steps))
        logging.info("a3_pool_size:"+str(opt.a3_pool_size))
        logging.info("a3_top_k:"+str(opt.a3_top_k))
        logging.info("a3_mt:"+str(opt.a3_mt))
        logging.info("a3_feature_loss_weight:"+str(opt.a3_feature_loss_weight))
        logging.info("a3_entropy_match_weight:"+str(opt.a3_entropy_match_weight))
        logging.info("a3_ema_loss_weight:"+str(opt.a3_ema_loss_weight))
        logging.info("a3_episodic:"+str(opt.a3_episodic))
        logging.info("a3_reset_on_scan_start:"+str(opt.a3_reset_on_scan_start))
    
    tb_writer = SummaryWriter( tbfile_dir  )

    with open(finalfile, 'a') as f:
        f.write(opt.expname+' '+opt.model+" \n")
    
    labmap=pre_labmap()
    labmap=labmap[opt.data_name]
    print("labmap:",labmap)

    # Auto-set nclass based on dataset if not explicitly provided
    dataset_nclass = {
        'ABDOMINAL': 5,
        'CARDIAC': 4,
        'PROSTATE': 2
    }
    if opt.data_name in dataset_nclass:
        expected_nclass = dataset_nclass[opt.data_name]
        if opt.nclass != expected_nclass:
            print(f"⚠️  WARNING: nclass={opt.nclass} but {opt.data_name} requires {expected_nclass} classes")
            print(f"   Auto-correcting to nclass={expected_nclass}")
            opt.nclass = expected_nclass

    smppm_requires_source_loader = (
        opt.phase == 'test'
        and opt.tta == 'sm_ppm'
        and opt.smppm_ablation_mode != 'source_free_proto'
    )
    smppm_source_free_proto = (
        opt.phase == 'test'
        and opt.tta == 'sm_ppm'
        and opt.smppm_ablation_mode == 'source_free_proto'
    )
    smppm_use_plain_source_set = smppm_requires_source_loader and opt.smppm_plain_source_loader

    if opt.data_name == 'ABDOMINAL':
        if opt.tr_domain=='SABSCT':
            tr_domain=['SABSCT']
            te_domain =[opt.target_domain if opt.target_domain is not None else 'CHAOST2']
        else:
            tr_domain=['CHAOST2']
            te_domain =[opt.target_domain if opt.target_domain is not None else 'SABSCT']

        if smppm_use_plain_source_set:
            train_set = ABD.get_training_plain(modality = tr_domain, norm_func = None, opt = opt)
        else:
            train_set  = ABD.get_training(modality = tr_domain ,norm_func = None,opt = opt)
        tr_valset  = ABD.get_trval(modality = tr_domain, norm_func = train_set.normalize_op,opt = opt) 
        tr_teset   = ABD.get_trtest(modality = tr_domain, norm_func = train_set.normalize_op,opt = opt)
        test_set   = ABD.get_test(modality = te_domain, norm_func = None,opt = opt) 
       
        label_name          = ABD.LABEL_NAME

    elif opt.data_name == 'PROSTATE':
        if PROS is None:
            raise ImportError("PROSTATE dataset support requires dataloaders/ProstateDataset.py, which is not present.")
        tr_domain=[opt.tr_domain]
        train_set  = PROS.get_training(modality = tr_domain , opt = opt)
        tr_valset  = PROS.get_trval(modality = tr_domain, opt = opt)
        tr_teset   = PROS.get_trtest(modality = tr_domain, opt = opt)
        test_set   = PROS.get_test(tr_modality = tr_domain, opt = opt)

        label_name      = PROS.LABEL_NAME

    elif opt.data_name == 'CARDIAC':
        if opt.tr_domain=='LGE':
            tr_domain=['LGE']
            te_domain=[opt.target_domain if opt.target_domain is not None else 'bSSFP']
        else:
            tr_domain=['bSSFP']
            te_domain=[opt.target_domain if opt.target_domain is not None else 'LGE']
        if smppm_use_plain_source_set:
            train_set = cardiac_cls.get_training_plain(modality = tr_domain, opt = opt)
        else:
            train_set       = cardiac_cls.get_training(modality = tr_domain , opt = opt)
        tr_valset  = cardiac_cls.get_trval(modality = tr_domain , opt = opt)
        tr_teset  = cardiac_cls.get_trtest(modality = tr_domain , opt = opt)#as dataset split,cardiac didn't have this
        test_set        = cardiac_cls.get_test(modality = te_domain , opt = opt)
        
        label_name      = cardiac_cls.LABEL_NAME

    else:
        print('not implement this dataset',opt.data_name)

    # Auto-adjust prefetch_factor for the always-on GIP/CLP augmentation path.
    effective_prefetch_factor = opt.prefetch_factor if opt.num_workers > 0 else None
    if effective_prefetch_factor is not None and effective_prefetch_factor > 2:
        print(f"⚠️  WARNING: LocationScaleAugmentation detected with prefetch_factor={effective_prefetch_factor}")
        print(f"   Reducing to prefetch_factor=2 to prevent deadlock with GIP/CLP augmentation")
        effective_prefetch_factor = 2

    source_set_for_loader = train_set
    if smppm_use_plain_source_set and opt.data_name == 'PROSTATE' and hasattr(PROS, 'get_training_plain'):
        source_set_for_loader = PROS.get_training_plain(modality=tr_domain, opt=opt)
    elif smppm_use_plain_source_set and opt.data_name == 'PROSTATE':
        print(
            "SM-PPM plain source loader requested but unavailable for PROSTATE; "
            "falling back to the augmented training loader."
        )

    source_train_batch_size = opt.batchSize
    if opt.phase == 'test' and opt.tta == 'asm':
        source_train_batch_size = opt.asm_src_batch_size
        print(
            "ASM source loader: using labeled source-domain training split "
            f"with batch_size={source_train_batch_size}, shuffle=True, drop_last=True. "
            "Target labels are not used for adaptation."
        )
    elif smppm_requires_source_loader:
        source_train_batch_size = opt.smppm_src_batch_size
        print(
            "SM-PPM source loader: using labeled source-domain training split "
            f"with batch_size={source_train_batch_size}, shuffle=True, drop_last=True. "
            f"ablation_mode={opt.smppm_ablation_mode}. "
            f"plain_source_loader={opt.smppm_plain_source_loader}. "
            "Target labels are not used for adaptation."
        )
    elif smppm_source_free_proto:
        print(
            "SM-PPM source_free_proto: no source_loader will be passed to the "
            "adapter; no source labels are used for adaptation."
        )
    elif opt.phase == 'test' and opt.tta == 'gtta':
        source_train_batch_size = opt.gtta_src_batch_size
        print(
            "GTTA source loader: using labeled source-domain training split "
            f"with batch_size={source_train_batch_size}, shuffle=True, drop_last=True. "
            "Target labels are not used for adaptation."
        )

    needs_train_loader = (
        opt.phase != 'test'
        or opt.tta in ['asm', 'gtta']
        or smppm_requires_source_loader
    )
    if needs_train_loader:
        train_loader_dataset = source_set_for_loader if smppm_requires_source_loader else train_set
        train_loader_kwargs = {
            "dataset": train_loader_dataset,
            "num_workers": opt.num_workers,
            "batch_size": source_train_batch_size,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
        }
        if opt.num_workers > 0:
            train_loader_kwargs["worker_init_fn"] = worker_init_fn
            train_loader_kwargs["prefetch_factor"] = effective_prefetch_factor
            train_loader_kwargs["persistent_workers"] = True
        train_loader = DataLoader(**train_loader_kwargs)
    else:
        train_loader = None

    eval_loader_kwargs = {
        "num_workers": opt.num_workers,
        "batch_size": 1,
        "shuffle": False,
        "pin_memory": True,
    }
    if opt.num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = effective_prefetch_factor
        eval_loader_kwargs["persistent_workers"] = True
    trval_loader=DataLoader(dataset = tr_valset, **eval_loader_kwargs)
    trte_loader=DataLoader(dataset = tr_teset, **eval_loader_kwargs)
    test_loader = DataLoader(dataset = test_set, **eval_loader_kwargs)

    # ========== TEST MODE ==========
    if opt.phase == 'test':
        print(f"\n{'='*80}")
        print(f"TEST MODE")
        print(f"TTA: {opt.tta}")
        if opt.tta == 'tent':
            print(f"TENT config: lr={opt.tent_lr}, steps={opt.tent_steps}")
        elif opt.tta == 'dg_tta':
            print(
                "DG-TTA config: MedSeg-TTA output-consistency adapter; "
                f"lr={opt.dgtta_lr}, steps={opt.dgtta_steps}, "
                f"strength={opt.dgtta_transform_strength}, entropy_weight={opt.dgtta_entropy_weight}, "
                f"bn_l2_reg={opt.dgtta_bn_l2_reg}, episodic={opt.dgtta_episodic}"
            )
        elif opt.tta == 'norm_alpha':
            print(f"norm_alpha config: alpha={opt.bn_alpha}")
        elif opt.tta == 'cotta':
            print(f"CoTTA config: lr={opt.cotta_lr}, steps={opt.cotta_steps}, mt={opt.cotta_mt}, rst={opt.cotta_rst}, ap={opt.cotta_ap}")
        elif opt.tta == 'memo':
            print(
                "MEMO config: "
                f"lr={opt.memo_lr}, steps={opt.memo_steps}, "
                f"n_aug={opt.memo_n_augmentations}, "
                f"include_identity={opt.memo_include_identity}, "
                f"hflip_p={opt.memo_hflip_p}, "
                f"scope={opt.memo_update_scope}"
            )
        elif opt.tta == 'asm':
            print(
                "ASM config: source-dependent supervised TTA; "
                "target images provide style/statistics only, target labels are evaluation-only. "
                f"lr={opt.asm_lr}, steps={opt.asm_steps}, inner_steps={opt.asm_inner_steps}, "
                f"lambda_reg={opt.asm_lambda_reg}, sampling_step={opt.asm_sampling_step}, "
                f"src_batch_size={opt.asm_src_batch_size}, style_backend={opt.asm_style_backend}, "
                f"episodic={opt.asm_episodic}"
            )
        elif opt.tta == 'sm_ppm':
            print(
                "SM-PPM config: "
                f"ablation_mode={opt.smppm_ablation_mode}, "
                "target labels are evaluation-only. "
                f"lr={opt.smppm_lr}, momentum={opt.smppm_momentum}, wd={opt.smppm_wd}, "
                f"steps={opt.smppm_steps}, src_batch_size={opt.smppm_src_batch_size}, "
                f"patch_size={opt.smppm_patch_size}, feature_size={opt.smppm_feature_size}, "
                f"episodic={opt.smppm_episodic}, "
                f"source_free_tau={opt.smppm_source_free_tau}, "
                f"source_free_entropy_threshold={opt.smppm_source_free_entropy_threshold}, "
                f"source_free_entropy_weight={opt.smppm_source_free_entropy_weight}, "
                f"source_free_lambda_proto={opt.smppm_source_free_lambda_proto}, "
                f"plain_source_loader={opt.smppm_plain_source_loader}"
            )
        elif opt.tta == 'gtta':
            print(
                "GTTA config: source-dependent supervised TTA with medical class-aware AdaIN; "
                "target labels are evaluation-only. "
                f"lr={opt.gtta_lr}, momentum={opt.gtta_momentum}, wd={opt.gtta_wd}, "
                f"steps={opt.gtta_steps}, src_batch_size={opt.gtta_src_batch_size}, "
                f"lambda_ce_trg={opt.gtta_lambda_ce_trg}, pseudo_momentum={opt.gtta_pseudo_momentum}, "
                f"style_alpha={opt.gtta_style_alpha}, include_original={opt.gtta_include_original}, "
                f"episodic={opt.gtta_episodic}"
            )
        elif opt.tta == 'gold':
            print(
                "GOLD config: source-free TTA; target labels are evaluation-only. "
                f"lr={opt.gold_lr}, steps={opt.gold_steps}, rank={opt.gold_rank}, "
                f"tau={opt.gold_tau}, alpha={opt.gold_alpha}, t_eig={opt.gold_t_eig}, "
                f"mt={opt.gold_mt}, s_lr={opt.gold_s_lr}, adapter_scale={opt.gold_adapter_scale}, "
                f"rst={opt.gold_rst}, ap={opt.gold_ap}, episodic={opt.gold_episodic}"
            )
        elif opt.tta == 'vptta':
            print(
                "VPTTA config: source-free frequency prompt TTA; "
                "target labels are evaluation-only. "
                f"optimizer={opt.vptta_optimizer}, lr={opt.vptta_lr}, "
                f"steps={opt.vptta_steps}, memory_size={opt.vptta_memory_size}, "
                f"neighbor={opt.vptta_neighbor}, prompt_alpha={opt.vptta_prompt_alpha}, "
                f"prompt_size={opt.vptta_prompt_size}, warm_n={opt.vptta_warm_n}"
            )
        elif opt.tta == 'pass':
            print(
                "PASS config: source-free style/shape prompt TTA; "
                "target labels are evaluation-only. "
                f"optimizer={opt.pass_optimizer}, lr={opt.pass_lr}, steps={opt.pass_steps}, "
                f"bn_alpha={opt.pass_bn_alpha}, bn_layers={opt.pass_bn_layers}, "
                f"entropy_weight={opt.pass_entropy_weight}, ema_decay={opt.pass_ema_decay}, "
                f"source_fallback={opt.pass_use_source_fallback}, episodic={opt.pass_episodic}, "
                f"prompt_size={opt.pass_prompt_size}"
            )
        elif opt.tta == 'samtta':
            print(
                "SAM-TTA config: source-free Bezier input transform plus EMA teacher-student TTA; "
                "target labels are evaluation-only. "
                f"lr={opt.samtta_lr}, transform_lr={opt.samtta_transform_lr}, "
                f"steps={opt.samtta_steps}, ema={opt.samtta_ema_momentum}, "
                f"dpc_w={opt.samtta_dpc_weight}, feature_w={opt.samtta_feature_weight}, "
                f"entropy_w={opt.samtta_entropy_weight}, "
                f"transform_reg_w={opt.samtta_transform_reg_weight}, "
                f"scope={opt.samtta_update_scope}, episodic={opt.samtta_episodic}"
            )
        elif opt.tta == 'spmo':
            print(
                "SPMO config: source-free shape-moment TTA; "
                "a frozen source model supplies per-slice size/moment priors, target labels are evaluation-only. "
                f"lr={opt.spmo_lr}, steps={opt.spmo_steps}, "
                f"entropy_w={opt.spmo_entropy_weight}, prior_w={opt.spmo_prior_weight}, "
                f"moment_w={opt.spmo_moment_weight}, moment_mode={opt.spmo_moment_mode}, "
                f"source_pseudo={opt.spmo_source_pseudo}, scope={opt.spmo_update_scope}, "
                f"episodic={opt.spmo_episodic}"
            )
        elif opt.tta == 'sictta':
            print(
                "SicTTA config: source-free single-image continual TTA; "
                "target labels are evaluation-only. "
                f"max_lens={opt.sictta_max_lens}, topk={opt.sictta_topk}, "
                f"threshold={opt.sictta_threshold}, select_points={opt.sictta_select_points}, "
                f"episodic={opt.sictta_episodic}"
            )
        elif opt.tta == 'a3_tta':
            print(
                "A3-TTA config: source-free online anchor alignment TTA; "
                "target labels are evaluation-only. "
                f"lr={opt.a3_lr}, steps={opt.a3_steps}, pool_size={opt.a3_pool_size}, "
                f"top_k={opt.a3_top_k}, mt={opt.a3_mt}, "
                f"feature_w={opt.a3_feature_loss_weight}, "
                f"entropy_w={opt.a3_entropy_match_weight}, "
                f"ema_w={opt.a3_ema_loss_weight}, "
                f"episodic={opt.a3_episodic}, reset_on_scan_start={opt.a3_reset_on_scan_start}"
            )
        print(f"{'='*80}\n")

        # Determine checkpoint path
        if opt.resume_path is not None:
            reload_model_fid = opt.resume_path
        elif opt.resume_epoch is not None:
            reload_model_fid = os.path.join(snap_dir, f'{opt.resume_epoch}_net_Seg.pth')
        else:
            # Find latest checkpoint
            ckpt_files = glob.glob(os.path.join(snap_dir, '*_net_Seg.pth'))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint found in {snap_dir}")
            reload_model_fid = max(ckpt_files, key=os.path.getctime)

        print(f"Loading checkpoint: {reload_model_fid}")

        if not os.path.exists(reload_model_fid):
            raise FileNotFoundError(f"Checkpoint not found: {reload_model_fid}")

        # Instantiate trainer
        source_dependent_loader = train_loader if (
            opt.tta in ['asm', 'gtta'] or smppm_requires_source_loader
        ) else None
        model = Train_process(opt, reloaddir=reload_model_fid, istest=1, source_loader=source_dependent_loader)

        with torch.no_grad():
            print("\n" + "="*80)
            print("Testing on target domain...")
            print("="*80)
            with open(finalfile, 'a') as f:
                f.write("Test mode evaluation\n")

            type1='testfinal'
            preds = prediction_wrapper(tb_writer, type1, logdir, model, test_loader, 0, label_name, save_prediction=opt.save_prediction)

            if opt.save_prediction:
                pred_dir = os.path.join(logdir, 'pred')
                os.makedirs(pred_dir, exist_ok=True)
                xx = 0
                for scan_id, comp in preds.items():
                    _pred = comp['pred'].detach().data.cpu().numpy()
                    _img = comp['img'].detach().data.cpu().numpy()
                    _gt = comp['gth'].detach().data.cpu().numpy()
                    for yy in range(_pred.shape[0]):
                        if len(np.unique(_gt[yy])) > 1:
                            save_teimgs(_pred[yy], _img[yy], _gt[yy], pred_dir, xx, opt, labmap)
                            xx += 1
                    if xx >= 50:
                        break

            if opt.eval_source_domain:
                print("\n" + "="*80)
                print("Testing on source domain...")
                print("="*80)
                if opt.tta != 'none':
                    print(f"Reloading checkpoint before source-domain evaluation to avoid carrying target {opt.tta} state.")
                    model = Train_process(opt, reloaddir=reload_model_fid, istest=1, source_loader=source_dependent_loader)
                with open(finalfile, 'a') as f:
                    f.write("\n\nTest on source domain\n")
                type1 = 'tetrainfinal'
                tmp = prediction_wrapper(tb_writer, type1, logdir, model, trte_loader, 0, label_name, save_prediction=opt.save_prediction)
            else:
                print("\nSkipping source-domain evaluation (--eval_source_domain false).")

        print(f"\nTest completed! Results saved in: {logdir}")
        exit(0)
    # ========== END TEST MODE ==========

    model = Train_process(opt,'0',0)

    total_steps,iternum = 0,0
    best_val = float('-inf')
    best_epoch = -1

    for epoch in range(1, opt.all_epoch + 1):
        epoch_start_time = time.time()

        # Training progress bar for each epoch
        train_pbar = tqdm(enumerate(train_loader), total = train_loader.dataset.size // opt.batchSize - 1,
                         desc=f'Epoch {epoch}/{opt.all_epoch}')

        for i, train_batch in train_pbar:
                total_steps=total_steps+opt.batchSize
                iternum=iternum+1
                # avoid batchsize issues caused by fetching last training batch
                if train_batch["label"].shape[0] != opt.batchSize:
                    continue

                tr_log=model.tr_func(train_batch,epoch)

                for key, x in tr_log.items():
                    tb_writer.add_scalar(key, x,iternum)

                    with open(osp.join(logdir, 'train', key+'.csv'), 'a') as f:
                        log1=[[iternum] + [x.item()]]
                        log1=map(str, log1)
                        f.write(','.join(log1) + '\n')

                # Update progress bar with paper-mainline losses only.
                total_loss = model.loss_all.item()
                if opt.use_cgsd and getattr(model, 'optimizer_cgsd', None) is not None:
                    total_loss += model.loss_cgsd.item()

                postfix_dict = {
                    'loss': f'{total_loss:.4f}',
                    'dice': f'{model.loss_dice.item():.4f}',
                    'lr': f'{model.get_lr().item():.5f}'
                }
                if hasattr(opt, 'use_saam') and opt.use_saam:
                    postfix_dict['saam'] = f'{model.loss_saam.item():.4f}'
                    # Display key SAAM statistics in the progress bar.
                    if hasattr(model, 'saam_stats') and model.saam_stats:
                        if 'effective_pixels' in model.saam_stats:
                            # Show float with 1 decimal to avoid truncating tiny values to 0
                            postfix_dict['eff_pix'] = f'{model.saam_stats["effective_pixels"]:.1f}'
                        if 'topk_selected_ratio' in model.saam_stats:
                            postfix_dict['topk%'] = f'{model.saam_stats["topk_selected_ratio"]*100:.1f}'
                if opt.use_cgsd:
                    postfix_dict['str'] = f'{model.loss_str.item():.4f}'
                train_pbar.set_postfix(postfix_dict)

                # Log detailed info to file only
                log_str = "Tr-Epoch:{},Iter:{},Lr:{:.5f}--loss:{:.5f} seg:{:.5f} dc:{:.5f} ce:{:.5f} ".format(
                    epoch, iternum, model.get_lr().item(), total_loss, model.loss_seg.item(),
                    model.loss_dice.item(), model.loss_ce.item())
                log_str += "segW:{:.5f} segV2:{:.5f} ".format(model.loss_seg1.item(), model.loss_seg2.item())
                if hasattr(model, 'loss_seg0'):
                    log_str += "seg0:{:.5f} ".format(model.loss_seg0.item())
                if opt.use_cgsd:
                    log_str += "str:{:.5f} sty:{:.5f} cgsd:{:.5f} ".format(
                        model.loss_str.item(), model.loss_sty.item(), model.loss_cgsd.item())
                if hasattr(opt, 'use_saam') and opt.use_saam:
                    log_str += "saam:{:.5f} saam_01:{:.5f} saam_02:{:.5f} ".format(
                        model.loss_saam.item(), model.loss_saam_01.item(), model.loss_saam_02.item())
                    if hasattr(model, 'saam_stats') and model.saam_stats:
                        stats = model.saam_stats
                        log_str += "topk:{:.2%} eff_pix:{:.1f} ".format(
                            stats.get('topk_selected_ratio', 0), stats.get('effective_pixels', 0))
                if getattr(model, 'rccs_applied', False):
                    log_str += "rccs:1 "
                logger.info(log_str)
                
    
                if iternum % opt.display_freq == 0:
                    tr_viz = model.get_img_tr()
                    save_trimgs(tr_viz,logdir,iternum,opt,labmap)
                   
     
 
        #val for tr_val
        if epoch % opt.validation_freq == 0:
            with torch.no_grad():
                type1='val'
                tmp= prediction_wrapper(tb_writer,type1,logdir,model, trval_loader,  epoch, label_name, save_prediction =False)
                if len(val_metric) > 0:
                    try:
                        curr_val = float(val_metric[-1][1])
                        if curr_val > best_val:
                            best_val = curr_val
                            best_epoch = epoch
                            print(f'New best val ({best_val:.4f}) at epoch {best_epoch}, saving best checkpoint.')
                            logger.info(f'Best val updated: epoch {best_epoch}, score {best_val:.4f}')
                            # Save canonical best checkpoint
                            model.save(snap_dir,'best')
                            # Also save with epoch in filename for traceability
                            model.save(snap_dir,f'best_ep{best_epoch}')
                    except Exception as e:
                        print(f'Warning: failed to update best val checkpoint due to {e}')


        # Save model periodically and at the end
        if epoch % opt.save_freq == 0 or epoch == opt.all_epoch:
            print('Saving model at epoch %d, iters %d' %(epoch, iternum))
            model.save(snap_dir,epoch)

        lr = opt.lr * (1 - epoch /opt.all_epoch )
        model.optimizer_seg.param_groups[0]['lr']=lr
        if getattr(model, 'optimizer_cgsd', None) is not None:
            cgsd_base_lr = opt.cgsd_lr if opt.cgsd_lr is not None else opt.lr
            model.optimizer_cgsd.param_groups[0]['lr'] = cgsd_base_lr * (1 - epoch / opt.all_epoch)

        # Monitor ChannelGate distribution when CGSD is enabled.
        if epoch % 10 == 0 and opt.use_cgsd:
            if hasattr(model.netseg, 'chan_gate'):
                # Compute gate statistics correctly for both sigmoid and softmax modes.
                logits = model.netseg.chan_gate.logits.detach().cpu()
                use_temperature = getattr(opt, 'use_temperature', 0)

                if use_temperature:
                    # Softmax mode: logits shape = [2, C, 1, 1]
                    gate_tau = getattr(opt, 'gate_tau', 0.1)
                    weights = torch.softmax(logits / gate_tau, dim=0)
                    # Track only the structure-channel weights (row 0).
                    m_values = weights[0]  # shape = [C, 1, 1]
                else:
                    # Sigmoid mode: logits shape = [1, C, 1, 1]
                    m_values = torch.sigmoid(logits)  # shape = [1, C, 1, 1]

                m_mean = m_values.mean().item()
                m_std = m_values.std().item()
                m_min = m_values.min().item()
                m_max = m_values.max().item()

                # Print to console
                print(f"\n[ChannelGate Stats] mean={m_mean:.3f}, std={m_std:.3f}, min={m_min:.3f}, max={m_max:.3f}")

                # Log to file
                logger.info(f"ChannelGate - Epoch {epoch}: mean={m_mean:.3f}, std={m_std:.3f}, min={m_min:.3f}, max={m_max:.3f}")

                # Log to TensorBoard
                tb_writer.add_scalar('ChannelGate/mean', m_mean, epoch)
                tb_writer.add_scalar('ChannelGate/std', m_std, epoch)
                tb_writer.add_scalar('ChannelGate/min', m_min, epoch)
                tb_writer.add_scalar('ChannelGate/max', m_max, epoch)

                # Health check warnings
                if m_mean > 0.9:
                    print(f"  ⚠️  WARNING: Gate mean > 0.9 - possible degeneration (all channels assigned to structure)")
                    logger.warning(f"ChannelGate degeneration: mean={m_mean:.3f} (threshold: 0.9)")
                elif m_mean < 0.1:
                    print(f"  ⚠️  WARNING: Gate mean < 0.1 - possible degeneration (all channels assigned to style)")
                    logger.warning(f"ChannelGate degeneration: mean={m_mean:.3f} (threshold: 0.1)")

                if m_std < 0.1:
                    print(f"  ⚠️  WARNING: Gate std < 0.1 - low diversity (all channels similar)")
                    logger.warning(f"ChannelGate low diversity: std={m_std:.3f} (threshold: 0.1)")
                else:
                    print(f"  ✓ Gate distribution healthy")

        print(f'\nEpoch {epoch}/{opt.all_epoch} completed in {int(time.time() - epoch_start_time)}s')

    # Save the latest model after training completes
    print(f'\nTraining completed. Saving final model...')
    model.save(snap_dir, 'latest')
    print(f'Latest model saved at {snap_dir}/latest_net_Seg.pth')

    if(total_steps>=0):
        print('final test epoch %d, iters %d' %(opt.all_epoch, total_steps))
        reload_model_fid = os.path.join(snap_dir, f'{opt.all_epoch}_net_Seg.pth')
        print("reload_model_fid:",reload_model_fid)
        source_dependent_loader = train_loader if opt.tta in ['asm', 'sm_ppm', 'gtta'] else None
        model1=Train_process(opt,reloaddir=reload_model_fid,istest=1, source_loader=source_dependent_loader)

        with torch.no_grad():
            with open(finalfile, 'a') as f:
                f.write("test testset final\n")
            type1='testfinal'
            preds= prediction_wrapper(tb_writer,type1,logdir,model1, test_loader,opt.all_epoch, label_name, save_prediction = opt.save_prediction)

            if opt.save_prediction==True:
                pred_dir = os.path.join(logdir, 'pred')
                os.makedirs(pred_dir, exist_ok=True)
                xx=0
                for scan_id, comp in preds.items():
                    _pred = comp['pred'].detach().data.cpu().numpy()
                    _img = comp['img'].detach().data.cpu().numpy()
                    _gt = comp['gth'].detach().data.cpu().numpy()
                    for yy in range(_pred.shape[0]):
                       if len(np.unique(_gt[yy]))>1:
                           save_teimgs(_pred[yy],_img[yy],_gt[yy],pred_dir,xx,opt,labmap)
                           xx=xx+1

                    if xx>=50:
                        break
                    
            print('\ntest for source domain')
            if opt.tta != 'none':
                print(f"Reloading checkpoint before source-domain evaluation to avoid carrying target {opt.tta} state.")
                model1 = Train_process(opt, reloaddir=reload_model_fid, istest=1, source_loader=source_dependent_loader)
            with open(finalfile, 'a') as f:
                 f.write("\n\ntest for source domain \n")          
            type1='tetrainfinal'  
            tmp= prediction_wrapper(tb_writer,type1,logdir,model1, trte_loader, opt.all_epoch, label_name, save_prediction = opt.save_prediction)
          
