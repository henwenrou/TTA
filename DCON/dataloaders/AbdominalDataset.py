
import glob
import numpy as np
import dataloaders.niftiio as nio
import dataloaders.transform_utils as trans
import torch
import os
import torch.utils.data as torch_data
import math
import itertools
from dataloaders.niftiio import read_nii_bysitk
from dataloaders.location_scale_augmentation import LocationScaleAugmentation

DATA_ROOT = os.environ.get(
    'SAA_DATA_ROOT',
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
)
BASEDIR = os.path.join(DATA_ROOT, 'abdominal')
LABEL_NAME = ["bg", "liver", "rk", "lk", "spleen"]


class NormalizeOp:
    """Normalization operation with accessible mean and std"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x_in):
        return (x_in - self.mean) / self.std

def get_normalize_op(modality, fids):# modality:   CT or MR,fids for the fold
    def get_CT_statistics(scan_fids):
        total_val = 0
        n_pix = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_val += in_img.sum()
            n_pix += np.prod(in_img.shape)
            del in_img
        meanval = total_val / n_pix

        total_var = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2 )
            del in_img
        var_all = total_var / n_pix

        global_std = var_all ** 0.5

        return meanval, global_std



    ct_mean, ct_std = get_CT_statistics(fids)

    return NormalizeOp(ct_mean, ct_std)



class AbdominalDataset(torch_data.Dataset):
    def __init__(self,  mode, transforms, base_dir, domains: list,  idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3, extern_norm_fn = None, opt=None):


        super(AbdominalDataset, self).__init__()
        self.transforms=transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.domains = domains
        self.all_label_names = LABEL_NAME
        self.nclass = len(LABEL_NAME)
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.idx_pct = idx_pct
        self.opt=opt
        self.use_sgf = bool(getattr(self.opt, 'use_sgf', 0))
        self.sgf_view2_only = bool(getattr(self.opt, 'sgf_view2_only', 0)) and self.use_sgf
        self.sgf_intensity_tfx = trans.get_intensity_transformer(trans.tr_aug) if self.use_sgf else None

        if self.is_train:
            print(f'Applying GIP/CLP location-scale augmentation on {mode} split')
            self.location_scale = LocationScaleAugmentation(vrange=(0.,1.), background_threshold=0.01)
        else:
            self.location_scale = None

        self.img_pids = {}
        for _domain in self.domains: 
            self.img_pids[_domain] = sorted([ fid.split("_")[-1].split(".nii.gz")[0] for fid in glob.glob(self._base_dir + "/" +  _domain  + "/processed/image_*.nii.gz") ], key = lambda x: int(x))

        self.scan_ids = self.__get_scanids(idx_pct) 
        self.info_by_scan = None
        self.sample_list = self.__search_samples(self.scan_ids) 

        self.pid_curr_load = self.scan_ids
       
        if extern_norm_fn is None:
            self.normalize_op = get_normalize_op(self.domains[0], [ itm['img_fid'] for _, itm in self.sample_list[self.domains[0]].items() ])
            print(f'{self.phase}_{self.domains[0]}: Using fold data statistics for normalization')
        else:
           
            self.normalize_op = extern_norm_fn

        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using scan ids {self.pid_curr_load}')

       
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset) 

    def __get_scanids(self, idx_pct):

        tr_trids,tr_valids,tr_teids,te_teids={},{},{},{}

        for _domain in self.domains:
            dset_size   = len(self.img_pids[_domain])
            tr_size     = round(dset_size * idx_pct[0])
            val_size    = math.floor(dset_size * idx_pct[1])
            te_size     = dset_size - tr_size - val_size

            tr_teids[_domain]     = self.img_pids[_domain][: te_size]
            tr_valids[_domain]    = self.img_pids[_domain][te_size: te_size + val_size]
            tr_trids[_domain]     = self.img_pids[_domain][te_size + val_size: ]
            te_teids[_domain] = list(itertools.chain(tr_trids[_domain], tr_teids[_domain], tr_valids[_domain]))

        if self.phase == 'train':
            return tr_trids
        elif self.phase == 'trainsup':
            return tr_trids
        elif self.phase == 'trval':
            return tr_valids
        elif self.phase == 'trtest':
            return tr_teids
        elif self.phase == 'test':
            return te_teids
        elif self.phase=='testsup':
            return tr_teids
      

    def __search_samples(self, scan_ids):
        """search for filenames for images and masks
        """
        out_list = {}
        for _domain, id_list in scan_ids.items():
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}

                _img_fid = os.path.join(self._base_dir, _domain , 'processed'  ,f'image_{curr_id}.nii.gz')
                _lb_fid  = os.path.join(self._base_dir, _domain , 'processed', f'label_{curr_id}.nii.gz')

                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                out_list[_domain][str(curr_id)] = curr_dict

        return out_list


    def __read_dataset(self):
        """
        Read the dataset into memory
        """

        out_list = []
        self.info_by_scan = {} 
        glb_idx = 0 

        for _domain, _sample_list in self.sample_list.items():
            for scan_id, itm in _sample_list.items():
                if scan_id not in self.pid_curr_load[_domain]:
                    continue

                img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info = True)
                self.info_by_scan[_domain + '_' + scan_id] = _info

                img = np.float32(img)
                vol_mean = self.normalize_op.mean if self.normalize_op.mean is not None else 0.0
                vol_std = self.normalize_op.std if self.normalize_op.std not in [None, 0] else 1.0
                vol_info = {
                    'vol_vmin': img.min(),
                    'vol_vmax': img.max(),
                    'vol_mean': vol_mean,
                    'vol_std': vol_std
                }
                img = self.normalize_op(img)

                lb = nio.read_nii_bysitk(itm["lbs_fid"])
                lb = np.float32(lb)

                img     = np.transpose(img, (1,2,0))
                lb      = np.transpose(lb, (1,2,0))

                assert img.shape[-1] == lb.shape[-1]


                out_list.append( {"img": img[..., 0: 1],
                               "lb":lb[..., 0: 0 + 1],
                               "is_start": True,
                               "is_end": False,
                               "domain": _domain,
                               "nframe": img.shape[-1],
                               "scan_id": _domain + "_" + scan_id,
                               "z_id":0,
                               "vol_info": vol_info})
                glb_idx += 1

                for ii in range(1, img.shape[-1] - 1):
                    out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii + 1],
                               "is_start": False,
                               "is_end": False,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii,
                               "vol_info": vol_info
                               })
                    glb_idx += 1

                ii += 1
                out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii+ 1],
                               "is_start": False,
                               "is_end": True,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii,
                               "vol_info": vol_info
                               })
                glb_idx += 1

        return out_list
    

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]

        is_start    = curr_dict["is_start"]
        is_end      = curr_dict["is_end"]
        nframe      = np.int32(curr_dict["nframe"])
        scan_id     = curr_dict["scan_id"]
        z_id        = curr_dict["z_id"]

        sample = {"is_start": is_start,"is_end": is_end,"nframe": nframe,"scan_id": scan_id,"z_id": z_id}

        if self.phase!='train':
            img = curr_dict['img']
            lb = curr_dict['lb']
            imgori=curr_dict['img']
          
            img = np.float32(img)
            lb = np.float32(lb)
            imgori= np.float32(imgori)
    
            img = np.transpose(img, (2, 0, 1))
            lb  = np.transpose(lb, (2, 0, 1))
            imgori= np.transpose(imgori, (2, 0, 1))

            img = torch.from_numpy( img )
            lb  = torch.from_numpy( lb )
            imgori= torch.from_numpy( imgori )

            if self.tile_z_dim > 1:
               img = img.repeat( [ self.tile_z_dim, 1, 1] )
               imgori = imgori.repeat( [ self.tile_z_dim, 1, 1] )
        

            sample['base_view'] = img
            sample['label'] = lb
            sample['anchor_view'] = imgori
        
        else:
            imgori = curr_dict['img']
            img = curr_dict["img"].copy()
            lb_data = curr_dict["lb"].copy()

            img_denorm = np.clip(self.denorm_(img, curr_dict['vol_info']), 0.0, 1.0)

            gip = self.location_scale.Global_Location_Scale_Augmentation(img_denorm.copy())
            gip = np.clip(gip, 0.0, 1.0)
            gip = self.renorm_(gip, curr_dict['vol_info'])

            clp = self.location_scale.Local_Location_Scale_Augmentation(img_denorm.copy(), lb_data.astype(np.int32))
            clp = np.clip(clp, 0.0, 1.0)
            clp = self.renorm_(clp, curr_dict['vol_info'])

            comp = np.concatenate([curr_dict["img"], gip, clp, curr_dict["lb"]], axis=-1)
            timg, lb = self.transforms(comp, c_img=3, c_label=1, nclass=self.nclass, is_train=self.is_train, use_onehot=False)
            _, gip_geo, clp_geo = np.split(timg, 3, axis=-1)

            if self.sgf_intensity_tfx is not None:
                gip_geo = self.sgf_intensity_tfx(gip_geo)
                clp_geo = self.sgf_intensity_tfx(clp_geo)

            base_view = np.float32(gip_geo)
            strong_view = np.float32(clp_geo)
            gip_geo = np.float32(gip_geo)
            clp_geo = np.float32(clp_geo)
            imgori = np.float32(imgori)
            lb = np.float32(lb)
            lb = np.clip(lb, 0, self.nclass - 1)

            base_view = np.transpose(base_view, (2, 0, 1))
            strong_view = np.transpose(strong_view, (2, 0, 1))
            gip_geo = np.transpose(gip_geo, (2, 0, 1))
            clp_geo = np.transpose(clp_geo, (2, 0, 1))
            imgori = np.transpose(imgori, (2, 0, 1))
            lb = np.transpose(lb, (2, 0, 1))

            base_view = torch.from_numpy(base_view)
            strong_view = torch.from_numpy(strong_view)
            gip_geo = torch.from_numpy(gip_geo)
            clp_geo = torch.from_numpy(clp_geo)
            imgori = torch.from_numpy(imgori)
            lb = torch.from_numpy(lb)

            if self.tile_z_dim > 1:
                base_view = base_view.repeat([self.tile_z_dim, 1, 1])
                strong_view = strong_view.repeat([self.tile_z_dim, 1, 1])
                gip_geo = gip_geo.repeat([self.tile_z_dim, 1, 1])
                clp_geo = clp_geo.repeat([self.tile_z_dim, 1, 1])
                imgori = imgori.repeat([self.tile_z_dim, 1, 1])

            sample['base_view'] = base_view
            sample['strong_view'] = strong_view
            sample['label'] = lb
            sample['anchor_view'] = imgori


        return sample

    def __len__(self):
        return len(self.actual_dataset)

    def denorm_(self, img, vol_info):
        """Denormalize image from z-score to 0-1 range"""
        vmin, vmax, vmean, vstd = vol_info['vol_vmin'], vol_info['vol_vmax'], vol_info['vol_mean'], vol_info['vol_std']
        # Prevent division by zero
        range_val = vmax - vmin
        if range_val == 0:
            range_val = 1.0
        return ((img * vstd + vmean) - vmin) / range_val

    def renorm_(self, img, vol_info):
        """Renormalize image from 0-1 range back to z-score"""
        vmin, vmax, vmean, vstd = vol_info['vol_vmin'], vol_info['vol_vmax'], vol_info['vol_mean'], vol_info['vol_std']
        # Prevent division by zero
        if vstd == 0:
            vstd = 1.0
        return ((img * (vmax - vmin) + vmin) - vmean) / vstd

def get_training(modality,norm_func, opt):
    print("get_train abd:",modality)

    tr_func  = trans.transform_with_label(trans.pre_aug)
    # Mainline training uses tile_z_dim=3 to convert 1-channel slices to 3-channel input.
    return AbdominalDataset(mode = 'train',transforms = tr_func,domains = modality,base_dir = BASEDIR,extern_norm_fn =norm_func,opt=opt, tile_z_dim=3)

def get_training_plain(modality, norm_func, opt):
    print("get_train_plain abd:",modality)
    return AbdominalDataset(mode = 'trainsup',transforms = None,domains = modality,base_dir = BASEDIR,extern_norm_fn = norm_func,opt=opt, tile_z_dim=3)

def get_trval(modality, norm_func, opt):
    print("get_trval abd:",modality)
    return AbdominalDataset(mode = 'trval',transforms = None,domains = modality,base_dir = BASEDIR,extern_norm_fn = norm_func,opt=opt, tile_z_dim=3)

def get_trtest(modality, norm_func, opt):
    print("get_trtest abd:",modality)
    return AbdominalDataset(mode = 'trtest',transforms = None,domains = modality,base_dir = BASEDIR,extern_norm_fn = norm_func,opt=opt, tile_z_dim=3)

def get_test(modality, norm_func,opt):
     print("get_test abd:",modality)
     return AbdominalDataset(mode = 'test',transforms = None,domains = modality,base_dir = BASEDIR,extern_norm_fn = norm_func,opt=opt, tile_z_dim=3)
