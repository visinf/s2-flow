import functools
import os

import cv2
import numpy as np
import torch
import torchvision
from util.utils import tensor2label

pil2tensor = torchvision.transforms.functional.to_tensor
background_idx = 0
hair_idx = 13
skin_idx = 1
ulip_idx = 11
llip_idx = 12
mouth_idx = 10
nose_idx = 2
lbrow = 6
rbrow = 7
eye_g_idx = 3
l_eye_idx = 4
r_eye_idx = 5

import lmdb
import pyarrow as pa


class MyDatasetLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, dataset, latents):
        self.db_path = db_path
        self.env = None
        self.latents = latents
        self.length = len(latents)

    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                                readonly=True, lock=False,
                                readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            # self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    @staticmethod
    def desearilize(lmdb_value):
        return pa.deserialize((lmdb_value))

    def get_item(self, index):
        if self.env is None:
            self._init_db()

        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        imgid, img_arr, gray_sm_arr, color_sm_arr, img_shape = pa.deserialize(byteflow)
        gray_sm = np.frombuffer(gray_sm_arr, dtype=np.uint8).reshape((512, 512)).copy()
        # gray_sm = torch.as_tensor(gray_sm, dtype=torch.int64)
        return np.asarray(gray_sm)

    def __getitem__(self, index):
        # x = torch.tensor(self.latents[index])
        gray_sm = self.get_item(index)
        return gray_sm 

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


# Helper
class AugmentDataset:
    def __init__(self, nbrs, smiling_nbrs, gender_attrs, glass_attrs, cls="easy"):
        self.nbrs = nbrs
        self.cls = cls
        self.smiling_nbrs = smiling_nbrs 
        self.gender_attrs = gender_attrs
        self.glass_attrs = glass_attrs
        self.edit_idx = {-1: "None", 0: "remove_glasses", 1: "add_glasses_reading", 2: "swp_eyebrows", 3: "swp_nose", 4: "swp_mouth", 5: "swp_hair"}
        sg_latents = np.load("./data/all_latents.pickle", allow_pickle=True)['Latent'][:10000]
        self.ffhq_dataset = MyDatasetLMDB("./data/dataset/ffhq/ffhq_lmdb.db", "FFHQ", sg_latents)

    def cvt_gray_color_sm(self, gray_sm):
        img = tensor2label(gray_sm.unsqueeze(0), 19)
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        img = torchvision.transforms.functional.to_pil_image(img)
        return img

    def swp_eyebrows(self, base_sm, tgt_sm):
        base_idx = functools.reduce(np.logical_or, (base_sm==lbrow, base_sm==rbrow))
        base_sm[base_idx] = skin_idx
        tgt_idx = functools.reduce(np.logical_or, (tgt_sm==lbrow, tgt_sm==rbrow))
        tgt_sm[~(tgt_idx)] = 0
        return base_sm, tgt_sm

    def swp_nose(self, base_sm, tgt_sm):
        base_idx = base_sm==nose_idx 
        base_sm[base_idx] = skin_idx
        tgt_idx = tgt_sm==nose_idx
        tgt_sm[~(tgt_idx)] = background_idx
        return base_sm, tgt_sm

    # tgt_sm should not have glasses
    def remove_glasses(self, base_sm, tgt_sm):
        base_idx = base_sm==eye_g_idx
        # idx = functools.reduce(np.logical_or, (base_segmap_copy == l_eye_idx, base_segmap_copy == r_eye_idx))
        base_sm[base_idx] = skin_idx
        tgt_idx = functools.reduce(np.logical_or, (tgt_sm==l_eye_idx, tgt_sm==r_eye_idx))
        tgt_sm[~(tgt_idx)] = background_idx
        return base_sm, tgt_sm

    # base_sm should not have glasses and tgt_sm should have glasses
    def add_glasses_reading(self, base_sm, tgt_sm, reading=True):
        if reading:
            tgt_idx = functools.reduce(np.logical_or, (tgt_sm==eye_g_idx, tgt_sm==l_eye_idx, tgt_sm==r_eye_idx))
            tgt_sm[~(tgt_idx)] = 0
        else:
            base_idx = functools.reduce(np.logical_or, (base_sm==l_eye_idx, base_sm==r_eye_idx)) 
            base_sm[base_idx] = eye_g_idx
            tgt_idx = tgt_sm==eye_g_idx
            tgt_sm[~(tgt_idx)] = 0
        return base_sm, tgt_sm

    def swp_mouth(self, base_sm, tgt_sm):
        base_idx = functools.reduce(np.logical_or, (base_sm==ulip_idx, base_sm==llip_idx, base_sm==mouth_idx))
        base_sm[base_idx] = skin_idx
        tgt_idx = functools.reduce(np.logical_or, (tgt_sm==ulip_idx, tgt_sm==llip_idx, tgt_sm==mouth_idx))
        tgt_sm[~(tgt_idx)] = 0
        return base_sm, tgt_sm

    def swp_hair(self, base_sm, tgt_sm):
        base_sm = base_sm + 100
        base_sm[base_sm==113] =  0
        base_sm[base_sm==100] = 1
        mask = np.asarray(~(tgt_sm==hair_idx) & ~(tgt_sm==background_idx) & ~(tgt_sm==14) & ~(tgt_sm==15) & ~(tgt_sm==17) & ~(tgt_sm==18), np.uint8)
        tgt_sm[~(tgt_sm== hair_idx)] = background_idx
        dst = cv2.inpaint(tgt_sm, mask, 3, cv2.INPAINT_NS)
        dst = np.where(dst > 0, 2, 1)
        tmp = np.maximum(base_sm, dst)
        tmp[tmp==1] = 100
        tmp[tmp==2] = 113
        tmp -= 100
        return tmp, tmp

    def get_no_glass(self, idx, nbrs):
        idxs = [idx]
        no_glass = {idx: self.glass_attrs[idx] == "NoGlasses"}
        for item in nbrs:
            no_glass.update({item: self.glass_attrs[item] == "NoGlasses"})
        return no_glass

    def make_remove_glasses(self, itemid, gt_base_sm=None):
        if itemid in self.nbrs[self.cls]:
            nbrs = self.nbrs[self.cls][itemid]
        else:
            return None

        no_glass = self.get_no_glass(itemid, self.nbrs[self.cls][itemid])
        flag = 0
        if no_glass[itemid] == False: # Wearning Glasses
            nbrs = self.nbrs[self.cls][itemid]
            if not len(nbrs) > 0 :
                return None
            if gt_base_sm is None:
                base_sm = self.ffhq_dataset[itemid]

                base_sm = np.copy(base_sm)
            else:
                base_sm = np.copy(gt_base_sm)
            for item in nbrs:
                cidx = item
                if no_glass[cidx] == True: # Not Wearing glasses
                    tgt_sm = self.ffhq_dataset[cidx]
                    tgt_sm = np.copy(tgt_sm)
                    base_sm_res, tgt_sm_res = self.remove_glasses(base_sm, tgt_sm)
                    flag = 1
                    break
            if flag == 1:
                res_sm = np.maximum(base_sm_res, tgt_sm_res)
                img = tensor2label(torch.tensor(res_sm).to(torch.uint8).unsqueeze(0), 19)
                img = np.array(img, dtype=np.float32)
                img = torch.from_numpy(img)
                base_sm = res_sm
                color_sm = torchvision.transforms.functional.to_pil_image(torch.tensor(img))
                gray_sm = torch.as_tensor(np.array(res_sm), dtype=torch.int64).unsqueeze(0)
                return gray_sm, color_sm
            if flag==0:
                return None
        else:
            return None
    
    def make_add_glasses(self, itemid, gt_base_sm=None):
        if itemid in self.nbrs[self.cls]:
            nbrs = self.nbrs[self.cls][itemid]
        else:
            return None
        if not len(nbrs) > 0 :
            return None
        if gt_base_sm is None:
            base_sm = self.ffhq_dataset[itemid]
            base_sm = np.copy(base_sm)
        else:
            base_sm = np.copy(gt_base_sm)
        flag = 0
        no_glass = self.get_no_glass(itemid, self.nbrs[self.cls][itemid])
        if no_glass[itemid] == True: # Not wearing glasses
            np.random.shuffle(nbrs)
            for item in nbrs:
                cidx = item
                if no_glass[cidx] == False: # Wearing glasses
                    tgt_sm = self.ffhq_dataset[cidx]
                    tgt_sm = np.copy(tgt_sm)
                    if self.glass_attrs[cidx] == "ReadingGlasses":
                        base_sm_res, tgt_sm_res = self.add_glasses_reading(base_sm, tgt_sm, reading=True)
                    else:
                        base_sm_res, tgt_sm_res = self.add_glasses_reading(base_sm, tgt_sm, reading=False)
                    flag = 1
                    break
        if flag==0:
            return None
        res_sm = np.maximum(base_sm_res, tgt_sm_res)
        img = tensor2label(torch.tensor(res_sm).to(torch.uint8).unsqueeze(0), 19)
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        base_sm = res_sm
        color_sm = torchvision.transforms.functional.to_pil_image(torch.tensor(img))
        gray_sm = torch.as_tensor(np.array(res_sm), dtype=torch.int64).unsqueeze(0)
        return gray_sm, color_sm

    def make_nose(self, itemid, gt_base_sm=None):
        if itemid in self.nbrs[self.cls]:
            nbrs = self.nbrs[self.cls][itemid]
        else:
            return None
        if not len(nbrs) > 0 :
            return None

        if gt_base_sm is None:
            base_sm = self.ffhq_dataset[itemid]
            base_sm = np.copy(base_sm)
        else:
            base_sm = np.copy(gt_base_sm)
        cidx = np.random.choice(nbrs)
        tgt_sm = self.ffhq_dataset[cidx]
        tgt_sm = np.copy(tgt_sm)
        # Transfer people smiling 
        base_sm_res, tgt_sm_res = self.swp_nose(base_sm, tgt_sm)

        res_sm = np.maximum(base_sm_res, tgt_sm_res)
        img = tensor2label(torch.tensor(res_sm).to(torch.uint8).unsqueeze(0), 19)
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        base_sm = res_sm
        color_sm = torchvision.transforms.functional.to_pil_image(torch.tensor(img))
        gray_sm = torch.as_tensor(np.array(res_sm), dtype=torch.int64).unsqueeze(0)
        return gray_sm, color_sm

    def make_smile(self, itemid, gt_base_sm=None):
        if itemid in self.smiling_nbrs[self.cls]:
            nbrs = self.smiling_nbrs[self.cls][itemid]
        else:
            return None
        if not len(nbrs) > 0:
            return None
        if gt_base_sm is None:
            base_sm = self.ffhq_dataset[itemid]
            base_sm = np.copy(base_sm)
        else:
            base_sm = np.copy(gt_base_sm)

        cidx = np.random.choice(nbrs)
        tgt_sm = self.ffhq_dataset[cidx]
        tgt_sm = np.copy(tgt_sm)
        # Transfer people smiling 
        base_sm_res, tgt_sm_res = self.swp_mouth(base_sm, tgt_sm)

        res_sm = np.maximum(base_sm_res, tgt_sm_res)
        img = tensor2label(torch.tensor(res_sm).to(torch.uint8).unsqueeze(0), 19)
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        base_sm = res_sm
        color_sm = torchvision.transforms.functional.to_pil_image(torch.tensor(img))
        gray_sm = torch.as_tensor(np.array(res_sm), dtype=torch.int64).unsqueeze(0)
        return gray_sm, color_sm

    def make_eyebrows(self, itemid, gt_base_sm=None):
        if itemid in self.nbrs[self.cls]:
            nbrs = self.nbrs[self.cls][itemid]
        else:
            return None
        if not len(nbrs) > 0 :
            return None
        if gt_base_sm is None:
            base_sm = self.ffhq_dataset[itemid]
            base_sm = np.copy(base_sm)
        else:
            base_sm = np.copy(gt_base_sm)

        cidx = np.random.choice(nbrs)
        tgt_sm = self.ffhq_dataset[cidx]
        tgt_sm = np.copy(tgt_sm)
        # Transfer people smiling 
        base_sm_res, tgt_sm_res = self.swp_eyebrows(base_sm, tgt_sm)

        res_sm = np.maximum(base_sm_res, tgt_sm_res)
        img = tensor2label(torch.tensor(res_sm).to(torch.uint8).unsqueeze(0), 19)
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        base_sm = res_sm
        color_sm = torchvision.transforms.functional.to_pil_image(torch.tensor(img))
        gray_sm = torch.as_tensor(np.array(res_sm), dtype=torch.int64).unsqueeze(0)
        return gray_sm, color_sm

    def make_hair(self, itemid, gt_base_sm=None):
        flag = 0
        if itemid in self.nbrs[self.cls]:
            nbrs = self.nbrs[self.cls][itemid]
        else:
            return None
        if not len(nbrs) > 0 :
            return None
        if gt_base_sm is None:
            base_sm = self.ffhq_dataset[itemid]
            base_sm = np.copy(base_sm)
        else:
            base_sm = np.copy(gt_base_sm)

        np.random.shuffle(nbrs)
        for item in nbrs:
            if self.gender_attrs[itemid] == self.gender_attrs[item]:
                cidx = item
                tgt_sm = self.ffhq_dataset[cidx]
                tgt_sm = np.copy(tgt_sm)
                flag = 1
                break 
            # Transfer people smiling 
        if flag==0:
            return None  

        base_sm_res, tgt_sm_res = self.swp_hair(base_sm, tgt_sm)
        res_sm = np.maximum(base_sm_res, tgt_sm_res)
        img = tensor2label(torch.tensor(res_sm).to(torch.uint8).unsqueeze(0), 19)
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        base_sm = res_sm
        color_sm = torchvision.transforms.functional.to_pil_image(torch.tensor(img))
        gray_sm = torch.as_tensor(np.array(res_sm), dtype=torch.int64).unsqueeze(0)
        return gray_sm, color_sm

    def make_edits(self, itemid, gray_sm, color_sm):
        fns = ["make_smile", "make_add_glasses", "make_remove_glasses", "make_hair"]
        gray_sm_l, color_sm_l = [], []
        gray_sm_orig, color_sm_orig = gray_sm, color_sm
        idxs= [0, 1, 2, 3]
        gray_sm = None
        color_sm = None
        for idx in idxs:
            item = eval(f"self.{fns[idx]}")(itemid, gray_sm_orig.squeeze(0))
            if item is not None:
                gray_sm, color_sm = item
                color_sm = torchvision.transforms.functional.to_tensor(color_sm)
                gray_sm = torch.as_tensor(gray_sm, dtype=torch.int64)
                gray_sm_l.append(gray_sm.squeeze(0)) 
                color_sm_l.append(color_sm)
            else:
                gray_sm_l.append(gray_sm_orig.squeeze(0))
                color_sm_l.append(color_sm_orig)
                
        return torch.stack(gray_sm_l), torch.stack(color_sm_l)

    # Sequential Edits
    def make_edits_v1(self, itemid, num_edits, gray_sm, color_sm):
        fns = ["make_smile",  "make_hair"]
        gray_sm_l, color_sm_l = [], []
        edit_opts = []
        gray_sm_orig, color_sm_orig = gray_sm, color_sm
        gray_sm_l.append(gray_sm_orig)
        color_sm_l.append(color_sm_orig)
        idxs = [0, 1]
        gray_sm = gray_sm_orig
        color_sm = None
        for idx in idxs:
            if gray_sm is not None:
                gray_sm = gray_sm.squeeze(0)
            item = eval(f"self.{fns[idx]}")(itemid, gray_sm.squeeze(0))
            if item is not None:
                gray_sm, color_sm = item
                if idx == 0:
                    edit_opts.append("smile")
                if idx == 1:
                    edit_opts.append("hair")
                gray_sm_l.append(gray_sm) 
                color_sm_l.append(torchvision.transforms.functional.to_tensor(color_sm))
            else:
                gray_sm = None
                break
        return gray_sm_l, color_sm_l, edit_opts

def instantiate_editor_faces(cls="easy"):
    # Load attrs
    attrs = np.load("./data/all_att.pickle", allow_pickle=True)['Attribute'][0]
    glass_attrs = []
    for idx in range(0, len(attrs)):
        if len(attrs[idx]) > 0:
            if isinstance(attrs[idx], list):
                glass_attrs.append(attrs[idx][0]['faceAttributes']['glasses'])
            else:
                glass_attrs.append(attrs[idx]['faceAttributes']['glasses'])
        else:
            glass_attrs.append([])

    smile_attrs = []
    for idx in range(0, len(attrs)):
        if len(attrs[idx]) > 0:
            if isinstance(attrs[idx], list):
                if attrs[idx][0]['faceAttributes']['smile'] > 0.5:
                    smile_attrs.append(True)
                else:
                    smile_attrs.append(False)
            else:
                smile_attrs.append(False)
        else:
            smile_attrs.append(False)
    gender_attrs = []
    for idx in range(0, len(attrs)):
        if len(attrs[idx]) > 0:
            if isinstance(attrs[idx], list):
                if attrs[idx][0]['faceAttributes']['gender']:
                    gender_attrs.append(attrs[idx][0]['faceAttributes']['gender'])
            else:
                gender_attrs.append(attrs[idx]['faceAttributes']['gender'])
        else:
            gender_attrs.append([])
    if os.path.exists("./data/cat_10k.pkl"):
        categories = np.load("./data/cat_10k.pkl", allow_pickle=True)
    smiling_nbrs = {"easy": {}, "med": {}, "hard": {}}

    for key in ["easy", "med", "hard"]:
        candidate_dict = categories[key]
        for k, val in candidate_dict.items():
            for v in val:
                if smile_attrs[v] == True:
                    if k in smiling_nbrs[key].keys():
                        smiling_nbrs[key][k].append(v)
                    else:
                        smiling_nbrs[key].update({k:[v]})
    augment_dataset = AugmentDataset(categories, smiling_nbrs, glass_attrs=glass_attrs, gender_attrs=gender_attrs, cls=cls)    
    return augment_dataset
