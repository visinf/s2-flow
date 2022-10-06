import glob
import os

import numpy as np
import PIL
import torch
import torchvision
from torch.utils.data import Dataset

pil2tensor = torchvision.transforms.functional.to_tensor


def reverse_flow(prior, cnd_ntwk, latents, condition, device):
    # latents.shape: (?, 18, 512) --> (?, 1, 18, 512)
    latents = latents.unsqueeze(1)
    latents = latents.to(device)
    zero_padding = torch.zeros(len(latents), latents.shape[2], 1).to(latents)

    # gray_segmap.shape: [?, 3, 512, 512] --> [?, 3, 512, 512]
    cnd_ntwk = cnd_ntwk.to(device)
    condition = condition.to(device)
    segmaps_to_cond = cnd_ntwk(condition)
    
    # Reverse Flow
    w_new, _ = prior(latents.squeeze(1), segmaps_to_cond, zero_padding, reverse=True)
    return  w_new


def forward_flow(prior, cnd_ntwk, latents, condition, device):
    # latents.shape: (?, 1, 18, 512) --> (?, 1, 18, 512)
    latents = latents.to(device)
    zero_padding = torch.zeros(len(latents), latents.shape[2], 1).to(latents)
    # gray_segmap.shape: [?, 3, 512, 512] --> [?, 3, 512, 512]
    cnd_ntwk = cnd_ntwk.to(device)
    condition = condition.to(device)
    segmaps_to_cond = cnd_ntwk(condition)
    # segmaps_to_cond = torch.zeros_like(segmaps_to_cond).to(segmaps_to_cond.device)

    # Forward Flow
    z, delta_log_p2 = prior(latents.squeeze(1), segmaps_to_cond, zero_padding)
    return z, delta_log_p2



def gen_image_new_w(helper, w_new, img_sz):
    gen_images = []
    for idx, _w in enumerate(w_new):
        tmp_img = helper.generate_im_from_w_space(_w.unsqueeze(0))
        tmp_img = tmp_img.resize((img_sz[0], img_sz[1]), PIL.Image.BILINEAR)
        gen_images.append(tmp_img)
    return gen_images


class TestDataset(Dataset):
    def __init__(self, dataset, latents, img, color_sm, gray_sm, is_conditional, transform=None, is_version_1=False):
        self.latents = latents
        self.img_path = img
        self.color_sm_path = color_sm
        self.gray_sm_path = gray_sm
        fns= list(map(lambda x: int(x.split("/")[-1][:-4]), glob.glob(f"{self.color_sm_path}/*.png")))
        self.actual_idxs = sorted(fns)
        self.transform = transform
        if dataset == "FFHQ":
            if os.path.exists("./data/dataset/ffhq/ffhq_test/gray_sm"):
                self.gray_sm_unedited = "./data/dataset/ffhq/ffhq_test/gray_sm/"
                self.color_sm_unedited = "./data/dataset/ffhq/ffhq_test/color_sm/"
        self.is_conditional = is_conditional
        self.is_version_1 = is_version_1

    def __getitem__(self, index):
        # index = self.sidx + index
        index = self.actual_idxs[index]
        x = torch.tensor(self.latents[index])
        img = PIL.Image.open(f"{self.img_path}/{index}.png")
        img = img.resize((512, 512), PIL.Image.BILINEAR)
        img = pil2tensor(img)

        color_sm = PIL.Image.open(f"{self.color_sm_path}/{index}.png")
        color_sm = pil2tensor(color_sm)

        target = PIL.Image.open(f"{self.gray_sm_path}/{index}.png")
        gray_sm = torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)

        color_sm_unedit = pil2tensor(PIL.Image.open(f"{self.color_sm_unedited}/{index}.png"))
        target = PIL.Image.open(f"{self.gray_sm_unedited}/{index}.png")
        gray_sm_unedit = torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)
        
        if self.transform:
            x = self.transform(x)
        if self.is_version_1: 
            return x, img, gray_sm, color_sm, gray_sm_unedit, color_sm_unedit, index
        else:
            return x, img, gray_sm, color_sm, color_sm_unedit, index

    def __len__(self):
        return len(self.actual_idxs)


class DataSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)
