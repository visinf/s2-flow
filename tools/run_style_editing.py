import argparse
import json
import os
import pickle
import random

import numpy as np
import PIL
import torch
import torch.backends.cudnn as cudnn
import torchvision
from model.conditions import ConditionalAugmentation
from model.module.flow import cnf
from util.segmentation_util import BuildSegmentor

from test_helper import (DataSubset, forward_flow, gen_image_new_w, pil2tensor,
                         reverse_flow)


def x2y(args, prior, cnd_ntwk, segnet, x_loader, y_mean, path, fn, preverse=False):
    full_grid = []
    end_idx = 100
    num_rows = 10
    for x in x_loader:
        imgs = []
        x, sm, imgid = x
        # x = x.unsqueeze(1)
        z = forward_flow(prior, cnd_ntwk, x, sm, "cuda")[0]
        for alpha in np.linspace(0, 1, end_idx):
            # z = z.detach().cpu().numpy()
            z = (1 - alpha) * z + (alpha * torch.tensor(y_mean).to("cuda"))
            w = reverse_flow(prior, cnd_ntwk, z, sm, "cuda")
            # If you go to far results will divereg to mean. 
            if len(imgs) < num_rows:
                imgs.append(gen_image_new_w(segnet, w, (512, 512)))

        imgs = np.asarray(imgs).T
        imgs = list(map(lambda x: pil2tensor(x), imgs.reshape(-1)))
        imgs = torch.stack(imgs)
        full_grid.append(imgs)
        break

    full_grid = torch.cat(full_grid)
    write_imgs = torchvision.utils.make_grid(full_grid, nrow=num_rows, padding=0)
    torchvision.utils.save_image(write_imgs, f"{path}/{fn}.png")


def get_hair_color(attrs):
    # Get Mean z for Specific Attr i.e Hair Color
    hair_color = []
    for idx in range(0, len(attrs)):
        if len(attrs[idx]) > 0:
            if isinstance(attrs[idx], list):
                if len(attrs[idx][0]['faceAttributes']['hair']['hairColor']) == 0:
                        hair_color.append([])
                        continue 
                hair_color.append(attrs[idx][0]['faceAttributes']['hair']['hairColor'][0]['color'])
            else:
                if len(attrs[idx]['faceAttributes']['hair']['hairColor']) == 0:
                    hair_color.append([])
                    continue
                hair_color.append(attrs[idx]['faceAttributes']['hair']['hairColor'][0]['color'])
        else:
            hair_color.append([])
    hair_color = np.asarray(hair_color)
    return hair_color


def hair_walk(args, prior, cnd_ntwk, segnet, my_dataset, hair_color, path):
    os.makedirs(f"{path}", exist_ok=True)
    with open(f"/data/attr_experiments_real/v1/mean_zs_haircolor.pkl", "rb") as f:
        mean_dicts = pickle.load(f)

    k1, k2 = "red", "black"
    idx = np.where(hair_color == k1)[0] 
    dataloader = torch.utils.data.DataLoader(DataSubset(my_dataset, idx), shuffle=False, batch_size=5)
    print(f"Processing....{k1}2{k2}")
    x2y(args, prior, cnd_ntwk, segnet, dataloader, mean_dicts[k2], path, f"{k1}2{k2}")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, latents, keys=None, transform=None):
        self.latents = latents
        self.keys = keys
        self.transform = transform
        self.color_sm_path = "/data/dataset/ffhq/color_sm"

    def __getitem__(self, index):
        x = self.latents[index]
        color_sm = PIL.Image.open(f"{self.color_sm_path}/{index}.png")
        color_sm = pil2tensor(color_sm)

        if self.transform:
            x = self.transform(x)

        return x, color_sm, index

    def __len__(self):
        return len(self.latents)

def load_model(args, ckptdir):
    # Load a Trained flow model
    cond_size = args.cond_size
    prior = cnf(512, args.flow_modules, cond_size, 1, args.layer_type)
    prior = prior

    cnd_ntwk = ConditionalAugmentation(args.cond_size)
    cnd_ntwk = cnd_ntwk

    chkpt = torch.load(f"{ckptdir}/last_checkpoint.pt", map_location="cuda")
    prior.load_state_dict(chkpt['prior_ntwk'])
    cnd_ntwk.load_state_dict(chkpt['cnd_ntwk'])

    prior = prior.eval()
    cnd_ntwk = cnd_ntwk.eval()

    return prior, cnd_ntwk

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="style-editing s2 flow")
    parser.add_argument("--exp_name", type=str, default="cs3_full")
    args = parser.parse_args()
    exp_name = args.exp_name

    # Load Yaml file
    with open(f"./experiments/{args.exp_name}/ckptdir/config.txt") as f:
        items = json.load(f)

    for key, val in items.items():
        setattr(args, key, val)

    # Seed Everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True

    args.exp_name = exp_name
    device = "cuda"
    # Load Helper Networks Generator and Segmentor
    print(f"{__file__}, args.dataset: {args.dataset}")
    segnet = BuildSegmentor(args, device=device)

    # Define Models
    cond_size = args.cond_size
    prior = cnf(512, args.flow_modules, cond_size, 1, args.layer_type)
    cnd_ntwk = ConditionalAugmentation(args.cond_size)
    cnd_ntwk = cnd_ntwk.to(device)
    params = list(prior.parameters()) + list(cnd_ntwk.parameters())

    ckptdir = f"./experiments/{args.exp_name}/ckptdir"

    if args.resume is True:
        chkpt = torch.load(f"{ckptdir}/last_checkpoint.pt", map_location="cuda")
        # Load prior weights
        prior.load_state_dict(chkpt['prior_ntwk'])
        # Load cnd_ntw weight
        cnd_ntwk.load_state_dict(chkpt['cnd_ntwk'])
        # Load optimizer state
        last_epoch = chkpt["epoch"]

    path = "./style_editing/"
    sg_latents = np.load("./data/all_latents.pickle", allow_pickle=True)['Latent'][:args.num_samples]
    my_dataset = MyDataset(latents=sg_latents)
    attrs = np.load("./data/all_att.pickle", allow_pickle=True)['Attribute'][0][:args.num_samples]
    hair_color = get_hair_color(attrs)
    hair_walk(args, prior, cnd_ntwk, segnet, my_dataset, hair_color, path)

