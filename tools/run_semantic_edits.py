import argparse
import glob
import json
import os
import random

import numpy as np
import PIL
import torch
import torch.backends.cudnn as cudnn
import torchvision
from model.conditions import ConditionalAugmentation
from model.module.flow import cnf
from torchvision.transforms.functional import to_tensor
from util.segmentation_util import BuildSegmentor

from test_helper import forward_flow, gen_image_new_w, reverse_flow

# Define Device
device = torch.device('cuda')


# Load the model
def make_edits(args, prior, cnd_ntwk, segnet, latent, edited_c_segmap, edited_g_segmap, prefix, fn, save_img=False):
    latent = latent.to(device)
    logits, test_c_segmap, test_g_segmap = segnet.generate_segmaps(latents=latent, img_sz=args.seg_map_sz)
    test_g_segmap = test_g_segmap.to(device)
    test_c_segmap = test_c_segmap.to(device)
    orig_segmap = torchvision.transforms.functional.to_pil_image(test_c_segmap[0])
    orig_img = segnet.generate_im_from_w_space(latent)
    orig_img = orig_img.resize((args.seg_map_sz, args.seg_map_sz), PIL.Image.BILINEAR)
    # Forward Flow
    z, _ = forward_flow(prior, cnd_ntwk, latent.unsqueeze(0), test_c_segmap, device)
    # Reverse Flow
    e_c_sm = torchvision.transforms.functional.to_tensor(edited_c_segmap)
    e_g_sm = torch.as_tensor(np.array(edited_g_segmap), dtype=torch.int64)

    w_new = reverse_flow(prior, cnd_ntwk, z, e_c_sm.unsqueeze(0), device)
    gen_images = gen_image_new_w(segnet, w_new, (args.seg_map_sz, args.seg_map_sz))

    gen_images_tensor = torch.stack(list(map(lambda x: to_tensor(x) , gen_images)))
    orig_images_tensor = torch.stack(list(map(lambda x: to_tensor(x) , [orig_img])))

    #predict segmaps
    gen_images_tensor, orig_images_tensor = gen_images_tensor.to(device), orig_images_tensor.to(device)
    grid_img = [orig_segmap, edited_c_segmap, orig_img, gen_images[0]]
    grid_img = torch.stack(list(map(lambda x: torchvision.transforms.functional.to_tensor(x), grid_img)))
    if save_img:
        torchvision.utils.save_image(grid_img, f"{prefix}_grid_{fn}.png", padding=0)
    else:
        return grid_img


# Load Segmentation Maps Edited 
def run_exp(args, prior, cnd_ntwk, segnet, path, save_path, return_img=False):
    imgs = []
    latents = torch.tensor(np.load("./data/all_latents.pickle", allow_pickle=True)["Latent"]).squeeze(0)
    fns= glob.glob(f"{path}/color/*.png")
    fns_gray = glob.glob(f"{path}/gray/*.png")
    idx = 0
    for fn in fns:
        img_idx = fn.split("/")[-1][:-4]
        prefix = f"{save_path}/{img_idx}"
        cur_latent = latents[int(img_idx)]
        cur_segmap = PIL.Image.open(fn)
        cur_segmap_gray = PIL.Image.open(f"{path}/gray/{img_idx}.png")
        if return_img:
            grid_img = make_edits(args, prior, cnd_ntwk, segnet, cur_latent, cur_segmap, cur_segmap_gray, prefix, idx, True)
            imgs.append(grid_img)
            idx+= 1
        else:
            make_edits(args, prior, cnd_ntwk, segnet, cur_latent, cur_segmap, cur_segmap_gray, prefix, idx, True)
            idx+=1
    if return_img:
        return torch.vstack(imgs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="semantic-editing s2 flow")
    parser.add_argument("--exp_name", type=str, default="cs3_full")
    parser.add_argument("--exp_path", type=str, required=True)

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

    save_path = f"{args.exp_path}/results/{args.exp_name}"
    os.makedirs(save_path, exist_ok=True)
    tmp = run_exp(args, prior, cnd_ntwk, segnet, args.exp_path, save_path, False)
    # torchvision.utils.save_image(tmp, "./tmp.png")

