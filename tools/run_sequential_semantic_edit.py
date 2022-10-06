import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from model.conditions import ConditionalAugmentation
from model.module.flow import cnf
from util.segmentation_util import BuildSegmentor

from test_helper import (TestDataset, forward_flow, gen_image_new_w,
                         reverse_flow)
from tools.editing_functions import instantiate_editor_faces

pil2tensor = torchvision.transforms.functional.to_tensor


class ComputeMetrics:
    def __init__(self, args, prior, cnd_ntwk, segnet, is_conditional, cls, device="cpu"):
        self.args = args
        self.prior = prior.eval()
        self.cnd_ntwk = cnd_ntwk.eval().to("cuda")
        self.segnet = segnet
        self.device = device

        self.is_conditional = is_conditional
        self.cls = cls

        self.pred_arr = []

    def setup(self, **kwargs):
        self.vis_edit_idx = kwargs["vis_edit_idx"]
        self.start_idx = 0
        path_sm_img =  "/data/dataset/ffhq/ffhq_test/imgs"
        path_sm = "/data/dataset/ffhq/ffhq_test"
        path_sm_color = f"{path_sm}/color_sm/"
        path_sm_gray = f"{path_sm}/gray_sm"


        latents = np.load("./data/all_latents.pickle", allow_pickle=True)['Latent'] # w latent
        if kwargs["is_version_1"]:
            self.test_dataset = TestDataset(self.args.dataset, latents, path_sm_img, path_sm_color, path_sm_gray, self.is_conditional, None, True)
        else:
            self.test_dataset = TestDataset(self.args.dataset, latents, path_sm_img, path_sm_color, path_sm_gray, self.is_conditional)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.batch, drop_last=False, shuffle=False)
        self.augment_dataset = instantiate_editor_faces(self.cls)

    def visualize_edits_v1(self):
        os.makedirs(f"./results_sequential/{self.args.exp_name}/", exist_ok=True)
        count = 0
        grid_img = []
        all_latent = {}
        all_edits_ops = {}
        gen_imgs = []
        for item in self.test_loader:
            latent, gt_img, _, _, orig_gray_sm, orig_color_sm, imgid = item # gt_color_sm: edited version of orig_color_sm
            for idxi in range(0, len(latent)):
                gray_sms, color_sms, edits_ops= self.augment_dataset.make_edits_v1(imgid[idxi].item(), 1, orig_gray_sm[idxi], orig_color_sm[idxi])
                if len(edits_ops) == 0:
                    continue
                w = latent[idxi].unsqueeze(0)
                gen_imgs.append(gt_img[idxi])
                for idxj in range(0, len(color_sms)-1):
                    z = forward_flow(self.prior, self.cnd_ntwk, w, color_sms[idxj].unsqueeze(0), self.device)[0]
                    w_new  = reverse_flow(self.prior, self.cnd_ntwk, z.unsqueeze(0), color_sms[idxj+1].unsqueeze(0), self.device, )
                    imgs = gen_image_new_w(self.segnet, w_new[0], (self.args.seg_map_sz, self.args.seg_map_sz))
                    gen_imgs.append(pil2tensor(imgs[0]))
                    w = w_new

                gen_imgs = torch.stack(gen_imgs)
                color_sms = torch.stack(color_sms)

                grid_img = torch.cat([color_sms, gen_imgs])
                grid_img = torchvision.utils.make_grid(grid_img, nrow=3)
                torchvision.utils.save_image(grid_img, f"./results_sequential/{self.args.exp_name}/{imgid[idxi]}.png")
                gen_imgs = []
                grid_img = []

                tmp = w_new.clone()
                all_latent.update({imgid[idxi].item(): [tmp.detach().cpu().numpy(), color_sms[-1].detach().cpu().numpy()]})
                all_edits_ops.update({imgid[idxi].item(): edits_ops})
            count += 1
            if count > 5:
                break

def test(prior, segnet, cnd_ntwk, device, args):
    vis_edit_idx = np.arange(0, 1000) # Arbitary Offset from sidx Images to Edit
    prior.eval()
    cnd_ntwk.eval()
    with torch.no_grad():
        # Conditional Sampling Results
        args.batch = 2
        cond_metrics = ComputeMetrics(args, prior, cnd_ntwk, segnet, is_conditional=False, cls="easy", device=device)
        test = "True"
        cond_metrics.setup(test=test, vis_edit_idx=vis_edit_idx, is_version_1=True)
        cond_metrics.visualize_edits_v1()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sequential semantic editing s2 flow")
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

    test(prior, segnet, cnd_ntwk, device, args)
