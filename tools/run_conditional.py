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

from test_helper import TestDataset, gen_image_new_w, reverse_flow
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
        if os.path.exists("./data/dataset/ffhq/ffhq_test/imgs"):
            path_sm_img =  "./data/dataset/ffhq/ffhq_test/imgs"


        if self.is_conditional: # Load 1000 vector from the z latent space
            num_samples = 1000
            # latents = torch.load(f"/visinf/home/ksingh/StyleFlow/dataset/ffhq/test/latents/{self.args.prior_type}/sample_z.pt") # z latent
            latents = torch.load(f"./data/dataset/ffhq/test/latents/{self.args.prior_type}/sample_z.pt") # z latent
            latents = torch.randn_like(latents)
            if self.args.dataset == "FFHQ":
                if os.path.exists("./data/dataset/ffhq/ffhq_test/color_sm"):
                    path_sm_color = "./data/dataset/ffhq/ffhq_test/color_sm"
                    path_sm_gray = "./data/dataset/ffhq/ffhq_test/gray_sm"

        latents = np.load("./data/all_latents.pickle", allow_pickle=True)['Latent']# w latent
        if kwargs["is_version_1"]:
            self.test_dataset = TestDataset(self.args.dataset, latents, path_sm_img, path_sm_color, path_sm_gray, self.is_conditional, None, True)
        else:
            self.test_dataset = TestDataset(self.args.dataset, latents, path_sm_img, path_sm_color, path_sm_gray, self.is_conditional)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.batch, drop_last=False, shuffle=False)
        self.augment_dataset = instantiate_editor_faces(self.cls)

    # Some cases might have no change in semantic mask. 
    # This is due to the fact that some semantic changes are not possible like removing glasses from person wearing no glasses.
    def visualize_content(self):
        for idx in range(0, 10):
            grid_imgs = []
            gt_color_sm = []
            item = self.test_dataset[idx]
            _, _, _, _, orig_gray_sm, orig_color_sm, imgid = item # gt_color_sm: edited version of orig_color_sm
            gray_sms, color_sms = self.augment_dataset.make_edits(imgid, orig_gray_sm, orig_color_sm)

            latents = torch.load(f"./data/dataset/ffhq/test/latents/{self.args.prior_type}/sample_z.pt")[0] # z latent
            latents = latents.repeat(len(gray_sms), 1, 1)
            gt_color_sm.append(item[5].unsqueeze(0).to(torch.float).to(self.device))
            gt_color_sm = torch.cat(gt_color_sm)

            w_new  = reverse_flow(self.prior, self.cnd_ntwk, latents, color_sms, self.device)

            imgs = gen_image_new_w(self.segnet, w_new, (self.args.seg_map_sz, self.args.seg_map_sz))
            imgs = torch.stack(list(map(lambda x: pil2tensor(x), imgs)))
            grid_imgs = torch.vstack((imgs.detach().cpu(), color_sms.detach().cpu()))
            write_imgs = torchvision.utils.make_grid(grid_imgs, nrow=4, padding=0)
            torchvision.utils.save_image(write_imgs, f"./results/{self.args.exp_name}/random_imgs_style_fixed_{idx}.png")
        return write_imgs

    def visualize_style(self):
        grid_imgs = []
        idxs = [12, 19, 51, 32, 19, 12, 85, 84, 72, 69, 93]
        for i in range(0, len(idxs)):
            latents = torch.load(f"./data/dataset/ffhq/test/latents/{self.args.prior_type}/sample_z.pt")[i*10: (i+1)*10] # z latent
            idx = idxs[i]
            item = self.test_dataset[idx] # arbitary offset from sidx # segmap fixed
            gt_color_sm = item[3].unsqueeze(0).repeat(len(latents), 1, 1, 1)
            w_new  = reverse_flow(self.prior, self.cnd_ntwk, latents, gt_color_sm, self.device)
            imgs = gen_image_new_w(self.segnet, w_new, (self.args.seg_map_sz, self.args.seg_map_sz))
            imgs = torch.stack(list(map(lambda x: pil2tensor(x), imgs)))
            grid_imgs = torch.vstack((imgs, gt_color_sm))
            write_imgs = torchvision.utils.make_grid(grid_imgs, nrow=10)
            torchvision.utils.save_image(write_imgs, f"./results/{self.args.exp_name}/random_imgs_segmap_fixed_{i}.png")
        return write_imgs

    def visualize_conditional(self, gen_img, color_sm_img):
        gen_images_tensor = torch.stack(gen_img)
        seg_images_tensor = torch.stack(color_sm_img)
        write_imgs = torch.vstack((gen_images_tensor, seg_images_tensor))
        return write_imgs
    
    def compute_metrics(self):
        for idx, item in enumerate(self.test_loader):
            latent, gt_img, gt_gray_sm, gt_color_sm, orig_color_sm, _, _ = item # gt_color_sm: edited version of orig_color_sm
            gen_images = []
            seg_images = []
            batch_z = latent.squeeze(1)# latent from z space 
            w_new  = reverse_flow(self.prior, self.cnd_ntwk, batch_z, gt_color_sm, self.device)
            imgs = gen_image_new_w(self.segnet, w_new, (self.args.seg_map_sz, self.args.seg_map_sz))
            gen_images.extend(list(map(torchvision.transforms.functional.to_tensor, imgs)))
            seg_images.extend(gt_color_sm.detach().cpu())

            if self.is_conditional:
                cond_img = self.visualize_conditional(gen_images, seg_images)
            return cond_img

def test(prior, segnet, cnd_ntwk, device, args):
    os.makedirs(f"./results/{args.exp_name}", exist_ok=True)
    vis_edit_idx = np.arange(0, 1000) # Arbitary Offset from sidx Images to Edit
    prior.eval()
    cnd_ntwk.eval()
    with torch.no_grad():
        # Conditional Sampling Results
        cond_metrics = ComputeMetrics(args, prior, cnd_ntwk, segnet, is_conditional=True, cls="easy", device=device)
        test = "True"
        cond_metrics.setup(num_edits=1, test=test, vis_edit_idx=vis_edit_idx, is_version_1=True)
        cond_img = cond_metrics.compute_metrics()
        os.makedirs(f"./results/{args.exp_name}/cond_test", exist_ok=True)
        cond_img = torchvision.utils.make_grid(cond_img, nrow=8)
        torchvision.utils.save_image(cond_img, f"./results/{args.exp_name}/cond_test/cond_gen.png")

        vis_style_img = cond_metrics.visualize_style() # Content fixed 
        vis_cont_img = cond_metrics.visualize_content() # Style fixed



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cond_and_disentanglment s2-flow")
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

    test(prior, segnet, cnd_ntwk, device, args)
