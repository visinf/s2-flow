import os

import numpy as np
import PIL
import requests
import torch
import torchvision.transforms as transforms
from ext import dnnlib, legacy

from util import deeplab
from util.utils import download_file, generate_label


class BuildSegmentor:
    def __init__(self, args, device="cuda:0"):
        self.device = device
        self.dataset = args.dataset
        print(f"{__file__}, args.dataset: {args.dataset}")
        if args.dataset == "FFHQ":
            # Load Generator Network
            network_pkl = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
            if os.path.exists("/usr/app/stylegan/stylegan2-ffhq-config-f.pkl"):
                print("Found local StyleGan2 !")
                network_pkl = "/usr/app/stylegan/stylegan2-ffhq-config-f.pkl" # Local load, avoiding to re-download 360Mb each time
            else:
                network_pkl = network_pkl

            # Load Segmentor Network
            resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM', file_path='./pretrained_model/deeplab_model/R-101-GN-WS.pth.tar', file_size=178260167, file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')
            deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY', file_path='./pretrained_model/deeplab_model/deeplab_model.pth', file_size=464446305, file_md5='8e8345b1b9d95e02780f9bed76cc0293')
            assert torch.cuda.is_available()
            torch.backends.cudnn.benchmark = True
            model_fname = './pretrained_model/deeplab_model/deeplab_model.pth'

            # if not os.path.isfile(resnet_file_spec['file_path']):
            # print('Downloading backbone Resnet Model parameters')
            #    with requests.Session() as session:
            #        download_file(session, resnet_file_spec)
            #    print('Done!')

            segnet = getattr(deeplab, 'resnet101')(
                        pretrained=True,
                        num_classes=19,
                        num_groups=32,
                        weight_std=True,
                        beta=False)

            segnet = segnet.to(self.device)
            segnet.eval()
            # if not os.path.isfile(deeplab_file_spec['file_path']):
            #    print('Downloading DeeplabV3 Model parameters')
            #    with requests.Session() as session:
            #        download_file(session, deeplab_file_spec)
            #    print('Done!')

            checkpoint = torch.load(model_fname)
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
            segnet.load_state_dict(state_dict)

        self.segnet = segnet

        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            Gs = legacy.load_network_pkl(f)['G_ema']
        self.Gs = Gs
        self.Gs.eval()
        self.Gs = self.Gs.to(self.device)

    def generate_im_from_random_seed(self, seed=22, truncation_psi=0.7, noise_mode="const"):
        Gs = self.Gs
        seeds = [seed]
        label = torch.zeros([1, Gs.c_dim], device=self.device)
        for seed_idx, seed in enumerate(seeds):
            # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, Gs.z_dim)).to(self.device)
            img = Gs(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        return images, z

    def generate_im_from_w_space(self, w, noise_mode="const", return_tensor=False):
        w = torch.tensor(w, device=self.device) # pylint: disable=not-callable
        img = self.Gs.synthesis(w, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if not return_tensor:
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')        
        else:
            pass
        return img

    def generate_segmaps(self, latents=None, img_sz=512, num_latents=10, return_img=False):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Synthesize the result of a W projection.
        if latents is not None:
            latents = latents.to(self.device)
            imgs = []
            if latents.shape[1:] == (self.Gs.num_ws, self.Gs.w_dim):
                pass
            else:
                latents = latents.repeat(1, 18, 1)

            img = self.generate_im_from_w_space(latents, return_tensor=True)

            imgs = list(map(lambda x: PIL.Image.fromarray(x.cpu().numpy(), "RGB").resize((img_sz, img_sz), PIL.Image.BILINEAR), img))
        else:
            # Generate 100 Random Images
            seeds = np.random.choice(1000, num_latents)
            imgs = []
            for seed in seeds:
                img = self.generate_im_from_random_seed(seed=seed)
                imgs.append(img)
            imgs = [img.resize((img_sz, img_sz), PIL.Image.BILINEAR) for img in imgs]

        imgs = torch.stack(list(map(lambda x: to_tensor(x), imgs)))
        imgs = imgs.to(self.device)
        # logits.shape: ?, 19, 512, 512
        if self.dataset == "FFHQ":
            logits = self.segnet(imgs)

        _, gray_img = torch.max(logits, 1)

        # Downscale image for memory reasons
        if self.dataset == "FFHQ":
            color_img = generate_label(logits, img_sz)
        if return_img is False:
            return logits, color_img, gray_img
        else:
            return logits, color_img, gray_img, imgs
    
    def segment_image(self, imgs, img_sz):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            imgs = torch.stack(list(map(lambda x: to_tensor(x), imgs)))
            imgs = imgs.to(self.device)
            if self.dataset == "FFHQ":
                logits = self.segnet(imgs)
            _, gray_img = torch.max(logits, dim=1)

            if self.dataset == "FFHQ":
                color_img = generate_label(logits, img_sz)

        return logits, color_img, gray_img
