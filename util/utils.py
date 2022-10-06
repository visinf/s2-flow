import glob
import hashlib
import html
import os
import uuid

import cv2
import lmdb
import numpy as np
import PIL
import PIL.Image
import PIL.ImageFile
import requests
import scipy
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms

pil2tensor = torchvision.transforms.functional.to_tensor
# Don't sort error
import pyarrow as pa

color_list = [(0, 0, 0), (204, 0, 0), (76, 153, 0), 
              (204, 204, 0), (204, 0, 204), (51, 51, 255), 
              (0, 255, 255), (0, 0, 255), (255, 0, 0),
              (102, 51, 0), (102, 204, 0), (255, 255, 0), 
              (0, 0, 153), (0, 0, 204), (255, 51, 153), 
              (0, 204, 204), (0, 51, 0), (255, 153, 51), 
              (0, 204, 0)]

def gen_image_new_w(helper, w_new, img_sz):
    gen_images = []
    for idx, _w in enumerate(w_new):
        tmp_img = helper.generate_im_from_w_space(_w.unsqueeze(0))
        tmp_img = tmp_img.resize((img_sz[0], img_sz[1]), PIL.Image.BILINEAR)
        gen_images.append(tmp_img)
    return gen_images

class AverageMeter(object):
    """Computes and stores the average and current value.
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_metrics(metrics_dict):
    out_str = ""
    for key, val in metrics_dict.items():
        if isinstance(val, AverageMeter):
            out_str += f"{key}:  {val.val} "
        else:
            out_str += f"{key}:  {val} "
    print(out_str)

def write_metrics_tf(metrics_dict, writer, tb_step):
    for key, val in metrics_dict.items():
        if isinstance(val, AverageMeter):
            writer.add_scalar(str(key), val.val, tb_step)
        else:
            writer.add_scalar(str(key), val, tb_step)

class MyDataset(Dataset):
    def __init__(self, dataset, latents, edit_sm, editor):
        self.latents = latents
        if dataset == "FFHQ":
            base_path = "/visinf/home/ksingh/StyleFlow/dataset/ffhq/"
            self.cat = np.load("./hardness_vis/cat.pkl", allow_pickle=True)
            self.editor = editor
            self.edit_sm = edit_sm
            # Change this outside of the if blk 
            self.easy = self.cat["easy"]
            self.med= self.cat["med"]
            self.hard = self.cat["hard"]
        
        self.img_path = f"{base_path}/imgs"
        self.gray_sm_path = f"{base_path}/gray_sm"
        self.color_sm_path = f"{base_path}/color_sm"
      

    def transform(self, gray_sm, color_sm, itemid):
        if self.edit_sm == "edit":
            num_edits = np.random.choice([0, 1, 2, 3])
            # num_edits = 0 
            if num_edits == 0:
                gray_sm_e, color_sm_e = gray_sm, color_sm
            else:
                gray_sm_e, color_sm_e , _ = self.editor.make_edits(itemid, num_edits)
                color_sm_e = pil2tensor(color_sm_e)
            return gray_sm_e, color_sm_e 
        elif self.edit_sm == "random":
            def get_sm(idx):
                if idx == -1:
                    return ()
                color_sm = PIL.Image.open(f"{self.color_sm_path}/{idx}.png")
                color_sm = pil2tensor(color_sm)
                target = PIL.Image.open(f"{self.gray_sm_path}/{idx}.png")
                gray_sm = torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)
                return (gray_sm, color_sm)
            
            easy_idx, med_idx, hard_idx = [itemid, itemid, itemid]  
            if itemid in self.easy:
                easy_idx = np.random.choice(self.cat["easy"][itemid])
            if itemid in self.med:
                med_idx = np.random.choice(self.cat["med"][itemid])
            if itemid in self.hard:
                hard_idx = np.random.choice(self.cat["hard"][itemid])

            easy_sm = get_sm(easy_idx)
            med_sm = get_sm(med_idx)
            hard_sm = get_sm(hard_idx)
            return (easy_sm[0::2], med_sm[0::2], hard_sm[0::2]), (easy_sm[1::2], med_sm[1::2], hard_sm[1::2])
        else:
            color_sm = torchvision.transforms.functional.normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(color_sm)
            return gray_sm, color_sm 

    def __getitem__(self, index):
        x = torch.tensor(self.latents[index])
        img = PIL.Image.open(f"{self.img_path}/{index}.png")
        img = img.resize((512, 512), PIL.Image.BILINEAR)
        img = pil2tensor(img)

        color_sm = PIL.Image.open(f"{self.color_sm_path}/{index}.png")
        color_sm = pil2tensor(color_sm)
        color_sm = torchvision.transforms.functional.normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(color_sm)

        target = PIL.Image.open(f"{self.gray_sm_path}/{index}.png")
        gray_sm = torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)
        
        gray_sm_e, color_sm_e = self.transform(gray_sm, color_sm, index)
        return x, img, gray_sm, color_sm, gray_sm_e, color_sm_e

    def __len__(self):
        return len(self.latents)

class MyDatasetLMDB(Dataset):
    def __init__(self, db_path, dataset, latents, editor, edit_sm, sm_size=(512, 512)):
        self.db_path = db_path
        self.env = None
        self.latents = latents
        self.length = len(latents)
        self.dataset = dataset 
        self.sm_size = sm_size

        if dataset == "FFHQ":
            base_path = "/fastdata/ksingh/dataset/ffhq/"
            self.editor = editor
            self.edit_sm = edit_sm
            if self.edit_sm is not None:
                self.cat = np.load("/fastdata/ksingh/dataset/ffhq/cat_9k.pkl", allow_pickle=True)
                # Change this outside of the if blk 
                self.easy = self.cat["easy"]
                self.med= self.cat["med"]
                self.hard = self.cat["hard"]

        if self.env is None:
            self._init_db()
    
    def transform(self, gray_sm, color_sm, itemid):
        if self.edit_sm == "edit": # we make this edit in the mail file
            if self.dataset == "FFHQ":
                # print("Inside Edit-Here")
                easy_sm = self.editor.make_edits_v1(itemid, 1, gray_sm, color_sm)
                med_sm = self.editor.make_edits_v1(itemid, 2, gray_sm, color_sm)
                hard_sm = self.editor.make_edits_v1(itemid, 3, gray_sm, color_sm)
                return (easy_sm[0::2], med_sm[0::2], hard_sm[0::2]), (easy_sm[1::2], med_sm[1::2], hard_sm[1::2])

        elif self.edit_sm == "random":
            # print("Inside Random-Here")
            def get_sm(idx):
                if idx == -1:
                    return ()
                _, gray_sm, color_sm, imgid = self.get_item(idx)
                # assert idx == imgid
                return (gray_sm, color_sm)

            easy_idx, med_idx, hard_idx = [itemid, itemid, itemid]  
            if itemid in self.easy:
                easy_idx = np.random.choice(self.cat["easy"][itemid])
            if itemid in self.med:
                med_idx = np.random.choice(self.cat["med"][itemid])
            if itemid in self.hard:
                hard_idx = np.random.choice(self.cat["hard"][itemid])
            easy_sm = get_sm(easy_idx)
            med_sm = get_sm(med_idx)
            hard_sm = get_sm(hard_idx)
            return (easy_sm[0::2], med_sm[0::2], hard_sm[0::2]), (easy_sm[1::2], med_sm[1::2], hard_sm[1::2])
        else:
            # print("Inside Random")
            return gray_sm, color_sm 

    
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
        img = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape).copy()
        img = torchvision.transforms.functional.to_tensor(img)

        color_sm = np.frombuffer(color_sm_arr, dtype=np.uint8).reshape(img_shape).copy()
        color_sm = torchvision.transforms.functional.to_tensor(color_sm)
 
        gray_sm = np.frombuffer(gray_sm_arr, dtype=np.uint8).reshape(self.sm_size).copy()
        gray_sm = torch.as_tensor(gray_sm, dtype=torch.int64).unsqueeze(0)

        return img, gray_sm, color_sm, imgid

    def __getitem__(self, index):
        x = torch.tensor(self.latents[index])
        img, gray_sm, color_sm, _ = self.get_item(index)
        if gray_sm.max() >= 150:
            print(f"{__file__}, gray_sm.max(): {gray_sm.max()}")
        gray_sm_e, color_sm_e = self.transform(gray_sm, color_sm, index)

        return index, x, img, gray_sm, color_sm, gray_sm_e, color_sm_e

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class TestDataset(Dataset):
    def __init__(self, dataset, latents, img, color_sm, gray_sm, is_conditional, transform=None, is_version_1=False):
        self.latents = latents
        self.img_path = img
        self.color_sm_path = color_sm
        self.gray_sm_path = gray_sm

        import glob

        fns= list(map(lambda x: int(x.split("/")[-1][:-4]), glob.glob(f"{self.color_sm_path}/*.png")))
        self.actual_idxs = sorted(fns)
        self.transform = transform
        if dataset == "FFHQ":
            if os.path.exists("/data/dataset/ffhq/ffhq_test/gray_sm"):
                self.gray_sm_unedited = "/data/dataset/ffhq/ffhq_test/gray_sm/"
                self.color_sm_unedit = "/data/dataset/ffhq/ffhq_test/color_sm/"
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

        color_sm_unedit = pil2tensor(PIL.Image.open(f"{self.color_sm_unedit}/{index}.png"))
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

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                    (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                    (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                    (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153), 
                    (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)], 
                    dtype=np.uint8) 
    return cmap

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (image_numpy + 1) / 2.0
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]

    return image_numpy

class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    #label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy

def generate_label(inputs, imsize):

    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))
    # TODO: Remove dtype                
    label_batch = np.array(label_batch, dtype=np.float32)
    label_batch = torch.from_numpy(label_batch)	

    return label_batch

def segmap_to_img(parsing_annos, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    outs = []
    for parsing_anno in parsing_annos:
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) 

        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = (pi, pi, pi)

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

        # Save result or not
        if save_im:
            cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno_color)
            cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            outs.append(vis_parsing_anno_color.transpose(2, 0, 1))

    return  torch.from_numpy(np.stack(outs)).type(torch.float32), num_of_class

def download_file(session, file_spec, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            break

        except:
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'export=download' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

def concat_images(image_list, size, shape=None):
    width, height = size, size
    images = []
    segmaps = []
    for idx in range(len(image_list[0])):
        img = image_list[1][idx].copy()
        segmap = image_list[0][idx].copy()
        img = img.resize((width, height))
        # images.append(img.resize(width, height))
        images.append(img)
        segmaps.append(segmap)
               
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = PIL.Image.new('RGB', image_size)
    
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            if col == 0:
                image.paste(images[row], offset)
            if col == 1:
                image.paste(segmaps[row], offset)

    return np.array(image)
