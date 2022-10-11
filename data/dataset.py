import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import cv2
import torch
import numpy as np
from os.path import exists, join, split, splitext
from os import listdir
from torchvision import transforms as T


from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        # update the dataloader here
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image # the input image
        ret['mask_image'] = mask_img
        ret['mask'] = mask # the mask is not usefull in our task now
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

# just a simple dataset that removes the mask settings
class DenoisingDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[288, 288]):
        with open(data_root, "r") as f:
            imgs = f.readlines()
        imgs = [img.replace("\n", "") for img in imgs]
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        # we don't want to resize, we will crop
        # but we have to make sure the crop is at 
        # the same location at both train and gt images
        self.tfs = transforms.Compose([
                transforms.RandomCrop((image_size[0], image_size[1])),
        ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        train = self.imgs[index]
        gt = train.replace("train", "gt")
        aa = self.to_tensor(Image.open(train))
        if aa.shape[0] == 3:
            aa = aa.mean(dim = 0, keepdim = True)
        bb = self.to_tensor(Image.open(gt))
        if bb.shape[0] == 3:
            bb = bb.mean(dim = 0, keepdim = True)
        img = torch.cat((aa, bb))
        img = self.tfs(img)
        cond_img, img = torch.chunk(img, 2, 0)
        cond_img = self.normalize(cond_img)
        img = self.normalize(img)
        ret['gt_image'] = img
        ret['cond_image'] = cond_img # the input image
        ret['path'] = train.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

class ShadingDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[256, 256], resize = 512, val = False):
        self.img_path = data_root
        self.resize = resize
        self.val = val
        if exists(join(self.img_path, "img_list_dm.txt")) == False:
            self.scan_imgs()
        with open(join(self.img_path, "img_list_dm.txt"), 'r') as f:
            imgs = f.readlines()

        imgs = [img.replace("\n", "") for img in imgs]
        if val:
            self.imgs = imgs[-8:]
        else:
            self.imgs = imgs[:-8]
        self.image_size = image_size
        self.to_dir_label = {"right": 0.25, "left":0.5, "back":0.75, "top":1.0}
        self.lable_flip = {0.25:0.5, 0.5:0.25}

    def remove_alpha(self, img, gray = False):
        # assum the img is numpy array
        # but we should be able to deal with different kind input images
        if len(img.shape) == 3:
            h, w, c = img.shape
            if c == 4:
                alpha = np.expand_dims(img[:, :, 3], -1) / 255
                whit_bg = np.ones((h, w, 3)) * 255
                img_res = img[:, :, :3] * alpha + whit_bg * (1 - alpha)
                if gray:
                    img_res = img_res.mean(axis = -1)
            else:
                img_res = img
        else:
            img_res = img
        return img_res

    def __len__(self):
        return len(self.imgs)

    def scan_imgs(self):
        # helper function to scan the full dataset
        print("Log:\tscan the %s"%self.img_path)
        imgs = []
        img_path = join(self.img_path, "img")
        for img in listdir(img_path):
            if "line" in img:
                if exists(join(img_path, img.replace("line", "flat"))) and\
                    exists(join(img_path, img.replace("line", "shadow"))):
                    imgs.append(join(img_path, img))
        with open(join(self.img_path, "img_list_dm.txt"), 'w') as f:
            f.write('\n'.join(imgs))
        print("Log:\tdone")

    def down_sample(self, img):
        dw = int(img.shape[1] / 2)
        dh = int(img.shape[0] / 2)
        img_d2x = cv2.resize(img, (dw, dh), interpolation = cv2.INTER_NEAREST)
        return img_d2x
    
    def random_bbox(self, img):
        h, w = img.shape[0], img.shape[1]
        # we generate top, left, bottom, right
        t = np.random.randint(0, h - self.image_size[0] - 1)
        l = np.random.randint(0, w - self.image_size[1] - 1)
        return (t, l, t + self.image_size[0], l + self.image_size[1])

    def crop(self, imgs, bbox):
        t, l, b, r = bbox
        res = []
        for img in imgs:
            res.append(img[t:b, l:r, ...])
        return res

    def random_flip(self, imgs, label, p = 0.5):
        # we only consider the horizontal flip
        dice = np.random.uniform()
        if dice < p:
            flipped = []
            for img in imgs:
                # flip the image and label
                flipped.append(np.flip(img, axis = 1))
            label = self.lable_flip.get(label, label)
        else:
            # don't change anything
            flipped = imgs
        return flipped, label
    
    def resize_hw(self, h, w):
        # we resize the shorter edge to the target size
        if h > w:
            ratio =  h / w
            h = int(self.resize * ratio)
            w = self.resize
        else:
            ratio = w / h
            w = int(self.resize * ratio)
            h = self.resize
        return h, w

    def to_tensor(self, img_np, normalize = True):
        # assume the input is always grayscal
        if normalize:
            transforms = T.Compose(
                    [
                        T.ToTensor(),
                        T.Normalize(0.5, 0.5, inplace = True)
                    ]
                )
        else:
            transforms = T.Compose(
                    [
                        T.ToTensor()
                    ]
                )
        return transforms(img_np)

    def __getitem__(self, index):
        ret = {}

        # get image path
        line_path = self.imgs[index].strip("\n")
        flat_path = line_path.replace("line", "flat")
        shad_path = line_path.replace("line", "shadow")
        
        # get label
        _, n = split(line_path)
        label = n.split("_")[1]
        label = self.to_dir_label[label]

        # open images
        assert exists(line_path), \
            f'No line art found for the ID {index}: {line_path}'
        assert exists(flat_path), \
            f'No flat found for the ID {index}: {flat_path}'
        assert exists(shad_path), \
            f'No shadow found for the ID {index}: {shad_path}'
        line_np = np.array(Image.open(line_path))
        flat_np = np.array(Image.open(flat_path))
        shad_np = np.array(Image.open(shad_path))

        # merge line and flat
        flat_np = self.remove_alpha(flat_np)
        flat_np = flat_np * (1 - np.expand_dims(line_np[:, :, 3], axis = -1) / 255)
        # flat_np = cv2.cvtColor(flat_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # line_np = 255 - line_np[:, :, 3] # remove alpha channel, but yes, we use alpha channel as the line drawing
        shad_np = self.remove_alpha(shad_np, gray = True) 
        _, shad_np = cv2.threshold(shad_np, 127, 255, cv2.THRESH_BINARY)

        # resize opened images
        h, w = shad_np.shape
        h, w = self.resize_hw(h, w)
        flat_np = cv2.resize(flat_np, (w, h), interpolation = cv2.INTER_AREA)
        # line_np = cv2.resize(line_np, (w, h), interpolation = cv2.INTER_AREA)
        shad_np = cv2.resize(shad_np, (w, h), interpolation = cv2.INTER_NEAREST)

        # random flip and crop to patches
        if self.val == False:
            img_list, label = self.random_flip([flat_np, shad_np], label)
            flat_np, shad_np = img_list
            bbox = self.random_bbox(flat_np)
            flat_np, shad_np = self.crop([flat_np, shad_np], bbox)
        
        # clip values
        flat_np = flat_np.clip(0, 255)
        shad_np = shad_np.clip(0, 255)

        # to tensors
        label = torch.Tensor([label]).expand(1, flat_np.shape[0], flat_np.shape[1])
        flat = self.to_tensor(flat_np / 255)
        flat = torch.cat((flat, label), dim = 0)
        shad = self.to_tensor(1 - shad_np / 255)

        # we should keep this part unchanged
        ret['gt_image'] = shad.float()
        ret['cond_image'] = flat.float() # the input image
        ret['path'] = line_path.rsplit("/")[-1].rsplit("\\")[-1]
        # ret['path'] = None
        return ret


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


