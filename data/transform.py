import collections
import math
import numbers
import random

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class RandomHorizontalFlip(object):
    """
    Random horizontal flip.

    prob = 0.5
    """

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, img):
        if (self.prob is None and random.random() < 0.5) or self.prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        
        return img


class RandomVerticalFlip(object):
    """
    Random vertical flip.

    prob = 0.5
    """

    def __init__(self, prob=None):
        self.prob = prob
        
    def __call__(self, img):

        if (self.prob is None and random.random() < 0.5) or self.prob < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomPairedCrop(object):

    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        """
        Get parameters for ``crop`` for a random crop.
        Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
        Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        img1 = img[0]
        img2 = img[1]
        depth = img[2] 
        
        i, j, th, tw = self.get_params(img1, self.size)

        img1 = F.crop(img1, i, j, th, tw)

        if depth is not None:
            depth = F.crop(depth, i, j, th, tw)
        if img2 is not None:
            img2 = F.crop(img2, i, j, th, tw)
        return img1, img2, depth 

        
class RandomImgAugment(object):
    """Randomly shift gamma"""

    def __init__(self, no_flip,no_augment,size=None):

        self.flip = not no_flip
        self.augment = not no_augment
        self.size = size


    def __call__(self, inputs):

        imgs = inputs[0]
        depth = inputs[1]
        phase = inputs[2]
        fb = inputs[3]

        h = imgs[("color", 0, -1)].height
        w = imgs[("color", 0, -1)].width
        w0 = w

        if self.size == [-1]:
            divisor = 32.0
            h = int(math.ceil(h/divisor) * divisor)
            w = int(math.ceil(w/divisor) * divisor)
            self.size = (h, w)
       
        scale_transform = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])

        for i in [0, -1, 1]:
            imgs[("color",i, -1)] = scale_transform(imgs[("color",i, -1)])

        if fb is not None:
            scale = float(self.size[1]) / float(w0)
            fb = fb * scale

        # if phase == 'test':
        #     return imgs, depth, fb
        if depth is not None:
           scale_transform_d = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
           depth = scale_transform_d(depth)


        if not self.size == 0:
    
            if depth is not None:
                arr_depth = np.array(depth, dtype=np.float32)
                arr_depth /= 65535.0  # cm->m, /10
                arr_depth[arr_depth<0.0] = 0.0
                depth = Image.fromarray(arr_depth, 'F')

        if self.flip:
            
            flip_prob = random.random()
            flip_transform = transforms.Compose([RandomHorizontalFlip(flip_prob)])
            if flip_prob < 0.5:
                for i in [0, -1, 1]:
                    imgs[("color", i, -1)] = flip_transform(imgs[("color", i, -1)])
            if depth is not None:
                depth = flip_transform(depth)

        if depth is not None:
            depth = np.array(depth, dtype=np.float32)   
            depth = depth * 2.0
            depth -= 1.0

        if self.augment:
            if random.random() < 0.5:

                brightness = random.uniform(0.8, 1.0)
                contrast = random.uniform(0.8, 1.0)
                saturation = random.uniform(0.8, 1.0)
                for i in [0, -1, 1]:
                    imgs[("color", i, -1)] = F.adjust_brightness(imgs[("color", i, -1)], brightness)
                    imgs[("color", i, -1)] = F.adjust_contrast(imgs[("color", i, -1)], brightness)
                    imgs[("color", i, -1)] = F.adjust_saturation(imgs[("color", i, -1)], brightness)

        return imgs, depth, fb

class DepthToTensor(object):
    def __call__(self, input):
        # tensors = [], [0, 1] -> [-1, 1]
        arr_input = np.array(input)
        tensors = torch.from_numpy(arr_input.reshape((1, arr_input.shape[0], arr_input.shape[1]))).float()
        return tensors

class PoseToTensor(object):
    def __call__(self, input):
        # tensors = [], [0, 1] -> [-1, 1]
        arr_input = np.array(input).astype(np.float64)
        tensors = torch.from_numpy(arr_input)
        return tensors

