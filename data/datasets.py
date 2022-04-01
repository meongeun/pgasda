import os
import re
import cv2
import glob
import PIL
import math
import torch
import random
import collections
import torchvision.transforms.functional as F
import PIL.Image as pil

import os.path as osp
import numpy as np

from itertools import islice

from PIL import Image
from PIL import ImageOps
from matplotlib import pyplot as plt
from torch.utils import data

from data.mono_dataset import MonoDataset
from data.transform import RandomHorizontalFlip
from utils.dataset_util import KITTI
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        dd = {}
        {dd.update(d[i]) for d in self.datasets if d is not None}

        return dd

    def __len__(self):
        return max(len(d) for d in self.datasets if d is not None)

class KittiGeneral(data.Dataset):
    def __init__(self, root='./datasets', data_file='tgt_train.list', phase='train', img_transform=None, depth_transform = None,
        joint_transform = None, size=None):
        self.root = root
        self.data_file = data_file
        self.files = []
        self.phase = phase
        self.to_tensor = transforms.ToTensor()
        self.frame_idxs = [0, -1, 1]
        self.size = size
        self.img_transform = img_transform
        self.depth_transform = depth_transform
        self.joint_transform = joint_transform

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)

        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

    def __len__(self):
        return len(self.files)

    def img_aug(self, inputs, color_aug):
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = f
                inputs[(n + "_aug", im, i)] = color_aug(f)

    def PIL2tensor(self, inputs):
        for key in inputs.keys():
            if str(type(inputs[key]))=="<class 'PIL.Image.Image'>":
                inputs[key] = self.to_tensor(inputs[key])

    def scale(self, inputs, include_depth=False):
        '''
        Args:
            inputs - dict
                        [('color', )]
                        [('depth_gt', )]
        '''
        for key in inputs.keys():
            if str(type(inputs[key]))=="<class 'PIL.Image.Image'>":
                h = inputs[key].height
                w = inputs[key].width
                break

        if self.size is None:
            divisor = 32.0
            h = int(math.ceil(h/divisor) * divisor)
            w = int(math.ceil(w/divisor) * divisor)
            self.size = (h, w)
       
        scale_transform = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
        for key in inputs.keys():
            if str(type(inputs[key]))=="<class 'PIL.Image.Image'>":
                inputs[key] = scale_transform(inputs[key])
                inputs[key] = self.to_tensor(inputs[key])


class VKittiDataset(KittiGeneral):
    def __init__(self, root='./datasets', data_file='src_train.list', phase='train', img_transform = None, depth_transform = None, joint_transform = None, pose_transform=None, size=None):
        super(VKittiDataset, self).__init__(root=root, data_file=data_file, phase=phase, size=size, img_transform = img_transform, depth_transform = depth_transform, joint_transform = joint_transform)
        self.pose_transform = pose_transform

        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_info = data.split(' ')                
                pose_info = data_info[0].split('/')
                # pose = read_txt("gt/"+pose_info[1]+"_"+pose_info[2]+".txt")
                self.files.append({
                    "rgb": data_info[0],
                    "depth": data_info[1],
                    "pose": "gt/"+pose_info[1]+"_"+pose_info[2]+".txt",
                    "frame_index": int(re.sub(r'[^0-9]', '', pose_info[3]))
                    })
    
    def read_data(self, datafiles):
        assert osp.exists(osp.join(self.root, datafiles['rgb'])), "Image does not exist"
        rgb = Image.open(osp.join(self.root, datafiles['rgb'])).convert('RGB')
        assert osp.exists(osp.join(self.root, datafiles['depth'])), "Depth does not exist"                
        depth = Image.open(osp.join(self.root, datafiles['depth']))
        assert osp.exists(osp.join(self.root, datafiles['pose'])), "Depth does not exist"
        q,t = self.read_pose_txt(osp.join(self.root, datafiles['pose']), datafiles['frame_index'])

        return rgb, depth, q,t

    def read_pose_txt(self, filename, frame_index):
        f = open(filename, 'r')
        lines = f.readlines()
        e = lines[frame_index+1].split(" ")
        e = [float(i) for i in e]
        r = np.array([[e[1], e[2], e[3]],[e[5], e[6], e[7]], [e[9], e[10], e[11]]])
        t = np.array([e[4], e[8], e[12]])
        q = R.from_matrix(r).as_quat()

        return q, t

    def __getitem__(self, index):
        data = {}
        imgs = {}
        line = self.files[index]
        frame_index = line['frame_index']
        depth = Image.open(osp.join(self.root, line['depth']))

        for i in self.frame_idxs:
            fpath = "/".join(line['rgb'].split("/")[:-1]) + "/{:05d}.png".format(frame_index + i)
            data[("color", i, -1)] = Image.open(osp.join(self.root, fpath)).convert('RGB')
            imgs[("color", i, -1)] = data[("color", i, -1)]

        if self.joint_transform is not None:
            if self.phase == 'train':
                imgs, depth, _ = self.joint_transform((imgs, depth, 'train', None))
            else:
                imgs, depth, _ = self.joint_transform((imgs, depth, 'test', None))
        for i in self.frame_idxs:
            if self.img_transform is not None:
                data[("color", i, -1)] = self.img_transform(imgs[("color", i, -1)])

        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        # if self.phase =='test':
        #     data = {}
        #     data['img'] = l_img
        #     data['depth'] = depth
        #     return data

        if depth is not None:
            data['depth'] = depth

        q = {}
        t = {}
        for f_i in self.frame_idxs:
            q[f_i], t[f_i] = self.read_pose_txt(osp.join(self.root, line['pose']), line['frame_index'] + f_i)

        data[("axisangle", 0, -1)] = q[0] - q[-1]
        data[("translation", 0, -1)] = t[0] - t[-1]

        data[("axisangle", 0, 1)] = q[1] - q[0]
        data[("translation", 0, 1)] = t[1] - t[0]

        if self.pose_transform is not None:
            for key in data.keys():
                if "axisangle" in key or "translation" in key:
                    data[key] = self.pose_transform(data[key]).float()

        #for k, v in data.items():
            #print(k, v.dtype)
            #data[k] = v.double()
            #data[k] = v.type(torch.DoubleTensor)
        
        return {'src': data}

class KittiDataset(KittiGeneral):
    def __init__(self, root='./datasets', data_file='tgt_train.list', phase='train',img_transform=None,
                     depth_transform=None, joint_transform=None, pose_transform=None, size=None):
        super(KittiDataset, self).__init__(root=root, data_file=data_file, phase=phase, size=size, img_transform=img_transform, depth_transform=depth_transform,
                                                joint_transform=joint_transform)

        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_info = data.split(' ')
                rgb_info =data_info[0].split("/")
                self.files.append({
                    "rgb": data_info[0],
                    "cam_intrin": data_info[2],
                    "depth": data_info[3],
                    "frame_index": int(re.sub(r'[^0-9]', '', rgb_info[4]))
                    })

    def read_data(self, datafiles, h, w):
        kitti = KITTI()
        # assert osp.exists(osp.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
        # k = kitti.get_k(osp.join(self.root, datafiles['cam_intrin']))
        k = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        #float32
        k[0,:] *= w
        k[1,:] *= h
        inv_k = np.linalg.pinv(k)

        k = torch.from_numpy(k)
        inv_k = torch.from_numpy(inv_k)

        assert osp.exists(osp.join(self.root, datafiles['depth'])), "Depth does not exist"
        depth = kitti.get_depth(calib_dir=osp.join(self.root, datafiles['cam_intrin']), velo_file_name=osp.join(self.root, datafiles['depth']), im_shape=[h, w])
        # debug
        # fpath = osp.join(self.root, datafiles['rgb']).replace('.png', '_disp.jpeg')
        # depth = Image.open(fpath).convert("L")

        assert osp.exists(osp.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
        fb = kitti.get_fb(osp.join(self.root, datafiles['cam_intrin']))

        return fb, depth, k, inv_k

    def disp_to_depth(self, disp):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        depth = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))*2.0 -1.0

        plt.figure()
        f, ax = plt.subplots(1, 1)
        # big_img = show_multi_img(imgs[key])
        ax.imshow(depth)
        ax.axis('off')
        ax.set_title('tgt_depth_gt')

        plt.savefig(os.path.join('./', 'tgt_depth_gt'))

        return depth

    def joint_kitti_transform(self, inputs):
        imgs = inputs[0]
        depth = inputs[1]
        phase = inputs[2]
        fb = inputs[3]

        h = imgs[("color", 0, -1)].height
        w = imgs[("color", 0, -1)].width
        w0 = w

        if self.size == [-1]:
            divisor = 32.0
            h = int(math.ceil(h / divisor) * divisor)
            w = int(math.ceil(w / divisor) * divisor)
            self.size = (h, w)

        scale_transform = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])

        for i in [0, -1, 1]:
            imgs[("color", i, -1)] = scale_transform(imgs[("color", i, -1)])

        if fb is not None:
            scale = float(self.size[1]) / float(w0)
            fb = fb * scale

        if phase == 'test':
            return imgs, depth, fb

        if depth is not None:
            scale_transform_d = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
            depth = scale_transform_d(depth)

        flip_prob = random.random()
        flip_transform = transforms.Compose([RandomHorizontalFlip(flip_prob)])
        if flip_prob < 0.5:
            for i in [0, -1, 1]:
                imgs[("color", i, -1)] = flip_transform(imgs[("color", i, -1)])
        if depth is not None:
            depth = flip_transform(depth)

        if not self.size == 0:

            if depth is not None:
                arr_depth = np.array(depth, dtype=np.float32)
                depth = self.disp_to_depth(arr_depth)

        if depth is not None:
            depth = np.array(depth, dtype=np.float32)
            depth = depth * 2.0
            depth -= 1.0


        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.0)
            contrast = random.uniform(0.8, 1.0)
            saturation = random.uniform(0.8, 1.0)
            for i in [0, -1, 1]:
                imgs[("color", i, -1)] = F.adjust_brightness(imgs[("color", i, -1)], brightness)
                imgs[("color", i, -1)] = F.adjust_contrast(imgs[("color", i, -1)], brightness)
                imgs[("color", i, -1)] = F.adjust_saturation(imgs[("color", i, -1)], brightness)

        return imgs, depth, fb

    def __getitem__(self, index):
        data = {}
        imgs = {}
        line = self.files[index]
        frame_index = line['frame_index']

        for i in self.frame_idxs:
            fpath = "/".join(line['rgb'].split("/")[:-1]) + "/{:010d}.png".format(frame_index + i)
            data[("color", i, -1)] = Image.open(osp.join(self.root, fpath)).convert('RGB')
            imgs[("color", i, -1)] = data[("color", i, -1)]

        h, w = data[("color", 0, -1)].size
        fb, depth, data["K"], data["inv_K"] = self.read_data(line, h, w)

        if self.joint_transform is not None:
            if self.phase == 'train':
                # imgs, depth, fb = self.joint_transform((imgs, None, 'train', fb))
                # debug
                imgs, _, fb = self.joint_kitti_transform((imgs, None, 'train', fb))

            else:
                imgs, _, fb = self.joint_transform((imgs, None, 'train', fb))
                # l_img, r_img, _, fb = self.joint_transform((imgs, None, 'test', fb))

        for i in self.frame_idxs:
            if self.img_transform is not None:
                data[("color", i, -1)] = self.img_transform(imgs[("color", i, -1)])


        # if self.depth_transform is not None and depth is not None:
        #     depth = self.depth_transform(depth)
        # print(depth)
        if self.phase == 'test':
        #     data = {}
        #     data['left_img'] = l_img
        #     data['right_img'] = r_img
            data['depth'] = depth
        #     data['fb'] = fb
            return data

        # if depth is not None:
        #     data['depth'] = depth


        return {'tgt': data}

class MonoKITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(MonoKITTIDataset, self).__init__(*args, **kwargs)
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}


    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class KITTIOdomDataset(MonoKITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        # print('image_path',image_path)
        return image_path

def get_dataset(root, data_file='train.list', phase='train', dataset='kitti', pose_transform=None, img_transform=None, depth_transform=None,
                            joint_transform=None, size=None):
                # (root, data_file='train.list', dataset='kitti', phase='train', pose_transform=None, test_dataset='kitti', size=None):

    DEFINED_DATASET = {'KITTI', 'VKITTI'}
    assert dataset.upper() in DEFINED_DATASET
    name2obj = {'KITTI': KittiDataset,
                'VKITTI': VKittiDataset,
        }
    #TODO 
    if phase == 'test' :
       name2obj['KITTI'] = KittiDataset

    return name2obj[dataset.upper()](root=root, data_file=data_file, phase=phase, pose_transform=pose_transform, img_transform=img_transform, depth_transform=depth_transform,
                            joint_transform=joint_transform, size=size)

