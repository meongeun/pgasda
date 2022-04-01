from __future__ import print_function
import torch
import time
import numpy as np
from PIL import Image
import os

# save image to the disk
from matplotlib import pyplot as plt


def save_images(visuals, results_dir, ind):
    
    for label, im_data in visuals.items():
        
        img_path = os.path.join(results_dir, '%.3d_%s.png' % (ind, label))
        if 'depth' in label:
            pass
        else:
            image_numpy = tensor2im(im_data)
            save_image(image_numpy, img_path, 'RGB')

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1)
    image_numpy = image_numpy / (2.0 / 255.0)
    return image_numpy.astype(imtype)

def tensor2depth(input_depth, imtype=np.int32):
    if isinstance(input_depth, torch.Tensor):
        depth_tensor = input_depth.data
    else:
        return input_depth
    depth_numpy = depth_tensor[0].cpu().float().numpy()
    # print(depth_numpy.max(), depth_numpy.min())
    depth_numpy += 1.0
    depth_numpy /= 2.0
    depth_numpy *= 65535.0
    depth_numpy = depth_numpy.reshape((depth_numpy.shape[1], depth_numpy.shape[2]))
    return depth_numpy.astype(imtype)

def tensor2depthmap(input_depth, imtype=np.int32):
    if isinstance(input_depth, torch.Tensor):
        depth_tensor = input_depth.data
    else:
        return input_depth
    depth_numpy = depth_tensor[0].cpu().float().numpy()
    # depth_numpy += 1.0
    # depth_numpy /= 2.0
    # depth_numpy *= 65535.0
    depth_numpy = depth_numpy.reshape((depth_numpy.shape[1], depth_numpy.shape[2]))
    return depth_numpy.astype(imtype)

def tensor2depthnorm(input_depth, fpath, name):
    if isinstance(input_depth, torch.Tensor):
        depth_tensor = input_depth.data
    else:
        return input_depth
    depth_numpy = depth_tensor[0].cpu().float().numpy()
    # print(np.max(depth_numpy), np.min(depth_numpy))
    sample = np.transpose(depth_numpy, [1,2,0])
    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
    plt.figure()
    f, ax = plt.subplots(1, 1)
    # big_img = show_multi_img(imgs[key])
    ax.imshow(sample[:,:,0])
    ax.axis('off')
    ax.set_title(name)

    plt.savefig(os.path.join(fpath,name))

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, imtype):
    image_pil = Image.fromarray(image_numpy, imtype)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

class SaveResults:
    def __init__(self, opt):
       
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.expr_name, 'image')
        mkdirs(self.img_dir) 
        self.log_name = os.path.join(opt.checkpoints_dir, opt.expr_name, 'loss_log.txt')
        self.index = 0
        self.ind = 0
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def save_current_results(self, visuals, epoch):
        fpath = os.path.join(self.img_dir,'epoch%.3d' % epoch)
        if not os.path.isdir(fpath):
            mkdirs(fpath)

        for label, keys in visuals.items():
                for key in keys.keys():
                    if "img" in label:
                        name = 'epoch%.3d/%d_%s_%s.png' % (epoch, self.ind, label, key)
                        if "color" in key:
                            img_path = os.path.join(self.img_dir, name)
                            image_numpy = tensor2im(keys[key])
                            save_image(image_numpy, img_path, 'RGB')
                            self.index += 1
                    elif "gt" in label:
                        name = 'epoch%.3d/%d_%s_%s.png' % (epoch, self.ind, label, key)
                        name2 = '%d_%s_%s_norm.png' % (self.index, label, key)
                        if "depth" in key:
                            img_path = os.path.join(self.img_dir, name)
                            depth_numpy = tensor2depth(keys[key])
                            tensor2depthnorm(keys[key], fpath, name2)
                            # depth_numpy = tensor2depthmap(keys[key])
                            # depth_numpy = tensor2im(keys[key])
                            save_image(depth_numpy, img_path, 'I')
                            self.index += 1
                        else:
                            pass
                            # save_txt()
                    elif "gen" in label:
                        name = 'epoch%.3d/%d_%s_%s.png' % (epoch, self.ind, label, key)
                        name2 = '%d_%s_%s_norm.png' % (self.index, label, key)
                        if "depth" in key:
                            img_path = os.path.join(self.img_dir, name)
                            depth_numpy = tensor2depth(keys[key])
                            tensor2depthnorm(keys[key], fpath, name2)
                            # depth_numpy = tensor2depthmap(keys[key])
                            # depth_numpy = tensor2im(keys[key])
                            save_image(depth_numpy, img_path, 'I')
                            self.index += 1
                    else:
                        continue
        self.ind += 1


            

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, lr, losses, t, t_data):
          
        message = '(epoch: %d, iters: %d, lr: %e, time: %.3f, data: %.3f) ' % (epoch, i, lr, t, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
