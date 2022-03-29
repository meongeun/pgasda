import torch.nn
from options.train_options import TrainOptions
from data import create_dataloader
import numpy as np
from matplotlib import pyplot as plt

import cv2

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

def show_multi_img(multi_img, gap=5):
    ndata, h, w, c = multi_img.shape

    big_h = ndata*h+ndata*(gap-1)

    big_img = np.ones([big_h, w, c], dtype=np.float32)

    for didx in range(ndata):
        for hidx in range(h):
            for widx in range(w):
                for cidx in range(c):
                    big_img[didx*(h + gap) + hidx][widx][cidx] = multi_img[didx][hidx][widx][cidx]

    return big_img


def show_imgs_batch(batch_data, path):
    # print(batch_data)
    # exit()
    key_list = batch_data.keys()

    imgs = dict()

    for key in key_list:
        sample = batch_data[key].numpy()
        if sample.ndim==4:
            sample = np.transpose(sample, [0,2,3,1])
            sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
            imgs[key] = sample 

    plt.figure()
    f, ax = plt.subplots(1, len(imgs))
    for idx, key in enumerate(imgs.keys()):
        big_img = show_multi_img(imgs[key])
        ax[idx].imshow(big_img)
        ax[idx].axis('off')
        ax[idx].set_title(key)

    plt.savefig(path)
    print(f"image saved in {path}")


if __name__ == '__main__':
    opt = TrainOptions().parse()
    src_train_loader, tgt_train_loader = create_dataloader(opt)
    train_dataset_size = len(src_train_loader)+ len(tgt_train_loader)
    print('#training images = %d' % train_dataset_size)
    # print(src_train_loader)
    for ind, data in enumerate(src_train_loader):
        data = data['src']
        print(data.keys())
        show_imgs_batch(data, f'{ind}.png')
        break
    #     print(data.keys())
        # show_img_depth(f'check', 0, data['src']['img'], data['src']['depth'])
    #     break
