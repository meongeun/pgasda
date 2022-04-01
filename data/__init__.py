import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from data.datasets import get_dataset, ConcatDataset
from data.transform import RandomImgAugment, DepthToTensor, PoseToTensor


def create_test_dataloader(args):

    joint_transform_list = [RandomImgAugment(True, True, args.loadSize)]
    img_transform_list = [ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])]

    joint_transform = Compose(joint_transform_list)
    
    img_transform = Compose(img_transform_list)
    
    depth_transform = Compose([DepthToTensor()])

    dataset = get_dataset(root=args.root, data_file=args.test_datafile, phase='test',
                        dataset=args.tgt_dataset, img_transform=img_transform, joint_transform=joint_transform,
                        depth_transform=None, size=args.loadSize)
    # for i in dataset:
    #     print(1,i['tgt'][('color', 0, -1)].shape)
    #     exit()
    loader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, drop_last=True,num_workers=int(args.nThreads),pin_memory=True)
    
    return loader

def create_train_dataloader(args):
    pose_transform = Compose([PoseToTensor()])
    joint_transform_list = [RandomImgAugment(args.no_flip, args.no_augment, args.loadSize)]
    img_transform_list = [ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])]

    joint_transform = Compose(joint_transform_list)

    img_transform = Compose(img_transform_list)

    depth_transform = Compose([DepthToTensor()])


    src_dataset = get_dataset(root=args.src_root, data_file=args.src_train_datafile, phase='train',
                            dataset=args.src_dataset, pose_transform=pose_transform, img_transform=img_transform, depth_transform=depth_transform,
                            joint_transform=joint_transform, size=args.loadSize)

        
    tgt_dataset = get_dataset(root=args.tgt_root, data_file=args.tgt_train_datafile, phase='train',
                            dataset=args.tgt_dataset, img_transform=img_transform, depth_transform=depth_transform,
                            joint_transform=joint_transform, size=args.loadSize)


    # loader = torch.utils.data.DataLoader(
    #                             ConcatDataset(
    #                                 src_dataset,
    #                                 tgt_dataset,
    #                             ),
    #                             batch_size=args.batchSize, shuffle=True,
    #                             num_workers=int(args.nThreads),
    #                             pin_memory=True)
    # print(type(src_dataset), type(tgt_dataset))

    # for i in src_dataset:
    #     print(i['src'].keys())
    # for i in tgt_dataset:
    #     print(i['tgt'].keys())
    # exit()
    src_train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batchSize, drop_last=True, shuffle=True,num_workers=int(args.nThreads),pin_memory=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_dataset, batch_size=args.batchSize, drop_last=True, shuffle=True,num_workers=int(args.nThreads), pin_memory=True)

    return src_train_loader, tgt_train_loader


def create_dataloader(args):
    if not args.isTrain:
        return create_test_dataloader(args)

    else:
        return create_train_dataloader(args)
   
