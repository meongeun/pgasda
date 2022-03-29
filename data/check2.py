class KittiDataset(KittiGeneral):
    def __init__(self, root='./datasets', data_file='tgt_train.list', phase='train', img_transform=None, joint_transform=None, depth_transform=None, pose_transform=None):
        super(KittiDataset, self).__init__(root=root, data_file=data_file, phase=phase, img_transform=img_transform, joint_transform=joint_transform, depth_transform=depth_transform):
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
        self.isvkitti = False

    def read_data(self, datafiles, h, w):
        kitti = KITTI()
        # assert osp.exists(osp.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
        # k = kitti.get_k(osp.join(self.root, datafiles['cam_intrin']))
        k = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        k = torch.from_numpy(k)
        inv_k = np.linalg.pinv(k)
        inv_k = torch.from_numpy(inv_k)

        assert osp.exists(osp.join(self.root, datafiles['depth'])), "Depth does not exist"
        depth = kitti.get_depth(osp.join(self.root, datafiles['cam_intrin']),
                                osp.join(self.root, datafiles['depth']), [h, w])

        return depth, k,inv_k


    def __getitem__(self, index):
        data = {}
        do_color_aug = self.phase == 'train' and random.random() > 0.5
        do_flip = self.phase == 'train' and random.random() > 0.5
        line = self.files[index]
        frame_index = line['frame_index']
        if frame_index == 0:
            return {'tgt': data}

        for i in self.frame_idxs:
            fpath = "/".join(line['rgb'].split("/")[:-1]) + "/{:010d}.png".format(frame_index + i)
            try:
                frgb = Image.open(osp.join(self.root, fpath)).convert('RGB')
                data[("color", i, -1)] = frgb
            except:
                return {'tgt':data}
            # data[("color", i, -1)] = Image.open(osp.join(self.root, line[frame_index+i]['color'])).convert('RGB')
            if do_flip:
                data[("color", i, -1)] = data[("color", i, -1)].transpose(PIL.Image.FLIP_LEFT_RIGHT)

        w = data[("color", 0, -1)].size[0]
        h = data[("color", 0, -1)].size[1]

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            self.img_aug(data, color_aug)

        depth, data["K"], data["inv_K"] = self.read_data(line, h, w)

        if self.phase == 'test':
            # data["depth_gt"] = depth
            data["depth_gt"] = np.expand_dims(depth, 0)
            data["depth_gt"] = torch.from_numpy(data["depth_gt"].astype(np.float32))
            return data

        return {'tgt': data}

