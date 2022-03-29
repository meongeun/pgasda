class VKittiDataset(KittiGeneral):
    def __init__(self, root='./datasets', data_file='src_train.list', phase='train', img_transform=None, depth_transform=None, pose_transform=None, joint_transform=None):
        super(vKittiDataset, self).__init__(root=root, data_file=data_file, phase=phase, img_transform=img_transform, joint_transform=joint_transform, depth_transform=depth_transform):
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
        pose = self.read_pose_txt(osp.join(self.root, datafiles['pose']), datafiles['frame_index'])

        return rgb, depth, pose

    def read_pose_txt(self, filename, frame_index):
        f = open(filename, 'r')
        lines = f.readlines()
        e = lines[frame_index+1].split(" ")
        r = [[e[1], e[2], e[3]],[e[5], e[6], e[7]], [e[9], e[10], e[11]]]
        t = [e[4], e[8], e[12]]
        pose = np.array([[e[1], e[2], e[3], e[4]],[e[5], e[6], e[7], e[8]], [e[9], e[10], e[11], e[12]], [0, 0, 0, 1]])
        return pose
        # pass

    def __getitem__(self, index):
        data = {}
        do_color_aug = self.phase == 'train' and random.random() > 0.5
        do_flip = self.phase == 'train' and random.random() > 0.5
        line = self.files[index]
        frame_index = line['frame_index']
        if frame_index == 0:
            return {'src': data}

        for i in self.frame_idxs:
            # print(frame_index + i)
            fpath = "/".join(line['rgb'].split("/")[:-1]) + "/{:05d}.png".format(frame_index+i)
            try:
                frgb = Image.open(osp.join(self.root, fpath)).convert('RGB')
                data[("color", i, -1)] = frgb
            except:
                return {'src': data}
            # data[("color", i, -1)] = Image.open(osp.join(self.root, line[frame_index+i]['color'])).convert('RGB')
            if do_flip:
                data[("color", i, -1)] = data[("color", i, -1)].transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            self.img_aug(data, color_aug)

        data["depth_gt"] = Image.open(osp.join(self.root, line['depth']))
        if do_flip:
            data["depth_gt"] = np.fliplr(data["depth_gt"])
        data["depth_gt"] = np.expand_dims(data["depth_gt"], 0)
        data["depth_gt"] = torch.from_numpy(data["depth_gt"].astype(np.float32))

        if self.phase == 'test':
            return data

        data["pose_gt"] = self.read_pose_txt(osp.join(self.root, line['pose']), line['frame_index'])
        if self.pose_transform is not None:
            data["pose_gt"] = self.pose_transform(data["pose_gt"])

        return {'src': data}

