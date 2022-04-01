import sys
import numpy as np
import torch
import itertools

from matplotlib import pyplot as plt

from debug import show_imgs_batch
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
import torch.nn.functional as F
from utils import dataset_util
from scipy.spatial.transform import Rotation as R
from networks import PoseCNN, PoseEncoderDecoder, DepthEncoderDecoder
import torchvision.transforms.functional

from .layers import *


class MonoFTModel(BaseModel):

    def name(self):
        return 'MonoFTModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_R_Depth', type=float, default=1.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_S_Depth', type=float, default=0.01, help='weight for smooth loss')

            parser.add_argument('--lambda_R_Pose', type=float, default=1.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_S_Pose', type=float, default=0.01, help='weight for smooth loss')
            
            parser.add_argument('--lambda_R_Img', type=float, default=1.0,help='weight for image reconstruction')
            
            parser.add_argument('--g_tgt_premodel', type=str, default="./cyclegan/G_Tgt.pth",help='pretrained G_Tgt model')

            parser.add_argument('--t_pose_premodel', type=str, default='',
                                help='models are saved here')
            parser.add_argument('--t_depth_premodel', type=str, default='',
                                help='models are saved here')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.src_img ={}
        self.tgt_img = {}
        self.fake_src_img = {}
        self.fake_tgt_img = {}

        self.src_gen = {}
        self.fake_src_gen = {}
        self.src_gt = {}
        self.tgt_gt = {}


        if self.isTrain:
            self.loss_names = ['R_Depth_Src',  'R_Pose_Src', 'S_Depth_Tgt', 'R_Img_Tgt' ]
          
        if self.isTrain:
            self.visual_names = ['src_img','src_gt','src_gen','tgt_img','fake_src_img','fake_src_gen', 'warp_fake_src_img']
        else:
            self.visual_names = ['pred', 'img']

        if self.isTrain:
            self.model_names = ['G_Depth_T', 'G_Pose_T', 'G_Tgt']
        else:
            self.model_names = ['G_Depth_T', 'G_Tgt']

        self.num_input_frames = len(self.opt.frame_idxs)
        self.num_pose_frames = 2
        # if self.opt.pose_model_input == "pairs" else self.num_input_frames

        # self.netG_Depth_T = networks.init_net(networks.UNetGenerator(norm='batch') , init_type='normal', gpu_ids=opt.gpu_ids)
        self.netG_Depth_T = networks.init_net(DepthEncoderDecoder(num_layers=opt.num_layers, pretrained=True).to(self.device))

        self.netG_Pose_T = networks.init_net(PoseEncoderDecoder(num_layers=opt.num_layers, pretrained=True, num_input_images=self.num_pose_frames, num_frames_to_predict_for=1, stride=1), init_type='normal', gpu_ids=opt.gpu_ids)
        self.netG_Pose_S = networks.init_net(PoseEncoderDecoder(num_layers=opt.num_layers, pretrained=True, num_input_images=self.num_pose_frames, num_frames_to_predict_for=1, stride=1), init_type='normal', gpu_ids=opt.gpu_ids)

        #self.netG_Pose_T = networks.init_net(PoseCNN(opt.num_frames), init_type='normal', gpu_ids=opt.gpu_ids)
        #self.netG_Pose_S = networks.init_net(PoseCNN(opt.num_frames), init_type='normal', gpu_ids=opt.gpu_ids)

        self.netG_Tgt = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            if self.opt.t_pose_premodel:
                self.init_with_pretrained_model('G_Pose_T', self.opt.t_pose_premodel)
            if self.opt.t_depth_premodel:
                # 처음만
                # self.init_with_first_pretrained_depth_model()
                # 이후에
                self.init_with_pretrained_depth_model('G_Depth_T', self.opt.t_depth_premodel)
            self.init_with_pretrained_model('G_Tgt', self.opt.g_tgt_premodel)
            self.netG_Tgt.eval()
         
        if self.isTrain:
            # define loss functions
            self.criterionDepthReg = torch.nn.L1Loss()
            self.criterionPoseReg = torch.nn.MSELoss()

            self.criterionSmooth = networks.SmoothLoss()
            self.criterionImgRecon = networks.ReconImgLoss()

            self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netG_Depth_T.parameters(), self.netG_Pose_T.parameters()),
                                                lr=opt.lr_task, betas=(0.9, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)

    def set_input(self, src_data, tgt_data):
        if self.isTrain:
            self.src_img[('color', 0, -1)] = src_data[('color', 0, -1)].to(self.device)
            self.src_img[('color', -1, -1)] = src_data[('color', -1, -1)].to(self.device)
            self.src_img[('color', 1, -1)] = src_data[('color', 1, -1)].to(self.device)

            self.src_gt['depth'] = src_data['depth'].to(self.device)
            self.src_gt[("axisangle", 0, -1)] = src_data[("axisangle", 0, -1)].to(self.device)
            self.src_gt[("translation", 0, -1)] = src_data[("translation", 0, -1)].to(self.device)

            self.src_gt[("axisangle", 0, 1)] = src_data[("axisangle", 0, 1)].to(self.device)
            self.src_gt[("translation", 0, 1)] = src_data[("translation", 0, 1)].to(self.device)

            self.tgt_img[('color', -1, -1)] = tgt_data[('color',-1,-1)].to(self.device)
            self.tgt_img[('color', 0, -1)] = tgt_data[('color', 0, -1)].to(self.device)
            self.tgt_img[('color', 1, -1)] = tgt_data[('color', 1, -1)].to(self.device)

            if 'depth_gt' in tgt_data.keys():
                self.tgt_gen['depth_gt'] = tgt_data['depth_gt'].to(self.device)

            self.tgt_img_K = tgt_data['K'].float().to(self.device)
            self.tgt_img_inv_K = tgt_data['inv_K'].float().to(self.device)

            self.num = self.src_img[('color', 0, -1)].shape[0]

            b, _, h, w = tgt_data[('color', 0, -1)].shape

            self.backproject_depth = BackprojectDepth(b, h, w)
            self.backproject_depth.to(self.device)

            self.project_3d = Project3D(b, h, w)
            self.project_3d.to(self.device)

        # TODO
        else: self.img =  tgt_data[('color', 0, -1)].to(self.device)
            # input_['left_img'].to(self.device)

    def forward(self):

        if self.isTrain:
            self.fake_src_img[('color', -1, -1)] = self.netG_Tgt(self.tgt_img[('color', -1, -1)]).detach()
            self.fake_src_img[('color', 0, -1)] = self.netG_Tgt(self.tgt_img[('color', 0, -1)]).detach()
            self.fake_src_img[('color', 1, -1)] = self.netG_Tgt(self.tgt_img[('color', 1, -1)]).detach()

            self.out = list(self.netG_Depth_T(torch.cat((self.src_img[('color', 0, -1)], self.fake_src_img[('color', 0, -1)]), 0)).values())

            # src image generated depth [-1]
            self.src_gen['depth'] = self.out[-1].narrow(0, 0, self.num)
            # r2s image generate depth
            self.fake_src_gen['depth'] = self.out[-1].narrow(0, self.num, self.num)


            for f_i in self.opt.frame_idxs[1:]:
                if f_i<0:
                    src_pose_inputs = torch.cat([self.src_img[('color', f_i, -1)], self.src_img[('color', 0, -1)]], axis=1)
                    fake_src_pose_inputs = torch.cat([self.fake_src_img[('color', f_i, -1)], self.fake_src_img[('color', 0, -1)]],axis=1)
                else:
                    src_pose_inputs = torch.cat([self.src_img[('color', 0, -1)], self.src_img[('color', f_i, -1)]], axis=1)
                    fake_src_pose_inputs = torch.cat([self.fake_src_img[('color', 0, -1)], self.fake_src_img[('color', f_i, -1)]], axis=1)
                # generate src pose
                self.src_gen[('axisangle', 0, f_i)], self.src_gen[('translation', 0, f_i)] = self.netG_Pose_S(src_pose_inputs)
                self.src_gen[('axisangle', 0, f_i)], self.src_gen[('translation', 0, f_i)] = self.src_gen[('axisangle', 0, f_i)].double(), self.src_gen[('translation', 0, f_i)].double() 
                # generate r2s pose
                self.fake_src_gen[('axisangle', 0, f_i)], self.fake_src_gen[('translation', 0, f_i)] = self.netG_Pose_T(fake_src_pose_inputs)
                self.fake_src_gen[('axisangle', 0, f_i)], self.fake_src_gen[('translation', 0, f_i)] = self.fake_src_gen[('axisangle', 0, f_i)].double(), self.fake_src_gen[('translation', 0, f_i)].double()
                # generate cam_T_cam
                # self.fake_src_gen[("cam_T_cam", 0, f_i)] = transformation_from_parameters(self.fake_src_gen[("axisangle", 0, f_i)][:, 0], self.fake_src_gen[("translation", 0, f_i)][:, 0])

        else:
            self.img_trans = self.netG_Tgt(self.img)
            self.pred = self.netG_Depth_T(self.img_trans)[-1]


    def backward_G(self):
        lambda_R_Depth = self.opt.lambda_R_Depth
        lambda_R_Img = self.opt.lambda_R_Img
        lambda_S_Depth = self.opt.lambda_S_Depth
        lambda_R_Pose = self.opt.lambda_R_Pose

        # Depth L1 loss for s2r image gen depth - src image depth gt
        self.loss_R_Depth_Src = 0.0
        src_gt_depths = dataset_util.scale_pyramid(self.src_gt['depth'], 4)

        for (src_gen_depth, gt_depth) in zip(self.out, src_gt_depths):
            self.loss_R_Depth_Src += self.criterionDepthReg(src_gen_depth[:self.num, :, :, :], gt_depth) * lambda_R_Depth

        # Pose L2 loss for s2r image gen depth - src image depth gt
        self.loss_R_Pose_Src = 0.0
        for f_i in self.opt.frame_idxs[1:]:
            src_q = []
            for ind, i in enumerate(self.src_gen[("axisangle", 0, f_i)].squeeze()):
                row = R.from_rotvec(i.cpu().detach().numpy()).as_quat()
                src_q.append(row)
            src_q = torch.from_numpy(np.array(src_q)).float().to(self.device)
            src_t = self.src_gen[("translation", 0, f_i)].view([-1,3]).float()
            self.loss_R_Pose_Src += lambda_R_Pose *((self.criterionPoseReg(src_q,self.src_gt[("axisangle", 0, f_i)]) + self.criterionPoseReg(src_t, self.src_gt[("translation", 0, f_i)])))

        # Recon loss
        # tgt image image depthmap, t-1 img , pose 정보 활용해서  -> t img로 wrap
        self.loss_R_Img_Tgt, self.warp_fake_src_img = self.criterionImgRecon(self.backproject_depth, self.project_3d, self.fake_src_img, self.fake_src_gen, self.tgt_img_K, self.tgt_img_inv_K, self.device)

        # Depth Smoothness loss for s2r image gen depth - tgt_img[('color', 0, -1)]
        fake_src_imgs = dataset_util.scale_pyramid(self.fake_src_img[('color', 0, -1)], 4)
        i = 0
        self.loss_S_Depth_Tgt = 0.0
        for (fake_src_gen_depth, img) in zip(self.out, fake_src_imgs):
            self.loss_S_Depth_Tgt += self.criterionSmooth(fake_src_gen_depth[self.num:,:,:,:], img) * lambda_S_Depth / 2**i
            i += 1

        # Pose loss 추가
        self.loss_G_Tgt = self.loss_S_Depth_Tgt + self.loss_R_Depth_Src + lambda_R_Img*self.loss_R_Img_Tgt + self.loss_R_Pose_Src
        
        self.loss_G_Tgt.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G_task.zero_grad()
        self.backward_G()
        self.optimizer_G_task.step()
