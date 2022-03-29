import numpy as np
import torch
import itertools

from scipy.spatial.transform import Rotation as R

from networks import PoseCNN, PoseEncoderDecoder
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
import torch.nn.functional as F
from utils import dataset_util
from .layers import BackprojectDepth, Project3D, transformation_from_parameters


class FSModel(BaseModel):
    def name(self):
        return 'FSModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_R_Depth', type=float, default=1.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_S_Depth', type=float, default=0.01, help='weight for smooth loss')

            parser.add_argument('--lambda_R_Pose', type=float, default=1.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_S_Pose', type=float, default=0.01, help='weight for smooth loss')

            parser.add_argument('--lambda_R_Img', type=float, default=1.0,help='weight for image reconstruction')
            
            parser.add_argument('--g_src_premodel', type=str, default=" ",help='pretrained G_Src model')

            parser.add_argument('--s_pose_premodel', type=str, default='/media/data1/results/pgasda',
                                help='models are saved here')
            parser.add_argument('--s_depth_premodel', type=str, default='/media/data1/results/pgasda', help='models are saved here')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.src_img ={}
        self.tgt_img = {}
        self.fake_src_img = {}
        self.fake_tgt_img = {}

        self.fake_tgt_gen = {}
        self.tgt_gen = {}

        self.src_gt = {}
        self.tgt_gt = {}

        if self.isTrain:
            self.loss_names = ['R_Depth_Src',  'R_Pose_Src', 'S_Depth_Tgt', 'R_Img_Tgt' ]
          
        if self.isTrain:
            self.visual_names = ['src_img', 'fake_tgt_img', 'src_gt', 'fake_tgt_gen', 'tgt_img', 'tgt_gen', 'warp_tgt_img']
        else:
            self.visual_names = ['pred', 'img']

        if self.isTrain:
            self.model_names = ['G_Depth_S', 'G_Pose_S', 'G_Src']

        else:
            self.model_names = ['G_Depth_S', 'G_Src']

        self.num_pose_frames = 2

        self.netG_Depth_S = networks.init_net(networks.UNetGenerator(norm='batch'), init_type='normal',
                                              gpu_ids=opt.gpu_ids)
        # self.netG_Pose_T = networks.init_net(PoseCNN(opt.num_frames), init_type='normal', gpu_ids=opt.gpu_ids)
        # self.netG_Pose_S = networks.init_net(PoseCNN(opt.num_frames), init_type='normal', gpu_ids=opt.gpu_ids)

        self.netG_Pose_T = networks.init_net(PoseEncoderDecoder(num_layers=opt.num_layers, pretrained=False, num_input_images=self.num_pose_frames, num_frames_to_predict_for=1, stride=1), init_type='normal', gpu_ids=opt.gpu_ids)
        self.netG_Pose_S = networks.init_net(PoseEncoderDecoder(num_layers=opt.num_layers, pretrained=False, num_input_images=self.num_pose_frames, num_frames_to_predict_for=1, stride=1), init_type='normal', gpu_ids=opt.gpu_ids)


        self.netG_Src = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming',gpu_ids=opt.gpu_ids)


        if self.isTrain:
            self.init_with_pretrained_model('G_Src', self.opt.g_src_premodel)
            # self.init_with_pretrained_model('G_Pose_S', self.opt.s_pose_premodel)
            # self.init_with_pretrained_model('G_Depth_S', self.opt.s_depth_premodel)
            
            self.netG_Src.eval()
         
        if self.isTrain:
            # define loss functions
            self.criterionDepthReg = torch.nn.L1Loss()
            self.criterionPoseReg = torch.nn.MSELoss()

            self.criterionSmooth = networks.SmoothLoss()
            self.criterionImgRecon = networks.ReconImgLoss()

            self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netG_Depth_S.parameters(), self.netG_Pose_S.parameters()),
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

            self.tgt_img[('color', -1, -1)] = tgt_data[('color', -1, -1)].to(self.device)
            self.tgt_img[('color', 0, -1)] = tgt_data[('color', 0, -1)].to(self.device)
            self.tgt_img[('color', 1, -1)] = tgt_data[('color', 1, -1)].to(self.device)

            if 'depth_gt' in tgt_data.keys():
                self.tgt_gen['depth_gt'] = tgt_data['depth_gt'].to(self.device)

            self.tgt_img_K = tgt_data['K'].to(self.device)
            self.tgt_img_inv_K = tgt_data['inv_K'].to(self.device)

            self.num = self.src_img[('color', 0, -1)].shape[0]

            b, _, h, w = tgt_data[('color', 0, -1)].shape

            self.backproject_depth = BackprojectDepth(b, h, w)
            self.backproject_depth.to(self.device)

            self.project_3d = Project3D(b, h, w)
            self.project_3d.to(self.device)

        # else:
        #     self.img = input['left_img'].to(self.device)

    def forward(self):

        if self.isTrain:
            self.fake_tgt_img[('color', -1, -1)] = self.netG_Src(self.src_img[('color', -1, -1)]).detach()
            self.fake_tgt_img[('color', 0, -1)] = self.netG_Src(self.src_img[('color', 0, -1)]).detach()
            self.fake_tgt_img[('color', 1, -1)] = self.netG_Src(self.src_img[('color', 1, -1)]).detach()

            self.out = self.netG_Depth_S(
                torch.cat((self.fake_tgt_img[('color', 0, -1)], self.tgt_img[('color', 0, -1)]), 0))
            # src image generated depth
            self.fake_tgt_gen['depth'] = self.out[-1].narrow(0, 0, self.num)
            # r2s image generate depth
            self.tgt_gen['depth'] = self.out[-1].narrow(0, self.num, self.num)

            for f_i in self.opt.frame_idxs[1:]:
                if f_i<0:
                    fake_tgt_pose_inputs = torch.cat([self.fake_tgt_img[('color', f_i, -1)], self.fake_tgt_img[('color', 0, -1)]], axis=1)
                    tgt_pose_inputs = torch.cat([self.tgt_img[('color', f_i, -1)], self.tgt_img[('color', 0, -1)]],axis=1)
                else:
                    fake_tgt_pose_inputs = torch.cat([self.fake_tgt_img[('color', 0, -1)], self.fake_tgt_img[('color', f_i, -1)]], axis=1)
                    tgt_pose_inputs = torch.cat([self.tgt_img[('color', 0, -1)], self.tgt_img[('color', f_i, -1)]],axis=1)
                # generate src pose
                self.fake_tgt_gen[('axisangle', 0, f_i)], self.fake_tgt_gen[('translation', 0, f_i)] = self.netG_Pose_S(fake_tgt_pose_inputs)
                # generate r2s pose
                self.tgt_gen[('axisangle', 0, f_i)], self.tgt_gen[('translation', 0, f_i)] = self.netG_Pose_T(tgt_pose_inputs)
                # generate cam_T_cam
                # self.tgt_gen[("cam_T_cam", 0, f_i)] = transformation_from_parameters(self.tgt_gen[("axisangle", 0, f_i)][:, 0], self.tgt_gen[("translation", 0, f_i)][:, 0])

        else:
            self.pred = self.netG_Depth_S(self.img)[-1]

    def backward_G(self):

        lambda_R_Depth = self.opt.lambda_R_Depth
        lambda_R_Img = self.opt.lambda_R_Img
        lambda_S_Depth = self.opt.lambda_S_Depth
        lambda_R_Pose = self.opt.lambda_R_Pose

        # Depth L1 loss for s2r image gen depth - src image depth gt
        self.loss_R_Depth_Src = 0.0
        real_depths = dataset_util.scale_pyramid(self.src_gt['depth'], 4)
        for (fake_tgt_gen_depth, real_depth) in zip(self.out, real_depths):
            self.loss_R_Depth_Src += self.criterionDepthReg(fake_tgt_gen_depth[:self.num,:,:,:], real_depth) * lambda_R_Depth

        # Pose L2 loss for s2r image gen depth - src image depth gt
        self.loss_R_Pose_Src = 0.0
        for f_i in self.opt.frame_idxs[1:]:
            fake_tgt_gen_q = []
            for ind, i in enumerate(self.tgt_gen[("axisangle", 0, f_i)].squeeze()):
                row = R.from_rotvec(i.cpu().detach().numpy()).as_quat()
                fake_tgt_gen_q.append(row)
            fake_tgt_gen_q = torch.from_numpy(np.array(fake_tgt_gen_q)).to(self.device)
            fake_tgt_gen_t = self.tgt_gen[("translation", 0, f_i)].view([-1, 3])
            self.src_gt[("translation", 0, f_i)].requires_grad = True
            # if torch.from_numpy(k).shape == self.src_gt[('axisangle', 0, f_i)].shape and ts.shape == self.src_gt[('translation', 0, f_i)].shape:
            self.loss_R_Pose_Src += lambda_R_Pose * (self.criterionPoseReg(fake_tgt_gen_q,self.src_gt[("axisangle", 0, f_i)]) +
                self.criterionPoseReg(fake_tgt_gen_t, self.src_gt[("translation", 0, f_i)]))


        # Recon loss
        # tgt image image depthmap, t-1 img , pose 정보 활용해서  -> t img로 wrap
        self.loss_R_Img_Tgt, self.warp_tgt_img = self.criterionImgRecon(self.backproject_depth, self.project_3d,self.tgt_img, self.tgt_gen, self.tgt_img_K,self.tgt_img_inv_K, self.device)

        # Depth Smoothness loss for s2r image gen depth - tgt_img[('color', 0, -1)]
        tgt_imgs = dataset_util.scale_pyramid(self.tgt_img[('color', 0, -1)], 4)
        i = 0
        self.loss_S_Depth_Tgt = 0.0
        for (gen_depth, img) in zip(self.out, tgt_imgs):
            self.loss_S_Depth_Tgt += self.criterionSmooth(gen_depth[self.num:, :, :, :],img) * lambda_S_Depth / 2 ** i
            i += 1


        self.loss_G_Src = lambda_R_Img*self.loss_R_Img_Tgt + self.loss_S_Depth_Tgt + self.loss_R_Depth_Src+ self.loss_R_Pose_Src
        self.loss_G_Src = self.loss_G_Src.type(torch.DoubleTensor)
        self.loss_G_Src.backward()

    def optimize_parameters(self):
        
        self.forward()
        self.optimizer_G_task.zero_grad()
        self.backward_G()
        self.optimizer_G_task.step()
