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
from .layers import BackprojectDepth, Project3D


class GASDAModel(BaseModel):
    def name(self):
        return 'GASDAModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_R_Depth', type=float, default=30.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_R_Pose', type=float, default=30.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_C_Depth', type=float, default=30.0, help='weight for consistency')
            parser.add_argument('--lambda_C_Pose', type=float, default=30.0, help='weight for consistency')

            parser.add_argument('--lambda_S_Depth', type=float, default=0.01,
                                help='weight for smooth loss')

            parser.add_argument('--lambda_R_Img', type=float, default=50.0, help='weight for image reconstruction')
            # cyclegan
            parser.add_argument('--lambda_Src', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_Tgt', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=30.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

            parser.add_argument('--s_depth_premodel', type=str, default=" ",
                                help='pretrained depth estimation model')
            parser.add_argument('--t_depth_premodel', type=str, default=" ",
                                help='pretrained depth estimation model')
            parser.add_argument('--s_pose_premodel', type=str, default=" ",
                                help='pretrained depth estimation model')
            parser.add_argument('--t_pose_premodel', type=str, default=" ",
                                help='pretrained depth estimation model')

            parser.add_argument('--g_src_premodel', type=str, default=" ",
                                help='pretrained G_Src model')
            parser.add_argument('--g_tgt_premodel', type=str, default=" ",
                                help='pretrained G_Tgt model')
            parser.add_argument('--d_src_premodel', type=str, default=" ",
                                help='pretrained D_Src model')
            parser.add_argument('--d_tgt_premodel', type=str, default=" ",
                                help='pretrained D_Tgt model')

            parser.add_argument('--freeze_bn', action='store_true', help='freeze the bn in mde')
            parser.add_argument('--freeze_in', action='store_true', help='freeze the in in cyclegan')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.src_img ={}
        self.tgt_img = {}
        self.fake_src_img = {}
        self.fake_tgt_img = {}

        self.src_gen = {}
        self.tgt_gen = {}
        self.src_gt = {}
        self.tgt_gt = {}
        self.idt_tgt_img = {}
        self.idt_src_img = {}
        self.rec_tgt_img = {}
        self.rec_src_img = {}
        self.fake_src_gen = {}
        self.fake_tgt_gen = {}

        if self.isTrain:
            self.loss_names = ['R_Depth_Src_S', 'R_Pose_Src_S','S_Depth_Tgt_S', 'R_Img_Tgt_S', 'C_Depth_Tgt','C_Pose_Tgt']
            self.loss_names += ['R_Depth_Src_T', 'S_Depth_Tgt_T', 'R_Pose_Src_T', 'R_Img_Tgt_T']
            self.loss_names += ['D_Src', 'G_Src', 'cycle_Src', 'idt_Src', 'D_Tgt', 'G_Tgt', 'cycle_Tgt', 'idt_Tgt']

        if self.isTrain:
            visual_names_src = ['src_img', 'fake_tgt_img', 'src_gt', 'src_gen', 'fake_tgt_gen']
            visual_names_tgt = ['tgt_img', 'fake_src_img', 'tgt_gen', 'fake_src_gen' ]
            if self.opt.lambda_identity > 0.0:
                visual_names_src.append('idt_src_img')
                visual_names_tgt.append('idt_tgt_img')
            self.visual_names = visual_names_src + visual_names_tgt
        else:
            self.visual_names = ['pred', 'tgt_img', 'img_trans']

        if self.isTrain:
            self.model_names = ['G_Depth_S', 'G_Pose_S', 'G_Depth_T', 'G_Pose_T']
            self.model_names += ['G_Src', 'G_Tgt', 'D_Src', 'D_Tgt']
        else:
            self.model_names = ['G_Depth_S', 'G_Depth_T', 'G_Tgt']

        self.netG_Depth_S = networks.init_net(networks.UNetGenerator(norm='batch'), init_type='kaiming', gpu_ids=opt.gpu_ids)
        self.netG_Depth_T = networks.init_net(networks.UNetGenerator(norm='batch'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        self.num_pose_frames = 2

        self.netG_Pose_T = networks.init_net(
            PoseEncoderDecoder(num_layers=opt.num_layers, pretrained=False, num_input_images=self.num_pose_frames,
                               num_frames_to_predict_for=1, stride=1), init_type='normal', gpu_ids=opt.gpu_ids)
        self.netG_Pose_S = networks.init_net(
            PoseEncoderDecoder(num_layers=opt.num_layers, pretrained=False, num_input_images=self.num_pose_frames,
                               num_frames_to_predict_for=1, stride=1), init_type='normal', gpu_ids=opt.gpu_ids)

        # self.netG_Pose_T = networks.init_net(PoseCNN(opt.num_frames), init_type='normal', gpu_ids=opt.gpu_ids)
        # self.netG_Pose_S = networks.init_net(PoseCNN(opt.num_frames), init_type='normal', gpu_ids=opt.gpu_ids)

        self.netG_Src = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)
        self.netG_Tgt = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            self.netD_Src = networks.init_net(networks.Discriminator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)
            self.netD_Tgt = networks.init_net(networks.Discriminator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

            self.init_with_pretrained_model('G_Depth_S', self.opt.s_depth_premodel)
            self.init_with_pretrained_model('G_Depth_T', self.opt.t_depth_premodel)
            self.init_with_pretrained_model('G_Pose_S', self.opt.s_pose_premodel)
            self.init_with_pretrained_model('G_Pose_T', self.opt.t_pose_premodel)
            self.init_with_pretrained_model('G_Src', self.opt.g_src_premodel)
            self.init_with_pretrained_model('G_Tgt', self.opt.g_tgt_premodel)
            self.init_with_pretrained_model('D_Src', self.opt.d_src_premodel)
            self.init_with_pretrained_model('D_Tgt', self.opt.d_tgt_premodel)
         
        if self.isTrain:
            # define loss functions
            self.criterionDepthReg = torch.nn.L1Loss()
            self.criterionDepthCons = torch.nn.L1Loss()
            self.criterionPoseReg = torch.nn.MSELoss()
            self.criterionPoseCons = torch.nn.MSELoss()

            self.criterionSmooth = networks.SmoothLoss()
            self.criterionImgRecon = networks.ReconImgLoss()
            self.criterionLR = torch.nn.L1Loss()

            self.fake_src_pool = ImagePool(opt.pool_size)
            self.fake_tgt_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G_task = torch.optim.Adam(
                itertools.chain(self.netG_Depth_S.parameters(), self.netG_Pose_S.parameters(),
                                self.netG_Depth_T.parameters(), self.netG_Pose_T.parameters()),
                lr=opt.lr_task, betas=(0.95, 0.999))
            self.optimizer_G_trans = torch.optim.Adam(
                itertools.chain(self.netG_Src.parameters()),
                lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Src.parameters(), self.netD_Tgt.parameters()),
                                                lr=opt.lr_trans, betas=(0.5, 0.9))
            # , self.netG_Tgt.parameters()
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)
            self.optimizers.append(self.optimizer_G_trans)
            self.optimizers.append(self.optimizer_D)
            if opt.freeze_bn:
                self.netG_Depth_S.apply(networks.freeze_bn)
                self.netG_Depth_T.apply(networks.freeze_bn)
                # TODO pose

            if opt.freeze_in:
                self.netG_Src.apply(networks.freeze_in)
                self.netG_Tgt.apply(networks.freeze_in)
                # TODO pose

    def set_input(self, src_data, tgt_data):

        if self.isTrain:
            # src data
            self.src_gt['depth'] = src_data['depth'].to(self.device)
            self.src_gt[("axisangle", 0, -1)] = src_data[("axisangle", 0, -1)].to(self.device)
            self.src_gt[("translation", 0, -1)] = src_data[("translation", 0, -1)].to(self.device)

            self.src_gt[("axisangle", 0, 1)] = src_data[("axisangle", 0, 1)].to(self.device)
            self.src_gt[("translation", 0, 1)] = src_data[("translation", 0, 1)].to(self.device)

            self.src_img[('color', 0, -1)] = src_data[('color', 0, -1)].to(self.device)
            self.src_img[('color', -1, -1)] = src_data[('color', -1, -1)].to(self.device)
            self.src_img[('color', 1, -1)] = src_data[('color', 1, -1)].to(self.device)

            # tgt data
            self.tgt_img[('color', -1, -1)] = tgt_data[('color', -1, -1)].to(self.device)
            self.tgt_img[('color', 0, -1)] = tgt_data[('color', 0, -1)].to(self.device)
            self.tgt_img[('color', 1, -1)] = tgt_data[('color', 1, -1)].to(self.device)

            # self.tgt_img['fb'] = tgt_data['fb'].to(self.device)
            b, _, h, w = tgt_data[('color', 0, -1)].shape

            self.tgt_backproject_depth = BackprojectDepth(b, h, w)
            self.tgt_backproject_depth.to(self.device)

            self.tgt_project_3d = Project3D(b, h, w)
            self.tgt_project_3d.to(self.device)

            b, _, h, w = src_data[('color', 0, -1)].shape
            self.src_backproject_depth = BackprojectDepth(b, h, w)
            self.src_backproject_depth.to(self.device)

            self.src_project_3d = Project3D(b, h, w)
            self.src_project_3d.to(self.device)

            if 'depth_gt' in tgt_data.keys():
                self.tgt_gen['depth_gt'] = tgt_data['depth_gt'].to(self.device)

            self.tgt_img_K = tgt_data['K'].float().to(self.device)
            self.tgt_img_inv_K = tgt_data['inv_K'].float().to(self.device)

            self.num = self.src_img[('color', 0, -1)].shape[0]
        # TODO test
        else:
            # self.img = input['left_img'].to(self.device)
            self.tgt_img[('color', -1, -1)] = tgt_data[('color', -1, -1)].to(self.device)
            self.tgt_img[('color', 0, -1)] = tgt_data[('color', 0, -1)].to(self.device)
            self.tgt_img[('color', 1, -1)] = tgt_data[('color', 1, -1)].to(self.device)

    def forward(self):

        if self.isTrain:
            pass
    
        else:
            self.pred_s = self.netG_Depth_S(self.tgt_img[('color', 0, -1)])[-1]
            self.img_trans = self.netG_Tgt(self.tgt_img[('color', 0, -1)])
            self.pred_t = self.netG_Depth_T(self.img_trans)[-1]
            self.pred = 0.5 * (self.pred_s + self.pred_t)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_Src(self):
        self.loss_D_Src = 0.0
        for i in self.opt.frame_idxs:
            fake_tgt = self.fake_tgt_pool.query(self.fake_tgt_img[('color', i, -1)])
            self.loss_D_Src += self.backward_D_basic(self.netD_Src, self.tgt_img[('color', i, -1)], fake_tgt)

    def backward_D_Tgt(self):
        self.loss_D_Tgt = 0.0
        for i in self.opt.frame_idxs:
            fake_src = self.fake_src_pool.query(self.fake_src_img[('color', i, -1)])
            self.loss_D_Tgt = self.backward_D_basic(self.netD_Tgt, self.src_img[('color', i, -1)], fake_src)

    def backward_G(self):

        lambda_R_Depth = self.opt.lambda_R_Depth
        lambda_R_Pose = self.opt.lambda_R_Pose
        lambda_R_Img = self.opt.lambda_R_Img
        lambda_S_Depth = self.opt.lambda_S_Depth
        lambda_C_Depth = self.opt.lambda_C_Depth
        lambda_C_Pose = self.opt.lambda_C_Pose
        lambda_idt = self.opt.lambda_identity
        lambda_Src = self.opt.lambda_Src
        lambda_Tgt = self.opt.lambda_Tgt

        # =========================== prerequisite ==========================
        for i in self.opt.frame_idxs:
            # s2r img
            self.fake_tgt_img[('color', i, -1)] = self.netG_Src(self.src_img[('color', i, -1)])

            # r2s img
            self.fake_src_img[('color', i, -1)] = self.netG_Tgt(self.tgt_img[('color', i, -1)])

        for f_i in self.opt.frame_idxs[1:]:
            if f_i < 0:
                fake_src_pose_inputs = torch.cat(
                    [self.fake_src_img[('color', f_i, -1)], self.fake_src_img[('color', 0, -1)]], axis=1)
                fake_tgt_pose_inputs = torch.cat(
                    [self.fake_tgt_img[('color', f_i, -1)], self.fake_tgt_img[('color', 0, -1)]], axis=1)
                src_pose_inputs = torch.cat([self.src_img[('color', f_i, -1)], self.src_img[('color', 0, -1)]], axis=1)
                tgt_pose_inputs = torch.cat([self.tgt_img[('color', f_i, -1)], self.tgt_img[('color', 0, -1)]], axis=1)
            else:
                fake_src_pose_inputs = torch.cat(
                    [self.fake_src_img[('color', f_i, -1)], self.fake_src_img[('color', 0, -1)]], axis=1)
                fake_tgt_pose_inputs = torch.cat(
                    [self.fake_tgt_img[('color', f_i, -1)], self.fake_tgt_img[('color', 0, -1)]], axis=1)
                src_pose_inputs = torch.cat([self.src_img[('color', 0, -1)], self.src_img[('color', f_i, -1)]], axis=1)
                tgt_pose_inputs = torch.cat([self.tgt_img[('color', 0, -1)], self.tgt_img[('color', f_i, -1)]], axis=1)
            # generate src pose
            self.src_gen[('axisangle', 0, f_i)], self.src_gen[('translation', 0, f_i)] = self.netG_Pose_S(
                src_pose_inputs)
            # generate r2s pose
            self.tgt_gen[('axisangle', 0, f_i)], self.tgt_gen[('translation', 0, f_i)] = self.netG_Pose_T(
                tgt_pose_inputs)
            # generate src pose
            self.fake_src_gen[('axisangle', 0, f_i)], self.fake_src_gen[('translation', 0, f_i)] = self.netG_Pose_S(
                fake_src_pose_inputs)
            # generate r2s pose
            self.fake_tgt_gen[('axisangle', 0, f_i)], self.fake_tgt_gen[('translation', 0, f_i)] = self.netG_Pose_T(
                fake_tgt_pose_inputs)
        # =========================== synthetic ==========================
        i = 0
        self.idt_tgt_img[('color', i, -1)] = self.netG_Tgt(self.src_img[('color', i, -1)])
        self.rec_src_img[('color', i, -1)] = self.netG_Tgt(self.fake_tgt_img[('color', i, -1)])
        self.fake_tgt_out = self.netG_Depth_S(self.fake_tgt_img[('color', i, -1)])
        self.src_out = self.netG_Depth_T(self.src_img[('color', i, -1)])
        self.src_gen['depth'] = self.src_out[-1]
        self.fake_tgt_gen['depth'] = self.fake_tgt_out[-1]

        self.loss_G_Src = self.criterionGAN(self.netD_Src(self.fake_tgt_img[('color', i, -1)]), True)

        self.loss_cycle_Src = self.criterionCycle(self.rec_src_img[('color', i, -1)],
                                                  self.src_img[('color', i, -1)])
        self.loss_idt_Tgt = self.criterionIdt(self.idt_tgt_img[('color', i, -1)],
                                              self.src_img[('color', i, -1)]) * lambda_Src * lambda_idt

        # depth l1 loss
        self.loss_R_Depth_Src_S = 0.0
        real_depths = dataset_util.scale_pyramid(self.src_gt['depth'], 4)
        for (fake_tgt_gen_depth, real_depth) in zip(self.fake_tgt_out, real_depths):
            self.loss_R_Depth_Src_S += self.criterionDepthReg(fake_tgt_gen_depth, real_depth) * lambda_R_Depth

        self.loss_R_Depth_Src_T = 0.0
        for (src_gen_depth, real_depth) in zip(self.src_out, real_depths):
            self.loss_R_Depth_Src_T += self.criterionDepthReg(src_gen_depth, real_depth) * lambda_R_Depth

        # pose l2 loss
        self.loss_R_Pose_Src_S = 0.0
        for f_i in self.opt.frame_idxs[1:]:
            src_gen_q = []
            for ind, i in enumerate(self.src_gen[("axisangle", 0, f_i)].squeeze()):
                row = R.from_rotvec(i.cpu().detach().numpy()).as_quat()
                src_gen_q.append(row)
            src_gen_q = torch.from_numpy(np.array(src_gen_q)).to(self.device)
            src_gen_t = self.src_gen[("translation", 0, f_i)].view([-1, 3])
            # self.src_gt[("translation", 0, f_i)].requires_grad = True
            self.loss_R_Pose_Src_S += lambda_R_Pose * (self.criterionPoseReg(src_gen_q,self.src_gt[("axisangle", 0,f_i)])
                                                       + self.criterionPoseReg(src_gen_t, self.src_gt[("translation", 0, f_i)]))

        self.loss_R_Pose_Src_T = 0.0
        for f_i in self.opt.frame_idxs[1:]:
            fake_tgt_gen_q = []
            for ind, i in enumerate(self.fake_tgt_gen[("axisangle", 0, f_i)].squeeze()):
                row = R.from_rotvec(i.cpu().detach().numpy()).as_quat()
                fake_tgt_gen_q.append(row)
            fake_tgt_gen_q = torch.from_numpy(np.array(fake_tgt_gen_q)).to(self.device)
            fake_tgt_gen_t = self.fake_tgt_gen[("translation", 0, f_i)].view([-1, 3])
            # self.src_gt[("translation", 0, f_i)].requires_grad = True
            self.loss_R_Pose_Src_T += lambda_R_Pose * (
                        self.criterionPoseReg(fake_tgt_gen_q,self.src_gt[("axisangle", 0, f_i)]) + self.criterionPoseReg(fake_tgt_gen_t,self.src_gt[("translation",0,f_i)]))

        self.loss = self.loss_G_Src + self.loss_R_Depth_Src_T + self.loss_cycle_Src +  self.loss_idt_Tgt  + self.loss_R_Depth_Src_S + self.loss_R_Pose_Src_S + self.loss_R_Pose_Src_T
        # 
        self.loss.backward()

        # ============================= real =============================
        i = 0
        self.idt_src_img[('color', i, -1)] = self.netG_Src(self.tgt_img[('color', i, -1)])
        self.rec_tgt_img[('color', i, -1)] = self.netG_Src(self.fake_src_img[('color', i, -1)])
        self.tgt_out = self.netG_Depth_S(self.tgt_img[('color', i, -1)])
        self.fake_src_out = self.netG_Depth_T(self.fake_src_img[('color', i, -1)])
        self.fake_src_gen['depth'] = self.fake_src_out[-1]
        self.tgt_gen['depth'] = self.tgt_out[-1]

        self.loss_G_Tgt = self.criterionGAN(self.netD_Tgt(self.fake_src_img[('color', i, -1)]), True)
        self.loss_cycle_Tgt = self.criterionCycle(self.rec_tgt_img[('color', i, -1)], self.tgt_img[('color', i, -1)])
        self.loss_idt_Src = self.criterionIdt(self.idt_src_img[('color', i, -1)],
                                              self.tgt_img[('color', i, -1)]) * lambda_Tgt * lambda_idt
        # geometry consistency
        # TODO
        self.loss_R_Img_Tgt_T, self.warp_tgt_img = self.criterionImgRecon(self.tgt_backproject_depth,self.tgt_project_3d,self.tgt_img, self.tgt_gen, self.tgt_img_K,self.tgt_img_inv_K, self.device)

        self.loss_R_Img_Tgt_S, self.warp_fake_src_img = self.criterionImgRecon(self.tgt_backproject_depth,self.tgt_project_3d,self.fake_src_img, self.fake_src_gen,self.tgt_img_K,self.tgt_img_inv_K, self.device)

        # smoothness
        tgt_imgs = dataset_util.scale_pyramid(self.tgt_img[('color', 0, -1)], 4)
        fake_src_imgs = dataset_util.scale_pyramid(self.fake_src_img[('color', 0, -1)], 4)
        i = 0
        self.loss_S_Depth_Tgt_S = 0.0
        for (tgt_gen_depth, img) in zip(self.tgt_out, tgt_imgs):
            self.loss_S_Depth_Tgt_S += self.criterionSmooth(tgt_gen_depth, img) * lambda_S_Depth / 2 ** i
            i += 1
        i = 0
        self.loss_S_Depth_Tgt_T = 0.0
        for (fake_src_gen_depth, img) in zip(self.fake_src_out, fake_src_imgs):
            self.loss_S_Depth_Tgt_T += self.criterionSmooth(fake_src_gen_depth, img) * lambda_S_Depth / 2 ** i
            i += 1

        # depth consistency
        self.loss_C_Depth_Tgt = 0.0
        for (gen_depth1, gen_depth2) in zip(self.tgt_out, self.fake_src_out):
            self.loss_C_Depth_Tgt += self.criterionDepthCons(gen_depth1, gen_depth2) * lambda_C_Depth

        # pose consistency
        self.loss_C_Pose_Tgt = 0.0
        # for (gen_depth1, gen_depth2) in zip(self.out_s, self.out_t):
        for f_i in self.opt.frame_idxs[1:]:
            tgt_gen_q = []
            fake_src_gen_q = []
            for i, (t, s) in enumerate(zip(self.tgt_gen[("axisangle", 0, f_i)].squeeze(),self.fake_src_gen[("axisangle", 0, f_i)].squeeze())):
                row_t = R.from_rotvec(t.cpu().detach().numpy()).as_quat()
                tgt_gen_q.append(row_t)
                row_s = R.from_rotvec(s.cpu().detach().numpy()).as_quat()
                fake_src_gen_q.append(row_s)
            tgt_gen_q = torch.from_numpy(np.array(tgt_gen_q)).to(self.device)
            fake_src_gen_q = torch.from_numpy(np.array(fake_src_gen_q)).to(self.device)
            tgt_gen_t = self.tgt_gen[("translation", 0, f_i)].view([-1, 3])
            fake_src_gen_t = self.fake_src_gen[("translation", 0, f_i)].view([-1, 3])
            # self.fake_src_gen[("translation", 0, f_i)].requires_grad = True
            self.loss_C_Pose_Tgt += lambda_C_Pose * (self.criterionPoseReg(tgt_gen_q, fake_src_gen_q) + self.criterionPoseReg(fake_src_gen_t, tgt_gen_t))

        self.loss_G = self.loss_G_Tgt + self.loss_cycle_Tgt + self.loss_idt_Src + lambda_R_Img*self.loss_R_Img_Tgt_T + lambda_R_Img*self.loss_R_Img_Tgt_S + self.loss_S_Depth_Tgt_T + self.loss_S_Depth_Tgt_S + self.loss_C_Depth_Tgt + self.loss_C_Pose_Tgt
        self.loss_G.backward()

        self.tgt_gen_depth = (self.fake_src_gen['depth'] + self.tgt_gen['depth']) / 2.0
        self.src_gen_depth = (self.src_gen['depth'] + self.fake_tgt_gen['depth']) / 2.0

    def optimize_parameters(self):
        
        self.forward()
        self.set_requires_grad([self.netD_Src, self.netD_Tgt], False)
        self.optimizer_G_trans.zero_grad()
        self.optimizer_G_task.zero_grad()
        self.backward_G()
        self.optimizer_G_trans.step()
        self.optimizer_G_task.step()

        self.set_requires_grad([self.netD_Src, self.netD_Tgt], True)
        self.optimizer_D.zero_grad()
        self.backward_D_Src()
        self.backward_D_Tgt()
        self.optimizer_D.step()
