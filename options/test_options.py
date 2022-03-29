from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default=f'{self.folder}/results/pgasda/results/',
                            help='saves results here.')
        parser.add_argument('--root', type=str, default=f'{self.folder}/datasets/kitti', help='data root')
        parser.add_argument('--test_datafile', type=str, default='test2.txt', help='stores data list, in root')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--save', action='store_true', help='save results')
        parser.add_argument('--test_dataset', type=str, default='kitti', help='kitti|stereo|make3d')
        # for pose
        parser.add_argument('--load_weights_folder', type=str, default=f'{self.folder}/results/pgasda/vkitti2kitti_gasda/',
                            help="name of model to load")
        parser.add_argument('--data_path', type=str, default=f'{self.folder}/datasets/kitti_odom/',
                            help="path to the training data")
        parser.add_argument('--split_path', type=str, default=f'{self.folder}/datasets/',
                            help="path to the splits")
        parser.add_argument('--eval_split',type=str,
                                 default="odom_9",
                                 choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10",
                                     "oxford_eigen_zhou_night","oxford_eigen_zhou_day"],
                                 help="which split to run eval on")
        parser.add_argument('--ts', type=str, default='T', choices = ['T', 'S'],
                            help="Pose Net for T or S ")
        self.isTrain = False
        return parser
