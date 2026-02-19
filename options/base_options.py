from typing import Optional
from typing import List
from typing import Literal
import argparse
import os
import util
import torch
from tap import Tap

class BaseOptions():
    is_train: bool = True
    
    def __init__(self):
        self.initialized = False

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')

        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0)
        parser.add_argument('--blur_sig', default='0.5')
        parser.add_argument('--jpg_prob', type=float, default=0)
        parser.add_argument('--jpg_method', default='cv2')
        parser.add_argument('--jpg_qual', default='75')

        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--classes', default='', help='image classes to train on')
        parser.add_argument('--multiclass', default='', help='image classes to train on')
        parser.add_argument('--class_bal', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=16, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        self.initialized = True
        return parser

    def gather_options(self) -> argparse.Namespace:
        # initialize parser with basic options
        if self.initialized:
            parser = self.parser
        else:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt: argparse.Namespace):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options: bool = True) -> argparse.Namespace:

        opt = self.gather_options()
        opt.isTrain = self.is_train   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        opt.classes = opt.classes.split(',')
        opt.multiclass = opt.multiclass.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
    
class TypedBaseOptions(Tap):
    """Tap-based options with the same flags/behavior as BaseOptions.

    This is intended as a drop-in replacement for the argparse Namespace used
    throughout the codebase (attributes like opt.isTrain, opt.gpu_ids, etc).
    """

    # NOTE: keep names aligned with legacy argparse flags.
    is_train: bool = True

    arch: Literal['clip', 'rn50'] = "clip" # Backbone architecture for the global branch

    # Data options
    models: List[str] # GenImage models to train on
    dataroot: str = "./data" # Path to dataset root, which should have subfolders for each model (e.g. ./data/imagenet_ai_0508_adm). Can also be a gcs path (e.g. gs://my-bucket/data) if using --gcp_project_name
    jpeg_p: float = 0.5 # Probability of applying JPEG compression
    blur_p: float = 0.5 # Probability of applying blur
    hflip_p: float = 0.0 # Probability of applying horizontal flip
    blur_sigma: tuple[float, float] = (0.1, 2.0) #  Range for blur sigma when applying blur
    jpeg_qual: tuple[int, int] = (10, 50) # Range for JPEG quality when applying JPEG compression
    batch_size: int = 32 # Batch size for training
    workers: int = 4 # Number of worker processes for data loading
    gcp_project_name: Optional[str] = None # GCP project name for loading data from GCS (Can be None, in which case will load from local filesystem)
    load_size: int = 256 # Size to scale images to before cropping
    crop_size: int = 224 # Size to crop images to for the global branch during training
    
    experiment_name: str = "" # Name of the experiment, used for saving checkpoints and logs
    checkpoints_dir: str = "./checkpoints" # Directory to save checkpoints and logs. Can be a gcs path (e.g. gs://my-bucket/checkpoints) if using --gcp_project_name
    fs: Literal['local', 'gcs'] = "local" # Whether to load/save data from the local filesystem or Google Cloud Storage (GCS). It will be inferred from the presence of gcp_project_name and the format of dataroot/checkpoints_dir.
    
    def process_args(self):
        if self.gcp_project_name is not None:
            self.fs = 'gcs'
        else:
            self.fs = 'local'
            if self.dataroot.startswith('gs://') or self.checkpoints_dir.startswith('gs://'):
                raise ValueError("When --gcp_project_name is not set, dataroot and checkpoints_dir should be local paths, not gcs paths (e.g. ./data, not gs://my-bucket/data)")
        if self.fs == 'gcs':
            if not self.dataroot.startswith('gs://'):
                raise ValueError("When using --gcp_project_name, dataroot should be a gcs path (e.g. gs://my-bucket/data)")
            if not self.checkpoints_dir.startswith('gs://'):
                raise ValueError("When using --gcp_project_name, checkpoints_dir should be a gcs path (e.g. gs://my-bucket/checkpoints)")

        
    

