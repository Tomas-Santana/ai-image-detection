import argparse
import os
import util
import torch
from tap import Tap
#import models
#import data


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
        if not self.initialized:
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

    mode: str = "binary"
    arch: str = "res50"

    # data augmentation
    rz_interp: str = "bilinear"
    blur_prob: float = 0.0
    blur_sig: str = "0.5"
    jpg_prob: float = 0.0
    jpg_method: str = "cv2"
    jpg_qual: str = "75"

    dataroot: str = "./dataset/"
    classes: str = ""
    multiclass: str = ""
    class_bal: bool = False
    batch_size: int = 64
    loadSize: int = 224
    cropSize: int = 224
    gpu_ids: str = "0"
    name: str = ""
    epoch: str = "latest"
    num_threads: int = 16
    checkpoints_dir: str = "./checkpoints"
    serial_batches: bool = False
    resize_or_crop: str = "scale_and_crop"
    no_flip: bool = False
    init_type: str = "normal"
    init_gain: float = 0.02
    suffix: str = ""

    def _user_items(self):
        items = []
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue
            items.append((k, v))
        return items

    def print_options(self):
        message = ""
        message += "----------------- Options ---------------\n"

        defaults = type(self)()
        for k, v in sorted(self._user_items(), key=lambda kv: kv[0]):
            comment = ""
            default = getattr(defaults, k, None)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        expr_dir = os.path.join(self.checkpoints_dir, self.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self, print_options: bool = True):
        # Parse CLI into this object.
        self.parse_args()

        # Match legacy attribute expected by BaseModel/Trainer.
        self.isTrain = self.is_train

        # process suffix
        if self.suffix:
            suffix = ("_" + self.suffix.format(**vars(self))) if self.suffix != "" else ""
            self.name = self.name + suffix

        # Print/save options BEFORE post-processing to preserve legacy output.
        if print_options:
            self.print_options()

        return self
