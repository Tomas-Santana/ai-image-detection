from typing import Literal
from .base_options import BaseOptions, TypedBaseOptions
from .data_options import DatasetOptions
import argparse


class TrainOptions(BaseOptions):
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=10)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=100, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--loadpath', type=str, default='/projects/yanju/0ICIP22/11_2b_APSM_AFF_ori/checkpoints/p_trained/model_epoch_best.pth')

        self.is_train = True
        return parser
    
class TypedTrainOptions(TypedBaseOptions):
    model_path: str = "" # Path to a checkpoint to load for training (e.g. for fine-tuning or resuming training). Can be a gcs path (e.g. gs://my-bucket/checkpoints/model_epoch_best.pth) if using --gcp_project_name
    is_train: bool = True
    earlystop_epoch: int = 10 # number of epochs to wait for improvement before early stopping
    continue_train: bool = False # resume training from model_path
    new_optim: bool = False # do not load optimizer state when resuming
    checkpoint_freq: int = 5 # frequency of saving checkpoints (in epochs)
    niter: int = 10 # number of epochs to train for
    lr: float = 0.0001 # learning rate for the optimizer
    optim: Literal['sgd', 'adam'] = 'adam' # optimizer to use
    beta1: float = 0.9 # beta1 for Adam
    train_split: str = 'train' # dataset split to use for training
    val_split: str = 'val' # dataset split to use for validation

    @property
    def train_dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options(split=self.train_split)

    @property
    def val_dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options(split=self.val_split)

    
