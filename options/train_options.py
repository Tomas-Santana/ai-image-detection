from typing import Literal
from .base_options import BaseOptions
from .data_options import DatasetOptions


class TrainOptions(BaseOptions):
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
    results_dir: str = './results' # Folder where evaluation csv results are stored (local path or Azure Blob URL prefix)

    def process_args(self):
        super().process_args()
        if not self.models:
            raise ValueError("--models is required for training")

    @property
    def train_dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options(split=self.train_split)

    @property
    def val_dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options(split=self.val_split)

    
