from typing import Literal
from .base_options import BaseOptions
from .data_options import DatasetOptions


class TrainOptions(BaseOptions):
    model_path: str = "" # Path to a checkpoint to load for training (e.g. for fine-tuning or resuming training). Supports local path, gcs path (with --gcp_project_name), or Azure Blob URL
    is_train: bool = True
    earlystop_epoch: int = 5 # number of epochs to wait for improvement before early stopping
    continue_train: bool = False # resume training from model_path
    new_optim: bool = False # do not load optimizer state when resuming
    checkpoint_freq: int = 5 # frequency of saving checkpoints (in iters)
    save_model_freq: int = 1 # frequency of saving checkpoints (in epochs)
    niter: int = 10 # number of epochs to train for
    lr: float = 0.0001 # learning rate for the optimizer
    weight_decay: float = 1e-3 # weight decay for the optimizer
    optim: Literal['sgd', 'adam'] = 'adam' # optimizer to use
    beta1: float = 0.9 # beta1 for Adam
    unfreeze_last_layers: bool = False # Whether to unfreeze the last layer of CLIP during training
    train_split: str = 'train' # dataset split to use for training
    val_split: str = 'val' # dataset split to use for validation
    results_dir: str = './results' # Folder where evaluation csv results are stored (local path or Azure Blob URL prefix)
    save_results_to_google_sheets: bool = False # Append one row per epoch to a worksheet named after experiment_name
    google_sheets_spreadsheet_id: str = '' # Target Google Sheets spreadsheet id
    google_sheets_credentials_path: str = '' # Optional path to a Google service-account json key file

    def process_args(self):
        super().process_args()
        if not self.models:
            raise ValueError("--models is required for training")
        if self.save_results_to_google_sheets:
            if not self.google_sheets_spreadsheet_id:
                raise ValueError("--google_sheets_spreadsheet_id is required when --save_results_to_google_sheets is enabled")
            if not self.experiment_name:
                raise ValueError("--experiment_name is required when --save_results_to_google_sheets is enabled")

    @property
    def train_dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options(split=self.train_split)

    @property
    def val_dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options(split=self.val_split)

    
