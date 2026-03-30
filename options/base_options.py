from typing import Optional
from typing import List
from typing import Literal
from urllib.parse import urlsplit
from .data_options import DatasetOptions, Models
import torch
from tap import Tap


def _is_http_url(path: str) -> bool:
    return path.startswith('https://') or path.startswith('http://')


def _is_azure_blob_url(path: str) -> bool:
    if not _is_http_url(path):
        return False
    return 'blob.core.windows.net' in urlsplit(path).netloc

class BaseOptions(Tap):
    """Tap-based options with the same flags/behavior as BaseOptions.

    This is intended as a drop-in replacement for the argparse Namespace used
    throughout the codebase (attributes like opt.isTrain, opt.gpu_ids, etc).
    """
    is_train: bool = True
    gpu_ids: list[int] = [0]

    # Data options
    models: List[Models] = [] # GenImage models to use. If empty in test mode, models can be auto-discovered from dataroot.
    dataroot: str = "./DATAROOT" # Path to dataset root, which should have subfolders for each model (e.g. ./DATAROOT/imagenet_ai_0508_adm). Can also be a gcs path (gs://my-bucket/data) or Azure container URL root (https://<account>.blob.core.windows.net/<container>/<optional-prefix>)
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
    checkpoints_dir: str = "./checkpoints" # Directory to save checkpoints and logs. Can be a gcs path (e.g. gs://my-bucket/checkpoints) if using --gcp_project_name or an Azure Blob URL (e.g. https://<account>.blob.core.windows.net/<container>/<optional-prefix>)
    fs: Literal['local', 'gcs', 'azure'] = "local" # Whether to load/save data from the local filesystem, Google Cloud Storage (GCS), or Azure Blob Storage. It is inferred from gcp_project_name and dataroot format.
    use_wds: bool = False # Use WebDataset
    
    def process_args(self):
        if self.gcp_project_name is not None:
            self.fs = 'gcs'
        elif _is_http_url(self.dataroot):
            self.fs = 'azure'
        else:
            self.fs = 'local'
            if self.dataroot.startswith('gs://') or self.checkpoints_dir.startswith('gs://'):
                raise ValueError("When --gcp_project_name is not set, dataroot and checkpoints_dir should be local paths, not gcs paths (e.g. ./data, not gs://my-bucket/data)")
            if self.dataroot.startswith('az://') or self.checkpoints_dir.startswith('az://'):
                raise ValueError("For Azure, set dataroot to a full container URL (e.g. https://<account>.blob.core.windows.net/<container>/<optional-prefix>) instead of az:// paths")
        if self.fs == 'gcs':
            if not self.dataroot.startswith('gs://'):
                raise ValueError("When using --gcp_project_name, dataroot should be a gcs path (e.g. gs://my-bucket/data)")
            if not self.checkpoints_dir.startswith('gs://'):
                raise ValueError("When using --gcp_project_name, checkpoints_dir should be a gcs path (e.g. gs://my-bucket/checkpoints)")
        if self.fs == 'azure':
            if not _is_azure_blob_url(self.dataroot):
                raise ValueError("When using Azure Blob Storage, dataroot should be a full container URL (e.g. https://<account>.blob.core.windows.net/<container>/<optional-prefix>)")
            if self.checkpoints_dir.startswith('gs://') or self.checkpoints_dir.startswith('az://'):
                raise ValueError("When using Azure Blob Storage, checkpoints_dir should be a local path or an Azure Blob URL (e.g. ./checkpoints or https://<account>.blob.core.windows.net/<container>/<optional-prefix>)")

        if _is_http_url(self.checkpoints_dir) and not _is_azure_blob_url(self.checkpoints_dir):
            raise ValueError("checkpoints_dir supports http(s) only for Azure Blob URLs (e.g. https://<account>.blob.core.windows.net/<container>/<optional-prefix>)")

        if self.gpu_ids == [-1] or not torch.cuda.is_available():
            self.gpu_ids = []
        else:
            torch.cuda.set_device(self.gpu_ids[0])

    def get_dataset_options(self, split: str = "train") -> DatasetOptions:
        return DatasetOptions(
            models=self.models,
            dataroot=self.dataroot,
            split=split,
            transforms={
                "jpeg": self.jpeg_p,
                "blur": self.blur_p,
                "hflip": self.hflip_p,
            },
            blur_sigma=self.blur_sigma,
            jpeg_quality=self.jpeg_qual,
            batch_size=self.batch_size,
            workers=self.workers,
            gcp_project_name=self.gcp_project_name,
            use_wds=self.use_wds,
        )
        
    @property
    def dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options()
        
    

