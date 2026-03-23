from .base_options import BaseOptions
from .data_options import DatasetOptions

class TestOptions(BaseOptions):
    model_path: str = "" # Path to model checkpoint used for evaluation
    is_train: bool = False
    test_split: str = 'val' # Dataset split to use for evaluation
    results_dir: str = './results' # Folder where evaluation csv results are stored (local path or Azure Blob URL prefix)
    no_resize: bool = False
    no_crop: bool = False
    eval: bool = True

    def process_args(self):
        super().process_args()
        if not self.model_path:
            raise ValueError("--model_path must be provided for testing")

    @property
    def test_dataset_options(self) -> DatasetOptions:
        return self.get_dataset_options(split=self.test_split)
