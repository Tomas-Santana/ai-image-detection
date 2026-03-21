from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import v2
import os
from options.data_options import DatasetOptions
from PIL import Image
from io import BytesIO
from dataflux_pytorch import dataflux_mapstyle_dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _join_data_path(root: str, *parts: str) -> str:
    clean_parts = [part.strip("/\\") for part in parts if part]
    if root.startswith("gs://"):
        return "/".join([root.rstrip("/"), *clean_parts])
    return os.path.join(root, *clean_parts)


class Processor:
    """
    Returns:
      - input_img:  normalized tensor, used for local patch crops
      - cropped_img: normalized 224x224 tensor, used for global CLIP branch
      - scale: (H, W) of the image that `input_img` corresponds to
    """

    def __init__(
        self,
        opt: DatasetOptions,
        *,
        train: bool,
        input_size: int = 512,
        crop_size: int = 224,
    ):
        self._train = train
        self._crop_size = crop_size

        self._to_image = v2.ToImage()
        
        self._augment = v2.Compose(
            [
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=5, sigma=opt.blur_sigma)],
                    p=opt.transforms.get("blur", 0),
                ),
                v2.RandomApply(
                    [v2.JPEG(quality=opt.jpeg_quality)], 
                    p=opt.transforms.get("jpeg", 0)
                ),
                v2.RandomHorizontalFlip(
                    p=opt.transforms.get("hflip", 0)
                ),
            ]
        )
        
        self._make_square = v2.Compose(
            [
                v2.Resize(input_size, antialias=True),
                v2.CenterCrop(input_size),
            ]
        )

        self._make_cropped = v2.Compose(
            [
                v2.Resize(crop_size, antialias=True),
                v2.CenterCrop(crop_size),
            ]
        )

        self._to_float = v2.ToDtype(torch.float32, scale=True)
        self._norm = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(
        self, pil_img: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self._to_image(pil_img)
        
        if self._train:
            img = self._augment(img)
            
        base = self._make_square(img)
        _, h, w = base.shape
            
        input_img = self._norm(self._to_float(base))

        cropped = self._make_cropped(base)
        cropped_img = self._norm(self._to_float(cropped))

        scale = torch.tensor([h, w])
        return input_img, cropped_img, scale


def patch_collate_train(batch):
    input_img = torch.stack([item[0] for item in batch], dim=0)
    cropped_img = torch.stack([item[1] for item in batch], dim=0)
    target = torch.tensor([item[2] for item in batch])
    scale = torch.stack([item[3] for item in batch], dim=0)
    return [input_img, cropped_img, target, scale]


def patch_collate_test(batch):
    input_img = torch.stack([item[0] for item in batch], dim=0)
    cropped_img = torch.stack([item[1] for item in batch], dim=0)
    target = torch.tensor([item[2] for item in batch])
    scale = torch.stack([item[3] for item in batch], dim=0)
    filename = [item[4] for item in batch]
    return [input_img, cropped_img, target, scale, filename]

class GenImageDataset(Dataset):
    """Like GenImageDataset, but returns the old GLFF tuple.

    Output matches data/dataset_train.py:
      (input_img, cropped_img, target, scale, img_name)
    """

    def __init__(
        self,
        root: str,
        opt: DatasetOptions,
        *,
        train: bool = True,
        input_size: int = 512,
        crop_size: int = 224,
    ):
        super().__init__()
        self.root = root
        self.processor = Processor(
            opt, train=train, input_size=input_size, crop_size=crop_size
        )

        self.images: list[str] = []
        self.labels: list[int] = []
        for label_idx, folder in enumerate(["ai", "nature"]):
            folder_path = os.path.join(self.root, folder)
            if os.path.exists(folder_path):
                imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
                self.images.extend(imgs)
                self.labels.extend([int(label_idx)] * len(imgs))

        self.image_len = len(self.images)

    def rgb_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index: int):
        try:
            img_path = self.images[index]
            image = self.rgb_loader(img_path)
            label = self.labels[index]
        except Exception:
            new_index = max(0, index - 1)
            img_path = self.images[new_index]
            image = self.rgb_loader(img_path)
            label = self.labels[new_index]

        input_img, cropped_img, scale = self.processor(image)
        return input_img, cropped_img, label, scale, img_path

    def __len__(self):
        return self.image_len


def load_dataflux_mapstyle_dataset(
    model_path: str,
    opt: DatasetOptions,
    *,
    train: bool = True,
    input_size: int = 256,
    crop_size: int = 224,
) -> dataflux_mapstyle_dataset.DataFluxMapStyleDataset:
    bucket_name = model_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(model_path.replace("gs://", "").split("/")[1:])

    processor = Processor(opt, train=train, input_size=input_size, crop_size=crop_size)

    def format_fn(path: str, bytes_content: bytes):
        img: Image.Image = Image.open(BytesIO(bytes_content)).convert("RGB")
        label = 1 if "nature" in path else 0
        input_img, cropped_img, scale = processor(img)
        return input_img, cropped_img, int(label), scale, path

    return dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
        project_name=opt.gcp_project_name,
        bucket_name=bucket_name,
        config=dataflux_mapstyle_dataset.Config(prefix=prefix),
        data_format_fn=format_fn,
    )


def load_dataset(
    model_path: str,
    opt: DatasetOptions,
    *,
    train: bool = True,
    input_size: int = 256,
    crop_size: int = 224,
) -> Dataset:
    if model_path.startswith("gs://"):
        return load_dataflux_mapstyle_dataset(
            model_path,
            opt,
            train=train,
            input_size=input_size,
            crop_size=crop_size,
        )
    return GenImageDataset(
        model_path, opt, train=train, input_size=input_size, crop_size=crop_size
    )


def get_loader(
    opt: DatasetOptions,
    *,
    train: bool = True,
    input_size: int = 256,
    crop_size: int = 224,
    include_filenames: bool = False,
) -> DataLoader:
    """
    Each batch item is:
      input_img:  [3, input_size, input_size]
      cropped_img:[3, crop_size, crop_size]
      target:     int (0=ai, 1=nature)
      scale:      [2] tensor (H, W) matching input_img
      img_name:   string path
    """

    datasets: list[Dataset] = []
    for model in opt.models:
        model_path = _join_data_path(opt.dataroot, model, opt.split)
        datasets.append(
            load_dataset(
                model_path,
                opt,
                train=train,
                input_size=input_size,
                crop_size=crop_size,
            )
        )

    dataset = torch.utils.data.ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=train,
        collate_fn=patch_collate_test if include_filenames else patch_collate_train,
        num_workers=opt.workers,
        pin_memory=True,
    )