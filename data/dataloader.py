from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import v2
import os
import glob
import urllib.request
import xml.etree.ElementTree as ET
from options.data_options import DatasetOptions
from PIL import Image
from io import BytesIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dataflux_pytorch import dataflux_mapstyle_dataset
from azstoragetorch.datasets import BlobDataset, Blob
import webdataset as wds


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
LABEL_AI = 0
LABEL_NATURE = 1


def _join_data_path(root: str, *parts: str) -> str:
    clean_parts = [part.strip("/\\") for part in parts if part]
    if root.startswith("https://") or root.startswith("http://"):
        parsed = urlsplit(root)
        base_path = parsed.path.rstrip('/\\')

        if base_path and clean_parts:
            joined_path = '/'.join([base_path, *clean_parts])
        elif base_path:
            joined_path = base_path
        elif clean_parts:
            joined_path = '/' + '/'.join(clean_parts)
        else:
            joined_path = '/'

        if not joined_path.startswith('/'):
            joined_path = '/' + joined_path

        return urlunsplit((parsed.scheme, parsed.netloc, joined_path, parsed.query, parsed.fragment))

    if root.startswith("gs://") or root.startswith("az://"):
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
        force_augment: bool = False,
    ):
        self._train = train
        self._force_augment = force_augment
        self._crop_size = crop_size

        self._to_image = v2.ToImage()
        
        self._augment = v2.Compose(
            [
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=3, sigma=opt.blur_sigma)],
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
        self._norm = v2.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def __call__(
        self, pil_img: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self._to_image(pil_img)
        
        if self._train or self._force_augment:
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
        force_augment: bool = False,
    ):
        super().__init__()
        self.root = root
        self.processor = Processor(
            opt, train=train, input_size=input_size, crop_size=crop_size, force_augment=force_augment
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


def load_azstoragetorch_blob_dataset(
    model_path: str,
    opt: DatasetOptions,
    *,
    train: bool = True,
    input_size: int = 256,
    crop_size: int = 224,
    force_augment: bool = False,
) -> BlobDataset:
    # model_path format: https://<account>.blob.core.windows.net/<container>/<prefix>
    parsed = urlsplit(model_path)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError(
            f"Invalid Azure model path '{model_path}': expected full container URL path with https://<account>.blob.core.windows.net/<container>/<prefix>"
        )
    path_parts = [p for p in parsed.path.split("/") if p]
    if not path_parts:
        raise ValueError(
            f"Invalid Azure model path '{model_path}': container name must not be empty"
        )
    container_name = path_parts[0]
    prefix = "/".join(path_parts[1:])

    container_url = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/{container_name}",
            parsed.query,
            parsed.fragment,
        )
    )

    processor = Processor(opt, train=train, input_size=input_size, crop_size=crop_size, force_augment=force_augment)

    def transform_fn(blob: Blob):
        with blob.reader() as f:
            bytes_content = f.read()
        img: Image.Image = Image.open(BytesIO(bytes_content)).convert("RGB")
        blob_name = blob.blob_name
        label = LABEL_NATURE if "nature" in blob_name else LABEL_AI
        input_img, cropped_img, scale = processor(img)
        return input_img, cropped_img, int(label), scale, blob_name

    return BlobDataset.from_container_url(
        container_url,
        prefix=prefix if prefix else None,
        transform=transform_fn, # pyright: ignore[reportArgumentType]
    )


def load_dataflux_mapstyle_dataset(
    model_path: str,
    opt: DatasetOptions,
    *,
    train: bool = True,
    input_size: int = 256,
    crop_size: int = 224,
    force_augment: bool = False,
) -> dataflux_mapstyle_dataset.DataFluxMapStyleDataset:
    bucket_name = model_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(model_path.replace("gs://", "").split("/")[1:])

    processor = Processor(opt, train=train, input_size=input_size, crop_size=crop_size, force_augment=force_augment)

    def format_fn(path: str, bytes_content: bytes):
        img: Image.Image = Image.open(BytesIO(bytes_content)).convert("RGB")
        label = LABEL_NATURE if "nature" in path else LABEL_AI
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
    force_augment: bool = False,
) -> Dataset:
    if model_path.startswith("gs://"):
        return load_dataflux_mapstyle_dataset(
            model_path,
            opt,
            train=train,
            input_size=input_size,
            crop_size=crop_size,
            force_augment=force_augment,
        )
    if model_path.startswith("https://") or model_path.startswith("http://"):
        return load_azstoragetorch_blob_dataset(
            model_path,
            opt,
            train=train,
            input_size=input_size,
            crop_size=crop_size,
            force_augment=force_augment,
        )
    return GenImageDataset(
        model_path, opt, train=train, input_size=input_size, crop_size=crop_size, force_augment=force_augment
    )


def _list_azure_wds_shards(dataroot_url: str, model: str, split_name: str) -> list[str]:
    parsed = urlsplit(dataroot_url)
    path_parts = [p for p in parsed.path.split("/") if p]
    if not path_parts:
        return []

    container_name = path_parts[0]
    base_prefix = "/".join(path_parts[1:]).strip("/")
    prefix_parts = [part for part in [base_prefix, model, f"{split_name}-"] if part]
    blob_prefix = "/".join(prefix_parts)

    sas_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    marker = ""
    shard_urls: list[str] = []

    while True:
        query_pairs: list[tuple[str, str]] = [
            ("restype", "container"),
            ("comp", "list"),
            ("prefix", blob_prefix),
        ]
        if marker:
            query_pairs.append(("marker", marker))
        query_pairs.extend(sas_pairs)

        list_url = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"/{container_name}",
                urlencode(query_pairs),
                "",
            )
        )

        with urllib.request.urlopen(list_url, timeout=30) as response:
            payload = response.read()

        root = ET.fromstring(payload)
        for name_node in root.findall(".//{*}Blob/{*}Name"):
            blob_name = name_node.text or ""
            if not blob_name.endswith(".tar"):
                continue
            blob_url = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    f"/{container_name}/{blob_name}",
                    parsed.query,
                    parsed.fragment,
                )
            )
            shard_urls.append(blob_url)

        marker = root.findtext(".//{*}NextMarker") or ""
        if not marker:
            break

    return sorted(shard_urls)


def is_not_none(x):
    return x is not None

class WDSDecoder:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, sample):
        key = sample.get("__key__", "")
        img_pil = (
            sample.get("jpg")
            or sample.get("jpeg")
            or sample.get("png")
            or sample.get("webp")
        )
        label_int = sample.get("cls")
        if img_pil is None:
            return None

        # Prefer key-derived labels so old/new shards stay consistent.
        lowered_key = str(key).lower()
        if lowered_key.startswith("nature/") or "/nature/" in lowered_key:
            label_int = LABEL_NATURE
        elif lowered_key.startswith("ai/") or "/ai/" in lowered_key:
            label_int = LABEL_AI

        if label_int is None:
            return None
        try:
            label_int = int(label_int)
        except (ValueError, TypeError):
            pass
        input_img, cropped_img, scale = self.processor(img_pil)
        return input_img, cropped_img, label_int, scale, key

def get_loader(
    opt: DatasetOptions,
    *,
    train: bool = True,
    input_size: int = 256,
    crop_size: int = 224,
    include_filenames: bool = False,
    force_augment: bool = False,
) -> DataLoader:
    """
    Each batch item is:
      input_img:  [3, input_size, input_size]
      cropped_img:[3, crop_size, crop_size]
      target:     int (0=ai, 1=nature)
      scale:      [2] tensor (H, W) matching input_img
      img_name:   string path
    """

    if getattr(opt, "use_wds", False):
        urls = []
        split_name = "train" if train else "val"
        for model in opt.models:
            if opt.dataroot.startswith("https://") or opt.dataroot.startswith("http://"):
                urls.extend(_list_azure_wds_shards(opt.dataroot, model, split_name))
            else:
                search_pattern = _join_data_path(opt.dataroot, model, f"{split_name}-*.tar")
                sorted_matched_files = sorted(glob.glob(search_pattern))
                urls.extend(sorted_matched_files)

        if not urls:
            raise ValueError(
                f"No WebDataset shards found for split '{split_name}'. Check dataroot, models, and SAS permissions."
            )
        
        processor = Processor(opt, train=train, input_size=input_size, crop_size=crop_size, force_augment=force_augment)
        decoder = WDSDecoder(processor)
        loader_workers = min(max(1, opt.workers), len(urls))

        dataset = wds.WebDataset( # type: ignore
            urls,
            nodesplitter=wds.split_by_node, # type: ignore
            shardshuffle=1000 if train else False,
            handler=wds.ignore_and_continue, # type: ignore
            empty_check=False,
        ) # type: ignore
        if train:
            shuffle_buffer = max(1000, opt.batch_size * 200)
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.decode("pil", handler=wds.ignore_and_continue).map(decoder, handler=wds.ignore_and_continue).select(is_not_none) # type: ignore
        if getattr(opt, "wds_cache_in_ram", False):
            dataset = dataset.mcached() # type: ignore
        return DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=loader_workers,
            pin_memory=True,
            collate_fn=patch_collate_test if include_filenames else patch_collate_train,
        )

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
                force_augment=force_augment,
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