from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from torchvision.transforms import v2
import os
from options.data_options import DatasetOptions
from PIL import Image
from io import BytesIO
from dataflux_pytorch import dataflux_mapstyle_dataset


def build_processing_pipeline(opt: DatasetOptions):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=5, sigma=opt.blur_sigma)],
                p=opt.transforms.get("blur", 0),
            ),
            v2.RandomApply(
                [v2.JPEG(quality=opt.jpeg_quality)], p=opt.transforms.get("jpeg", 0)
            ),
            v2.Resize((256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),  # Convierte a [0, 1]
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class GenImageDataset(Dataset):
    def __init__(self, root: str, pipeline: v2.Compose | transforms.Compose):
        super().__init__()
        self.pipeline = pipeline
        self.root = root

        self.images = []
        self.labels = []

        for label_idx, folder in enumerate(["ai", "nature"]):
            folder_path = os.path.join(self.root, folder)
            if os.path.exists(folder_path):
                imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
                self.images.extend(imgs)
                self.labels.extend([float(label_idx)] * len(imgs))

        self.image_len = len(self.images)

    def rgb_loader(self, path) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        try:
            image = self.rgb_loader(self.images[index])
            label = self.labels[index]
        except Exception:
            new_index = index - 1
            image = self.rgb_loader(self.images[max(0, new_index)])
            label = self.labels[max(0, new_index)]

        image = self.pipeline(image)
        return image, label

    def __len__(self):
        return self.image_len


def load_dataflux_mapstyle_dataset(
    opt: DatasetOptions,
) -> dataflux_mapstyle_dataset.DataFluxMapStyleDataset:
    bucket_name = opt.dataroot.replace("gs://", "").split("/")[0]
    prefix = "/".join(opt.dataroot.replace("gs://", "").split("/")[1:])

    pipeline = build_processing_pipeline(opt)

    def format_fn(path: str, bytes_content: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        img: Image.Image = Image.open(BytesIO(bytes_content)).convert("RGB")
        label = 1 if "nature" in path else 0
        return pipeline(img), torch.tensor(label)

    return dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
        project_name=opt.gcp_project_name,
        bucket_name=bucket_name,
        config=dataflux_mapstyle_dataset.Config(prefix=prefix),
        data_format_fn=format_fn,
    )


def load_dataset(model_path: str, opt: DatasetOptions) -> Dataset:
    if model_path.startswith("gs://"):
        return load_dataflux_mapstyle_dataset(opt)
    else:
        return GenImageDataset(model_path, build_processing_pipeline(opt))


def get_loader(opt: DatasetOptions):
    datasets: list[Dataset] = []
    for model in opt.models:
        model_path = os.path.join(opt.dataroot, model)
        datasets.append(load_dataset(model_path, opt))

    train_dataset = torch.utils.data.ConcatDataset(datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
    )
    return train_loader


"""
=================
CODE FROM THE ORIGINAL SSP
Probably not needed
=================
"""

"""mp = {
    0: "imagenet_ai_0508_adm",
    1: "imagenet_ai_0419_biggan",
    2: "imagenet_glide",
    3: "imagenet_midjourney",
    4: "imagenet_ai_0424_sdv5",
    5: "imagenet_ai_0419_vqdm",
    6: "imagenet_ai_0424_wukong",
}


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def sample_randint(s):
    if len(s) == 1:
        return s[0]
    return rd.randint(s[0], s[1])


def gaussian_blur_gray(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def data_augment(img, opt: DatasetOptions):
    img = np.array(img)

    if random() < opt.transforms.get("blur", 0):
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.transforms.get("jpeg", 0):
        method = sample_discrete(opt.jpg_method)
        qual = sample_randint(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


class GenImageValDataset(Dataset):
    def __init__(self, image_root, image_dir, is_real, opt):
        super().__init__()
        self.opt = opt
        self.root = os.path.join(image_root, image_dir, "val")
        if is_real:
            self.img_path = os.path.join(self.root, "nature")
            self.img_list = [
                os.path.join(self.img_path, f) for f in os.listdir(self.img_path)
            ]
            self.img_len = len(self.img_list)
            self.labels = torch.ones(self.img_len)
        else:
            self.img_path = os.path.join(self.root, "ai")
            self.img_list = [
                os.path.join(self.img_path, f) for f in os.listdir(self.img_path)
            ]
            self.img_len = len(self.img_list)
            self.labels = torch.zeros(self.img_len)

    def rgb_loader(self, path) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        image = self.rgb_loader(self.img_list[index])
        label = self.labels[index]
        # image = processing(image, self.opt)
        return image, label

    def __len__(self):
        return self.img_len


def get_single_loader(opt, image_dir, is_real):
    val_dataset = GenImageValDataset(
        opt.image_root, image_dir=image_dir, is_real=is_real, opt=opt
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return val_loader, len(val_dataset)


def get_val_loader(opt):
    choices = opt.choices
    loader = []
    for i, choice in enumerate(choices):
        datainfo = dict()
        if choice == 0 or choice == 1:
            print("val on:", mp[i])
            datainfo["name"] = mp[i]
            datainfo["val_ai_loader"], datainfo["ai_size"] = get_single_loader(
                opt, datainfo["name"], is_real=False
            )
            datainfo["val_nature_loader"], datainfo["nature_size"] = get_single_loader(
                opt, datainfo["name"], is_real=True
            )
            loader.append(datainfo)
    return loader
"""