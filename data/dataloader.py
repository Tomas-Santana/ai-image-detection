from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from torchvision.transforms import v2
import os
from options.data_options import DatasetOptions
from PIL import Image
from io import BytesIO
from dataflux_pytorch import dataflux_mapstyle_dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        input_size: int = 256,
        crop_size: int = 224,
    ):
        self._train = train
        self._input_size = input_size
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
        _, h, w = img.shape
        
        if self._train:
            img = self._augment(img)
            
        input_img = self._norm(self._to_float(img))

        cropped = self._make_cropped(img)
        cropped_img = self._norm(self._to_float(cropped))

        scale = torch.tensor([h, w])
        return input_img, cropped_img, scale

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
        input_size: int = 256,
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
    opt: DatasetOptions,
    *,
    train: bool = True,
    input_size: int = 256,
    crop_size: int = 224,
) -> dataflux_mapstyle_dataset.DataFluxMapStyleDataset:
    bucket_name = opt.dataroot.replace("gs://", "").split("/")[0]
    prefix = "/".join(opt.dataroot.replace("gs://", "").split("/")[1:])

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
            opt, train=train, input_size=input_size, crop_size=crop_size
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
        model_path = os.path.join(opt.dataroot, model)
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
        num_workers=opt.workers,
        pin_memory=True,
    )


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
