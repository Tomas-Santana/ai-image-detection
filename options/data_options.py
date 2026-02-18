from typing import Literal, Dict, Optional
from dataclasses import dataclass

TransformName = Literal["jpeg", "blur", "hflip"]

TransformOpt = Dict[TransformName, float]

Models = Literal[
    "imagenet_ai_0508_adm",
    "imagenet_ai_0419_biggan",
    "imagenet_glide",
    "imagenet_midjourney",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_wukong",
]


@dataclass
class DatasetOptions:
    models: list[Models]
    dataroot: str = "./data"
    transforms: TransformOpt = {"jpeg": 0.5, "blur": 0.3}
    blur_sigma: tuple[float, float] = (0.1, 2.0)
    jpeg_quality: tuple[int, int] = (10, 50)
    batch_size: int = 32
    workers: int = 4
    gcp_project_name: Optional[str] = None
