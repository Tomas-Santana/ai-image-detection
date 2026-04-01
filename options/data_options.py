from typing import Literal, Dict, Optional
from dataclasses import dataclass, field

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
    dataroot: str = "./DATAROOT"
    split: str = "train"
    transforms: TransformOpt = field(
        default_factory=lambda: {"jpeg": 0.1, "blur": 0.1, "hflip": 0.0}
    )
    blur_sigma: tuple[float, float] = (0.1, 2.0)
    jpeg_quality: tuple[int, int] = (75, 90)
    batch_size: int = 32
    workers: int = 4
    gcp_project_name: Optional[str] = None
    use_wds: bool = False
    wds_cache_in_ram: bool = False
