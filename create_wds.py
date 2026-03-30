import os
import webdataset as wds
import random
from itertools import zip_longest


def _collect_class_images(split_path: str, cls_name: str) -> list[str]:
    cls_path = os.path.join(split_path, cls_name)
    if not os.path.exists(cls_path):
        return []

    images: list[str] = []
    for root, dirs, files in os.walk(cls_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                images.append(os.path.join(root, file))
    return images


def _iter_interleaved_paths(class_to_files: dict[str, list[str]]):
    # Interleave per-class shuffled lists so each shard/batch sees a better class mix.
    per_class_iters = [iter(paths) for paths in class_to_files.values()]
    for group in zip_longest(*per_class_iters):
        for item in group:
            if item is not None:
                yield item

def process_dataset():
    input_base = "DATAROOT"
    output_base = "DATAROOT_WDS"
    
    # Class to integer mapping used throughout train/test loaders.
    class_to_idx = {"nature": 1, "ai": 0}

    # Iterate over all datasets in DATAROOT
    if not os.path.exists(input_base):
        print(f"Directory {input_base} not found.")
        return

    datasets = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]
    
    for dataset in datasets:
        dataset_path = os.path.join(input_base, dataset)
        
        # Process train and val splits
        for split in ["train", "val"]:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue
            
            # Create output directory for shards
            out_dir = os.path.join(output_base, dataset)
            os.makedirs(out_dir, exist_ok=True)
            
            # Will produce files like DATAROOT_WDS/imagenet_ai_0419_biggan/train-000000.tar
            pattern = os.path.join(out_dir, f"{split}-%06d.tar")

            class_to_files: dict[str, list[str]] = {}
            for cls_name in class_to_idx:
                files = _collect_class_images(split_path, cls_name)
                random.shuffle(files)
                class_to_files[cls_name] = files

            ordered_paths = list(_iter_interleaved_paths(class_to_files))
            
            # Using ShardWriter to automatically split into chunks of ~1000 samples or ~100MB
            with wds.ShardWriter(pattern, maxcount=1000) as sink: # type:ignore
                for filepath in ordered_paths:
                    relpath = os.path.relpath(filepath, split_path)
                    relpath = relpath.replace("\\", "/")
                    cls_name = relpath.split("/", 1)[0]
                    cls_label = class_to_idx[cls_name]

                    with open(filepath, "rb") as stream:
                        image_data = stream.read()

                    filename = os.path.basename(filepath)
                    ext = filename.split('.')[-1].lower()
                    key = os.path.splitext(relpath)[0]

                    sink.write({
                        "__key__": key,
                        ext: image_data,
                        "cls": cls_label,
                    })
            print(f"Finished processing {dataset} - {split}")

if __name__ == "__main__":
    process_dataset()
