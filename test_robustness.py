import csv
import io
from collections import OrderedDict
from typing import Any, Literal, Sized, cast
from itertools import product

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve, auc
from tqdm import tqdm

from data.dataloader import get_loader
from networks.patch_model import Patch5Model
from options.data_options import Models
from options.test_options import TestOptions
from storage.base import BaseFS
from storage.default import get_storage_fs


def _best_threshold_by_accuracy(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        return float("nan"), float("nan")

    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    acc_by_threshold = (tpr * positives + (1.0 - fpr) * negatives) / (positives + negatives)
    best_index = int(np.argmax(acc_by_threshold))
    return float(thresholds[best_index]), float(acc_by_threshold[best_index])


def validate(model: torch.nn.Module, data_loader) -> tuple[float, float, float, int, float, float]:
    device = next(model.parameters()).device
    amp = cast(Any, torch.amp)
    use_amp = device.type == "cuda"

    with torch.no_grad():
        y_true: list[float] = []
        y_pred: list[float] = []

        for data in tqdm(data_loader, desc="Testing", leave=False):
            input_img = data[0]
            cropped_img = data[1].to(device)
            label = data[2].to(device)
            scale = data[3].to(device)

            with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits = model(input_img, cropped_img, scale)
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr > 0.5)
    unique_labels = np.unique(y_true_arr)
    best_threshold, best_acc = _best_threshold_by_accuracy(y_true_arr, y_pred_arr)

    if np.any(y_true_arr == 1):
        ap = average_precision_score(y_true_arr, y_pred_arr)
    else:
        ap = float('nan')

    if unique_labels.size >= 2:
        fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = float('nan')
    num_images = int(y_true_arr.shape[0])
    return float(acc), float(roc_auc), float(ap), num_images, float(best_threshold), float(best_acc)


def _load_model(
    checkpoint_path: str,
    device: torch.device,
    fs: BaseFS,
    backbone: Literal["clip", "resnet"],
) -> torch.nn.Module:
    model = Patch5Model()
    checkpoint_bytes = fs.read_bytes(checkpoint_path)
    state_dict = torch.load(io.BytesIO(checkpoint_bytes), map_location=device)

    model_state = state_dict["model"]
    clean_state = OrderedDict()
    for key, value in model_state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = value

    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()
    return model


def _resolve_models(opt: TestOptions, fs: BaseFS) -> list[Models]:
    if opt.models:
        return sorted(opt.models)

    models = fs.list_model_names(opt.dataroot)
    if not models:
        raise ValueError(
            f"No models found under dataroot '{opt.dataroot}'. Expected folders like <dataroot>/<model>/train and <dataroot>/<model>/val"
        )
    return cast(list[Models], models)


def _build_results_csv(rows: list[dict[str, float | int | str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "model",
            "split",
            "blur_sigma",
            "jpeg_quality",
            "num_images",
            "accuracy",
            "roc_auc",
            "average_precision",
            "best_threshold",
            "best_accuracy",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    opt = TestOptions().parse_args()
    checkpoint_fs = get_storage_fs(opt.model_path)
    dataroot_fs = get_storage_fs(opt.dataroot)
    results_fs = get_storage_fs(opt.results_dir)

    device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
    model = _load_model(opt.model_path, device, checkpoint_fs, opt.backbone)

    models = _resolve_models(opt, dataroot_fs)
    print("Models to evaluate:", ", ".join(models))

    # Define the sets of degradation parameters
    blur_sigmas = [0.0, 1.0, 2.0, 3.0]
    jpeg_qualities = [100, 95, 90, 85, 80, 70]

    rows: list[dict[str, float | int | str]] = []
    
    for model_name in models:
        # We test cross combinations: blur=0 with different jpeg OR jpeg=100 with different blur
        # To avoid a massive grid, let's do the typical approach:
        # Test Blur independently (with jpeg=100) and Test JPEG independently (with blur=0)
        # We'll just merge them in a checklist
        
        test_cases = []
        for sigma in blur_sigmas:
            test_cases.append((sigma, 100))
        for quality in jpeg_qualities:
            if quality != 100:  # avoid duplicate (0.0, 100)
                test_cases.append((0.0, quality))
                
        # Make unique cases, preserving order
        test_cases = list(dict.fromkeys(test_cases))

        for sigma, quality in test_cases:
            dataset_options = opt.get_dataset_options(split=opt.test_split)
            dataset_options.models = [model_name]
            
            # Configure data augmentation specifically for robust testing
            dataset_options.transforms = {
                "blur": 1.0 if sigma > 0 else 0.0,
                "jpeg": 1.0 if quality < 100 else 0.0,
                "hflip": 0.0,
            }
            # v2.GaussianBlur throws ValueError if sigma=0.0, so we pass a small >0 value 
            # when sigma is 0 since the probability of applying it is 0.0 anyway.
            dataset_options.blur_sigma = sigma if sigma > 0 else (0.1, 0.1)
            dataset_options.jpeg_quality = quality

            loader = get_loader(
                dataset_options,
                train=False,
                input_size=opt.load_size,
                crop_size=opt.crop_size,
                force_augment=True, # Force transformations on test set
            )

            print(f"Testing [model={model_name}] | Blur Sigma={sigma} | JPEG Quality={quality}")
            acc, roc_auc, ap, num_images, best_threshold, best_acc = validate(model, loader)

            print(
                f" -> acc@0.5={acc:.4f}, best_thr={best_threshold:.4f}, "
                f"acc@best={{best_acc:.4f}}, auc={roc_auc:.4f}, ap={ap:.4f}"
            )

            rows.append(
                {
                    "model": model_name,
                    "split": opt.test_split,
                    "blur_sigma": float(sigma),
                    "jpeg_quality": int(quality),
                    "num_images": num_images,
                    "accuracy": float(acc),
                    "roc_auc": float(roc_auc),
                    "average_precision": float(ap),
                    "best_threshold": float(best_threshold),
                    "best_accuracy": float(best_acc),
                }
            )

    csv_name = f"{opt.experiment_name}_robustness_results.csv" if opt.experiment_name else "robustness_results.csv"
    csv_path = results_fs.join_path(opt.results_dir, csv_name)
    csv_content = _build_results_csv(rows)
    results_fs.write_text(csv_path, csv_content)

    print(f"Saved robustness results to: {csv_path}")


if __name__ == "__main__":
    main()