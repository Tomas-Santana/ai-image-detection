import csv
import io
from collections import OrderedDict
from typing import Sized, cast

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


def validate(model: torch.nn.Module, data_loader) -> tuple[float, float, float, int]:
    device = next(model.parameters()).device
    try:
        print("number of validation images:", len(cast(Sized, data_loader.dataset)))
    except TypeError:
        print("number of validation images: unknown (iterable dataset)")

    with torch.no_grad():
        y_true: list[float] = []
        y_pred: list[float] = []

        for data in tqdm(data_loader, desc="Testing"):
            input_img = data[0]
            cropped_img = data[1].to(device)
            label = data[2].to(device)
            scale = data[3].to(device)

            logits = model(input_img, cropped_img, scale)
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr > 0.5)
    unique_labels = np.unique(y_true_arr)

    if np.any(y_true_arr == 1):
        ap = average_precision_score(y_true_arr, y_pred_arr)
    else:
        ap = float('nan')

    if unique_labels.size >= 2:
        fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = float('nan')
        print(
            f"Validation labels contain one class only ({unique_labels.tolist()}); roc_auc set to NaN for this run"
        )
    num_images = int(y_true_arr.shape[0])
    return float(acc), float(roc_auc), float(ap), num_images


def _load_model(checkpoint_path: str, device: torch.device, fs: BaseFS) -> torch.nn.Module:
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
        fieldnames=["model", "split", "num_images", "accuracy", "roc_auc", "average_precision"],
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
    model = _load_model(opt.model_path, device, checkpoint_fs)

    models = _resolve_models(opt, dataroot_fs)
    print("Models to evaluate:", ", ".join(models))

    rows: list[dict[str, float | int | str]] = []
    for model_name in models:
        dataset_options = opt.get_dataset_options(split=opt.test_split)
        dataset_options.models = [model_name]

        loader = get_loader(
            dataset_options,
            train=False,
            input_size=opt.load_size,
            crop_size=opt.crop_size,
        )

        acc, roc_auc, ap, num_images = validate(model, loader)

        print(
            f"[model={model_name}] acc={acc:.6f}, roc_auc={roc_auc:.6f}, ap={ap:.6f}, num_images={num_images}"
        )

        rows.append(
            {
                "model": model_name,
                "split": opt.test_split,
                "num_images": num_images,
                "accuracy": float(acc),
                "roc_auc": float(roc_auc),
                "average_precision": float(ap),
            }
        )

    csv_name = f"{opt.experiment_name}_results.csv" if opt.experiment_name else "results.csv"
    csv_path = results_fs.join_path(opt.results_dir, csv_name)
    csv_content = _build_results_csv(rows)
    results_fs.write_text(csv_path, csv_content)

    print(f"Saved per-model results to: {csv_path}")


if __name__ == "__main__":
    main()
