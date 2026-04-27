import argparse
import io
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from data.dataloader import Processor, CLIP_MEAN, CLIP_STD, DatasetOptions
from networks.patch_model import Patch5Model, Patch5ModelGlobalOnly
from storage.default import get_storage_fs


def _load_model(
    checkpoint_path: str,
    device: torch.device,
    fs,
    backbone: str,
    variant: str | None,
) -> torch.nn.Module:
    model = Patch5ModelGlobalOnly() if variant == "global-only" else Patch5Model()
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


def main():
    parser = argparse.ArgumentParser(description="Test a single image and visualize extracted patches.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--output_path", type=str, default="visualization.jpg", help="Path to save the visualization.")
    parser.add_argument("--variant", type=str, default=None, choices=["global-only", None])
    parser.add_argument("--backbone", type=str, default="clip", choices=["clip", "resnet"])
    parser.add_argument("--input_size", type=int, default=512, help="Input size for the image (local prep).")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for the global model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[{device.type.upper()}] Loading model from {args.model_path}...")
    checkpoint_fs = get_storage_fs(args.model_path)
    model = _load_model(args.model_path, device, checkpoint_fs, args.backbone, args.variant)
    
    print(f"Loading image {args.image_path}...")
    img_fs = get_storage_fs(args.image_path)
    img_bytes = img_fs.read_bytes(args.image_path)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    opt = DatasetOptions(models=[])
    processor = Processor(opt, train=False, input_size=args.input_size, crop_size=args.crop_size, force_augment=False)
    
    input_img, cropped_img, scale = processor(img_pil)
    
    input_img_t = input_img.unsqueeze(0).to(device)
    cropped_img_t = cropped_img.unsqueeze(0).to(device)
    scale_t = scale.unsqueeze(0).to(device)

    print("Running inference...")
    use_amp = device.type == "cuda"
    with torch.no_grad():
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False):
            logits = model(input_img_t, cropped_img_t, scale_t)
            probability = torch.sigmoid(logits).item()  # 1.0 = Nature, 0.0 = AI
            
            input_loc = None
            if not isinstance(model, Patch5ModelGlobalOnly):
                spatial_maps, _ = model.clip(cropped_img_t)
                early, mid, late = spatial_maps
                fused_global_maps = model.fusion(early, mid, late)
                
                input_loc_tensor, _ = model.COOI.get_coordinates(fused_global_maps, scale_t)
                input_loc = input_loc_tensor.cpu()

    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1).to(input_img.device)
    std = torch.tensor(CLIP_STD).view(3, 1, 1).to(input_img.device)
    
    img_show = input_img * std + mean
    img_show = img_show.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_show)
    ax.axis("off")
    
    classification_text = "NATURE (Real)" if probability >= 0.5 else "AI GENERATED (Fake)"
    confidence = probability if probability >= 0.5 else (1.0 - probability)
    color_title = 'darkgreen' if probability >= 0.5 else 'darkred'
    
    title = f"Prediction: {classification_text}\nConfidence: {confidence * 100:.2f}%"
    
    if input_loc is not None:
        title += f"  |  Local Patches: {input_loc.size(1)}"
        for proposal_no in range(input_loc.size(1)):
            t, left, b, r = input_loc[0, proposal_no].tolist()
            width = r - left
            height = b - t
            
            rect = patches.Rectangle(
                (left, t), width, height,
                linewidth=3.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(
                left, max(t - 10, 10), 
                f"Patch {proposal_no+1}", 
                color='white', fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='red', edgecolor='red', pad=1.5, alpha=0.7)
            )
            
    plt.title(title, fontsize=18, fontweight='bold', color=color_title, pad=20)
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches='tight', dpi=300)
    
    print(f"\n--- SUCCESS ---")
    print(f"Prediction: {classification_text} ({confidence * 100:.2f}%)")
    print(f"Visualization saved successfully to: {args.output_path}")

if __name__ == "__main__":
    main()
