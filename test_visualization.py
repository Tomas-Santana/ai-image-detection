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
    parser = argparse.ArgumentParser(description="Probar una sola imagen y visualizar los parches extraídos.")
    parser.add_argument("--image_path", type=str, required=True, help="Ruta a la imagen de entrada.")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al checkpoint del modelo entrenado.")
    parser.add_argument("--output_path", type=str, default="visualizacion.jpg", help="Ruta para guardar la visualización.")
    parser.add_argument("--variant", type=str, default=None, choices=["global-only", None])
    parser.add_argument("--backbone", type=str, default="clip", choices=["clip", "resnet"])
    parser.add_argument("--input_size", type=int, default=512, help="Tamaño de entrada para la imagen (prep local).")
    parser.add_argument("--crop_size", type=int, default=224, help="Tamaño de recorte para el modelo global.")
    parser.add_argument("--etiqueta_real", type=str, choices=["real", "ia"], default=None, help="Etiqueta real de la imagen ('real' o 'ia').")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[{device.type.upper()}] Cargando modelo desde {args.model_path}...")
    checkpoint_fs = get_storage_fs(args.model_path)
    model = _load_model(args.model_path, device, checkpoint_fs, args.backbone, args.variant)
    
    print(f"Cargando imagen {args.image_path}...")
    img_fs = get_storage_fs(args.image_path)
    img_bytes = img_fs.read_bytes(args.image_path)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    opt = DatasetOptions(models=[])
    processor = Processor(opt, train=False, input_size=args.input_size, crop_size=args.crop_size, force_augment=False)
    
    input_img, cropped_img, scale = processor(img_pil)
    
    input_img_t = input_img.unsqueeze(0).to(device)
    cropped_img_t = cropped_img.unsqueeze(0).to(device)
    scale_t = scale.unsqueeze(0).to(device)

    print("Ejecutando inferencia...")
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
    
    classification_text = "NATURALEZA (Real)" if probability >= 0.5 else "GENERADA POR IA (Falsa)"
    confidence = probability if probability >= 0.5 else (1.0 - probability)
    color_title = 'darkgreen' if probability >= 0.5 else 'darkred'
    
    title = f"Predicción: {classification_text}\nConfianza: {confidence * 100:.2f}%"
    
    texto_resultado = None
    if args.etiqueta_real:
        es_real_pred = probability >= 0.5
        es_real_true = args.etiqueta_real == "real"
        acierto = es_real_pred == es_real_true
        texto_resultado = "¡ACIERTO!" if acierto else "INCORRECTO"
        title += f"\nResultado final: {texto_resultado} (Etiqueta real: {args.etiqueta_real.upper()})"
    
    if input_loc is not None:
        title += f"  |  Parches Locales: {input_loc.size(1)}"
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
                f"Parche {proposal_no+1}", 
                color='white', fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='red', edgecolor='red', pad=1.5, alpha=0.7)
            )
            
    plt.title(title, fontsize=16, fontweight='bold', color=color_title, pad=20)
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"\n--- ÉXITO ---")
    print(f"Predicción: {classification_text} ({confidence * 100:.2f}%)")
    if texto_resultado:
        print(f"Evaluación del modelo: {texto_resultado}")
    print(f"Visualización guardada exitosamente en: {args.output_path}")

if __name__ == "__main__":
    main()
