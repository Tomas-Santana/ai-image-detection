import torch
import time
# Asegúrate de ejecutar este script desde la raíz del proyecto para que la importación funcione
from networks.patch_model import Patch5Model
from thop import profile, clever_format
import argparse

def measure_complexity(model, device="cuda"):
    print("=== Análisis de Complejidad Computacional ===")
    
    # 1. Calcular el número de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n[Parámetros]")
    print(f"Parámetros Totales:      {total_params:,}")
    print(f"Parámetros Entrenables:  {trainable_params:,}")

    # 2. Medir el tiempo de inferencia
    model.to(device)
    model.eval() # Modo de evaluación para desactivar Dropout, etc.

    batch_size = 1
    # Creación de tensores dummy basados en las firmas del método forward
    # input_img: Imagen de mayor resolución antes del crop (ej. 448x448 o mayor)
    dummy_input_img = torch.randn(batch_size, 3, 448, 448).to(device)
    # cropped_img: Imagen recortada/redimensionada que va a CLIP (típicamente 224x224)
    dummy_cropped_img = torch.randn(batch_size, 3, 224, 224).to(device)
    # scale: Utilizado por ViTCOOI.get_coordinates (suele ser un tensor de dimensión [B, 2])
    dummy_scale = torch.ones(batch_size, 2).to(device)

    print(f"\n[Tiempo de Inferencia (Batch Size = {batch_size}) en {device.upper()}]")
    
    # Fase de Warm-up (Vital en PyTorch y CUDA para inicializar los contextos y evitar medir los tiempos de carga iniciales)
    print("Realizando warm-up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input_img, dummy_cropped_img, dummy_scale)

    # Variables para medir el tiempo
    num_iterations = 50
    
    if device == "cuda":
        # Usamos torch.cuda.Event para mediciones asíncronas precisas en GPU
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        
        print("Midiendo iteraciones...")
        with torch.no_grad():
            for i in range(num_iterations):
                start_events[i].record()
                _ = model(dummy_input_img, dummy_cropped_img, dummy_scale)
                end_events[i].record()
                
        # Sincronizar para asegurar que todas las operaciones de GPU hayan terminado
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)] # Tiempos en milisegundos
        avg_time_ms = sum(times) / num_iterations
        
    else:
        # Medición tradicional para CPU
        print("Midiendo iteraciones...")
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                _ = model(dummy_input_img, dummy_cropped_img, dummy_scale)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000) # Convertir a milisegundos
        avg_time_ms = sum(times) / num_iterations

    print(f"Tiempo promedio de inferencia: {avg_time_ms:.2f} ms")
    print(f"Frames Por Segundo (FPS):      {1000 / avg_time_ms:.2f} img/s")
    

def measure_flops(model, device="cuda"):
    print("=== Análisis de FLOPs y MACs ===")
    model.to(device)
    model.eval()

    # Definir el tamaño del batch para la medición (usualmente se mide con batch_size=1)
    batch_size = 1
    
    # Crear los tensores dummy requeridos por tu método forward()
    dummy_input_img = torch.randn(batch_size, 3, 448, 448).to(device)
    dummy_cropped_img = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_scale = torch.ones(batch_size, 2).to(device)

    # Empaquetar las entradas en una tupla tal como lo requiere `thop`
    inputs = (dummy_input_img, dummy_cropped_img, dummy_scale)

    print("Calculando la complejidad computacional... (esto puede tardar unos segundos)")
    
    with torch.no_grad():
        # thop ejecuta el modelo y cuenta las operaciones
        macs, params = profile(model, inputs=inputs, verbose=False) #type: ignore

    # Convertir a formato legible (K, M, G - Miles, Millones, Billones/Giga)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
    flops_formatted = clever_format([macs * 2], "%.3f")

    print("\n[Resultados de Complejidad]")
    print(f"Parámetros calculados: {params_formatted}")
    print(f"MACs (Multiply-Accumulates): {macs_formatted}")
    print(f"FLOPs estimados: {flops_formatted}")
    
    print("\n*Nota: 1 operación MAC (a * b + c) equivale aproximadamente a 2 FLOPs (1 multiplicación + 1 suma).")


if __name__ == "__main__":
    # Prueba tu modelo principal. 
    # Cambia 'partial_unfreeze' a True para ver cómo cambian los parámetros entrenables.
    # partial unfreeze arg
    
    ap = argparse.ArgumentParser(description="Medir complejidad computacional del modelo Patch5Model")
    ap.add_argument("--partial-unfreeze", action="store_true")
    
    args = ap.parse_args()
    
    modelo_prueba = Patch5Model(partial_unfreeze=args.partial_unfreeze)
    
    # Detectar automáticamente si hay GPU disponible
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    
    measure_complexity(modelo_prueba, dispositivo)
    measure_flops(modelo_prueba, dispositivo)