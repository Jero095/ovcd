"""Fase 0 — prueba de plomeria HyperCap en CPU.

Construye BLIP-base con el encoder 3D-ConvSST (libre de mamba/CUDA) como
vision_model, carga un patch real 11x11 de Indian Pines y genera un caption.
El modelo NO esta entrenado -> el texto sera sin sentido; el objetivo es
demostrar que el camino HSI patch -> encoder 3D -> BLIP -> texto funciona
mecanicamente en esta maquina.
"""
import sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

sys.path.insert(0, "HyperCap")
from model.vision._3D_ConvSST import VisionTransformer

# --- Datos: patch real de Indian Pines -------------------------------------
data = sio.loadmat("HyperCap/Datasets/Indian Pines/indian_pines_data.mat")["data"]
gt = sio.loadmat("HyperCap/Datasets/Indian Pines/indian_pines_gt.mat")["gt"]
print("data:", data.shape, "| gt:", gt.shape, "| pixeles etiquetados:", int((gt > 0).sum()))

bands = data.shape[0]
PATCH = 11
pad = PATCH // 2
padded = np.pad(data, ((0, 0), (pad, pad), (pad, pad)), mode="constant")

ys, xs = np.where(gt > 0)
i, j = int(ys[100]), int(xs[100])  # un pixel etiquetado cualquiera
patch = padded[:, i:i + PATCH, j:j + PATCH].astype(np.float32)
print(f"patch en ({i},{j}) clase gt={int(gt[i, j])} -> shape {patch.shape}")

# --- Modelo: BLIP-base + 3D-ConvSST como vision encoder --------------------
from transformers import AutoProcessor, BlipForConditionalGeneration

class VisionModel(nn.Module):
    def __init__(self, bands: int):
        super().__init__()
        # conv6 del encoder produce 64 canales -> embed_dim=64 y adapter 64->768
        # (mismo patron que el tutorial usa con FAHM: fc = Linear(64, 768))
        self.base = VisionTransformer(in_chans=bands, embed_dim=64, n_heads=8)
        self.fc = nn.Linear(64, 768)

    def forward(self, pixel_values, **kwargs):
        x = self.fc(self.base(pixel_values))
        return [x.unsqueeze(1)]  # (B, 1, 768) como lista, igual que el tutorial

print("Descargando/cargando BLIP-base...")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.vision_model = VisionModel(bands)
model.eval()

x = torch.from_numpy(patch).unsqueeze(0)  # (1, 200, 11, 11)
with torch.no_grad():
    feats = model.vision_model(x)[0]
    print("vision encoder OK ->", tuple(feats.shape))
    out = model.generate(pixel_values=x, max_length=30)

caption = processor.decode(out[0], skip_special_tokens=True)
print("\nCaption generado (modelo SIN entrenar, texto sin sentido esperado):")
print(f"  '{caption}'")
print("\nPLOMERIA OK — HSI patch -> 3D-ConvSST -> BLIP -> texto funciona en CPU")
