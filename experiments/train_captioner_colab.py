"""Entrenamiento del captioner HyperCap (BLIP + 3D-ConvSST) — para Colab/Kaggle GPU.

Autocontenido: solo necesita el repo HyperCap clonado al lado. NO corre en CPU.

Uso en Google Colab (runtime GPU, p.ej. T4):

    !git clone https://github.com/arya-domain/HyperCap.git
    !pip install -q transformers datasets einops timm
    # subir este archivo a la sesión y:
    !python train_captioner_colab.py --dataset "Indian Pines" --epochs 50

Al terminar descarga el checkpoint y colócalo en weights/ del proyecto local:
    weights/blip_3dconvsst_indian_pines.pth

Diferencias deliberadas con Tutorial_Captioning_BLIP.py del repo:
1. Encoder 3D-ConvSST en vez de FAHM (FAHM requiere mamba_ssm, CUDA-only, y
   queremos INFERENCIA en CPU local).
2. BlipForConditionalGeneration stock de transformers (el model/blipcaptioning.py
   del repo no existe — solo hay un .pyc).
3. El cubo se normaliza min-max POR BANDA a [0,1] ANTES de extraer patches.
   El pipeline OVCD local alimenta cubos alineados normalizados [0,1]; entrenar
   igual evita un mismatch de dominio en inferencia.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "HyperCap")

PATCH_SIZE = 11
BLIP_NAME = "Salesforce/blip-image-captioning-base"


# ---------------------------------------------------------------------------
# Dataset: adaptado del tutorial de HyperCap, con normalización por banda
# ---------------------------------------------------------------------------
class HyperCapCaptionDataset(Dataset):
    def __init__(self, folder: str, processor, patch_size: int = PATCH_SIZE):
        self.processor = processor
        self.patch_size = patch_size

        folder = Path(folder)
        data_file = next(folder.glob("*_data.mat"))
        gt_file = next(folder.glob("*_gt.mat"))
        csv_file = next(folder.glob("*.csv"))
        map_file = folder / "class_map.json"

        data = sio.loadmat(str(data_file))["data"].astype(np.float32)  # (C, H, W)
        gt = sio.loadmat(str(gt_file))["gt"]
        captions = pd.read_csv(csv_file)
        with open(map_file) as f:
            class_map = json.load(f)

        # Normalización min-max POR BANDA a [0,1] (ver docstring del módulo)
        c_min = data.min(axis=(1, 2), keepdims=True)
        c_max = data.max(axis=(1, 2), keepdims=True)
        denom = np.where(c_max - c_min > 1e-8, c_max - c_min, 1.0)
        data = (data - c_min) / denom

        bands, height, width = data.shape
        self.bands = bands
        pad = patch_size // 2
        padded = np.pad(data, ((0, 0), (pad, pad), (pad, pad)), mode="constant")

        captions["Class name"] = captions["Class name"].str.strip().str.lower()
        pool = {name: g["Description"].tolist()
                for name, g in captions.groupby("Class name")}
        idx = {name: 0 for name in pool}

        self.patches, self.texts = [], []
        for i, j in product(range(height), range(width)):
            label = gt[i, j]
            if label == 0:
                continue
            cname = class_map.get(str(int(label) - 1), "Unknown").strip().lower()
            avail = pool.get(cname, ["No description available."])
            self.texts.append(avail[idx.get(cname, 0) % len(avail)])
            idx[cname] = idx.get(cname, 0) + 1
            self.patches.append(padded[:, i:i + patch_size, j:j + patch_size])

        print(f"{folder.name}: {len(self.patches)} patches, {bands} bandas")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, k):
        enc = self.processor(text=self.texts[k], padding="max_length",
                             return_tensors="pt")
        enc = {key: v.squeeze() for key, v in enc.items()}
        enc["pixel_values"] = torch.from_numpy(
            np.ascontiguousarray(self.patches[k])).float()
        return enc


# ---------------------------------------------------------------------------
# Modelo: BLIP-base + 3D-ConvSST (idéntico a stages/captioning/captioner.py)
# ---------------------------------------------------------------------------
class VisionModel(nn.Module):
    def __init__(self, bands: int):
        super().__init__()
        from model.vision._3D_ConvSST import VisionTransformer
        self.base = VisionTransformer(in_chans=bands, embed_dim=64, n_heads=8)
        self.fc = nn.Linear(64, 768)

    def forward(self, pixel_values, **kwargs):
        return [self.fc(self.base(pixel_values)).unsqueeze(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Indian Pines",
                    help="Subcarpeta de HyperCap/Datasets")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default=None, help="Ruta del checkpoint de salida")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Este script necesita GPU (usa Colab/Kaggle)."
    device = "cuda"
    torch.manual_seed(0)

    from transformers import AutoProcessor, BlipForConditionalGeneration
    processor = AutoProcessor.from_pretrained(BLIP_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_NAME)

    ds = HyperCapCaptionDataset(os.path.join("HyperCap", "Datasets", args.dataset),
                                processor)
    model.vision_model = VisionModel(ds.bands)
    model = model.to(device).train()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_path = args.out or f"blip_3dconvsst_{args.dataset.lower().replace(' ', '_')}.pth"

    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            out = model(input_ids=input_ids, pixel_values=pixel_values,
                        labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(out.loss)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / len(loader):.4f}")
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), out_path)
            print(f"  checkpoint guardado: {out_path}")

    print(f"\nListo. Descarga {out_path} y colócalo en weights/ del proyecto.")


if __name__ == "__main__":
    main()
