"""
Descarga únicamente una parte del dataset HyperGlobal-450K desde Hugging Face
(ej. EO1-part1.zip) para inspeccionarlo localmente.

Este script:
1. Descarga un solo archivo ZIP del dataset.
2. Lo extrae en una carpeta local.
3. Lista los archivos encontrados.
4. Busca automáticamente posibles metadatos (.hdr, .json, .xml, .txt).

Requisitos:
    pip install huggingface_hub

Uso:
    python download_hyperglobal_sample.py
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import zipfile
import os

# =========================================================
# CONFIGURACIÓN
# =========================================================

REPO_ID = "WHU-Sigma/HyperGlobal-450K"
REPO_TYPE = "dataset"

# Cambia este nombre si quieres otra parte:
# EO1-part1.zip, EO1-part2.zip, GF5-part1.zip, etc.
FILENAME = "EO1-part1.zip"

# Carpeta local donde se guardará todo
OUTPUT_DIR = Path("hyperglobal_sample")

# =========================================================
# DESCARGAR
# =========================================================

print(f"Descargando {FILENAME}...")

zip_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    filename=FILENAME,
    local_dir=OUTPUT_DIR,
    local_dir_use_symlinks=False,
)

print(f"Archivo descargado en: {zip_path}")

# =========================================================
# EXTRAER
# =========================================================

extract_dir = OUTPUT_DIR / FILENAME.replace(".zip", "")
extract_dir.mkdir(parents=True, exist_ok=True)

print(f"Extrayendo en: {extract_dir}")

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)

print("Extracción completada.")

# =========================================================
# LISTAR ARCHIVOS
# =========================================================

all_files = list(extract_dir.rglob("*"))
files_only = [f for f in all_files if f.is_file()]

print(f"\nTotal de archivos extraídos: {len(files_only)}")

print("\nPrimeros 30 archivos:")
for f in files_only[:30]:
    print(" -", f.relative_to(extract_dir))

# =========================================================
# BUSCAR METADATOS
# =========================================================

metadata_extensions = {
    ".hdr",
    ".json",
    ".xml",
    ".txt",
    ".csv",
    ".tif",
    ".tiff",
}

metadata_files = [
    f for f in files_only
    if f.suffix.lower() in metadata_extensions
]

print(f"\nArchivos potencialmente útiles como metadatos: {len(metadata_files)}")

for f in metadata_files[:50]:
    print(" -", f.relative_to(extract_dir))

# =========================================================
# MOSTRAR CONTENIDO DE UN .HDR O .JSON SI EXISTE
# =========================================================

for f in metadata_files:
    if f.suffix.lower() in {".hdr", ".json", ".txt"}:
        print(f"\nMostrando contenido de: {f.name}")
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                print(fp.read(2000))  # primeros 2000 caracteres
        except Exception as e:
            print("No se pudo leer:", e)
        break

print("\nProceso finalizado.")