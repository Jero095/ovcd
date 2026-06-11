"""Test de reproducibilidad: ejecuta el pipeline completo DOS veces y compara."""
import numpy as np
from pipeline import ovcd_pipeline

ARGS = dict(
    path_t1="hyperglobal/imgs/barbara/barbara_2013.mat",
    path_t2="hyperglobal/imgs/barbara/barbara_2014.mat",
    sam_checkpoint="sam_vit_h_4b8939.pth",
    hypersigma_checkpoint="weights/spat-vit-base-ultra-checkpoint-1599.pth",
    hsi_format="mat",
    device="cpu",
)

print("========== CORRIDA 1 ==========")
r1 = ovcd_pipeline(**ARGS, save_viz="resultado_repro_1.png")

print("========== CORRIDA 2 ==========")
r2 = ovcd_pipeline(**ARGS, save_viz="resultado_repro_2.png")

print("========== COMPARACION ==========")
cm_diff = float(np.abs(r1["change_map"] - r2["change_map"]).max())
bm_equal = bool(np.array_equal(r1["binary_map"], r2["binary_map"]))
n_equal = r1["n_regions"] == r2["n_regions"]
labels1 = [d["change_type"] for d in r1["detections"]]
labels2 = [d["change_type"] for d in r2["detections"]]
masks_equal = all(
    np.array_equal(d1["mask"], d2["mask"])
    for d1, d2 in zip(r1["detections"], r2["detections"])
)

print(f"change_map  max diff : {cm_diff}")
print(f"binary_map  identico : {bm_equal}")
print(f"n_regions            : {r1['n_regions']} vs {r2['n_regions']} -> {'OK' if n_equal else 'DIFIEREN'}")
print(f"etiquetas identicas  : {labels1 == labels2}")
print(f"mascaras identicas   : {masks_equal}")

if cm_diff == 0.0 and bm_equal and n_equal and labels1 == labels2 and masks_equal:
    print("\nREPRODUCIBLE OK — las dos corridas son identicas")
else:
    print("\nNO REPRODUCIBLE — hay diferencias entre corridas")
