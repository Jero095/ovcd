"""PASO 9 — Visualización."""

from __future__ import annotations

import numpy as np


def visualize_ovcd(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    change_map: np.ndarray,
    results: list[dict],
    save_path: str = "resultado_ovcd.png",
) -> None:
    """
    Genera una imagen de visualización con:
    - T1 y T2 en RGB
    - Mapa de cambio (heatmap)
    - T2 con máscaras SAM y etiquetas CLIP superpuestas
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb_t1)
    axes[0].set_title("T1 — RGB")
    axes[0].axis("off")

    axes[1].imshow(rgb_t2)
    axes[1].set_title("T2 — RGB")
    axes[1].axis("off")

    im = axes[2].imshow(change_map, cmap="hot", norm=Normalize(0, 1))
    axes[2].set_title("Mapa de cambio")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # T2 + máscaras
    overlay = rgb_t2.copy()
    colors = plt.get_cmap("tab10")
    legend_handles = []
    for i, res in enumerate(results):
        mask = res["mask"]
        color_rgb = (np.array(colors(i % 10)[:3]) * 255).astype(np.uint8)
        overlay[mask] = (overlay[mask] * 0.5 + color_rgb * 0.5).astype(np.uint8)

        label = res.get("change_type", f"region_{res['region_id']}")
        patch = mpatches.Patch(color=colors(i % 10), label=label)
        legend_handles.append(patch)

        # Centroide
        rows, cols = np.where(mask)
        if len(rows) > 0:
            cy, cx = int(rows.mean()), int(cols.mean())
            axes[3].text(cx, cy, str(res["region_id"]),
                         color="white", fontsize=7, ha="center", va="center",
                         bbox=dict(facecolor="black", alpha=0.4, pad=1))

    axes[3].imshow(overlay)
    axes[3].set_title("SAM + CLIP")
    axes[3].axis("off")
    if legend_handles:
        axes[3].legend(handles=legend_handles, loc="lower right",
                       fontsize=6, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualización guardada en: {save_path}")
