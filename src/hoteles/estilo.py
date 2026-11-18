"""Estilo gráfico compartido con la Entrega 1 (mismas paletas y tamaños)."""
import logging
from pathlib import Path

import matplotlib.pyplot as plt

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#898781"
VERDE, MORADO = "#1f9e74", "#7d5ba6"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e1e0d9"


def aplicar_estilo():
    for nombre in ("matplotlib.font_manager", "fontTools"):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": ["Inter", "Helvetica", "DejaVu Sans"], "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
        "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def guardar(fig, carpeta: Path, nombre: str):
    """Exporta la figura en PDF (para LaTeX) y PNG (para la presentación)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}.pdf")
    fig.savefig(carpeta / f"{nombre}.png")


def pct(x, dec=1):
    """Porcentaje con el signo escapado para LaTeX."""
    return f"{100 * x:.{dec}f}\\%"
