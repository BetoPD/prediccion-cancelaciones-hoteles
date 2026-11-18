"""Servicio de predicción: convierte reservas en formato simple a la entrada del modelo.

Se usa desde el servidor MCP. Aplica exactamente las mismas funciones `normalizar` y
`crear_variables` que el entrenamiento, para que no haya diferencias entre entrenamiento y servicio.
"""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .datos import MESES, crear_variables, normalizar

VALORES_POR_DEFECTO = {
    "adults": 2, "children": 0, "babies": 0, "meal": "BB", "market_segment": "Online TA",
    "distribution_channel": "TA/TO", "is_repeated_guest": 0, "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0, "reserved_room_type": "A", "deposit_type": "No Deposit",
    "agent": None, "company": None, "days_in_waiting_list": 0, "customer_type": "Transient",
    "total_of_special_requests": 0, "country": None, "required_car_parking_spaces": 0,
    "booking_changes": 0,
}
REQUERIDOS = ["hotel", "lead_time", "fecha_llegada", "stays_in_weekend_nights",
              "stays_in_week_nights", "adr"]


def ruta_modelo() -> Path:
    """`HOTELES_MODELO` si existe; si no, `models/modelo_final.joblib` del repositorio."""
    if os.environ.get("HOTELES_MODELO"):
        return Path(os.environ["HOTELES_MODELO"])
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        candidato = base / "models" / "modelo_final.joblib"
        if candidato.exists():
            return candidato
    raise FileNotFoundError("No se encontró models/modelo_final.joblib; define HOTELES_MODELO")


@lru_cache(maxsize=1)
def cargar_modelo() -> dict:
    """Artefacto: {'modelo': Pipeline, 'umbral': float, 'metadatos': dict}."""
    return joblib.load(ruta_modelo())


def preparar_entrada(reservas: list[dict]) -> pd.DataFrame:
    filas = []
    for i, r in enumerate(reservas):
        faltan = [c for c in REQUERIDOS if r.get(c) is None]
        if faltan:
            raise ValueError(f"Reserva {i}: faltan campos obligatorios {faltan}")
        filas.append({**VALORES_POR_DEFECTO, **{k: v for k, v in r.items() if v is not None}})
    df = pd.DataFrame(filas)
    fecha = pd.to_datetime(df["fecha_llegada"])
    df["fecha_llegada"] = fecha
    df["arrival_date_month"] = fecha.dt.month.map(lambda m: MESES[m - 1])
    df["arrival_date_week_number"] = fecha.dt.isocalendar().week.astype(int)
    df["arrival_date_day_of_month"] = fecha.dt.day
    return crear_variables(normalizar(df))


def nivel_riesgo(p: float, umbral: float) -> str:
    if p >= max(umbral, 0.7):
        return "alto"
    if p >= umbral:
        return "medio-alto"
    if p >= umbral / 2:
        return "medio"
    return "bajo"


def predecir(reservas: list[dict]) -> list[dict]:
    artefacto = cargar_modelo()
    X = preparar_entrada(reservas)
    probs = artefacto["modelo"].predict_proba(X)[:, 1]
    umbral = artefacto["umbral"]
    return [{"probabilidad_cancelacion": round(float(p), 4),
             "se_predice_cancelacion": bool(p >= umbral),
             "umbral_decision": round(umbral, 4),
             "nivel_riesgo": nivel_riesgo(float(p), umbral)} for p in probs]
