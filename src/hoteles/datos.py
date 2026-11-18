"""Carga, limpieza e ingeniería de características.

Todo lo de este módulo es una regla fija fila por fila (no aprende nada de los datos), así que se
puede aplicar antes de partir en entrenamiento/prueba sin fuga de información, y se reutiliza tal
cual al predecir una reserva nueva en el servidor MCP. Lo que sí aprende parámetros (imputación,
recorte de ADR, escalado, codificación) vive en `preprocesamiento.py`, dentro del Pipeline.
"""
from pathlib import Path

import numpy as np
import pandas as pd

Y = "is_canceled"

MESES = ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Bitácora EDA #1: revelan el desenlace -> se excluyen siempre.
FUGA = ["reservation_status", "reservation_status_date", "assigned_room_type"]
# Bitácora EDA #2: posible registro posterior a la reserva -> modelo con y sin ellas.
SOSPECHOSAS_NUM = ["required_car_parking_spaces", "booking_changes"]
SOSPECHOSAS_CAT = ["country"]

NUM_BASE = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
            "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
            "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
            "days_in_waiting_list", "adr", "total_of_special_requests"]
NUM_NUEVAS = ["noches", "huespedes", "estancia_cero", "proporcion_fin_semana", "tiene_ninos",
              "mes_sin", "mes_cos", "llegada_fin_semana", "tiene_agente", "tiene_empresa",
              "historial_reservas", "tasa_cancelacion_previa", "adr_por_persona"]
CAT = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type",
       "deposit_type", "agent", "company", "customer_type", "dia_semana"]
# Conteos con cola larga a la derecha (bitácora EDA #11): log1p solo para el modelo lineal.
SESGADAS = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
            "previous_cancellations", "previous_bookings_not_canceled", "days_in_waiting_list",
            "adr", "noches", "historial_reservas", "adr_por_persona"]


def columnas_modelo(con_sospechosas: bool = False):
    """Listas (numéricas, categóricas) que entran al modelo según el escenario."""
    num = NUM_BASE + NUM_NUEVAS + (SOSPECHOSAS_NUM if con_sospechosas else [])
    cat = CAT + (SOSPECHOSAS_CAT if con_sospechosas else [])
    return num, cat


def cargar_crudos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV igual que en la Entrega 1: `NULL` y `NA` se leen como faltantes."""
    return pd.read_csv(ruta, na_values=["NULL", "NA"], keep_default_na=False, low_memory=False)


def _codificar_id(serie: pd.Series, sin_valor: str, prefijo: str) -> pd.Series:
    """agent/company: nulo significa 'no aplica' (bitácora EDA #5) -> categoría propia."""
    v = pd.to_numeric(serie, errors="coerce")
    return pd.Series(np.where(v.isna(), sin_valor, prefijo + v.fillna(0).astype(int).astype(str)),
                     index=serie.index)


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Correcciones de valores fila por fila (se aplican igual en entrenamiento y en servicio)."""
    d = df.copy()
    if "fecha_llegada" not in d:
        d["fecha_llegada"] = pd.to_datetime(
            d["arrival_date_year"].astype(str) + "-" + d["arrival_date_month"] + "-"
            + d["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    d["meal"] = d["meal"].replace({"Undefined": "SC"})  # ambos significan "sin plan de alimentos"
    d["agent"] = _codificar_id(d["agent"], "Sin agente", "A")
    d["company"] = _codificar_id(d["company"], "Sin empresa", "E")
    if "country" in d:
        d["country"] = d["country"].fillna("Desconocido")  # bitácora EDA #6: nulo informativo
    return d


def limpiar(df: pd.DataFrame):
    """Limpieza del conjunto histórico. Devuelve (datos limpios, bitácora de cambios)."""
    d = df.copy()
    huespedes = d["adults"] + d["children"].fillna(0) + d["babies"]
    bitacora = [
        ("children nulo", "Se deja nulo; se imputa con la mediana dentro del Pipeline", int(d["children"].isna().sum())),
        ("country nulo", "Categoría «Desconocido» (la nulidad es informativa)", int(d["country"].isna().sum())),
        ("agent nulo", "Categoría «Sin agente» (no aplica)", int(d["agent"].isna().sum())),
        ("company nulo", "Categoría «Sin empresa» (no aplica)", int(d["company"].isna().sum())),
        ("meal = Undefined", "Se recodifica como SC (sin plan)", int(d["meal"].eq("Undefined").sum())),
        ("Cero huéspedes", "Se eliminan (reservas imposibles)", int((huespedes == 0).sum())),
        ("ADR < 0 o ADR atípico", "Recorte a [0, p99.9] ajustado en entrenamiento", int(((d["adr"] < 0) | (d["adr"] > 1000)).sum())),
    ]
    # Firma de duplicado (bitácora EDA #3): filas idénticas comparten grupo para la validación.
    d["grupo"] = d.groupby(list(df.columns), dropna=False, sort=False).ngroup()
    d = normalizar(d)
    d = d.loc[huespedes != 0].reset_index(drop=True)
    d = d.drop(columns=FUGA + ["arrival_date_year"])  # bitácora EDA #1 y #4
    bitacora = pd.DataFrame(bitacora, columns=["problema", "tratamiento", "filas"])
    return d, bitacora


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de características (reglas deterministas, sin parámetros aprendidos)."""
    d = df.copy()
    ninos = d["children"].fillna(0)
    d["noches"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["huespedes"] = d["adults"] + ninos + d["babies"]
    d["estancia_cero"] = (d["noches"] == 0).astype(int)
    d["proporcion_fin_semana"] = np.divide(d["stays_in_weekend_nights"], d["noches"],
                                           out=np.zeros(len(d)), where=d["noches"].to_numpy() > 0)
    d["tiene_ninos"] = ((ninos + d["babies"]) > 0).astype(int)
    # Mes como ángulo: diciembre y enero quedan cerca (codificación cíclica).
    mes = d["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MESES)})
    d["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    d["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    # De la fecha de llegada se obtiene el día de la semana (no viene en los datos originales).
    dow = d["fecha_llegada"].dt.dayofweek
    d["dia_semana"] = dow.map(dict(enumerate(DIAS)))
    d["llegada_fin_semana"] = (dow >= 4).astype(int)  # viernes a domingo
    d["tiene_agente"] = (d["agent"] != "Sin agente").astype(int)
    d["tiene_empresa"] = (d["company"] != "Sin empresa").astype(int)
    d["historial_reservas"] = d["previous_cancellations"] + d["previous_bookings_not_canceled"]
    d["tasa_cancelacion_previa"] = np.divide(d["previous_cancellations"], d["historial_reservas"],
                                             out=np.zeros(len(d)),
                                             where=d["historial_reservas"].to_numpy() > 0)
    d["adr_por_persona"] = d["adr"] / d["huespedes"].clip(lower=1)
    return d
