"""Servidor MCP que expone el modelo de cancelaciones hoteleras.

Local:   python mcp_server/server.py            -> http://localhost:8000/mcp
Render:  usa la variable PORT y escucha en 0.0.0.0 (ver render.yaml).
"""
import os
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from hoteles import prediccion

# El modelo vive en models/ del repositorio, junto a esta carpeta.
os.environ.setdefault("HOTELES_MODELO", str(Path(__file__).resolve().parents[1] / "models" / "modelo_final.joblib"))

INSTRUCCIONES = """
Estima la probabilidad de que una reserva hotelera se cancele, con un modelo entrenado con
119 mil reservas de dos hoteles de Portugal (Hotel Booking Demand, 2015-2017).
Usa `predecir_cancelacion` para una reserva, `predecir_lote` para varias, e `info_modelo` para
conocer el modelo, sus métricas y limitaciones. Solo hotel, lead_time, fecha_llegada, noches y
adr son obligatorios; el resto toma valores típicos si no se conocen. La probabilidad sirve para
priorizar reservas por riesgo, no es una certeza.
""".strip()

mcp = FastMCP(
    "prediccion-cancelaciones-hoteles",
    instructions=INSTRUCCIONES,
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8000")),
    stateless_http=True,
)


class Reserva(BaseModel):
    """Datos conocidos al momento de hacer la reserva (sin variables que revelan el desenlace)."""

    hotel: Literal["City Hotel", "Resort Hotel"] = Field(
        description="City Hotel = hotel urbano (Lisboa); Resort Hotel = resort (Algarve)")
    lead_time: int = Field(ge=0, le=800, description="Días entre la fecha de la reserva y la llegada")
    fecha_llegada: date = Field(description="Fecha de llegada (AAAA-MM-DD)")
    stays_in_weekend_nights: int = Field(ge=0, le=30, description="Noches de sábado o domingo")
    stays_in_week_nights: int = Field(ge=0, le=60, description="Noches de lunes a viernes")
    adr: float = Field(ge=0, le=1000, description="Tarifa diaria promedio (EUR por noche)")
    adults: int = Field(2, ge=0, le=10, description="Número de adultos")
    children: int = Field(0, ge=0, le=10, description="Número de niños")
    babies: int = Field(0, ge=0, le=10, description="Número de bebés")
    meal: Literal["BB", "HB", "FB", "SC"] = Field(
        "BB", description="BB = desayuno, HB = media pensión, FB = pensión completa, SC = sin alimentos")
    market_segment: Literal["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate",
                            "Complementary", "Aviation"] = Field(
        "Online TA", description="Segmento de mercado (TA = agencia de viajes, TO = operador turístico)")
    distribution_channel: Literal["TA/TO", "Direct", "Corporate", "GDS"] = Field(
        "TA/TO", description="Canal de distribución")
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"] = Field(
        "No Deposit", description="Tipo de depósito")
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"] = Field(
        "Transient", description="Tipo de cliente")
    reserved_room_type: Literal["A", "B", "C", "D", "E", "F", "G", "H", "L"] = Field(
        "A", description="Código (anonimizado) del tipo de habitación reservada")
    agent: int | None = Field(None, description="ID de la agencia; nulo si la reserva no vino de agencia")
    company: int | None = Field(None, description="ID de la empresa que paga; nulo si no aplica")
    is_repeated_guest: Literal[0, 1] = Field(0, description="1 si el huésped ya se hospedó antes")
    previous_cancellations: int = Field(0, ge=0, le=100, description="Cancelaciones previas del cliente")
    previous_bookings_not_canceled: int = Field(0, ge=0, le=100, description="Reservas previas no canceladas")
    days_in_waiting_list: int = Field(0, ge=0, le=500, description="Días en lista de espera")
    total_of_special_requests: int = Field(0, ge=0, le=10, description="Solicitudes especiales")


def _predecir(reservas: list[Reserva]) -> list[dict]:
    datos = [r.model_dump(mode="json") for r in reservas]
    return prediccion.predecir(datos)


@mcp.tool()
def predecir_cancelacion(reserva: Reserva) -> dict:
    """Probabilidad de que una reserva se cancele, decisión según el umbral del modelo y nivel de riesgo."""
    return _predecir([reserva])[0]


@mcp.tool()
def predecir_lote(reservas: Annotated[list[Reserva], Field(min_length=1, max_length=500)]) -> list[dict]:
    """Predicción para varias reservas (hasta 500), en el mismo orden en que se envían."""
    return _predecir(reservas)


@mcp.tool()
def info_modelo() -> dict:
    """Describe el modelo: algoritmo, umbral, métricas de validación y prueba, variables y limitaciones."""
    return prediccion.cargar_modelo()["metadatos"]


@mcp.custom_route("/salud", methods=["GET"])
async def salud(request: Request) -> JSONResponse:
    meta = prediccion.cargar_modelo()["metadatos"]
    return JSONResponse({"estado": "ok", "modelo": meta["modelo"], "version": meta["version_paquete"]})


if __name__ == "__main__":
    prediccion.cargar_modelo()  # falla al arrancar (no en la primera llamada) si falta el modelo
    mcp.run(transport="streamable-http")
