"""Prueba el servidor MCP con el cliente oficial.

Uso:  python mcp_server/probar_cliente.py [URL]   (por defecto http://localhost:8000/mcp)
"""
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/mcp"
RESERVA = {"hotel": "City Hotel", "lead_time": 250, "fecha_llegada": "2017-09-15",
           "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adr": 95,
           "market_segment": "Groups", "distribution_channel": "TA/TO", "deposit_type": "Non Refund",
           "agent": 1}


async def main():
    async with streamablehttp_client(URL) as (leer, escribir, _):
        async with ClientSession(leer, escribir) as sesion:
            await sesion.initialize()
            herramientas = await sesion.list_tools()
            print("Herramientas:", [h.name for h in herramientas.tools])
            for nombre, args in [("info_modelo", {}), ("predecir_cancelacion", {"reserva": RESERVA}),
                                 ("predecir_lote", {"reservas": [RESERVA, {**RESERVA, "lead_time": 5, "deposit_type": "No Deposit", "market_segment": "Direct", "distribution_channel": "Direct"}]})]:
                r = await sesion.call_tool(nombre, args)
                texto = r.content[0].text if r.content else ""
                print(f"\n== {nombre} (error={r.isError})\n{texto[:600]}")
                if r.isError:
                    sys.exit(1)


asyncio.run(main())
