# Predicción de cancelaciones de reservas hoteleras

Proyecto de equipo de **Aprendizaje Máquina I** (Maestría en Ciencia de Datos, Universidad
Panamericana), desarrollado con la metodología CRISP-DM sobre el conjunto *Hotel Booking Demand*
(Antonio, de Almeida y Nunes, 2019): 119,390 reservas de un hotel urbano (Lisboa) y un resort
(Algarve) con llegadas entre julio de 2015 y agosto de 2017.

**Objetivo:** estimar, con la información disponible el día anterior a la llegada, la probabilidad
de que una reserva se cancele, para reducir
el ingreso en riesgo y las habitaciones-noche perdidas.

## Entregas

| Entrega | Fase CRISP-DM | Cuaderno |
|---|---|---|
| 1 | Entendimiento del negocio y de los datos | [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb) |
| 2 | Preparación de los datos | [`notebooks/02_preparacion.ipynb`](notebooks/02_preparacion.ipynb) |
| 2 | Modelado | [`notebooks/03_modelado.ipynb`](notebooks/03_modelado.ipynb) |

## Resultados principales (Entrega 2)

**Preparación.** 180 filas imposibles eliminadas; 3 variables con fuga excluidas; nulos con
significado convertidos en categorías; 14 variables nuevas (día de la semana y estacionalidad
cíclica a partir de la fecha de llegada, tasa de cancelación previa, banderas de canal, ADR por
persona, frecuencia histórica de la agencia); partición temporal (prueba = llegadas de mayo a agosto de 2017); transformaciones
ajustadas solo con entrenamiento dentro de un `Pipeline`.

**Modelado** (clasificación binaria, PR-AUC como métrica principal, validación cruzada agrupada
por duplicados):

| Modelo | PR-AUC CV | PR-AUC prueba | ROC-AUC prueba |
|---|---|---|---|
| Línea base (prevalencia) | 0.363 | 0.406 | 0.500 |
| Regresión logística **implementada con NumPy** | 0.842 | 0.785 | 0.811 |
| Random Forest (ajustado) | 0.873 | 0.787 | 0.821 |
| **Gradient Boosting (ajustado, modelo final)** | **0.873** | **0.784** | **0.816** |
| Gradient Boosting + variables sospechosas de fuga | 0.926 | 0.834 | 0.876 |

- La regresión logística desde cero (sigmoide, entropía cruzada con L2, gradiente verificado
  numéricamente, descenso por gradiente con mini-lotes) iguala el desempeño de scikit-learn.
- Las variables sospechosas inflan la métrica 5 puntos: no estarían disponibles el día anterior a
  la llegada y se excluyen.
- La caída en prueba se explica por deriva temporal (la tasa de cancelación del resort sube de
  26 % a 36 %); se recomienda reentrenar y recalibrar el umbral periódicamente.
- El umbral (0.72) minimiza el costo esperado suponiendo que reubicar a un huésped cuesta 1.5 veces
  su tarifa. En la prueba, el modelo anticipa 2.2 de los 5.0 millones de ingreso en riesgo con 81 %
  de precisión y reduce 28 % el costo de las cancelaciones.
- Interpretación con importancia por permutación y valores SHAP: depósito, solicitudes especiales,
  anticipación, agencia, tipo de cliente y segmento.


## Estructura

```
data/raw/hotels.csv          datos originales (TidyTuesday 2020-02-11)
data/processed/              train/test limpios (se regeneran con el cuaderno 02)
notebooks/                   cuadernos de cada fase
src/hoteles/                 paquete de Python con el código reutilizable
  datos.py                   limpieza e ingeniería de características (reglas fijas)
  preprocesamiento.py        transformaciones que aprenden de los datos (dentro del Pipeline)
  logistica_numpy.py         regresión logística desde cero, solo con NumPy
  prediccion.py              carga del modelo y predicción para reservas nuevas
models/                      modelo final (.joblib) y metadatos (.json)
mcp_server/                  servidor MCP para usar el modelo desde Claude
report/figuras, report/tablas  figuras (PDF/PNG) y tablas LaTeX generadas por los cuadernos
render.yaml                  despliegue del servidor MCP en Render
```

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-notebooks.txt
pip install -e .
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 02_preparacion.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_modelado.ipynb
```

El cuaderno 03 tarda alrededor de 25 minutos (búsqueda de hiperparámetros con validación
cruzada) y sobrescribe `models/modelo_final.joblib`.

## Usar el modelo desde Claude (MCP)

El servidor `mcp_server/server.py` expone tres herramientas:

| Herramienta | Qué hace |
|---|---|
| `predecir_cancelacion` | Probabilidad de cancelación, decisión con el umbral del modelo y nivel de riesgo de una reserva |
| `predecir_lote` | Lo mismo para hasta 500 reservas |
| `info_modelo` | Algoritmo, hiperparámetros, umbral, métricas, variables y limitaciones |

Campos obligatorios: `hotel`, `lead_time`, `fecha_llegada`, `stays_in_weekend_nights`,
`stays_in_week_nights` y `adr`; los demás tienen valores por defecto típicos.

**Servidor público (Render):** `https://prediccion-cancelaciones-hoteles.onrender.com/mcp`

- En claude.ai: *Settings → Connectors → Add custom connector* y pegar la URL anterior.
- En Claude Code: `claude mcp add --transport http hoteles https://prediccion-cancelaciones-hoteles.onrender.com/mcp`
- Estado: `https://prediccion-cancelaciones-hoteles.onrender.com/salud`

El plan gratuito de Render apaga el servicio tras 15 minutos sin uso; la primera llamada después
puede tardar ~1 minuto en responder. El servidor no tiene autenticación (solo expone predicciones).

**Local:**

```bash
pip install -r requirements.txt && pip install .
python mcp_server/server.py                 # http://localhost:8000/mcp
python mcp_server/probar_cliente.py         # prueba las tres herramientas
```

Para Claude Code: `claude mcp add --transport http hoteles http://localhost:8000/mcp`.
