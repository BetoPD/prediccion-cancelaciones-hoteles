"""Transformaciones que aprenden parámetros: siempre dentro del Pipeline (ajuste solo en train)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .datos import SESGADAS, columnas_modelo


class RecortePercentil(BaseEstimator, TransformerMixin):
    """Recorta columnas a [0, percentil q] con el percentil aprendido en entrenamiento.

    Bitácora EDA #8: ADR = 5,400 es un error de captura y hay un ADR negativo. ADR = 0 es legítimo
    (cortesías), así que no se elimina nada: solo se acotan los extremos.
    """

    def __init__(self, columnas=("adr", "adr_por_persona"), q=0.999):
        self.columnas = columnas
        self.q = q

    def fit(self, X, y=None):
        self.limites_ = {c: float(X[c].quantile(self.q)) for c in self.columnas if c in X}
        return self

    def transform(self, X):
        X = X.copy()
        for c, lim in self.limites_.items():
            X[c] = X[c].clip(lower=0, upper=lim)
        return X


def construir_preprocesador(lineal: bool, con_sospechosas: bool = False) -> Pipeline:
    """Recorte de ADR -> imputación -> (log1p + escalado si el modelo es lineal) -> one-hot.

    Los árboles no necesitan escalado ni log; la regresión logística sí (gradiente estable y
    coeficientes comparables). Las categorías con menos de 0.5 % de las filas (agencias, empresas
    y países raros) se agrupan en una sola columna «infrequent» (bitácora EDA, §9.2).
    """
    num, cat = columnas_modelo(con_sospechosas)
    sesgadas = [c for c in num if c in SESGADAS]
    resto = [c for c in num if c not in SESGADAS]
    if lineal:
        t_sesgadas = Pipeline([("imputar", SimpleImputer(strategy="median")),
                               ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                               ("escalar", StandardScaler())])
        t_resto = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
    else:
        t_sesgadas = SimpleImputer(strategy="median")
        t_resto = SimpleImputer(strategy="median")
    t_cat = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("onehot", OneHotEncoder(min_frequency=0.005, handle_unknown="infrequent_if_exist",
                                 sparse_output=False)),
    ])
    columnas = ColumnTransformer([("sesgadas", t_sesgadas, sesgadas), ("numericas", t_resto, resto),
                                  ("categoricas", t_cat, cat)], verbose_feature_names_out=False)
    return Pipeline([("recorte", RecortePercentil()), ("columnas", columnas)])
