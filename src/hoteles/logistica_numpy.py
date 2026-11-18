"""Regresión logística implementada desde cero, solo con NumPy.

Modelo:
    z = X·w + b            (combinación lineal)
    p = σ(z) = 1/(1+e^-z)  (probabilidad de cancelación)

Función de pérdida (entropía cruzada ponderada + penalización L2):
    L(w, b) = -(1/Σs) Σ sᵢ [yᵢ log pᵢ + (1-yᵢ) log(1-pᵢ)] + (λ/2)·||w||²
donde sᵢ es el peso de la muestra (mayor para la clase minoritaria si pesos_clase="balanced").

Gradiente (se deriva a mano; la derivada de σ es σ(1-σ) y se simplifica con la del log):
    ∂L/∂w = (1/Σs) Xᵀ [s ⊙ (p - y)] + λ·w
    ∂L/∂b = (1/Σs) Σ sᵢ (pᵢ - yᵢ)

Optimización: descenso por gradiente con mini-lotes, barajando en cada época, y parada temprana
cuando la pérdida deja de bajar más de `tol` durante `paciencia` épocas.

La clase expone `fit`, `predict_proba`, `predict`, `get_params` y `set_params` para poder
colocarla al final de un Pipeline y usarla en validación cruzada, sin depender de scikit-learn.
"""
import numpy as np


class RegresionLogisticaNumpy:
    _estimator_type = "classifier"
    _PARAMETROS = ("tasa_aprendizaje", "lambda_l2", "epocas", "tam_lote", "pesos_clase", "tol",
                   "paciencia", "semilla")

    def __init__(self, tasa_aprendizaje=0.1, lambda_l2=1e-4, epocas=60, tam_lote=1024,
                 pesos_clase="balanced", tol=1e-4, paciencia=5, semilla=42):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.lambda_l2 = lambda_l2
        self.epocas = epocas
        self.tam_lote = tam_lote
        self.pesos_clase = pesos_clase
        self.tol = tol
        self.paciencia = paciencia
        self.semilla = semilla

    # --- interfaz de parámetros (la usan la validación cruzada y las búsquedas) -------------
    def get_params(self, deep=True):
        return {p: getattr(self, p) for p in self._PARAMETROS}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self._PARAMETROS:
                raise ValueError(f"Parámetro desconocido: {k}")
            setattr(self, k, v)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"RegresionLogisticaNumpy({args})"

    # --- matemáticas ------------------------------------------------------------------------
    @staticmethod
    def sigmoide(z):
        """σ(z) numéricamente estable: evita exp de números grandes positivos o negativos."""
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _pesos_muestra(self, y):
        if self.pesos_clase == "balanced":
            n, n1 = len(y), y.sum()
            w1, w0 = n / (2 * n1), n / (2 * (n - n1))  # misma fórmula que class_weight="balanced"
            return np.where(y == 1, w1, w0)
        return np.ones_like(y)

    def perdida(self, X, y, s=None):
        """Entropía cruzada ponderada + L2 (lo que el descenso por gradiente minimiza)."""
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        s = self._pesos_muestra(y) if s is None else s
        p = np.clip(self.sigmoide(X @ self.coef_ + self.intercept_), 1e-12, 1 - 1e-12)
        ll = -(s * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / s.sum()
        return ll + 0.5 * self.lambda_l2 * np.dot(self.coef_, self.coef_)

    def gradiente(self, X, y, s):
        """Gradiente analítico de la pérdida respecto a (w, b)."""
        p = self.sigmoide(X @ self.coef_ + self.intercept_)
        err = s * (p - y)
        norm = s.sum()
        return X.T @ err / norm + self.lambda_l2 * self.coef_, err.sum() / norm

    # --- entrenamiento y predicción ---------------------------------------------------------
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        n, d = X.shape
        rng = np.random.default_rng(self.semilla)
        s = self._pesos_muestra(y)
        self.classes_ = np.array([0, 1])
        self.coef_ = np.zeros(d)
        self.intercept_ = 0.0
        self.historial_perdida_ = [self.perdida(X, y, s)]
        mejor, sin_mejora = np.inf, 0
        for _ in range(self.epocas):
            orden = rng.permutation(n)
            for ini in range(0, n, self.tam_lote):
                lote = orden[ini:ini + self.tam_lote]
                g_w, g_b = self.gradiente(X[lote], y[lote], s[lote])
                self.coef_ -= self.tasa_aprendizaje * g_w
                self.intercept_ -= self.tasa_aprendizaje * g_b
            actual = self.perdida(X, y, s)
            self.historial_perdida_.append(actual)
            sin_mejora = sin_mejora + 1 if mejor - actual < self.tol else 0
            mejor = min(mejor, actual)
            if sin_mejora >= self.paciencia:
                break
        self.n_epocas_ = len(self.historial_perdida_) - 1
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = self.sigmoide(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, umbral=0.5):
        return (self.predict_proba(X)[:, 1] >= umbral).astype(int)
