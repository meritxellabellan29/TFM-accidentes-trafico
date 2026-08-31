"""
Funciones de interpretabilidad para el TFM de accidentes de tráfico en
Madrid -- usadas en 05_Interpretabilidad.ipynb para traducir los valores
SHAP (calculados sobre el modelo de referencia) a tablas listas para la
memoria, sin copiar ningún número a mano desde un gráfico.

Todo se extrae directamente del objeto `shap.Explanation` que devuelve
`explainer(muestra)` -- nombres de variables, magnitud de cada una, y el
propio valor base (log-odds) que usó el explainer -- para que la tabla
sea siempre un reflejo exacto del cálculo real, sin desincronizarse si se
vuelve a ejecutar el notebook con otra muestra o otro modelo.
"""
import numpy as np
import pandas as pd


def _log_odds_a_probabilidad(log_odds):
    """Sigmoide: convierte log-odds a probabilidad (0-1)."""
    return 1 / (1 + np.exp(-log_odds))


def shap_a_puntos_probabilidad(shap_value, log_odds_base):
    """Convierte una contribución SHAP (en log-odds) a un cambio aproximado
    en puntos de probabilidad, partiendo de un log-odds base concreto. El
    resultado depende del punto de partida (no linealidad de la sigmoide),
    así que no es un factor de conversión fijo -- es ilustrativo, para dar
    intuición de magnitud en la memoria, no un valor exacto por accidente."""
    prob_base = _log_odds_a_probabilidad(log_odds_base)
    prob_nueva = _log_odds_a_probabilidad(log_odds_base + shap_value)
    return (prob_nueva - prob_base) * 100


def tabla_importancia_shap(shap_values, top_n=15):
    """Extrae automáticamente de un objeto `shap.Explanation` (el que
    devuelve `explainer(muestra)`) la importancia media (|SHAP|, en
    log-odds) de cada variable, y la convierte a una estimación aproximada
    en puntos de probabilidad -- sin necesidad de mirar el gráfico y copiar
    valores a mano.

    El log-odds base se toma directamente de `shap_values.base_values`
    (el valor real usado por el propio explainer para esta muestra y este
    modelo), no de una tasa recalculada aparte -- así la conversión queda
    garantizada consistente con los valores SHAP que se están interpretando,
    incluso si el modelo o la muestra cambian entre ejecuciones.

    `shap_values` es el objeto `shap.Explanation` devuelto por
    `explainer(muestra)`; `top_n` limita la tabla a las variables más
    importantes por |SHAP| medio (`None` devuelve la tabla completa).
    Devuelve un DataFrame con columnas variable, SHAP_log_odds y
    puntos_probabilidad_aprox, ordenado de mayor a menor importancia."""
    feature_names = list(shap_values.feature_names)
    valores = np.asarray(shap_values.values)
    base_values = np.asarray(shap_values.base_values)

    # base_values normalmente es constante para todas las filas de la
    # muestra en modelos de árboles -- se promedia por robustez, por si
    # hubiera pequeñas variaciones numéricas entre filas
    log_odds_base = float(np.mean(base_values))

    importancia_media = np.abs(valores).mean(axis=0)

    tabla = pd.DataFrame({
        'variable': feature_names,
        'SHAP_log_odds': importancia_media,
        'puntos_probabilidad_aprox': [
            shap_a_puntos_probabilidad(v, log_odds_base) for v in importancia_media
        ],
    }).sort_values('SHAP_log_odds', ascending=False).reset_index(drop=True)

    tabla['SHAP_log_odds'] = tabla['SHAP_log_odds'].round(4)
    tabla['puntos_probabilidad_aprox'] = tabla['puntos_probabilidad_aprox'].round(2)

    if top_n:
        return tabla.head(top_n)
    return tabla


def tabla_shap_por_distrito(shap_values, muestra, pipe, columna='TASA_GRAVEDAD_HIST_DISTRITO'):
    """Traduce la contribución SHAP de un agregado histórico de distrito
    (p. ej. `TASA_GRAVEDAD_HIST_DISTRITO`) de vuelta al nombre real del
    distrito, usando el mapeo aprendido y guardado dentro del propio
    `PipelineAccidentes` (`pipe.mapeos_encoding`) en vez de recalcularlo o
    copiarlo a mano.

    Solo es exacto sobre filas de **test**: en `transform()` cada distrito
    recibe un único valor fijo (el mapeo completo aprendido en `fit()`), así
    que la relación valor -> distrito es 1 a 1. En **train** el valor viene
    de un *target encoding* out-of-fold (`smoothed_target_encode_oof`), por
    lo que un mismo distrito puede tener valores ligeramente distintos entre
    filas y el mapeo inverso no sería exacto -- de ahí que `muestra` deba
    proceder de `X_test_ref`.

    `shap_values` es el objeto devuelto por `explainer(muestra)`; `muestra`
    son esas mismas filas (`X_test_ref` o una muestra de ella) con la
    columna `columna` sin transformar; `pipe` es el `PipelineAccidentes` ya
    ajustado (`.load(...)`), con `mapeos_encoding` poblado en `fit()`.
    Devuelve un DataFrame con columnas DISTRITO, n, shap_medio y tasa_hist,
    ordenado de mayor a menor `shap_medio`."""
    col_original, mapping, _ = pipe.mapeos_encoding[columna]
    mapeo_inverso = {round(valor, 10): distrito for distrito, valor in mapping.items()}

    idx = list(muestra.columns).index(columna)
    distrito = muestra[columna].round(10).map(mapeo_inverso)

    sin_match = distrito.isna().sum()
    if sin_match > 0:
        raise ValueError(
            f'{sin_match} filas de "{columna}" no coinciden con ningún valor de '
            f'pipe.mapeos_encoding -- probablemente `muestra` incluye filas de train '
            f'(out-of-fold, no invertible) en vez de solo test.'
        )

    tabla = pd.DataFrame({
        col_original: distrito.values,
        'shap_valor': shap_values.values[:, idx],
        'tasa_hist': muestra[columna].values,
    })

    resumen = (
        tabla.groupby(col_original)
        .agg(n=('shap_valor', 'size'), shap_medio=('shap_valor', 'mean'), tasa_hist=('tasa_hist', 'first'))
        .sort_values('shap_medio', ascending=False)
        .reset_index()
        .rename(columns={col_original: 'DISTRITO'})
    )
    resumen['shap_medio'] = resumen['shap_medio'].round(4)
    resumen['tasa_hist'] = resumen['tasa_hist'].round(4)
    return resumen


def tabla_efecto_binario(shap_values, muestra, columnas, log_odds_base=None):
    """Para cada variable binaria en `columnas`, traduce su contribución
    SHAP a una probabilidad de gravedad media estimada por el modelo cuando
    la variable vale 1 frente a cuando vale 0.

    Es más intuitivo que `tabla_importancia_shap` para un público no
    técnico (informe de prevención): esta última promedia el valor
    absoluto de la contribución sobre todas las filas, mezclando en un
    único número los casos en los que la variable empuja el riesgo hacia
    arriba y hacia abajo. Aquí se separan ambos grupos y se expresan como
    un porcentaje directamente comparable con la tasa de gravedad global.

    `shap_values` y `muestra` deben ser el mismo objeto `shap.Explanation`
    y el mismo DataFrame usados para calcularlo (igual que en
    `tabla_shap_por_distrito`); `log_odds_base` por defecto se toma de
    `shap_values.base_values`, igual que en `tabla_importancia_shap`, para
    que ambas tablas sean siempre consistentes entre sí. Devuelve un
    DataFrame con columnas variable, n_con, n_sin, pct_gravedad_con,
    pct_gravedad_sin y diferencia_puntos, ordenado por esta última."""
    valores = np.asarray(shap_values.values)
    if log_odds_base is None:
        log_odds_base = float(np.mean(np.asarray(shap_values.base_values)))

    filas = []
    for col in columnas:
        idx = list(muestra.columns).index(col)
        es_uno = muestra[col].to_numpy() == 1
        shap_con = valores[es_uno, idx].mean() if es_uno.any() else np.nan
        shap_sin = valores[~es_uno, idx].mean() if (~es_uno).any() else np.nan
        pct_con = _log_odds_a_probabilidad(log_odds_base + shap_con) * 100
        pct_sin = _log_odds_a_probabilidad(log_odds_base + shap_sin) * 100
        filas.append({
            'variable': col,
            'n_con': int(es_uno.sum()),
            'n_sin': int((~es_uno).sum()),
            'pct_gravedad_con': round(pct_con, 2),
            'pct_gravedad_sin': round(pct_sin, 2),
            'diferencia_puntos': round(pct_con - pct_sin, 2),
        })

    return pd.DataFrame(filas).sort_values('diferencia_puntos', ascending=False).reset_index(drop=True)


def tabla_odds_ratios(modelo_logreg, feature_names, top_n=15):
    """Extrae automáticamente de una `LogisticRegression` binaria ya
    entrenada los *odds ratios* de cada variable -- `exp(coeficiente)` --
    para usarlos como validación cruzada de la importancia SHAP con un
    modelo de naturaleza distinta (lineal, coeficientes constantes) frente
    al ensemble de árboles.

    A diferencia de los puntos de probabilidad de `shap_a_puntos_probabilidad`
    (que dependen del punto de partida), el odds ratio es constante: "esta
    variable multiplica las odds de gravedad por X, manteniendo el resto
    constante" -- sin necesidad de fijar una tasa base.

    Aviso: las variables continuas sin escalar (p. ej. tasas o conteos) no
    son directamente comparables en magnitud con las binarias -- el
    coeficiente de una continua depende de su rango de valores. Válido para
    comparar variables binarias/dummy entre sí (como INCLUYE_MOTO,
    ACC_ATROPELLO), que es el uso que se le da en la memoria.

    `modelo_logreg` es un `LogisticRegression` ya entrenado (clasificación
    binaria); `feature_names` son los nombres de columna en el mismo orden
    que se usó para entrenar; `top_n` limita la tabla a las variables con
    mayor magnitud de coeficiente (`None` devuelve la tabla completa).
    Devuelve un DataFrame con columnas variable, coeficiente y odds_ratio,
    ordenado por magnitud de coeficiente de mayor a menor."""
    coeficientes = np.asarray(modelo_logreg.coef_).ravel()

    tabla = pd.DataFrame({
        'variable': list(feature_names),
        'coeficiente': coeficientes,
        'odds_ratio': np.exp(coeficientes),
    }).sort_values('coeficiente', key=np.abs, ascending=False).reset_index(drop=True)

    tabla['coeficiente'] = tabla['coeficiente'].round(4)
    tabla['odds_ratio'] = tabla['odds_ratio'].round(2)

    if top_n:
        return tabla.head(top_n)
    return tabla
