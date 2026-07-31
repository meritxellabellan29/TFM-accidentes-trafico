"""
Funciones de entrenamiento y evaluación para el TFM de accidentes de tráfico
en Madrid. Se usan igual para el modelo de triaje operativo (Pregunta 1:
variables disponibles en el momento del aviso) y para el modelo explicativo
(Pregunta 2: todas las variables) — lo único que cambia entre ambos es qué
matriz de features se les pasa, no el código de entrenamiento/evaluación.

Todas las métricas están pensadas para un problema con fuerte desbalanceo de
clases (~9.6% "grave"): se prioriza AUC-ROC, AUC-PR (average precision) y
calibración (Brier score) por encima de accuracy, que es engañosa con este
nivel de desbalanceo (el baseline naïf de "nunca grave" ya da 90.4%).

Elección de modelos candidatos (ver justificación completa en el notebook):
SVM y KNN se descartan como candidatos principales aquí — a diferencia de un
problema balanceado con pocas variables continuas (como en otras prácticas
de clasificación binaria), este dataset tiene fuerte desbalanceo, decenas de
columnas one-hot dispersas y ~58k filas de train, un escenario donde el
coste computacional y el mal encaje de esos dos algoritmos con variables
dummy los hace poco prácticos frente a modelos basados en árboles/boosting.

Las funciones gráficas (`graficar_*`) que originalmente vivían en este mismo
módulo se movieron a `src.utils.graficos_modelos` -- aquí solo queda
entrenamiento/evaluación, sin dependencias de matplotlib.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    recall_score, precision_score, f1_score,
)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False


def entrenar_modelos_baseline(X_train, y_train, random_state=42):
    """Entrena los modelos candidatos recomendados para un problema tabular
    con fuerte desbalanceo (~9.6% clase positiva) y features mayoritariamente
    one-hot/cíclicas:

    - Regresión logística con class_weight='balanced': baseline interpretable.
    - Random Forest con class_weight='balanced': robusto, buena base para
      interpretar importancia de variables (relevante para el modelo
      explicativo, Pregunta 2).
    - HistGradientBoostingClassifier con class_weight='balanced': mismo
      algoritmo que XGBoost/LightGBM, incluido en sklearn — capta
      interacciones no lineales entre variables (p. ej. ES_CRUCE x TIPO_VIA,
      ya detectada en el EDA) sin necesidad de especificarlas a mano, y no
      requiere escalado.
    - XGBoost real, solo si está instalado (`pip install xgboost`): se
      añade como alternativa si se dispone de la librería, usando
      `scale_pos_weight` (equivalente a class_weight='balanced' en XGBoost).

    Devuelve un dict {nombre: modelo_ya_entrenado}."""
    modelos = {
        'Regresión logística': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=random_state, n_jobs=-1
        ),
        'Gradient Boosting (Hist)': HistGradientBoostingClassifier(
            class_weight='balanced', random_state=random_state
        ),
    }

    if XGBOOST_DISPONIBLE:
        ratio_desbalanceo = (y_train == 0).sum() / (y_train == 1).sum()
        modelos['XGBoost'] = XGBClassifier(
            scale_pos_weight=ratio_desbalanceo, random_state=random_state,
            eval_metric='logloss', n_jobs=-1
        )

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
    return modelos


def explorar_hiperparametros(estimador_factory, grid, X_train, y_train, cv_splits=5):
    """Grid MANUAL y explícito (no distribuciones aleatorias): prueba cada
    combinación exacta del `grid` dado, entrenando sobre TODO train y
    validando con TimeSeriesSplit -- todo dentro de train, sin tocar test
    en ningún momento de esta exploración.

    El gap se calcula como AUC_train - AUC_cv (validación cruzada dentro de
    train), NO como AUC_train - AUC_test. Es una distinción importante: si
    se calculara el gap contra test para elegir entre muchas combinaciones,
    se estaría mirando el test repetidamente para decidir -- la misma fuga
    de información que la validación cruzada existe para evitar. El test
    debe evaluarse una única vez, al final, con la configuración ya elegida
    aquí (ver `evaluar_modelo`/`tabla_comparativa_modelos`).

    `estimador_factory` es una función que, dados los kwargs de una fila del
    grid, devuelve un estimador sin entrenar (p. ej.
    `lambda **kw: RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **kw)`).
    `grid` es un dict {parametro: [valores a probar]}, igual que
    `sklearn.model_selection.ParameterGrid`.

    Devuelve una tabla ordenada por AUC_cv descendente, con una columna
    `gap_train_cv` para poder elegir a ojo la combinación con mejor
    compromiso entre AUC y generalización."""
    from sklearn.model_selection import ParameterGrid, TimeSeriesSplit, cross_val_score

    cv = TimeSeriesSplit(n_splits=cv_splits)
    resultados = []

    for params in ParameterGrid(grid):
        modelo = estimador_factory(**params)
        auc_cv = cross_val_score(modelo, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1).mean()
        modelo.fit(X_train, y_train)
        auc_train = roc_auc_score(y_train, modelo.predict_proba(X_train)[:, 1])

        resultados.append({
            **params,
            'AUC_train': round(auc_train, 4),
            'AUC_cv': round(auc_cv, 4),
            'gap_train_cv': round(auc_train - auc_cv, 4),
        })

    return pd.DataFrame(resultados).sort_values('AUC_cv', ascending=False)


def tunear_modelos(X_train, y_train, n_iter=30, cv_splits=5, random_state=42, verbose=1):
    """Ajusta los 4 modelos candidatos con `RandomizedSearchCV`, usando
    `TimeSeriesSplit` como esquema de validación cruzada -- coherente con
    que el problema tiene estructura temporal (evita mezclar años dentro de
    la propia búsqueda de hiperparámetros).

    Importante: la búsqueda se hace ENTERAMENTE dentro de `X_train`/`y_train`
    -- el conjunto de test no se toca en ningún momento de este proceso.
    Solo después de llamar a esta función se debe evaluar el resultado en
    test, una única vez, con la configuración ya elegida -- mirar el test
    varias veces mientras se prueban combinaciones sería una fuga de
    información hacia la evaluación final (el mismo motivo por el que la
    validación cruzada existe: tunear sin gastar el conjunto de evaluación).

    Devuelve (modelos_tuneados, tabla_resumen): un dict {nombre: modelo ya
    reentrenado con la mejor configuración sobre TODO X_train} y una tabla
    con los mejores hiperparámetros y el AUC-ROC medio de validación
    cruzada de cada uno (no el de test)."""
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from scipy.stats import randint, uniform

    cv = TimeSeriesSplit(n_splits=cv_splits)

    espacios = {
        'Regresión logística': (
            LogisticRegression(class_weight='balanced', max_iter=2000, random_state=random_state),
            {'C': uniform(0.01, 10)},
        ),
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1),
            {
                'max_depth': randint(3, 15),
                'n_estimators': randint(100, 400),
                'max_samples': uniform(0.2, 0.7),
                'min_samples_leaf': randint(1, 50),
            },
        ),
        'Gradient Boosting (Hist)': (
            HistGradientBoostingClassifier(class_weight='balanced', random_state=random_state),
            {
                'max_leaf_nodes': randint(10, 63),
                'learning_rate': uniform(0.02, 0.28),
                'l2_regularization': uniform(0.0, 1.0),
                'max_iter': randint(80, 300),
                'min_samples_leaf': randint(10, 50),
            },
        ),
    }

    if XGBOOST_DISPONIBLE:
        ratio_desbalanceo = (y_train == 0).sum() / (y_train == 1).sum()
        espacios['XGBoost'] = (
            XGBClassifier(scale_pos_weight=ratio_desbalanceo, random_state=random_state,
                           eval_metric='logloss', n_jobs=-1),
            {
                'max_depth': randint(2, 8),
                'learning_rate': uniform(0.01, 0.29),
                'n_estimators': randint(80, 300),
                'reg_alpha': uniform(0.0, 1.0),
                'reg_lambda': uniform(0.5, 2.0),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
            },
        )

    modelos_tuneados = {}
    filas_resumen = {}

    for nombre, (estimador, espacio) in espacios.items():
        if verbose:
            print(f'Tuneando {nombre}...')
        busqueda = RandomizedSearchCV(
            estimador, param_distributions=espacio, n_iter=n_iter,
            scoring='roc_auc', cv=cv, random_state=random_state, n_jobs=-1, refit=True,
        )
        busqueda.fit(X_train, y_train)
        modelos_tuneados[nombre] = busqueda.best_estimator_
        filas_resumen[nombre] = {
            **busqueda.best_params_,
            'AUC-ROC_cv': round(busqueda.best_score_, 4),
        }
        if verbose:
            print(f'  Mejor AUC-ROC (CV): {busqueda.best_score_:.4f} | params: {busqueda.best_params_}')

    tabla_resumen = pd.DataFrame(filas_resumen).T
    return modelos_tuneados, tabla_resumen


def evaluar_modelo(modelo, X, y, umbral=0.5):
    """Calcula el conjunto de métricas relevante para un problema
    desbalanceado: AUC-ROC y AUC-PR (no dependen de un umbral), Brier score
    (calibración), y precision/recall/F1 en el umbral dado (por defecto 0.5
    — recordar que con este nivel de desbalanceo, 0.5 puede no ser el mejor
    punto de corte para un uso real de priorización)."""
    y_proba = modelo.predict_proba(X)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)

    return {
        'AUC-ROC': roc_auc_score(y, y_proba),
        'AUC-PR': average_precision_score(y, y_proba),
        'Brier score': brier_score_loss(y, y_proba),
        f'Precision (umbral={umbral})': precision_score(y, y_pred, zero_division=0),
        f'Recall (umbral={umbral})': recall_score(y, y_pred, zero_division=0),
        f'F1 (umbral={umbral})': f1_score(y, y_pred, zero_division=0),
    }


def tabla_comparativa_modelos(modelos, X_train, y_train, X_test, y_test, umbral=0.5):
    """Evalúa varios modelos ya entrenados EN TRAIN Y EN TEST — no solo en
    test — para poder medir el gap train-test (síntoma de overfitting). Un
    modelo con AUC-ROC ~0.99 en train y ~0.75 en test está sobreajustando,
    aunque su métrica de test por sí sola parezca razonable.

    Devuelve una tabla con columnas sufijadas '_train'/'_test' y una columna
    adicional 'gap_AUC-ROC' (train - test) para localizar overfitting de un
    vistazo."""
    filas = {}
    for nombre, modelo in modelos.items():
        m_train = evaluar_modelo(modelo, X_train, y_train, umbral=umbral)
        m_test = evaluar_modelo(modelo, X_test, y_test, umbral=umbral)
        fila = {f'{k}_train': v for k, v in m_train.items()}
        fila.update({f'{k}_test': v for k, v in m_test.items()})
        fila['gap_AUC-ROC'] = m_train['AUC-ROC'] - m_test['AUC-ROC']
        filas[nombre] = fila
    return pd.DataFrame(filas).T.round(4)


def curva_calibracion(y_test, y_proba, n_bins=10):
    """Calibración: para cada bin de probabilidad predicha, compara la
    frecuencia real de la clase positiva frente a la probabilidad media
    predicha en ese bin. Un modelo bien calibrado debería quedar cerca de la
    diagonal (frecuencia observada ≈ probabilidad predicha).

    Relevante especialmente aquí porque el split es temporal (train
    2012-2017, test 2018) y la tasa de gravedad real desciende año a año
    (~10.9% en 2012 a ~8.2% en 2018) — un modelo entrenado con una tasa base
    más alta podría sobreestimar sistemáticamente el riesgo en 2018."""
    frac_positivos, prob_media_predicha = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy='quantile')
    return pd.DataFrame({
        'prob_media_predicha': prob_media_predicha,
        'frac_positivos_real': frac_positivos,
    })


def comparar_tasa_base_train_test(y_train, y_test):
    """Compara la tasa de la clase positiva entre train y test — un
    desajuste grande aquí explicaría parte de una mala calibración,
    independientemente de la calidad del modelo."""
    return {
        'tasa_grave_train': y_train.mean(),
        'tasa_grave_test': y_test.mean(),
        'diferencia_pts': (y_test.mean() - y_train.mean()) * 100,
    }


def comparar_splits_temporal_vs_aleatorio(datos_temporal, datos_aleatorio, modelo_factory, random_state=42):
    """Entrena el MISMO modelo (misma configuración, vía `modelo_factory`)
    sobre dos splits distintos — temporal (train 2012-2017 / test 2018) y
    aleatorio estratificado — para separar dos cosas que se confunden
    fácilmente:

    - Si el AUC-ROC es similar en ambos splits pero la calibración (Brier
      score, diferencia de tasa base) solo empeora en el temporal, el
      problema es la deriva real de la tasa de gravedad a lo largo de los
      años, no un fallo del modelo ni del split en sí.
    - El split aleatorio, al ser estratificado, oculta por construcción esa
      deriva (train y test tienen casi la misma tasa base) — por eso NO
      sirve para detectar este problema, solo el temporal lo revela. Verlos
      uno al lado del otro deja esto explícito.

    `datos_temporal` y `datos_aleatorio` son tuplas
    (X_train, y_train, X_test, y_test). `modelo_factory` es una función sin
    argumentos que devuelve un modelo sin entrenar (p. ej.
    `lambda: HistGradientBoostingClassifier(class_weight='balanced', random_state=42)`),
    para garantizar que ambos splits usan exactamente la misma configuración."""
    filas = {}
    for nombre_split, (X_train, y_train, X_test, y_test) in [
        ('Temporal (train 2012-17 / test 2018)', datos_temporal),
        ('Aleatorio estratificado', datos_aleatorio),
    ]:
        modelo = modelo_factory()
        modelo.fit(X_train, y_train)
        m_train = evaluar_modelo(modelo, X_train, y_train)
        m_test = evaluar_modelo(modelo, X_test, y_test)
        tasas = comparar_tasa_base_train_test(y_train, y_test)
        filas[nombre_split] = {
            'AUC-ROC_train': m_train['AUC-ROC'], 'AUC-ROC_test': m_test['AUC-ROC'],
            'gap_AUC-ROC': m_train['AUC-ROC'] - m_test['AUC-ROC'],
            'Brier_test': m_test['Brier score'],
            'tasa_grave_train_%': tasas['tasa_grave_train'] * 100,
            'tasa_grave_test_%': tasas['tasa_grave_test'] * 100,
            'diferencia_tasa_pts': tasas['diferencia_pts'],
        }
    return pd.DataFrame(filas).T.round(4)


def validacion_temporal_expansiva(X, y, años, modelo_factory, año_min_train=2013, año_max=2018):
    """Validación walk-forward (ventana expansiva) por año natural: para
    cada año de validación Y, entrena con todos los datos de años
    anteriores a Y y valida sobre el año Y, avanzando año a año.

    Es el análogo temporal de `TimeSeriesSplit`, pero alineado con los años
    del dataset en vez de con particiones de tamaño arbitrario — más
    interpretable aquí, porque el objetivo real no es tunear hiperparámetros
    sino comprobar si el split final (train 2012-2017, test 2018) es
    representativo del comportamiento general del modelo a lo largo del
    tiempo, o si 2018 se comporta de forma atípica frente a otros años.

    `X`, `y`, `años` deben ser indexables por máscara booleana y compartir
    el mismo índice (p. ej. tres pandas Series/DataFrame alineados).

    Limitación importante, que conviene documentar en el notebook: las
    variables ya construidas (target encoding de DISTRITO, one-hot de
    TIPO_VIA/TIPO ACCIDENTE) se ajustaron una única vez sobre 2012-2017 en
    `03_Feature_Engineering.ipynb`, no se reajustan dentro de cada fold. Un
    walk-forward estrictamente sin fuga de información exigiría reajustar
    esas transformaciones en cada iteración usando solo los años de ese
    fold — aquí se usa la aproximación más simple (features fijas), que
    puede optimizar ligeramente los folds más tempranos."""
    resultados = []
    for año_val in range(año_min_train + 1, año_max + 1):
        mask_train = años < año_val
        mask_val = años == año_val
        if mask_train.sum() == 0 or mask_val.sum() == 0:
            continue
        modelo = modelo_factory()
        modelo.fit(X[mask_train], y[mask_train])
        y_proba = modelo.predict_proba(X[mask_val])[:, 1]
        resultados.append({
            'año_validacion': año_val,
            'n_train': int(mask_train.sum()),
            'n_val': int(mask_val.sum()),
            'tasa_grave_train_%': round(y[mask_train].mean() * 100, 2),
            'tasa_grave_val_%': round(y[mask_val].mean() * 100, 2),
            'AUC-ROC_val': round(roc_auc_score(y[mask_val], y_proba), 4),
        })
    return pd.DataFrame(resultados)


def entrenar_random_forest_tuneado(X_train, y_train, max_depth=5, n_estimators=100,
                                    max_samples=0.2, random_state=42):
    """Random Forest con los hiperparámetros resultantes del análisis de
    estabilidad del bagging (`analisis_estabilidad_bagging`): limitar
    `max_depth` es lo que realmente corrige el overfitting (gap ~0.30 sin
    límite -> ~0.025 con max_depth=5); una vez fijada la profundidad, variar
    `max_samples`/`n_estimators` apenas cambia el resultado, así que se usa
    la combinación más barata computacionalmente dentro de la zona estable."""
    modelo = RandomForestClassifier(
        max_depth=max_depth, n_estimators=n_estimators, max_samples=max_samples,
        class_weight='balanced', random_state=random_state, n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    return modelo


def tabla_train_test(modelo, X_train, y_train, X_test, y_test, umbral=0.5):
    """Evalúa un modelo ya entrenado en train y test, devolviendo una tabla
    de una sola fila por métrica (en vez de dos dicts sueltos por consola)
    -- pensada para las celdas de "baseline" de cada modelo individual,
    donde `tabla_comparativa_modelos` sería excesivo (esa está pensada para
    comparar varios modelos a la vez, no uno solo)."""
    m_train = evaluar_modelo(modelo, X_train, y_train, umbral=umbral)
    m_test = evaluar_modelo(modelo, X_test, y_test, umbral=umbral)
    tabla = pd.DataFrame({'Train': m_train, 'Test': m_test})
    tabla['Gap (train-test)'] = tabla['Train'] - tabla['Test']
    return tabla.round(4)


def tabla_hiperparametros(modelos, solo_relevantes=True):
    """Extrae los hiperparámetros REALES de cada modelo ya entrenado
    (`get_params()`), no los que se pasaron explícitamente al crearlo --
    incluye también los que se quedaron en su valor por defecto, que es
    precisamente donde suelen estar las causas de overfitting no detectadas
    (p. ej. `max_depth=None` en Random Forest, o `max_depth=6` por defecto
    en XGBoost, sin que nadie lo haya fijado a propósito).

    Con `solo_relevantes=True` (por defecto), filtra a un subconjunto de
    parámetros que suelen explicar diferencias de comportamiento entre
    modelos (profundidad, nº de estimadores, learning rate, regularización,
    manejo del desbalanceo) -- con `False`, muestra el `get_params()`
    completo de cada modelo, mucho más largo pero exhaustivo."""
    PARAMS_RELEVANTES = {
        'max_depth', 'max_iter', 'n_estimators', 'max_samples',
        'learning_rate', 'max_leaf_nodes', 'min_samples_leaf',
        'l2_regularization', 'reg_lambda', 'reg_alpha',
        'class_weight', 'scale_pos_weight', 'C', 'penalty',
        'subsample', 'colsample_bytree', 'early_stopping',
        'random_state',
    }

    filas = {}
    for nombre, modelo in modelos.items():
        params = modelo.get_params()
        if solo_relevantes:
            params = {k: v for k, v in params.items() if k in PARAMS_RELEVANTES}
        # Los valores None son informativos (p. ej. max_depth=None = sin límite,
        # justo lo que causaba el overfitting de Random Forest) -- se marcan
        # como texto explícito para no confundirlos con "parámetro no aplicable"
        params = {k: ('None (sin límite)' if v is None else v) for k, v in params.items()}
        filas[nombre] = params

    tabla = pd.DataFrame(filas)
    tabla = tabla.fillna('— (no aplica)')  # aquí sí: clave ausente = el modelo no tiene ese parámetro
    return tabla


def analisis_estabilidad_bagging(X_train, y_train, X_test, y_test,
                                  porcentajes=(0.2, 0.4, 0.6, 0.8, 1.0),
                                  n_arboles=(100, 300, 500),
                                  profundidades=(5, 10, None),
                                  random_state=42):
    """Grid de estabilidad del bagging para Random Forest: barre el
    porcentaje de muestra por árbol (`max_samples`), el número de árboles
    (`n_estimators`) y la profundidad máxima de cada árbol (`max_depth`),
    midiendo AUC-ROC en train y test y el gap entre ambos en cada
    combinación.

    Se añade `max_depth` como tercera dimensión porque variar solo
    `max_samples`/`n_estimators` no corrige el overfitting si cada árbol
    individual ya memoriza el train (árboles sin límite de profundidad,
    `max_depth=None`, por defecto en sklearn) — el bagging reduce varianza
    entre árboles, pero si todos parten ya sobreajustados, promediarlos no
    resuelve el problema de raíz. `None` se incluye como referencia para
    comparar contra el comportamiento por defecto.

    El objetivo del bagging es reducir el gap train-test manteniendo el AUC
    de test — se busca la combinación donde el gap se estabiliza en un valor
    bajo sin que el AUC de test empeore."""
    resultados = []
    for prof in profundidades:
        for pct in porcentajes:
            for n in n_arboles:
                modelo = RandomForestClassifier(
                    n_estimators=n, max_samples=pct, max_depth=prof,
                    class_weight='balanced', random_state=random_state, n_jobs=-1
                )
                modelo.fit(X_train, y_train)
                auc_train = roc_auc_score(y_train, modelo.predict_proba(X_train)[:, 1])
                auc_test = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
                resultados.append({
                    'max_depth': prof if prof is not None else 'Sin límite',
                    'max_samples': pct, 'n_estimators': n,
                    'AUC_train': round(auc_train, 4), 'AUC_test': round(auc_test, 4),
                    'gap': round(auc_train - auc_test, 4),
                })
    return pd.DataFrame(resultados).sort_values('gap')
