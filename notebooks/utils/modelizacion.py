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


def graficar_validacion_temporal(tabla, filename=None):
    """Grafica la evolución del AUC-ROC de validación año a año en el
    walk-forward, junto con la tasa de gravedad de cada año — para ver de
    un vistazo si el rendimiento se mantiene estable o si empeora conforme
    se valida en años más recientes.

    El eje del AUC-ROC se fija en un rango con contexto (ancla en 0.5,
    el nivel de un clasificador aleatorio) en vez de autoajustarse al rango
    exacto de los datos — si no, una variación de apenas 0.02 puntos puede
    parecer visualmente enorme cuando en realidad es ruido menor."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(tabla['año_validacion'], tabla['AUC-ROC_val'], marker='o', color='#2a78d6', linewidth=2, label='AUC-ROC (validación)')
    ax2.plot(tabla['año_validacion'], tabla['tasa_grave_val_%'], marker='s', color='#eb6834', linewidth=2, linestyle='--', label='% grave (validación)')

    ax1.set_xlabel('Año de validación')
    ax1.set_ylabel('AUC-ROC', color='#2a78d6')
    ax2.set_ylabel('% de gravedad', color='#eb6834')
    ax1.set_title('Validación walk-forward: AUC-ROC y tasa de gravedad por año')
    ax1.grid(alpha=0.3)

    # Años como enteros, sin decimales
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Rango del AUC-ROC con contexto: ancla en 0.5 (azar) para no exagerar
    # visualmente una variación de apenas 0.02 puntos
    auc_max = tabla['AUC-ROC_val'].max()
    ax1.set_ylim(0.5, max(0.85, auc_max + 0.05))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


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


def graficar_curvas_roc(modelos, X_test, y_test, titulo='Curvas ROC — comparación de modelos', filename=None):
    """Superpone la curva ROC de varios modelos ya entrenados sobre el mismo
    test, con el AUC de cada uno en la leyenda, y la diagonal de referencia
    (clasificador aleatorio, AUC=0.5)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, modelo in modelos.items():
        y_proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{nombre} (AUC={auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Aleatorio (AUC=0.5)')
    ax.set_xlabel('Tasa de falsos positivos (1 - especificidad)')
    ax.set_ylabel('Tasa de verdaderos positivos (recall)')
    ax.set_title(titulo)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def graficar_curvas_roc_train_test(modelos, X_train, y_train, X_test, y_test, filename=None):
    """Curva ROC train vs. test de todos los modelos a la vez, en una
    rejilla de subplots (uno por modelo) — para comparar de un vistazo qué
    modelos sobreajustan (curva de train muy por encima de la de test) y
    cuáles generalizan bien (ambas curvas próximas entre sí), sin tener que
    generar un gráfico separado por modelo."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    import math

    n = len(modelos)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        for nombre_split, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
            y_proba = modelo.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            auc = roc_auc_score(y, y_proba)
            ax.plot(fpr, tpr, label=f'{nombre_split} (AUC={auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1)
        ax.set_title(nombre, fontsize=11)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

    # Ocultar subplots sobrantes si el nº de modelos no llena la rejilla
    for ax in axes[len(modelos):]:
        ax.axis('off')

    fig.suptitle('Curvas ROC train vs. test — todos los modelos', fontsize=14)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


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
