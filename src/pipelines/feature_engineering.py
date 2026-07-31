"""
Funciones de feature engineering para el TFM de accidentes de tráfico en
Madrid. Se aplican siempre ajustando con `train` y transfiriendo el
resultado a `test`, para no filtrar información del futuro hacia el pasado.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def smoothed_target_encode(train_df, col, target, m):
    """Codificación de una categórica por la media (suavizada) de `target`
    en ese grupo, ajustada con TODO train. Se usa para el mapeo final que
    se aplica a test."""
    global_mean = train_df[target].mean()
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    smoothed = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
    return smoothed, global_mean


def smoothed_target_encode_oof(train_df, col, target, m, n_splits=5, random_state=42):
    """Versión out-of-fold para TRAIN: cada fila se codifica con un mapeo
    aprendido en los folds a los que NO pertenece, para que no contribuya a
    su propio valor (si no, cada fila "ve" un poco de su propia etiqueta a
    través de la media de su grupo — una fuga de información sutil pero
    real)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(index=train_df.index, dtype=float)
    global_mean = train_df[target].mean()

    for idx_fit, idx_hold in kf.split(train_df):
        fit_fold = train_df.iloc[idx_fit]
        agg = fit_fold.groupby(col)[target].agg(['mean', 'count'])
        smoothed_fold = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
        oof.iloc[idx_hold] = train_df.iloc[idx_hold][col].map(smoothed_fold).fillna(global_mean)

    return oof


def apply_encoding(df, col, mapping, global_mean, new_col):
    """Aplica el mapeo aprendido en train; categorías no vistas -> media global."""
    df = df.copy()
    df[new_col] = df[col].map(mapping).fillna(global_mean)
    return df


def codificar_distrito_historico(train, test, m=50):
    """Aplica smoothed target encoding de DISTRITO sobre GRAVE, INCLUYE_PEATON
    e INCLUYE_MOTO: train con out-of-fold, test con el mapeo completo de train.

    Recupera parte de la señal de las variables de persona sin romper la
    disponibilidad temporal — es información de contexto histórico del
    distrito, no del accidente actual."""
    train = train.copy()
    test = test.copy()
    especificaciones = [
        ('DISTRITO', 'GRAVE',          'TASA_GRAVEDAD_HIST_DISTRITO'),
        ('DISTRITO', 'INCLUYE_PEATON', 'PCT_HIST_PEATON_DISTRITO'),
        ('DISTRITO', 'INCLUYE_MOTO',   'PCT_HIST_MOTO_DISTRITO'),
    ]
    for col, target, nuevo_nombre in especificaciones:
        train[nuevo_nombre] = smoothed_target_encode_oof(train, col, target, m)
        mapping, global_mean = smoothed_target_encode(train, col, target, m)
        test = apply_encoding(test, col, mapping, global_mean, nuevo_nombre)
    return train, test


def codificar_ciclica(df, col, periodo):
    """Codifica una variable cíclica (p. ej. HORA con periodo 24, MES con
    periodo 12) en seno/coseno, para no imponer un salto artificial en la
    frontera del ciclo (23h -> 0h, diciembre -> enero)."""
    df = df.copy()
    df[f'{col}_SIN'] = np.sin(2 * np.pi * df[col] / periodo)
    df[f'{col}_COS'] = np.cos(2 * np.pi * df[col] / periodo)
    return df


def one_hot_fit_train(df, col, cats, prefix):
    """One-hot manual con columnas fijadas por `cats` (aprendidas de train):
    categorías no vistas en `df` quedan a 0 en todas las columnas —
    equivalente a la categoría de referencia — en vez de romper el pipeline."""
    dummies = pd.get_dummies(df[col], prefix=prefix)
    return dummies.reindex(columns=[f'{prefix}_{c}' for c in cats], fill_value=0).astype(int)


def pares_alta_colinealidad(X, umbral=0.6):
    """Devuelve los pares de variables con correlación absoluta > umbral,
    ordenados de mayor a menor — para revisar antes de usar un modelo
    lineal."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'var_1', 'level_1': 'var_2', 0: 'correlacion'})
        .query('correlacion > @umbral')
        .sort_values('correlacion', ascending=False)
    )
