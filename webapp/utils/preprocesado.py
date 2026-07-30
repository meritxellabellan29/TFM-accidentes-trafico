"""
Funciones de preprocesado a nivel accidente para el TFM de accidentes de
tráfico en Madrid. Se apoyan en `limpieza.py` (limpieza estructural a nivel
persona, compartida con el EDA) y añaden todo lo necesario para agregar de
persona -> accidente, tipar variables derivadas de texto libre, y separar
train/test.

Cada función hace una sola cosa, documentada con el porqué de la decisión
(no solo el qué), para que `02_Preprocessing2.ipynb` pueda leerse como una
secuencia de decisiones ya justificadas, en vez de mezclarlas con el código
que las implementa.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


GRUPOS_RIESGO_EDAD = [
    'DE 0 A 5 AÑOS', 'DE 6 A 9 AÑOS', 'DE 10 A 14 AÑOS', 'DE 15 A 17 AÑOS',
    'DE 65 A 69 AÑOS', 'DE 70 A 74 AÑOS', 'DE MAS DE 74 AÑOS'
]

ORDEN_LESIVIDAD = ['IL', 'HL', 'HG', 'MT']
GRAVES = ['HG', 'MT']

COLS_ACCIDENTE = [
    'Nº PARTE', 'FECHA', 'AÑO', 'MES', 'RANGO HORARIO', 'DIA SEMANA', 'DISTRITO', 'LUGAR ACCIDENTE',
    'TIPO ACCIDENTE', 'Nº VICTIMAS *',
    'CPFA Granizo', 'CPFA Hielo', 'CPFA Lluvia', 'CPFA Niebla', 'CPFA Seco', 'CPFA Nieve',
    'CPSV Mojada', 'CPSV Aceite', 'CPSV Barro', 'CPSV Grava Suelta', 'CPSV Hielo', 'CPSV Seca Y Limpia'
]

COLS_CATEGORICAS_MODELO = ['DISTRITO', 'TIPO ACCIDENTE', 'TIPO_VIA', 'DIA SEMANA']


def filtrar_lesividad_conocida(df, orden_lesividad=ORDEN_LESIVIDAD):
    """Filtra a personas con LESIVIDAD conocida y construye el target GRAVE
    a nivel persona. Devuelve también cuántos accidentes se perderían por
    completo si todas sus personas tuvieran LESIVIDAD desconocida (debería
    ser 0; si no lo es, el filtro no es seguro y hay que revisarlo antes de
    seguir)."""
    partes_todos = set(df['Nº PARTE'].unique())

    df_pers = df[df['LESIVIDAD'].isin(orden_lesividad)].copy()
    df_pers['GRAVE'] = df_pers['LESIVIDAD'].isin(GRAVES).astype(int)

    partes_con_dato = set(df_pers['Nº PARTE'].unique())
    n_accidentes_perdidos = len(partes_todos - partes_con_dato)

    stats = {
        'filas_eliminadas': len(df) - len(df_pers),
        'personas_tras_filtro': len(df_pers),
        'accidentes_antes': len(partes_todos),
        'accidentes_despues': len(partes_con_dato),
        'accidentes_perdidos': n_accidentes_perdidos,
    }
    return df_pers, stats


def agregar_flags_persona(df, df_pers, grupos_riesgo_edad=GRUPOS_RIESGO_EDAD):
    """Agrega de persona a accidente las variables de persona con mayor
    asociación con la gravedad (TIPO PERSONA, Tipo Vehiculo, Tramo Edad),
    como flags de implicación, en vez de perderlas al agregar.

    N_PERSONAS_IMPLICADAS se calcula sobre `df` completo (no sobre
    `df_pers`), porque de lo contrario infraestimaría el nº real de
    implicados en cualquier accidente con alguna persona de LESIVIDAD
    desconocida (~8.9% de los accidentes)."""
    n_personas_real = df.groupby('Nº PARTE').size().rename('N_PERSONAS_IMPLICADAS')

    agg_flags = df_pers.groupby('Nº PARTE').agg(
        GRAVE=('GRAVE', 'max'),
        INCLUYE_PEATON=('TIPO PERSONA', lambda s: (s == 'PEATON').any()),
        INCLUYE_MOTO=('Tipo Vehiculo', lambda s: s.isin(['MOTOCICLETA', 'CICLOMOTOR']).any()),
        INCLUYE_BICI=('Tipo Vehiculo', lambda s: (s == 'BICICLETA').any()),
        INCLUYE_EDAD_RIESGO=('Tramo Edad', lambda s: s.isin(grupos_riesgo_edad).any()),
    ).reset_index()

    agg_flags = agg_flags.merge(n_personas_real, on='Nº PARTE', how='left')

    for c in ['INCLUYE_PEATON', 'INCLUYE_MOTO', 'INCLUYE_BICI', 'INCLUYE_EDAD_RIESGO']:
        agg_flags[c] = agg_flags[c].astype(int)

    return agg_flags


def verificar_invariancia_por_accidente(df, cols=COLS_ACCIDENTE):
    """Comprueba que las columnas 'de accidente' (fecha, distrito, meteorología...)
    no varían dentro de un mismo Nº PARTE, antes de asumirlo y tomar el primer
    valor sin más. Devuelve un dict {columna: nº de accidentes inconsistentes}
    solo con las columnas que sí presentan alguna inconsistencia."""
    inconsistencias = {
        c: (df.groupby('Nº PARTE')[c].nunique() > 1).sum()
        for c in cols if c != 'Nº PARTE'
    }
    return {k: v for k, v in inconsistencias.items() if v > 0}


def construir_dataset_accidente(df, agg_flags, cols=COLS_ACCIDENTE):
    """Une las columnas invariantes por accidente (tomadas una única vez por
    Nº PARTE) con los flags agregados de persona, para producir el dataset
    final a nivel accidente."""
    df_accidente_info = df.drop_duplicates(subset='Nº PARTE')[cols]
    return df_accidente_info.merge(agg_flags, on='Nº PARTE', how='inner')


def convertir_binarias_si_no(df, cols):
    """Convierte columnas SI/NO a 1/0 explícito, verificando antes que no
    haya valores inesperados (para no convertir en silencio un error de
    los datos en un 0)."""
    df = df.copy()
    for c in cols:
        valores = set(df[c].unique())
        assert valores <= {'SI', 'NO'}, f'{c} tiene valores inesperados: {valores}'
        df[c] = (df[c] == 'SI').astype(int)
    return df


def extraer_hora_es_finde(df):
    """Deriva HORA (0-23, numérica) de RANGO HORARIO, y ES_FINDE de DIA SEMANA."""
    df = df.copy()
    df['HORA'] = df['RANGO HORARIO'].str.extract(r'DE\s+(\d{1,2}):')[0].astype(int)
    assert df['HORA'].between(0, 23).all(), 'HORA fuera de rango 0-23'
    df['ES_FINDE'] = df['DIA SEMANA'].isin(['SABADO', 'DOMINGO']).astype(int)
    return df


def detectar_cruce(df):
    """Añade ES_CRUCE: patrón 'CALLE 1 - CALLE 2' en LUGAR ACCIDENTE, que
    marca una intersección entre dos vías (~46% de los accidentes)."""
    df = df.copy()
    df['ES_CRUCE'] = df['LUGAR ACCIDENTE'].str.contains(' - ', na=False).astype(int)
    return df


def tipo_via(texto):
    """Clasifica LUGAR ACCIDENTE por tipo de vía, usando todo el texto (no
    solo el primer tramo antes de un posible '-') con un orden de prioridad
    de mayor a menor velocidad/relevancia esperada. Si el cruce mezcla dos
    tipos (p. ej. 'CALLE - AUTOVIA'), prevalece el de mayor prioridad — el
    orden de los tramos en el texto es alfabético en ~88% de los casos, no
    indica cuál vía es la "principal", así que no puede usarse como
    criterio."""
    if pd.isna(texto):
        return np.nan
    t = texto.upper()
    if 'AUTOVIA' in t or 'M-30' in t or 'M-40' in t:
        return 'AUTOVIA'
    if 'CARRETERA' in t:
        return 'CARRETERA'
    if 'RONDA' in t:
        return 'RONDA'
    if 'AVENIDA' in t:
        return 'AVENIDA'
    if 'PASEO' in t:
        return 'PASEO'
    if 'GLORIETA' in t:
        return 'GLORIETA'
    if 'CALLE' in t:
        return 'CALLE'
    if 'PLAZA' in t:
        return 'PLAZA'
    return 'OTROS'


def aplicar_tipo_via(df):
    """Añade TIPO_VIA a partir de LUGAR ACCIDENTE (ver `tipo_via`)."""
    df = df.copy()
    df['TIPO_VIA'] = df['LUGAR ACCIDENTE'].apply(tipo_via)
    return df


def imputar_categoricas_desconocido(df, cols=COLS_CATEGORICAS_MODELO):
    """Imputa nulos residuales en categóricas de modelo con la categoría
    explícita 'Desconocido' (no elimina filas) — coherente con que, en este
    dominio, el propio 'no asignado' resultó ser informativo en el EDA."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype('object').fillna('Desconocido')
    return df


def split_temporal(df, col_año='AÑO', año_corte=2017):
    """Split temporal: train = hasta año_corte (inclusive), test = posterior.
    Más honesto que un split aleatorio si el modelo se plantea como una
    herramienta entrenada con histórico para predecir accidentes futuros."""
    train = df[df[col_año] <= año_corte].copy()
    test = df[df[col_año] > año_corte].copy()
    return train, test


def split_aleatorio_estratificado(df, target='GRAVE', test_size=0.2, random_state=42):
    """Split aleatorio estratificado por `target` — referencia secundaria,
    con distribución de clases idéntica en train y test."""
    return train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
