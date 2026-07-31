"""
Funciones de limpieza estructural compartidas entre el EDA (01_EDA.ipynb) y el
preprocessing (02_Preprocessing.ipynb) del proyecto.

Cada paso de limpieza vive en su propia función de responsabilidad única;
`limpiar_database()` las encadena en el orden correcto. Esto permite:
  - testear/depurar cada paso por separado si algo falla,
  - reutilizar un paso concreto de forma aislada si hiciera falta,
  - leer de un vistazo qué hace cada cosa, sin un único bloque largo.
"""
import pandas as pd

VALORES_NULOS = {'NO ASIGNADO', 'NO ASIGNADA', 'DESCONOCIDO', 'DESCONOCIDA'}


def limpiar_texto(df):
    """Aplica strip() a todas las columnas de texto."""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    return df


def corregir_encoding_tramo_edad(df):
    """Corrige la inconsistencia de encoding detectada en Tramo Edad
    ('ANOS' sin tilde conviviendo con 'AÑOS' en el resto de tramos)."""
    df = df.copy()
    df['Tramo Edad'] = df['Tramo Edad'].str.replace('ANOS', 'AÑOS', regex=False)
    return df


def procesar_fecha(df):
    """Convierte FECHA a datetime y extrae AÑO y MES como columnas separadas."""
    df = df.copy()
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['AÑO'] = df['FECHA'].dt.year
    df['MES'] = df['FECHA'].dt.month
    return df


def excluir_testigos(df, verbose=True):
    """Excluye las filas de TIPO PERSONA == 'TESTIGO': un testigo no es un
    implicado con lesividad propia, no aporta información para el target."""
    n_antes = len(df)
    n_testigos = (df['TIPO PERSONA'] == 'TESTIGO').sum()
    df = df[df['TIPO PERSONA'] != 'TESTIGO'].copy()
    if verbose:
        print(f'Registros originales:       {n_antes:>8,}')
        print(f'Testigos excluidos:          {n_testigos:>8,}')
        print(f'Tras excluir testigos:       {len(df):>8,}')
    return df


def limpiar_database(df, verbose=True):
    """
    Aplica la limpieza estructural mínima común sobre un DataFrame ya cargado,
    encadenando los pasos individuales en el orden correcto:
      1. limpiar_texto            -> strip() de columnas de texto
      2. corregir_encoding_tramo_edad -> ANOS -> AÑOS
      3. procesar_fecha           -> FECHA a datetime, AÑO y MES
      4. excluir_testigos         -> quita TESTIGO

    No elimina duplicados ni convierte "NO ASIGNADO"/"DESCONOCIDO" en nulos:
    esos pasos se aplican por separado con `eliminar_duplicados()` y
    `normalizar_nulos()`, para que cada notebook decida explícitamente si
    los necesita.
    """
    df = limpiar_texto(df)
    df = corregir_encoding_tramo_edad(df)
    df = procesar_fecha(df)
    df = excluir_testigos(df, verbose=verbose)
    return df


def eliminar_duplicados(df, verbose=True):
    """Elimina duplicados exactos (todas las columnas). Debe aplicarse
    DESPUÉS de `limpiar_database` (el nº de duplicados depende de haber
    excluido ya los testigos)."""
    n_antes = len(df)
    n_duplicados = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    if verbose:
        print(f'Duplicados exactos eliminados: {n_duplicados:>8,} ({n_duplicados/n_antes*100:.2f}%)')
        print(f'Filas tras eliminar duplicados: {len(df):>8,}')
    return df


def normalizar_nulos(df, verbose=True):
    """Sustituye los valores de texto que representan 'sin dato' por NaN reales."""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace(VALORES_NULOS, pd.NA)
    if verbose:
        n_nulos = df.isna().sum().sum()
        print(f'Nulos tras normalizar valores "sin dato": {n_nulos:,}')
    return df
