"""
Pipeline de productivización end-to-end para el TFM de accidentes de
tráfico en Madrid. Envuelve limpieza + preprocesado + feature engineering
en un único objeto con `fit`/`transform`, para poder procesar un Excel
nuevo (mismo formato que el original) sin repetir manualmente cada
notebook.

Importante: los objetos "aprendidos" en train (mapeo de target encoding de
DISTRITO, categorías de TIPO_VIA/TIPO ACCIDENTE vistas en train, columna de
referencia del one-hot) se guardan explícitamente en `fit()` y se
reutilizan tal cual en `transform()` — nunca se recalculan sobre datos
nuevos, para no filtrar información del Excel nuevo hacia el modelo ya
entrenado.

Este pipeline cubre el escenario de "Excel nuevo con el mismo formato
histórico" (p. ej. una nueva publicación anual de datos, con LESIVIDAD ya
conocida por persona) — útil para revalidar el modelo sobre datos futuros.
NO es el mismo escenario que predecir un accidente individual en el
momento del aviso (donde LESIVIDAD no se conoce todavía, por definición):
para eso está `predecir_accidente_individual()` en este mismo módulo.
"""
import numpy as np
import pandas as pd
import joblib

from utils import limpieza
from utils import preprocesado as prep
from utils import feature_engineering as fe


class PipelineAccidentes:
    """Envuelve todo el proceso de limpieza + preprocesado + feature
    engineering, con los mapeos aprendidos en `fit()` guardados como
    atributos del objeto, para poder serializarlo con `save()`/`load()` y
    aplicarlo después a datos nuevos con `transform()`/`predecir()`."""

    def __init__(self, m_target_encoding=50):
        self.m = m_target_encoding
        self.ajustado = False

    def _limpiar_y_preprocesar(self, df_raw):
        """Pasos de limpieza + preprocesado hasta nivel accidente, comunes
        a fit() y transform() (no dependen de nada aprendido en train)."""
        df = limpieza.limpiar_database(df_raw.copy(), verbose=False)
        df = limpieza.eliminar_duplicados(df, verbose=False)
        df = limpieza.normalizar_nulos(df, verbose=False)

        df_pers, stats = prep.filtrar_lesividad_conocida(df)
        if stats['accidentes_perdidos'] > 0:
            print(f"⚠️  Aviso: {stats['accidentes_perdidos']} accidentes se han perdido "
                  f"por completo (todas las personas con LESIVIDAD desconocida).")

        agg_flags = prep.agregar_flags_persona(df, df_pers)
        df_final = prep.construir_dataset_accidente(df, agg_flags)

        # N_VICTIMAS_LOG: log1p sobre Nº VICTIMAS * (ya presente en COLS_ACCIDENTE),
        # igual que se calculaba en 02_Preprocesado.ipynb -- se centraliza aquí para
        # que fit() y transform() sean la única fuente de verdad de esta columna
        df_final['N_VICTIMAS_LOG'] = np.log1p(df_final['Nº VICTIMAS *'])

        cols_binarias = [c for c in prep.COLS_ACCIDENTE if c.startswith('CPFA') or c.startswith('CPSV')]
        df_final = prep.convertir_binarias_si_no(df_final, cols_binarias)
        df_final = prep.extraer_hora_es_finde(df_final)
        df_final = prep.detectar_cruce(df_final)
        df_final = prep.aplicar_tipo_via(df_final)
        df_final = prep.imputar_categoricas_desconocido(df_final)
        return df_final, cols_binarias

    def fit(self, df_raw):
        """Ajusta el pipeline sobre el dataset de entrenamiento original:
        aprende y guarda las categorías de TIPO_VIA/TIPO ACCIDENTE, la
        categoría de referencia de cada una, y el mapeo de target encoding
        de DISTRITO. Devuelve el dataframe a nivel accidente ya con todas
        las features, por si se quiere inspeccionar."""
        df_final, cols_binarias = self._limpiar_y_preprocesar(df_raw)

        self.ref_via = df_final['TIPO_VIA'].value_counts().idxmax()
        self.ref_acc = df_final['TIPO ACCIDENTE'].value_counts().idxmax()
        self.cats_tipo_via = [c for c in sorted(df_final['TIPO_VIA'].unique()) if c != self.ref_via]
        self.cats_tipo_acc = [c for c in sorted(df_final['TIPO ACCIDENTE'].unique()) if c != self.ref_acc]

        especificaciones = [
            ('DISTRITO', 'GRAVE', 'TASA_GRAVEDAD_HIST_DISTRITO'),
            ('DISTRITO', 'INCLUYE_PEATON', 'PCT_HIST_PEATON_DISTRITO'),
            ('DISTRITO', 'INCLUYE_MOTO', 'PCT_HIST_MOTO_DISTRITO'),
        ]
        self.mapeos_encoding = {}
        for col, target, nuevo in especificaciones:
            # Valor OOF (out-of-fold) para las features de ESTE dataframe de
            # entrenamiento -- evita que cada fila contribuya a su propio
            # valor (ver smoothed_target_encode_oof en feature_engineering.py)
            df_final[nuevo] = fe.smoothed_target_encode_oof(df_final, col, target, self.m)
            # Mapeo completo (con TODO el histórico disponible en fit) para
            # aplicar tal cual a datos futuros en transform() -- ahí no hace
            # falta OOF, porque los datos futuros nunca participaron en el ajuste
            mapping, global_mean = fe.smoothed_target_encode(df_final, col, target, self.m)
            self.mapeos_encoding[nuevo] = (col, mapping, global_mean)

        for nombre, periodo in [('HORA', 24), ('MES', 12)]:
            df_final = fe.codificar_ciclica(df_final, nombre, periodo)

        ohe_via = fe.one_hot_fit_train(df_final, 'TIPO_VIA', self.cats_tipo_via, 'VIA')
        ohe_acc = fe.one_hot_fit_train(df_final, 'TIPO ACCIDENTE', self.cats_tipo_acc, 'ACC')
        df_final[ohe_via.columns.tolist()] = ohe_via
        df_final[ohe_acc.columns.tolist()] = ohe_acc

        cpfa_a = [c for c in cols_binarias if c.startswith('CPFA') and c != 'CPFA Seco']
        cpsv_a = [c for c in cols_binarias if c.startswith('CPSV') and c != 'CPSV Seca Y Limpia']
        cols_base = (['ES_CRUCE', 'ES_FINDE', 'HORA_SIN', 'HORA_COS', 'MES_SIN', 'MES_COS'] + cpfa_a + cpsv_a
                     + ['TASA_GRAVEDAD_HIST_DISTRITO', 'PCT_HIST_PEATON_DISTRITO', 'PCT_HIST_MOTO_DISTRITO']
                     + ['INCLUYE_MOTO', 'INCLUYE_BICI'])
        self.features_operativo = cols_base + [f'VIA_{c}' for c in self.cats_tipo_via] + [f'ACC_{c}' for c in self.cats_tipo_acc]
        self.features_referencia = self.features_operativo + [
            'INCLUYE_PEATON',        # redundante en parte con TIPO ACCIDENTE=ATROPELLO, pero se deja para el modelo de referencia
            'INCLUYE_EDAD_RIESGO',
            'N_PERSONAS_IMPLICADAS',
            'N_VICTIMAS_LOG',
        ]

        self.ajustado = True
        return df_final

    def transform(self, df_raw_nuevo):
        """Aplica a un Excel NUEVO (mismo formato) todos los pasos de
        limpieza/preprocesado/feature engineering, usando los mapeos ya
        aprendidos en fit() -- nunca recalculados sobre el dato nuevo."""
        assert self.ajustado, "El pipeline no está ajustado: llama a .fit(df_train) o .load(path) primero."
        df_final, _ = self._limpiar_y_preprocesar(df_raw_nuevo)

        for nuevo, (col, mapping, global_mean) in self.mapeos_encoding.items():
            df_final[nuevo] = df_final[col].map(mapping).fillna(global_mean)

        n_categorias_nuevas_via = (~df_final['TIPO_VIA'].isin(self.cats_tipo_via + [self.ref_via])).sum()
        n_categorias_nuevas_acc = (~df_final['TIPO ACCIDENTE'].isin(self.cats_tipo_acc + [self.ref_acc])).sum()
        if n_categorias_nuevas_via or n_categorias_nuevas_acc:
            print(f"⚠️  Aviso: {n_categorias_nuevas_via} accidentes con TIPO_VIA no vista en train, "
                  f"{n_categorias_nuevas_acc} con TIPO ACCIDENTE no vista -- se codifican como la categoría de referencia.")

        for nombre, periodo in [('HORA', 24), ('MES', 12)]:
            df_final = fe.codificar_ciclica(df_final, nombre, periodo)

        ohe_via = fe.one_hot_fit_train(df_final, 'TIPO_VIA', self.cats_tipo_via, 'VIA')
        ohe_acc = fe.one_hot_fit_train(df_final, 'TIPO ACCIDENTE', self.cats_tipo_acc, 'ACC')
        df_final[ohe_via.columns.tolist()] = ohe_via
        df_final[ohe_acc.columns.tolist()] = ohe_acc

        return df_final

    def predecir(self, df_raw_nuevo, modelo, tipo='operativo'):
        """Aplica transform() y el modelo ya entrenado, devolviendo un
        dataframe con Nº PARTE, la probabilidad predicha, y GRAVE real (si
        está disponible en el Excel nuevo, útil para validar el modelo
        sobre un nuevo lote histórico)."""
        df_final = self.transform(df_raw_nuevo)
        cols = self.features_operativo if tipo == 'operativo' else self.features_referencia
        X = df_final[cols]
        proba = modelo.predict_proba(X)[:, 1]

        resultado = df_final[['Nº PARTE', 'GRAVE']].copy()
        resultado['probabilidad_grave'] = proba
        return resultado

    def predecir_accidente_individual(self, modelo, distrito, fecha_hora, tipo_accidente,
                                       tipo_via=None, lugar_texto=None, es_cruce=None,
                                       meteorologia='Seco', estado_firme='Seca Y Limpia',
                                       incluye_moto=False, incluye_bici=False):
        """Predice la probabilidad de gravedad de UN accidente a partir de
        los datos disponibles en el momento del aviso — sin depender de
        `LESIVIDAD` ni de ningún dato de las personas implicadas (ese es
        precisamente el punto: es el caso de uso real de la Pregunta 1).

        Parámetros
        ----------
        distrito : str
            Uno de los 21 distritos de Madrid (mismo texto que en el
            dataset original, p. ej. 'CENTRO', 'SALAMANCA'...).
        fecha_hora : datetime-like (str admitido, p. ej. '2024-03-15 14:30')
            Fecha y hora del aviso. Se usan para derivar HORA, MES y si es
            fin de semana.
        tipo_accidente : str
            Tipo de accidente tal como lo describiría el aviso (p. ej.
            'ATROPELLO', 'COLISIÓN DOBLE', 'CAÍDA MOTOCICLETA'...). Si no
            se reconoce (no vista en train), se trata como la categoría de
            referencia con un aviso por consola.
        tipo_via : str, opcional
            Categoría de vía ('CALLE', 'AVENIDA', 'AUTOVIA'...). Si no se
            indica, se puede derivar de `lugar_texto` con el clasificador
            de texto libre ya usado en el resto del proyecto.
        lugar_texto : str, opcional
            Descripción libre del lugar (p. ej. 'CALLE DE ALCALA - GRAN
            VIA'), como alternativa a pasar `tipo_via`/`es_cruce` ya
            calculados. Si se da, tiene prioridad sobre `tipo_via`.
        es_cruce : bool, opcional
            Si el accidente ocurre en un cruce. Si no se indica y hay
            `lugar_texto`, se detecta automáticamente (patrón ' - ').
            Si no hay ninguno de los dos, se asume False.
        meteorologia : str
            Una de: 'Seco', 'Lluvia', 'Nieve', 'Niebla', 'Granizo', 'Hielo'.
        estado_firme : str
            Uno de: 'Seca Y Limpia', 'Mojada', 'Aceite', 'Barro',
            'Grava Suelta', 'Hielo'.
        incluye_moto, incluye_bici : bool
            Si se sabe que hay una motocicleta/ciclomotor o una bicicleta
            implicada. No hace falta indicar si hay un peatón implicado:
            ya se deduce de `tipo_accidente='ATROPELLO'`.

        Devuelve
        --------
        dict con la probabilidad predicha y un resumen de los inputs
        usados, para poder revisar que se ha interpretado todo bien.
        """
        assert self.ajustado, "El pipeline no está ajustado: llama a .fit(df_train) o .load(path) primero."

        fecha_hora = pd.to_datetime(fecha_hora)
        hora, mes = fecha_hora.hour, fecha_hora.month
        es_finde = int(fecha_hora.dayofweek >= 5)

        if lugar_texto is not None:
            tipo_via_final = prep.tipo_via(lugar_texto)
            es_cruce_final = int(' - ' in lugar_texto.upper())
        else:
            tipo_via_final = tipo_via
            es_cruce_final = int(bool(es_cruce)) if es_cruce is not None else 0

        if tipo_via_final not in self.cats_tipo_via + [self.ref_via]:
            print(f"⚠️  TIPO_VIA '{tipo_via_final}' no vista en train -- se trata como referencia ({self.ref_via}).")
        if tipo_accidente not in self.cats_tipo_acc + [self.ref_acc]:
            print(f"⚠️  TIPO ACCIDENTE '{tipo_accidente}' no visto en train -- se trata como referencia ({self.ref_acc}).")

        fila = {}
        fila['ES_CRUCE'] = es_cruce_final
        fila['ES_FINDE'] = es_finde
        fila['HORA_SIN'] = np.sin(2 * np.pi * hora / 24)
        fila['HORA_COS'] = np.cos(2 * np.pi * hora / 24)
        fila['MES_SIN'] = np.sin(2 * np.pi * mes / 12)
        fila['MES_COS'] = np.cos(2 * np.pi * mes / 12)

        for col in self.features_operativo:
            if col.startswith('CPFA '):
                fila[col] = int(col == f'CPFA {meteorologia}')
            elif col.startswith('CPSV '):
                fila[col] = int(col == f'CPSV {estado_firme}')
            elif col.startswith('VIA_'):
                fila[col] = int(col == f'VIA_{tipo_via_final}')
            elif col.startswith('ACC_'):
                fila[col] = int(col == f'ACC_{tipo_accidente}')

        for nuevo, (col, mapping, global_mean) in self.mapeos_encoding.items():
            fila[nuevo] = mapping.get(distrito, global_mean)
        if distrito not in self.mapeos_encoding['TASA_GRAVEDAD_HIST_DISTRITO'][1].index:
            print(f"⚠️  DISTRITO '{distrito}' no visto en train -- se usa la media global.")

        fila['INCLUYE_MOTO'] = int(incluye_moto)
        fila['INCLUYE_BICI'] = int(incluye_bici)

        X_individual = pd.DataFrame([fila])[self.features_operativo]
        probabilidad = modelo.predict_proba(X_individual)[0, 1]

        return {
            'probabilidad_grave': round(float(probabilidad), 4),
            'inputs_interpretados': {
                'distrito': distrito, 'hora': hora, 'mes': mes, 'es_finde': bool(es_finde),
                'tipo_accidente': tipo_accidente, 'tipo_via': tipo_via_final, 'es_cruce': bool(es_cruce_final),
                'meteorologia': meteorologia, 'estado_firme': estado_firme,
                'incluye_moto': bool(incluye_moto), 'incluye_bici': bool(incluye_bici),
            },
        }

    def save(self, path):
        joblib.dump(self, path)
        print(f'Pipeline guardado en {path}')

    @staticmethod
    def load(path):
        return joblib.load(path)
