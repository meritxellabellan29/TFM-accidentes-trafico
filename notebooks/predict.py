"""
Script de predicción para accidentes de tráfico nuevos.

Dado un Excel con el mismo formato que el dataset original (nivel persona,
con LESIVIDAD conocida — p. ej. una nueva publicación anual de datos),
aplica el pipeline completo (limpieza + preprocesado + feature engineering)
ya ajustado en 04_Modelizacion.ipynb, y produce la probabilidad de gravedad
predicha por accidente.

Uso desde la carpeta notebooks/ (donde vive utils/):

    python predict.py --input ruta/al/excel_nuevo.xlsx --output predicciones.csv

Opciones:
    --tipo {operativo,referencia}   Qué modelo usar (por defecto: operativo)
    --pipeline RUTA                 Ruta al pipeline guardado (por defecto: ../models/pipeline_produccion.joblib)
    --modelo RUTA                   Ruta al modelo guardado (por defecto: ../models/modelo_<tipo>.joblib)
"""
import argparse
import sys

import pandas as pd
import joblib

from utils.pipeline_produccion import PipelineAccidentes


def main():
    parser = argparse.ArgumentParser(
        description='Predice la probabilidad de gravedad de accidentes de tráfico nuevos.'
    )
    parser.add_argument('--input', required=True, help='Ruta al Excel nuevo (mismo formato que el original)')
    parser.add_argument('--output', default='predicciones.csv', help='Ruta del CSV de salida')
    parser.add_argument('--tipo', choices=['operativo', 'referencia'], default='operativo',
                         help='Qué modelo usar: operativo (Pregunta 1) o referencia (Pregunta 2)')
    parser.add_argument('--pipeline', default='../models/pipeline_produccion.joblib',
                         help='Ruta al pipeline ya ajustado (PipelineAccidentes.save())')
    parser.add_argument('--modelo', default=None,
                         help='Ruta al modelo; por defecto ../models/modelo_<tipo>.joblib')
    args = parser.parse_args()

    modelo_path = args.modelo or f'../models/modelo_{args.tipo}.joblib'

    print(f'Cargando pipeline desde {args.pipeline} ...')
    try:
        pipe = PipelineAccidentes.load(args.pipeline)
    except FileNotFoundError:
        print(f'ERROR: no se encuentra el pipeline en {args.pipeline}. '
              f'¿Se ha ejecutado pipe.save(...) en 04_Modelizacion.ipynb?')
        sys.exit(1)

    print(f'Cargando modelo desde {modelo_path} ...')
    try:
        modelo = joblib.load(modelo_path)
    except FileNotFoundError:
        print(f'ERROR: no se encuentra el modelo en {modelo_path}.')
        sys.exit(1)

    print(f'Leyendo datos nuevos desde {args.input} ...')
    df_nuevo = pd.read_excel(args.input)
    print(f'  {len(df_nuevo):,} registros de persona leídos.')

    print('Procesando (limpieza + preprocesado + feature engineering) y prediciendo ...')
    resultado = pipe.predecir(df_nuevo, modelo, tipo=args.tipo)

    resultado.to_csv(args.output, index=False)
    print(f'\n✅ Predicciones guardadas en {args.output} ({len(resultado):,} accidentes)')
    print(f'   Probabilidad media predicha: {resultado["probabilidad_grave"].mean()*100:.2f}%')
    if resultado['GRAVE'].notna().all():
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(resultado['GRAVE'], resultado['probabilidad_grave'])
        print(f'   AUC-ROC sobre estos datos (si GRAVE es conocido): {auc:.4f}')


if __name__ == '__main__':
    main()
