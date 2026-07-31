"""
Script de predicción para accidentes de tráfico nuevos.

Dado un Excel con el mismo formato que el dataset original (nivel persona,
con LESIVIDAD conocida — p. ej. una nueva publicación anual de datos),
aplica el pipeline completo (limpieza + preprocesado + feature engineering)
ya ajustado en 04_Modelizacion.ipynb, y produce la probabilidad de gravedad
predicha por accidente.

Uso (desde cualquier directorio, las rutas por defecto se resuelven
respecto a la raíz del repo vía config/paths.yaml):

    python scripts/predict.py --input ruta/al/excel_nuevo.xlsx --output predicciones.csv

Opciones:
    --tipo {operativo,referencia}   Qué modelo usar (por defecto: el de config/model.yaml)
    --pipeline RUTA                 Ruta al pipeline guardado (por defecto: models/pipeline_produccion.joblib)
    --modelo RUTA                   Ruta al modelo guardado (por defecto: models/modelo_<tipo>.joblib)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import cfg
from src.pipelines.pipeline_produccion import PipelineAccidentes


def main():
    produccion = cfg.model['produccion']

    parser = argparse.ArgumentParser(
        description='Predice la probabilidad de gravedad de accidentes de tráfico nuevos.'
    )
    parser.add_argument('--input', required=True, help='Ruta al Excel nuevo (mismo formato que el original)')
    parser.add_argument('--output', default='predicciones.csv', help='Ruta del CSV de salida')
    parser.add_argument('--tipo', choices=['operativo', 'referencia'], default=produccion['tipo_defecto'],
                         help='Qué modelo usar: operativo (Pregunta 1) o referencia (Pregunta 2)')
    parser.add_argument('--pipeline', default=None,
                         help='Ruta al pipeline ya ajustado (PipelineAccidentes.save())')
    parser.add_argument('--modelo', default=None,
                         help='Ruta al modelo; por defecto models/modelo_<tipo>.joblib')
    args = parser.parse_args()

    pipeline_path = args.pipeline or cfg.ruta(cfg.paths['models_dir']) / produccion['pipeline']
    modelo_path = args.modelo or cfg.ruta(cfg.paths['models_dir']) / produccion[f'modelo_{args.tipo}']

    print(f'Cargando pipeline desde {pipeline_path} ...')
    try:
        pipe = PipelineAccidentes.load(str(pipeline_path))
    except FileNotFoundError:
        print(f'ERROR: no se encuentra el pipeline en {pipeline_path}. '
              f'¿Se ha ejecutado pipe.save(...) en 04_Modelizacion.ipynb?')
        sys.exit(1)

    print(f'Cargando modelo desde {modelo_path} ...')
    try:
        modelo = joblib.load(str(modelo_path))
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
