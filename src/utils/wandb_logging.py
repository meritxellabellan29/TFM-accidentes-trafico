"""
Envío de los resultados de los grids de hiperparámetros del **modelo
operativo** (Sección 1 de `04_Modelizacion.ipynb`, Pregunta 1) a Weights &
Biases, para poder compararlos en un panel de Parallel Coordinates en vez de
leer las tablas ordenadas a ojo. El modelo explicativo (Sección 2, Pregunta
2) no usa grid de hiperparámetros ni pasa por este módulo -- se entrena con
la configuración por defecto de cada algoritmo (`entrenar_modelos_baseline`).

No sustituye a `explorar_hiperparametros`/`analisis_estabilidad_bagging`
(src/models/entrenamiento.py) -- esas funciones siguen siendo la única
fuente de verdad de los resultados; este módulo solo toma la tabla que ya
devuelven y registra cada fila como un run.
"""
import os

import wandb

# Evita que cada wandb.init()/run.finish() imprima en el notebook los bloques
# "Tracking run...", "Syncing run...", "Run history/summary", "Synced X W&B
# file(s)..." -- con un run por fila de la tabla, eso satura la salida.
os.environ.setdefault("WANDB_SILENT", "true")


def registrar_resultados_wandb(tabla, project, modelo, columnas_metricas):
    """Registra cada fila de una tabla de resultados (una fila = una
    combinación de hiperparámetros ya evaluada) como un run de W&B.

    Las columnas que no están en `columnas_metricas` se registran como
    `config` (hiperparámetros); las de `columnas_metricas`, como el
    resultado (`wandb.log`). `modelo` se añade también a la config de cada
    run para poder comparar entre familias de modelos dentro del mismo
    `project` -- así el panel de parallel coordinates puede mostrar los
    runs de todos los modelos a la vez, con los hiperparámetros que no
    aplican a un modelo concreto quedando en blanco en su línea.

    `tabla` es el resultado de `explorar_hiperparametros` o
    `analisis_estabilidad_bagging`; `project` es el nombre del proyecto en
    W&B (el mismo para los distintos modelos que se quieran comparar
    juntos); `modelo` es el nombre de la familia de modelo de esta tabla
    (p. ej. 'RandomForest'), para distinguir los runs en el panel;
    `columnas_metricas` son las columnas de `tabla` que son resultado (AUC,
    gap...), no hiperparámetro.

    Si `WANDB_MODE=disabled` (p. ej. en un entorno sin acceso a red o sin
    cuenta configurada), no se intenta ni un solo `wandb.init()`: esta
    función es un no-op. El panel local equivalente (`graficar_parallel_coordinates`,
    en `graficos_modelos.py`) no depende de W&B y sigue generándose igual."""
    if os.environ.get('WANDB_MODE') == 'disabled':
        return
    columnas_config = [c for c in tabla.columns if c not in columnas_metricas]
    for _, fila in tabla.iterrows():
        run = wandb.init(project=project, group=modelo,
                          config={'modelo': modelo, **fila[columnas_config].to_dict()},
                          reinit=True)
        wandb.log(fila[columnas_metricas].to_dict())
        run.finish()
