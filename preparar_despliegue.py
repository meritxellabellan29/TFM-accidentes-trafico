"""
Sincroniza la carpeta webapp/ (autocontenida, para desplegar) con la fuente
de verdad real del proyecto: notebooks/utils/ y models/.

Nunca edites nada dentro de webapp/utils/ ni webapp/models/ a mano -- se
sobrescriben cada vez que se ejecuta este script. Si necesitas cambiar algo
del pipeline, edítalo en notebooks/utils/ (donde también lo usan tus
notebooks) y vuelve a ejecutar este script antes de desplegar.

Uso (desde la raíz del proyecto, TFM-accidentes-trafico/):
    venv\\Scripts\\python.exe preparar_despliegue.py
"""
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
ORIGEN_UTILS = RAIZ / "notebooks" / "utils"
ORIGEN_MODELS = RAIZ / "models"
DESTINO = RAIZ / "webapp"

ARCHIVOS_UTILS = [
    "limpieza.py",
    "preprocesado.py",
    "feature_engineering.py",
    "pipeline_produccion.py",
]
ARCHIVOS_MODELOS = [
    "pipeline_produccion.joblib",
    "modelo_operativo.joblib",
]


def sincronizar():
    destino_utils = DESTINO / "utils"
    destino_models = DESTINO / "models"
    destino_utils.mkdir(parents=True, exist_ok=True)
    destino_models.mkdir(parents=True, exist_ok=True)

    # __init__.py vacío, necesario para que "utils" sea un paquete importable
    (destino_utils / "__init__.py").touch()

    print("Sincronizando utils/ ...")
    for nombre in ARCHIVOS_UTILS:
        origen = ORIGEN_UTILS / nombre
        if not origen.exists():
            print(f"  ⚠️  No encontrado, se omite: {origen}")
            continue
        shutil.copy2(origen, destino_utils / nombre)
        print(f"  ✓ {nombre}")

    print("\nSincronizando models/ ...")
    for nombre in ARCHIVOS_MODELOS:
        origen = ORIGEN_MODELS / nombre
        if not origen.exists():
            print(f"  ⚠️  No encontrado, se omite: {origen} "
                  f"(¿ya ejecutaste 03_Feature_Engineering.ipynb y 04_Modelizacion.ipynb?)")
            continue
        shutil.copy2(origen, destino_models / nombre)
        print(f"  ✓ {nombre}")

    print(f"\n✅ webapp/ sincronizada con la versión actual de notebooks/utils/ y models/.")
    print("   Recuerda: no edites nada dentro de webapp/utils/ ni webapp/models/ a mano.")


if __name__ == "__main__":
    sincronizar()
