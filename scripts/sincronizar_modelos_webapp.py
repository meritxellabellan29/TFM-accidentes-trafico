"""
Copia los artefactos entrenados (models/*.joblib) a webapp/models/, para que
el despliegue de la webapp (Render/Railway/etc., con "Root Directory:
webapp/") tenga los modelos disponibles sin depender de generarlos en el
servidor de producción.

webapp/ ya no tiene su propia copia de utils/ -- importa el código
directamente de src/ (ver webapp/api.py) -- así que lo único que hace falta
sincronizar antes de desplegar son los binarios .joblib.

Nunca edites a mano los .joblib dentro de webapp/models/ -- se sobrescriben
cada vez que se ejecuta este script. Si necesitas regenerarlos, vuelve a
ejecutar 03_Feature_Engineering.ipynb / 04_Modelizacion.ipynb (que guardan en
models/) y luego este script.

Uso (desde la raíz del proyecto):
    python scripts/sincronizar_modelos_webapp.py
"""
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import cfg


def sincronizar():
    produccion = cfg.model['produccion']
    origen = cfg.ruta(cfg.paths['models_dir'])
    destino = cfg.ruta('webapp') / 'models'
    destino.mkdir(parents=True, exist_ok=True)

    archivos = [produccion['pipeline'], produccion['modelo_operativo'], produccion['modelo_referencia']]

    print('Sincronizando modelos hacia webapp/models/ ...')
    for nombre in archivos:
        ruta_origen = origen / nombre
        if not ruta_origen.exists():
            print(f'  ⚠️  No encontrado, se omite: {ruta_origen}')
            continue
        shutil.copy2(ruta_origen, destino / nombre)
        print(f'  ✓ {nombre}')

    print('\n✅ webapp/models/ sincronizada con la versión actual de models/.')


if __name__ == '__main__':
    sincronizar()
