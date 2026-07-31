"""
Carga la configuración del proyecto (config/*.yaml) y expone rutas absolutas
resueltas respecto a la raíz del repo, para que notebooks, scripts/ y webapp/
usen siempre los mismos valores sin importar desde qué directorio se ejecuten.
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def _cargar_yaml(nombre):
    with open(CONFIG_DIR / nombre, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    def __init__(self):
        self.paths = _cargar_yaml("paths.yaml")
        self.data = _cargar_yaml("data.yaml")
        self.model = _cargar_yaml("model.yaml")

    def ruta(self, relativa):
        """Convierte una ruta relativa a la raíz del repo (p. ej. la que
        viene de config/paths.yaml) en una ruta absoluta."""
        return REPO_ROOT / relativa


cfg = Config()
