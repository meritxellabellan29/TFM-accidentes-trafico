"""
Backend de la app de triaje de accidentes de tráfico (Madrid).

Sirve la interfaz (index.html) y expone un endpoint JSON que envuelve
pipe.predecir_accidente_individual() -- no reimplementa ninguna lógica del
modelo, solo la conecta con el formulario web.

Uso en local (desde esta misma carpeta):
    python api.py
Después, abrir http://localhost:5000 en el navegador.

En despliegue (Render, Railway, etc.), el servidor de producción (gunicorn)
importa "app" directamente de este archivo -- no hace falta tocar nada.

Estructura esperada (repo completo desplegado, no solo esta carpeta):
    src/pipelines/pipeline_produccion.py (+ el resto de src/)
    webapp/api.py
    webapp/index.html
    webapp/models/pipeline_produccion.joblib
    webapp/models/modelo_operativo.joblib
"""
import datetime
import pathlib
import sys

import joblib
from flask import Flask, jsonify, request, send_from_directory

# Rutas SIEMPRE relativas a este archivo, no al directorio desde el que se
# ejecute -- imprescindible para que funcione igual en local y en el
# servidor de despliegue, sea cual sea su directorio de trabajo actual.
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# src/ vive en la raíz del repo, un nivel por encima de webapp/
sys.path.append(str(BASE_DIR.parent))
from src.pipelines.pipeline_produccion import PipelineAccidentes

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

print("Cargando pipeline y modelo...")
pipe = PipelineAccidentes.load(str(MODELS_DIR / "pipeline_produccion.joblib"))
modelo = joblib.load(str(MODELS_DIR / "modelo_operativo.joblib"))
print("Listo.")


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/api/opciones")
def opciones():
    """Devuelve las listas de opciones (distritos, tipos de vía/accidente),
    derivadas dinámicamente del propio pipeline entrenado -- si algún día
    se reentrena con más categorías, el formulario se actualiza solo."""
    return jsonify({
        "distritos": sorted(pipe.mapeos_encoding["TASA_GRAVEDAD_HIST_DISTRITO"][1].index.tolist()),
        "tipos_via": sorted(pipe.cats_tipo_via + [pipe.ref_via]),
        "tipos_accidente": sorted(pipe.cats_tipo_acc + [pipe.ref_acc]),
        "meteorologia": ["Seco", "Lluvia", "Nieve", "Niebla", "Granizo", "Hielo"],
        "estado_firme": ["Seca Y Limpia", "Mojada", "Aceite", "Barro", "Grava Suelta", "Hielo"],
    })


@app.route("/api/predecir", methods=["POST"])
def predecir():
    datos = request.get_json()
    try:
        fecha_hora = datetime.datetime.fromisoformat(datos["fecha_hora"])

        resultado = pipe.predecir_accidente_individual(
            modelo,
            distrito=datos["distrito"],
            fecha_hora=fecha_hora,
            tipo_accidente=datos["tipo_accidente"],
            tipo_via=datos["tipo_via"],
            es_cruce=datos.get("es_cruce", False),
            meteorologia=datos.get("meteorologia", "Seco"),
            estado_firme=datos.get("estado_firme", "Seca Y Limpia"),
            incluye_moto=datos.get("incluye_moto", False),
            incluye_bici=datos.get("incluye_bici", False),
        )
        return jsonify({"ok": True, **resultado})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)

