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


def sugerir_recursos(probabilidad, inputs):
    """Sugerencia ORIENTATIVA de qué recursos reforzar, a partir de reglas de
    dominio sobre la probabilidad y el tipo de accidente -- NO es una salida
    del modelo: el dataset de entrenamiento no registra qué recursos se
    enviaron a cada accidente histórico, así que no hay con qué entrenar esa
    predicción. El 112 no aparece aquí como "recurso" porque es el canal de
    coordinación a través del cual se despachan los demás, no una alternativa
    a ellos -- se asume que ya se ha llamado.

    Policía Municipal se sugiere siempre (regulación de tráfico y atestado),
    con independencia de la gravedad. SAMUR se refuerza si la probabilidad de
    gravedad es alta o si hay un usuario vulnerable de la vía implicado
    (peatón, moto o bicicleta) -- ver Sección 6 de la memoria, donde estas
    son las variables con mayor peso en el modelo explicativo. Bomberos se
    apunta como posible necesidad de excarcelación solo en vuelco o colisión
    múltiple con probabilidad alta."""
    recursos = [{
        "recurso": "Policía Municipal",
        "motivo": "Regulación del tráfico y atestado del accidente",
    }]

    usuario_vulnerable = (
        inputs.get("tipo_accidente") == "ATROPELLO"
        or inputs.get("incluye_moto")
        or inputs.get("incluye_bici")
    )
    if probabilidad >= 0.5 or usuario_vulnerable:
        motivo = (
            "Probabilidad de gravedad alta"
            if probabilidad >= 0.5
            else "Usuario vulnerable de la vía implicado (peatón, moto o bicicleta)"
        )
        recursos.append({"recurso": "SAMUR — asistencia sanitaria reforzada", "motivo": motivo})

    if inputs.get("tipo_accidente") in ("VUELCO", "COLISIÓN MÚLTIPLE") and probabilidad >= 0.5:
        recursos.append({
            "recurso": "Bomberos — posible excarcelación",
            "motivo": f'{inputs["tipo_accidente"].capitalize()} con probabilidad de gravedad alta',
        })

    return recursos


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
        resultado["recursos_sugeridos"] = sugerir_recursos(
            resultado["probabilidad_grave"], resultado["inputs_interpretados"]
        )
        return jsonify({"ok": True, **resultado})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)

