"""
Aplicación de apoyo a la priorización de recursos de emergencia.

Uso: durante la llamada de un aviso de accidente, la persona que atiende va
rellenando el formulario con los datos disponibles en ese momento (sin
esperar a que lleguen los servicios de emergencia) y la app devuelve la
probabilidad de que el accidente resulte grave, calculada por el modelo
operativo entrenado en 04_Modelizacion.ipynb (Pregunta 1 del TFM).

Cómo ejecutarla (desde la raíz del repo):
    streamlit run scripts/streamlit_app.py

Requiere tener ya generados:
    models/pipeline_produccion.joblib   (guardado en 03_Feature_Engineering.ipynb)
    models/modelo_operativo.joblib      (guardado en 04_Modelizacion.ipynb)
"""
import datetime
import sys
from pathlib import Path

import joblib
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import cfg
from src.pipelines.pipeline_produccion import PipelineAccidentes

st.set_page_config(page_title="Triaje de accidentes — Madrid", page_icon="🚨", layout="centered")


@st.cache_resource
def cargar_artefactos():
    """Carga el pipeline y el modelo una sola vez (no en cada interacción del
    formulario) -- son los mismos artefactos ya guardados por los notebooks,
    nunca se reajustan aquí."""
    produccion = cfg.model['produccion']
    models_dir = cfg.ruta(cfg.paths['models_dir'])
    pipe = PipelineAccidentes.load(str(models_dir / produccion['pipeline']))
    modelo = joblib.load(str(models_dir / produccion['modelo_operativo']))
    return pipe, modelo


try:
    pipe, modelo = cargar_artefactos()
except FileNotFoundError as e:
    st.error(
        "No se encuentran los artefactos del modelo. Asegúrate de haber ejecutado "
        "`03_Feature_Engineering.ipynb` y `04_Modelizacion.ipynb` antes de lanzar esta app.\n\n"
        f"Detalle: {e}"
    )
    st.stop()

st.title("🚨 Triaje de accidentes de tráfico — Madrid")
st.caption(
    "Apoyo a la priorización de recursos de emergencia (SAMUR / Policía Municipal / "
    "Emergencias 112) con la información disponible en el momento del aviso."
)

# --- Listas de opciones, derivadas dinámicamente del propio pipeline entrenado ---
DISTRITOS = sorted(pipe.mapeos_encoding["TASA_GRAVEDAD_HIST_DISTRITO"][1].index.tolist())
TIPOS_VIA = sorted(pipe.cats_tipo_via + [pipe.ref_via])
TIPOS_ACCIDENTE = sorted(pipe.cats_tipo_acc + [pipe.ref_acc])
METEOROLOGIA = ["Seco", "Lluvia", "Nieve", "Niebla", "Granizo", "Hielo"]
ESTADO_FIRME = ["Seca Y Limpia", "Mojada", "Aceite", "Barro", "Grava Suelta", "Hielo"]

with st.form("formulario_aviso"):
    st.subheader("Datos del aviso")

    col1, col2 = st.columns(2)
    with col1:
        distrito = st.selectbox("Distrito", DISTRITOS)
        tipo_accidente = st.selectbox("Tipo de accidente", TIPOS_ACCIDENTE)
        tipo_via = st.selectbox("Tipo de vía", TIPOS_VIA)
    with col2:
        fecha = st.date_input("Fecha", value=datetime.date.today())
        hora = st.slider("Hora", 0, 23, datetime.datetime.now().hour)
        es_cruce = st.checkbox("Ocurre en un cruce")

    st.subheader("Condiciones")
    col3, col4 = st.columns(2)
    with col3:
        meteorologia = st.selectbox("Condición meteorológica", METEOROLOGIA)
    with col4:
        estado_firme = st.selectbox("Estado del firme", ESTADO_FIRME)

    st.subheader("Vehículos implicados (si se conoce)")
    col5, col6 = st.columns(2)
    with col5:
        incluye_moto = st.checkbox("Motocicleta o ciclomotor implicado")
    with col6:
        incluye_bici = st.checkbox("Bicicleta implicada")

    enviado = st.form_submit_button("Calcular probabilidad", use_container_width=True, type="primary")

if enviado:
    fecha_hora = datetime.datetime.combine(fecha, datetime.time(hour=hora))

    resultado = pipe.predecir_accidente_individual(
        modelo,
        distrito=distrito,
        fecha_hora=fecha_hora,
        tipo_accidente=tipo_accidente,
        tipo_via=tipo_via,
        es_cruce=es_cruce,
        meteorologia=meteorologia,
        estado_firme=estado_firme,
        incluye_moto=incluye_moto,
        incluye_bici=incluye_bici,
    )
    prob = resultado["probabilidad_grave"]

    st.divider()
    st.subheader("Resultado")

    if prob >= 0.5:
        st.error(f"### Probabilidad de gravedad: {prob*100:.1f}%")
        st.markdown("**Prioridad alta** — se recomienda reforzar el envío de recursos.")
    elif prob >= 0.25:
        st.warning(f"### Probabilidad de gravedad: {prob*100:.1f}%")
        st.markdown("**Prioridad media** — seguir el protocolo estándar, vigilar evolución.")
    else:
        st.success(f"### Probabilidad de gravedad: {prob*100:.1f}%")
        st.markdown("**Prioridad estándar** según la información disponible en el aviso.")

    st.caption(
        "Esta probabilidad es una señal de apoyo adicional, no un sustituto del juicio "
        "profesional del personal de emergencias. Modelo entrenado con datos históricos "
        "2012-2017 del Ayuntamiento de Madrid; ver limitaciones en la memoria del TFM."
    )

    with st.expander("Ver datos interpretados por el modelo"):
        st.json(resultado["inputs_interpretados"])
