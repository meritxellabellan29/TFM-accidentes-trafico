# TFM — Predicción de Gravedad en Accidentes de Tráfico

**Autor:** Meritxell Abellan  
**Dataset:** Accidentes de tráfico en Madrid (Open Data Ayuntamiento de Madrid)  
**Pregunta de investigación:** ¿Qué características de un accidente de tráfico están asociadas a una mayor probabilidad de que el accidente tenga consecuencias graves?

---

## Descripción del proyecto

Este TFM aplica técnicas de Machine Learning para predecir la **gravedad de un accidente de tráfico** a partir de variables contextuales: condiciones meteorológicas, estado del firme, tipo de accidente, hora, día de la semana, distrito, tipo de vehículo y perfil de los implicados.

- **Unidad de análisis:** accidente (identificado por `Nº PARTE`)
- **Variable objetivo:** gravedad del accidente, derivada de la máxima lesividad registrada entre las personas implicadas
- **Clases objetivo (tentativas):**
  - 0 — Sin heridos (todos ilesos)
  - 1 — Heridos leves
  - 2 — Heridos graves o fallecidos

---

## Estructura del repositorio

```
TFM/
│
├── data/
│   ├── raw/              # Datos originales sin modificar
│   └── processed/        # Datos limpios y transformados
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Análisis exploratorio descriptivo
│   ├── 02_Preprocesado.ipynb         # Limpieza y transformaciones
│   ├── 03_FeatureEngineering.ipynb   # Construcción del dataset a nivel accidente
│   ├── 04_Modelos.ipynb              # Entrenamiento y comparación de modelos
│   └── 05_Interpretabilidad.ipynb    # SHAP, importancia de variables
│
├── src/
│   ├── limpieza.py        # Funciones de limpieza reutilizables
│   ├── features.py        # Construcción de variables
│   ├── modelos.py         # Entrenamiento y evaluación
│   └── utils.py           # Funciones auxiliares (visualizaciones, etc.)
│
├── figures/               # Gráficos exportados
│
├── models/                # Modelos entrenados guardados (.pkl)
│
├── requirements.txt       # Dependencias del proyecto
└── README.md
```

---

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Fases del proyecto

| Fase | Notebook | Estado |
|------|----------|--------|
| Análisis exploratorio (EDA) | `01_EDA.ipynb` | 🔲 Pendiente |
| Limpieza y preprocesado | `02_Preprocesado.ipynb` | 🔲 Pendiente |
| Feature engineering | `03_FeatureEngineering.ipynb` | 🔲 Pendiente |
| Modelización | `04_Modelos.ipynb` | 🔲 Pendiente |
| Interpretabilidad | `05_Interpretabilidad.ipynb` | 🔲 Pendiente |

---

## Fuente de datos

Datos abiertos del Ayuntamiento de Madrid.  
Licencia: [Licencia de uso de datos abiertos del Ayuntamiento de Madrid](https://datos.madrid.es/portal/site/egob/menuitem.3efdb29b813ad8241e830cc2a8a409a/?vgnextoid=108804d4aab90410VgnVCM100000171f5a0aRCRD&vgnextchannel=b4c412b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default)
