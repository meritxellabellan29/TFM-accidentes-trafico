"""
Genera el informe de prevención vial dirigido a las autoridades (Sección de
Interpretabilidad, Pregunta 2), en HTML y PDF.

A diferencia de la memoria -- pensada para un tribunal técnico, con las
cifras del modelo explicativo (SHAP, log-odds) y sus matices de
calibración ya explicados-- este informe va dirigido a un público no
técnico y prioriza cifras que se sostienen sin ninguna nota al pie: las
tasas de gravedad reales observadas en los datos (2012-2018), agrupadas
por factor. La única sección que sí usa el modelo explicativo es la de
distritos, porque ahí el propio hallazgo -- qué distritos concentran más
riesgo del que cabría esperar por su tipo de vía y meteorología-- solo
existe una vez que se controla por esas otras variables; una tasa bruta
por distrito no lo mostraría (de hecho apunta en la dirección contraria,
como se explica en el propio informe).

Uso (desde la raíz del repo):
    python scripts/generar_informe_prevencion.py

Genera reports/informe_prevencion_vial.html y, si Microsoft Edge está
disponible en el sistema, también reports/informe_prevencion_vial.pdf
(mismo mecanismo de impresión sin cabecera/pie que las capturas de la
webapp -- ver Sección 7 de la memoria).
"""
import base64
import io
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import cfg
from src.pipelines.pipeline_produccion import PipelineAccidentes
from src.utils.interpretabilidad import tabla_shap_por_distrito

REPORTS_DIR = Path(__file__).resolve().parent.parent / 'reports'

COLOR_NAVY = '#1a2b4a'
COLOR_RED = '#cb181d'
COLOR_GRIS = '#5a6472'
COLOR_FONDO_CARD = '#f4f6f9'


def _fig_a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def calcular_hallazgos():
    """Calcula, a partir de los datos y modelos ya entrenados, todas las
    cifras que usa el informe -- ninguna se copia a mano."""
    df = pd.read_csv(cfg.ruta(cfg.paths['data']['processed_dir']) / 'accidentes_accidente_nivel_clean.csv')
    tasa_global = df['GRAVE'].mean() * 100

    factores_humanos = {}
    for col, etiqueta in [
        ('INCLUYE_PEATON', 'Peatón implicado'),
        ('INCLUYE_MOTO', 'Motocicleta o ciclomotor implicado'),
        ('INCLUYE_EDAD_RIESGO', 'Menor de 18 o mayor de 65 años implicado'),
    ]:
        con = df.loc[df[col] == 1, 'GRAVE'].mean() * 100
        sin = df.loc[df[col] == 0, 'GRAVE'].mean() * 100
        factores_humanos[col] = {
            'etiqueta': etiqueta,
            'pct_con': con,
            'pct_sin': sin,
            'ratio': con / sin,
            'n_con': int((df[col] == 1).sum()),
        }

    def banda_horaria(h):
        if 0 <= h <= 5:
            return 'Madrugada (00-06h)'
        if 6 <= h <= 11:
            return 'Mañana (06-12h)'
        if 12 <= h <= 17:
            return 'Tarde (12-18h)'
        return 'Noche (18-24h)'

    df['BANDA'] = df['HORA'].apply(banda_horaria)
    bandas = (df.groupby('BANDA')['GRAVE'].mean() * 100).reindex(
        ['Madrugada (00-06h)', 'Mañana (06-12h)', 'Tarde (12-18h)', 'Noche (18-24h)']
    )

    n_total_accidentes = len(df)

    top_distritos_graves = (
        df.groupby('DISTRITO')['GRAVE'].agg(n_graves='sum', pct='mean').assign(pct=lambda t: t['pct'] * 100)
        .sort_values('n_graves', ascending=False).head(3)
    )
    top_distritos_graves['n_graves'] = top_distritos_graves['n_graves'].astype(int)

    # -- Modelo explicativo: riesgo residual por distrito (SHAP) --------------
    path_features = cfg.ruta(cfg.paths['data']['features_dir'])
    X_train_ref = pd.read_csv(path_features / 'X_train_referencia.csv')
    X_test_ref = pd.read_csv(path_features / 'X_test_referencia.csv')
    modelo_ref = joblib.load(cfg.ruta(cfg.paths['models_dir']) / cfg.model['produccion']['modelo_referencia'])
    pipe = PipelineAccidentes.load(str(cfg.ruta(cfg.paths['models_dir']) / cfg.model['produccion']['pipeline']))

    n_muestra = cfg.model['shap']['n_muestra']
    muestra = X_test_ref.sample(n=min(n_muestra, len(X_test_ref)), random_state=cfg.data['random_state'])
    explainer = shap.Explainer(modelo_ref, X_train_ref)
    shap_values = explainer(muestra, check_additivity=False)

    tabla_distritos = tabla_shap_por_distrito(shap_values, muestra, pipe)
    top_riesgo_residual = tabla_distritos.head(3)

    distritos_coincidentes = sorted(
        set(top_distritos_graves.index) & set(top_riesgo_residual['DISTRITO'])
    )

    return {
        'tasa_global': tasa_global,
        'n_total_accidentes': n_total_accidentes,
        'factores_humanos': factores_humanos,
        'bandas': bandas,
        'top_distritos_graves': top_distritos_graves,
        'top_riesgo_residual': top_riesgo_residual,
        'distritos_coincidentes': distritos_coincidentes,
    }


def grafico_factores_humanos(factores_humanos, tasa_global):
    etiquetas = [f['etiqueta'] for f in factores_humanos.values()]
    valores_con = [f['pct_con'] for f in factores_humanos.values()]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y_pos = np.arange(len(etiquetas))
    ax.set_ylim(-0.8, len(etiquetas) - 0.4)
    ax.barh(y_pos, valores_con, color=COLOR_RED, height=0.5, zorder=3)
    ax.axvline(tasa_global, color=COLOR_GRIS, linestyle='--', linewidth=1.3, zorder=2)
    ax.text(tasa_global + 0.3, -0.7, f'Tasa global: {tasa_global:.1f}\u2009%',
            fontsize=9, color=COLOR_GRIS, va='top')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(etiquetas, fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(valores_con):
        ax.text(v + 0.3, i, f'{v:.1f}\u2009%', va='center', fontsize=10, fontweight='bold', color=COLOR_NAVY)
    ax.set_xlabel('% de accidentes graves o mortales cuando el factor está presente', fontsize=9, color=COLOR_GRIS)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, max(valores_con) * 1.3)
    fig.tight_layout()
    return _fig_a_base64(fig)


def grafico_bandas_horarias(bandas):
    colores = [COLOR_RED if 'Madrugada' in b else COLOR_GRIS for b in bandas.index]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x_pos = np.arange(len(bandas))
    ax.bar(x_pos, bandas.values, color=colores, width=0.55, zorder=3)
    for i, v in enumerate(bandas.values):
        ax.text(i, v + 0.15, f'{v:.1f}\u2009%', ha='center', fontsize=10, fontweight='bold', color=COLOR_NAVY)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bandas.index, fontsize=9)
    ax.set_ylabel('% de accidentes graves o mortales', fontsize=9, color=COLOR_GRIS)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(bandas.values) * 1.25)
    fig.tight_layout()
    return _fig_a_base64(fig)


def render_html(h):
    fh = h['factores_humanos']
    peaton, moto, edad = fh['INCLUYE_PEATON'], fh['INCLUYE_MOTO'], fh['INCLUYE_EDAD_RIESGO']
    madrugada_pct = h['bandas']['Madrugada (00-06h)']
    madrugada_ratio = madrugada_pct / h['tasa_global']

    img_factores = grafico_factores_humanos(fh, h['tasa_global'])
    img_bandas = grafico_bandas_horarias(h['bandas'])

    filas_graves = ''.join(
        f'<tr><td>{r.Index.title()}</td><td>{r.n_graves:,}</td><td>{r.pct:.1f}\u2009%</td></tr>'
        for r in h['top_distritos_graves'].itertuples()
    )
    filas_residual = ''.join(
        f'<tr><td>{r.DISTRITO.title()}</td><td>{r.tasa_hist * 100:.1f}\u2009%</td>'
        f'<td class="positivo">por encima de lo esperado</td></tr>'
        for r in h['top_riesgo_residual'].itertuples()
    )

    nombres_graves = [d.title() for d in h['top_distritos_graves'].index]
    nombres_residual = [d.title() for d in h['top_riesgo_residual']['DISTRITO']]
    coincidentes = [d.title() for d in h['distritos_coincidentes']]
    if coincidentes:
        texto_coincidencia = (
            f"Adem\u00e1s, {' y '.join(coincidentes)} figura{'n' if len(coincidentes) > 1 else ''} en ambas "
            "listas a la vez: no solo concentra" + ('n' if len(coincidentes) > 1 else '') + " m\u00e1s "
            "accidentes graves en cifra absoluta por su volumen de tr\u00e1fico, sino que, por accidente, "
            "el riesgo de que sea grave es tambi\u00e9n mayor de lo que explicar\u00edan por s\u00ed solas su tipo de "
            "v\u00eda y sus condiciones."
        )
    else:
        texto_coincidencia = ''

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Informe de prevención vial</title>
<style>
  @page {{ margin: 16mm 14mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Arial, Helvetica, sans-serif;
    color: #24303f;
    margin: 0;
    font-size: 13px;
    line-height: 1.5;
  }}
  header {{
    background: {COLOR_NAVY};
    color: white;
    padding: 22px 28px;
    border-radius: 10px;
    margin-bottom: 22px;
  }}
  header h1 {{ margin: 0 0 6px; font-size: 21px; }}
  header p {{ margin: 0; font-size: 12px; color: #c7d0de; }}
  h2 {{
    font-size: 15px;
    color: {COLOR_NAVY};
    border-bottom: 2px solid {COLOR_NAVY};
    padding-bottom: 4px;
    margin: 26px 0 12px;
  }}
  .cards {{ display: flex; gap: 12px; margin-bottom: 6px; }}
  .card {{
    flex: 1;
    background: {COLOR_FONDO_CARD};
    border-left: 4px solid {COLOR_RED};
    border-radius: 6px;
    padding: 12px 14px;
  }}
  .card .cifra {{ font-size: 22px; font-weight: bold; color: {COLOR_RED}; }}
  .card .desc {{ font-size: 11px; color: #4a5568; margin-top: 3px; }}
  .grid-2 {{ display: flex; gap: 20px; align-items: flex-start; }}
  .grid-2 > div {{ flex: 1; }}
  img {{ max-width: 100%; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
  th {{ text-align: left; color: {COLOR_GRIS}; font-weight: 600; border-bottom: 1px solid #ccd3dc; padding: 5px 6px; }}
  td {{ padding: 5px 6px; border-bottom: 1px solid #e7eaf0; }}
  td.positivo {{ color: {COLOR_RED}; font-weight: 600; }}
  ul.recomendaciones li {{ margin-bottom: 7px; }}
  .nota {{ background: {COLOR_FONDO_CARD}; border-radius: 6px; padding: 12px 16px; font-size: 10.5px; color: #56606e; margin-top: 26px; }}
  .caption {{ font-size: 10px; color: #7a8494; margin-top: 4px; }}
</style>
</head>
<body>

<header>
  <h1>Informe de apoyo a la prevención vial &mdash; Madrid</h1>
  <p>Basado en el análisis de interpretabilidad del modelo explicativo de gravedad de accidentes de tráfico &middot;
     Datos del Ayuntamiento de Madrid, periodo 2012&ndash;2018</p>
</header>

<h2>Resumen: dónde se concentra el riesgo</h2>
<div class="cards">
  <div class="card">
    <div class="cifra">{peaton['ratio']:.1f}&times;</div>
    <div class="desc">más riesgo de gravedad cuando hay un <b>peatón</b> implicado
    ({peaton['pct_con']:.1f}&thinsp;% frente a {peaton['pct_sin']:.1f}&thinsp;% sin peatón)</div>
  </div>
  <div class="card">
    <div class="cifra">{edad['ratio']:.1f}&times;</div>
    <div class="desc">más riesgo cuando hay un <b>menor de 18 o mayor de 65 años</b> implicado
    ({edad['pct_con']:.1f}&thinsp;% frente a {edad['pct_sin']:.1f}&thinsp;%)</div>
  </div>
  <div class="card">
    <div class="cifra">{moto['ratio']:.1f}&times;</div>
    <div class="desc">más riesgo cuando hay una <b>motocicleta o ciclomotor</b> implicado
    ({moto['pct_con']:.1f}&thinsp;% frente a {moto['pct_sin']:.1f}&thinsp;%)</div>
  </div>
  <div class="card">
    <div class="cifra">{madrugada_ratio:.1f}&times;</div>
    <div class="desc">más riesgo en la <b>franja de madrugada</b> (00&ndash;06h) que la tasa media
    ({madrugada_pct:.1f}&thinsp;% frente a {h['tasa_global']:.1f}&thinsp;% global)</div>
  </div>
</div>
<p class="caption">Porcentajes calculados directamente sobre los {h['n_total_accidentes']:,} accidentes
  históricos del periodo 2012&ndash;2018, comparando los que tienen y no tienen cada factor &mdash;no son estimaciones del modelo.</p>

<h2>Quién está en mayor riesgo</h2>
<div class="grid-2">
  <div>
    <img src="data:image/png;base64,{img_factores}">
    <p class="caption">Tasa de accidentes graves o mortales, con y sin cada factor implicado. La línea discontinua marca la tasa media global.</p>
  </div>
  <div>
    <p>Los tres factores humanos analizados &mdash;presencia de un peatón, de una motocicleta o de una persona en edad de riesgo&mdash;
    elevan de forma clara la probabilidad de que un accidente sea grave, muy por encima de la tasa media.</p>
    <p>El análisis de interpretabilidad del modelo explicativo (valores SHAP), que aísla el efecto de cada variable
    controlando simultáneamente por el resto &mdash;distrito, tipo de vía, meteorología, tipo de accidente&mdash;,
    confirma de forma independiente que la presencia de una motocicleta o de un peatón se encuentran entre los
    factores individuales de mayor peso en la predicción de gravedad, lo que descarta que la asociación observada
    se deba únicamente a otras variables de contexto.</p>
    <p><b>Recomendación:</b> reforzar protección de motociclistas y peatones (visibilidad, pasos regulados,
    campañas específicas) y prestar atención diferenciada a colectivos de menores y mayores de 65 años
    (rutas escolares seguras, adaptación de tiempos de cruce).</p>
  </div>
</div>

<h2>Cuándo ocurre el mayor riesgo</h2>
<div class="grid-2">
  <div>
    <img src="data:image/png;base64,{img_bandas}">
    <p class="caption">Tasa de accidentes graves o mortales por franja horaria.</p>
  </div>
  <div>
    <p>La franja de madrugada (00&ndash;06h) concentra, con diferencia, la mayor tasa de gravedad, pese a tener
    menos volumen de accidentes que el resto del día &mdash;es decir, cuando ocurre un accidente de madrugada,
    tiene más probabilidad de ser grave que en cualquier otra franja.</p>
    <p><b>Recomendación:</b> reforzar controles de velocidad y alcoholemia en horario nocturno/madrugada,
    especialmente en fines de semana.</p>
  </div>
</div>

<h2>Dónde priorizar: dos lecturas complementarias</h2>
<div class="grid-2">
  <div>
    <p><b>Mayor volumen de accidentes graves (cifra absoluta)</b></p>
    <table>
      <tr><th>Distrito</th><th>Accidentes graves</th><th>% graves</th></tr>
      {filas_graves}
    </table>
    <p class="caption">Útil para dimensionar recursos de emergencia ya desplegados: son los distritos con más
    tráfico y, por tanto, más accidentes graves en número absoluto.</p>
  </div>
  <div>
    <p><b>Mayor riesgo residual (según el modelo, una vez descontado tipo de vía, meteorología y tipo de accidente)</b></p>
    <table>
      <tr><th>Distrito</th><th>% graves (bruto)</th><th>Riesgo ajustado</th></tr>
      {filas_residual}
    </table>
    <p class="caption">El modelo indica que, cuando ocurre un accidente en estos distritos, el riesgo de que sea
    grave es mayor de lo que su tipo de vía y condiciones explicarían por sí solas &mdash;una señal de que otros
    factores no recogidos en los datos (velocidad real de la vía, diseño de las intersecciones) podrían estar
    influyendo, y merecen una auditoría específica. {texto_coincidencia}</p>
  </div>
</div>

<h2>Recomendaciones</h2>
<ul class="recomendaciones">
  <li><b>Peatones y motociclistas:</b> intensificar campañas de seguridad y revisión de infraestructura
  (pasos de peatones, carriles/protección para motos) en los puntos de mayor tránsito de estos colectivos.</li>
  <li><b>Menores y mayores de 65 años:</b> auditar rutas escolares y entornos con alta presencia de mayores
  (tiempos de semáforo, iluminación, señalización).</li>
  <li><b>Franja de madrugada:</b> reforzar controles de velocidad y alcoholemia entre las 00h y las 06h,
  con especial atención a fines de semana.</li>
  <li><b>Distritos con más accidentes graves en cifra absoluta</b> ({', '.join(nombres_graves)}):
  mantener y reforzar los recursos de emergencia ya desplegados.</li>
  <li><b>Distritos con riesgo residual elevado</b> ({', '.join(nombres_residual)}): auditoría específica de
  infraestructura viaria, más allá de lo que explican el tipo de vía y la meteorología.</li>
  <li><b>Nuevos medios de movilidad</b> (patinetes eléctricos y similares): incorporar su registro explícito
  en los partes de accidente, ya que su presencia era marginal en el periodo analizado (2012&ndash;2018) y
  hoy es ya relevante para la seguridad vial de la ciudad.</li>
</ul>

<div class="nota">
  <b>Nota metodológica.</b> Las cifras de las secciones &laquo;Quién&raquo; y &laquo;Cuándo&raquo; son tasas de
  gravedad observadas directamente sobre el histórico de accidentes 2012&ndash;2018 (Ayuntamiento de Madrid), no
  estimaciones del modelo. La sección de distritos combina esa misma cifra bruta con el resultado del modelo
  explicativo (valores SHAP sobre un Gradient Boosting, validado de forma cruzada con una regresión logística),
  que sí es necesario para aislar el riesgo específico de cada distrito del resto de variables. Este informe es
  una señal de apoyo a la decisión y no sustituye el criterio experto de los servicios de tráfico y prevención
  vial. Generado automáticamente a partir del pipeline reproducible del proyecto
  (<code>scripts/generar_informe_prevencion.py</code>).
</div>

</body></html>"""


def main():
    print('Calculando hallazgos a partir de los datos y modelos entrenados...')
    hallazgos = calcular_hallazgos()

    REPORTS_DIR.mkdir(exist_ok=True)
    html_path = REPORTS_DIR / 'informe_prevencion_vial.html'
    html_path.write_text(render_html(hallazgos), encoding='utf-8')
    print(f'HTML generado en {html_path}')

    edge = shutil.which('msedge') or next(
        (p for p in [
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        ] if Path(p).exists()),
        None,
    )
    if not edge:
        print('Microsoft Edge no encontrado: se deja solo el HTML (ábrelo y usa "Imprimir > Guardar como PDF").')
        return

    pdf_path = REPORTS_DIR / 'informe_prevencion_vial.pdf'
    perfil_temporal = REPORTS_DIR / '.edge_profile_temp'
    subprocess.run([
        edge, '--headless', '--disable-gpu', '--no-sandbox',
        f'--user-data-dir={perfil_temporal}',
        f'--print-to-pdf={pdf_path}', '--no-pdf-header-footer',
        html_path.resolve().as_uri(),
    ], check=True, capture_output=True)
    shutil.rmtree(perfil_temporal, ignore_errors=True)
    print(f'PDF generado en {pdf_path}')


if __name__ == '__main__':
    main()
