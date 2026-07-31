"""
Funciones de graficado reutilizables para el EDA de accidentes de tráfico.

Dos patrones se repetían de forma casi idéntica en distintas celdas del
notebook, cambiando solo la variable de agrupación:
  - `graficar_lesividad_categoria`: distribución de LESIVIDAD por una
    categórica (absolutos + % apilado), sin evolución de tasa. Usado para
    variables sin orden temporal (sexo, tipo de persona, tipo de vehículo...).
  - `graficar_lesividad_y_tasa_grave`: barras apiladas de LESIVIDAD + línea
    de tasa de gravedad en eje secundario. Usado para variables con un orden
    natural (año, mes, día de la semana, hora, tramo de edad...).

Extraer esta lógica evita que un ajuste de estilo (p. ej. el tamaño de
fuente de las anotaciones) tenga que repetirse a mano en cada celda que
usa el mismo tipo de gráfico.
"""
import matplotlib.pyplot as plt
import pandas as pd


def graficar_lesividad_categoria(df, columna, orden_lesividad, colores_lesividad, labels_lesividad,
                                  titulo_base='', xlabel='', filename=None,
                                  horizontal=False, rotation=0, figsize=(18, 5),
                                  ordenar_por_total=False):
    """
    Distribución de LESIVIDAD por una variable categórica: absolutos (izq.)
    y porcentaje apilado (dcha.), sin línea de tasa de gravedad.

    Sustituye el patrón repetido en las celdas de "LESIVIDAD por sexo",
    "LESIVIDAD por tipo de persona", "LESIVIDAD por tipo de vehículo", etc.

    Si `ordenar_por_total=True`, las categorías se ordenan por nº total de
    registros (de mayor a menor). En horizontal, esto coloca la categoría
    con más registros arriba del todo (matplotlib dibuja barh de abajo
    hacia arriba, así que se ordena ascendente internamente); en vertical,
    la categoría con más registros queda a la izquierda.
    """
    df_ = df.copy()
    df_['LESIVIDAD'] = df_['LESIVIDAD'].fillna('Sin dato')

    tabla = (df_.groupby([columna, 'LESIVIDAD']).size()
                .unstack(fill_value=0)
                .reindex(columns=orden_lesividad, fill_value=0))

    if ordenar_por_total:
        orden = tabla.sum(axis=1).sort_values(ascending=horizontal).index
        tabla = tabla.reindex(orden)

    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    colores = [colores_lesividad[c] for c in orden_lesividad]
    labels = list(labels_lesividad)
    kind = 'barh' if horizontal else 'bar'

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    tabla.plot(kind=kind, ax=axes[0], color=colores, edgecolor='white')
    axes[0].set_title(f'{titulo_base} (absolutos)', fontsize=14)
    axes[0].set_xlabel(xlabel if not horizontal else 'Nº de registros')
    axes[0].set_ylabel('Nº de registros' if not horizontal else xlabel)
    axes[0].tick_params(axis='x', rotation=rotation)
    axes[0].legend(title='Lesividad', labels=labels)

    tabla_pct.plot(kind=kind, stacked=True, ax=axes[1], color=colores, edgecolor='white')
    axes[1].set_title(f'{titulo_base} (%)', fontsize=14)
    if horizontal:
        axes[1].set_xlim(0, 100); axes[1].axvline(50, color='grey', linestyle='--', alpha=0.5)
    else:
        axes[1].set_ylim(0, 100); axes[1].axhline(50, color='grey', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=rotation)
    axes[1].legend(title='Lesividad', labels=labels, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    return tabla, tabla_pct


def graficar_lesividad_y_tasa_grave(df_sin_na, columna, orden_lesividad, colores_lesividad,
                                     orden_categorias=None, titulo='', xlabel='',
                                     filename=None, rotation=0, figsize=(12, 5)):
    """
    Barras apiladas de LESIVIDAD por categoría (eje izquierdo) + línea de
    tasa de gravedad (% HG+MT) en eje secundario, con anotaciones.

    Sustituye el patrón repetido en las celdas de evolución por año, mes,
    día de la semana, hora y tramo de edad.
    """
    tabla = (df_sin_na.groupby([columna, 'LESIVIDAD']).size()
                       .unstack(fill_value=0)
                       .reindex(columns=orden_lesividad, fill_value=0))
    tasa_grave = (df_sin_na.groupby(columna)['GRAVE']
                           .agg(['mean', 'count'])
                           .rename(columns={'mean': 'pct_grave', 'count': 'n'})
                           .assign(pct_grave=lambda x: x['pct_grave'] * 100))

    if orden_categorias is not None:
        tabla = tabla.reindex(orden_categorias)
        tasa_grave = tasa_grave.reindex(orden_categorias)

    colores = [colores_lesividad[c] for c in orden_lesividad]

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    tabla.plot(kind='bar', stacked=True, ax=ax1, color=colores, edgecolor='white', width=0.7)

    idx = range(len(tasa_grave))
    ax2.plot(idx, tasa_grave['pct_grave'], color='#e74c3c', marker='o',
             linewidth=2, markersize=6, label='% HG+MT', zorder=5)
    for i, (_, row) in enumerate(tasa_grave.iterrows()):
        ax2.annotate(f"{row['pct_grave']:.1f}%", (i, row['pct_grave']),
                     textcoords='offset points', xytext=(0, 8),
                     ha='center', fontsize=7, color='#e74c3c', fontweight='bold')

    ax1.set_title(titulo, fontsize=13)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Nº de registros')
    ax1.tick_params(axis='x', rotation=rotation, labelsize=8)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    ax2.set_ylim(0, tasa_grave['pct_grave'].max() * 2)
    ax2.yaxis.set_visible(False)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, title='Lesividad',
               bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, title_fontsize=9)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    return tabla, tasa_grave


def graficar_top_n_por_lesividad_grave(tabla_abs, tabla_pct, categorias_graves, orden_lesividad,
                                        colores_lesividad, labels_lesividad, n=15,
                                        titulo_abs='', titulo_pct='', suptitle=None,
                                        xlabel_abs='Nº de registros', filename=None, figsize=(22, 7)):
    """
    Selecciona las N categorías (filas de `tabla_abs`) con más registros en
    `categorias_graves` (p. ej. ['HG', 'MT']) y las grafica en horizontal,
    ordenadas de mayor a menor de arriba a abajo.

    A diferencia de `graficar_lesividad_categoria`, parte de tablas YA
    calculadas (no de un DataFrame crudo) y ordena/filtra por un subconjunto
    de columnas en vez de por el total — pensada para variables de muy alta
    cardinalidad (como LUGAR ACCIDENTE) donde antes de graficar hace falta
    seleccionar un top N por un criterio concreto de gravedad.
    """
    top_n = tabla_abs[categorias_graves].sum(axis=1).sort_values(ascending=False).head(n)
    orden = top_n.index.tolist()[::-1]  # invertido: barh dibuja de abajo hacia arriba

    tabla_top = tabla_abs.reindex(orden)
    tabla_pct_top = tabla_pct.reindex(orden)

    colores = [colores_lesividad[c] for c in orden_lesividad]
    labels = list(labels_lesividad)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    tabla_top.plot(kind='barh', stacked=True, ax=axes[0], color=colores, width=0.8, edgecolor='white')
    axes[0].set_title(titulo_abs, fontsize=13)
    axes[0].set_xlabel(xlabel_abs); axes[0].set_ylabel('')
    axes[0].legend(title='Lesividad', labels=labels, bbox_to_anchor=(1.01, 1), loc='upper left')

    tabla_pct_top.plot(kind='barh', stacked=True, ax=axes[1], color=colores, width=0.8, edgecolor='white')
    axes[1].set_title(titulo_pct, fontsize=13)
    axes[1].set_xlabel('%'); axes[1].set_xlim(0, 100); axes[1].set_ylabel('')
    axes[1].axvline(50, color='grey', linestyle='--', alpha=0.5, label='_nolegend_')
    axes[1].legend(title='Lesividad', labels=labels, bbox_to_anchor=(1.01, 1), loc='upper left')

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    return tabla_top, tabla_pct_top


def graficar_condicion_binaria(df_accidentes, df_sin_na, prefijo, color_barras, color_tasa,
                                titulo_frecuencia, titulo_tasa, filename=None,
                                min_n=100, figsize=(16, 5)):
    """
    Analiza un grupo de columnas binarias SI/NO con un prefijo común
    (p. ej. 'CPFA' para meteorología, 'CPSV' para estado del firme):
    frecuencia de cada condición + tasa de gravedad asociada.

    - Las frecuencias se calculan sobre `df_accidentes` (nivel accidente,
      ya que la condición meteorológica/de firme es una única por accidente).
    - La tasa de gravedad se calcula sobre `df_sin_na` (nivel persona, con
      la columna GRAVE), consistente con el resto de tasas de gravedad del
      EDA. Categorías con menos de `min_n` personas se excluyen del gráfico
      de tasas (poco fiables con muestra pequeña).
    """
    cols = [c for c in df_accidentes.columns if c.startswith(prefijo)]

    df_bin = df_accidentes[cols].apply(lambda x: x.str.strip()).replace({'SI': 1, 'NO': 0})
    counts = df_bin.sum().sort_values(ascending=False)

    tasas_graves = {}
    for col in cols:
        nombre = col.replace(f'{prefijo} ', '')
        mask = df_sin_na[col].str.strip() == 'SI'
        if mask.sum() > min_n:
            tasas_graves[nombre] = df_sin_na.loc[mask, 'GRAVE'].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    counts.plot(kind='bar', ax=axes[0], color=color_barras)
    axes[0].set_title(titulo_frecuencia, fontsize=13)
    axes[0].set_xticklabels([c.replace(f'{prefijo} ', '') for c in counts.index], rotation=30)
    axes[0].set_ylabel('Nº de accidentes')

    pd.Series(tasas_graves).sort_values(ascending=False).plot(kind='bar', ax=axes[1], color=color_tasa)
    axes[1].set_title(titulo_tasa, fontsize=13)
    axes[1].set_ylabel('% grave o fallecido')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()

    tabla = pd.DataFrame({
        'Nº accidentes': counts,
        'Porcentaje (%)': (counts / len(df_accidentes) * 100).round(2),
        '% graves': pd.Series({f'{prefijo} {k}': v for k, v in tasas_graves.items()}).round(2)
    })
    tabla.index = [i.replace(f'{prefijo} ', '') for i in tabla.index]
    print(tabla.to_string())

    return tabla
