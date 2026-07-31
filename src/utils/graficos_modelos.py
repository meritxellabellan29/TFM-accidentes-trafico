"""
Funciones gráficas de evaluación de modelos para el TFM de accidentes de
tráfico en Madrid (curvas ROC, matrices de confusión, validación
walk-forward). Extraídas del módulo de entrenamiento
(`src.models.entrenamiento`) para separar entrenamiento/evaluación de
visualización.
"""
from sklearn.metrics import roc_auc_score


def graficar_validacion_temporal(tabla, filename=None):
    """Grafica la evolución del AUC-ROC de validación año a año en el
    walk-forward, junto con la tasa de gravedad de cada año — para ver de
    un vistazo si el rendimiento se mantiene estable o si empeora conforme
    se valida en años más recientes.

    El eje del AUC-ROC se fija en un rango con contexto (ancla en 0.5,
    el nivel de un clasificador aleatorio) en vez de autoajustarse al rango
    exacto de los datos — si no, una variación de apenas 0.02 puntos puede
    parecer visualmente enorme cuando en realidad es ruido menor."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(tabla['año_validacion'], tabla['AUC-ROC_val'], marker='o', color='#2a78d6', linewidth=2, label='AUC-ROC (validación)')
    ax2.plot(tabla['año_validacion'], tabla['tasa_grave_val_%'], marker='s', color='#eb6834', linewidth=2, linestyle='--', label='% grave (validación)')

    ax1.set_xlabel('Año de validación')
    ax1.set_ylabel('AUC-ROC', color='#2a78d6')
    ax2.set_ylabel('% de gravedad', color='#eb6834')
    ax1.set_title('Validación walk-forward: AUC-ROC y tasa de gravedad por año')
    ax1.grid(alpha=0.3)

    # Años como enteros, sin decimales
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Rango del AUC-ROC con contexto: ancla en 0.5 (azar) para no exagerar
    # visualmente una variación de apenas 0.02 puntos
    auc_max = tabla['AUC-ROC_val'].max()
    ax1.set_ylim(0.5, max(0.85, auc_max + 0.05))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def graficar_matrices_confusion(modelos, X_test, y_test, umbral=0.5, filename=None):
    """Igual que `graficar_matriz_confusion`, pero para varios modelos a la
    vez, en una rejilla de subplots -- para comparar de un vistazo cómo
    reparte cada modelo sus aciertos/errores entre las dos clases, en vez
    de generar un gráfico separado por modelo."""
    import math
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    n = len(modelos)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    etiquetas = ['No grave', 'Grave']
    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        y_pred = (modelo.predict_proba(X_test)[:, 1] >= umbral).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks([0, 1]); ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(etiquetas, fontsize=9)
        ax.set_xlabel('Predicción', fontsize=9)
        ax.set_ylabel('Real', fontsize=9)
        ax.set_title(nombre, fontsize=11)

        for i in range(2):
            for j in range(2):
                color = 'white' if cm_pct[i, j] > 50 else 'black'
                ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                        ha='center', va='center', color=color, fontsize=10)

    for ax in axes[len(modelos):]:
        ax.axis('off')

    fig.suptitle(f'Matrices de confusión (umbral={umbral})', fontsize=14)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def graficar_matriz_confusion(modelo, X_test, y_test, umbral=0.5, filename=None):
    """Matriz de confusión con recuentos absolutos y porcentaje por fila
    (normalizado sobre la clase real, no sobre el total) -- con un
    desbalanceo del ~9.6% de la clase positiva, normalizar sobre el total
    haría casi invisible el bloque de "grave", por eso se normaliza por fila:
    de cada clase real, qué % se predijo correctamente y qué % no.

    Nota sobre el umbral: 0.5 no es sagrado -- es el punto de corte que
    convierte una probabilidad en una decisión binaria, y cambiarlo mueve
    precision y recall en direcciones opuestas (más umbral -> más precision,
    menos recall; y viceversa). Se deja como parámetro explícito para que
    se pueda ajustar según el caso de uso real (p. ej. en un triaje, puede
    interesar más un umbral bajo que maximice recall, aunque baje precision,
    para no dejar pasar accidentes graves)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_pred = (modelo.predict_proba(X_test)[:, 1] >= umbral).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    etiquetas = ['No grave', 'Grave']
    ax.set_xticks([0, 1]); ax.set_xticklabels(etiquetas)
    ax.set_yticks([0, 1]); ax.set_yticklabels(etiquetas)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusión (umbral={umbral})')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                     ha='center', va='center', color=color, fontsize=12)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return cm


def graficar_curvas_roc(modelos, X_test, y_test, titulo='Curvas ROC — comparación de modelos', filename=None):
    """Superpone la curva ROC de varios modelos ya entrenados sobre el mismo
    test, con el AUC de cada uno en la leyenda, y la diagonal de referencia
    (clasificador aleatorio, AUC=0.5)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, modelo in modelos.items():
        y_proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{nombre} (AUC={auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Aleatorio (AUC=0.5)')
    ax.set_xlabel('Tasa de falsos positivos (1 - especificidad)')
    ax.set_ylabel('Tasa de verdaderos positivos (recall)')
    ax.set_title(titulo)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def graficar_curvas_roc_train_test(modelos, X_train, y_train, X_test, y_test, filename=None):
    """Curva ROC train vs. test de TODOS los modelos a la vez, en una
    rejilla de subplots (uno por modelo) — para comparar de un vistazo qué
    modelos sobreajustan (curva de train muy por encima de la de test) y
    cuáles generalizan bien (ambas curvas próximas entre sí), sin tener que
    generar un gráfico separado por modelo."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    import math

    n = len(modelos)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        aucs = {}
        for nombre_split, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
            y_proba = modelo.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            auc = roc_auc_score(y, y_proba)
            aucs[nombre_split] = auc
            ax.plot(fpr, tpr, label=f'{nombre_split} (AUC={auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1)

        # Entrada extra en la leyenda, solo con el gap -- sin línea visible
        # asociada (marcador transparente), para no confundirla con una curva
        gap = aucs['Train'] - aucs['Test']
        ax.plot([], [], ' ', label=f'Gap (train-test): {gap:.3f}')

        ax.set_title(nombre, fontsize=11)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

    # Ocultar subplots sobrantes si el nº de modelos no llena la rejilla
    for ax in axes[len(modelos):]:
        ax.axis('off')

    fig.suptitle('Curvas ROC train vs. test — todos los modelos', fontsize=14)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def graficar_curva_roc_train_vs_test(modelo, X_train, y_train, X_test, y_test,
                                      nombre_modelo='Modelo', filename=None):
    """Curva ROC del MISMO modelo en train y test, superpuestas — forma
    visual directa de ver overfitting: si la curva de train está muy por
    encima de la de test (mucho más AUC), el modelo memoriza el train en
    vez de generalizar (ver discusión de gap train-test)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre_split, X, y, estilo in [('Train', X_train, y_train, '-'), ('Test', X_test, y_test, '-')]:
        y_proba = modelo.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, estilo, label=f'{nombre_split} (AUC={auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Aleatorio (AUC=0.5)')
    ax.set_xlabel('Tasa de falsos positivos (1 - especificidad)')
    ax.set_ylabel('Tasa de verdaderos positivos (recall)')
    ax.set_title(f'Curva ROC train vs. test — {nombre_modelo}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    return fig
