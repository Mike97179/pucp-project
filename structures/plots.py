"""
Gráficas de los experimentos.

matplotlib en modo Agg: escribe PNGs sin necesitar pantalla, que es lo que
WSL y Colab requieren.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import config

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#c98500', '#d55181']


def _save(fig, target, dpi=150):
    plt.tight_layout()
    fig.savefig(target, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Gráfica guardada: {target}')
    return target


def lr_vs_map(df, target):
    """Learning rate (escala log) contra mAP50 de detección y segmentación."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df['learning_rate'], df['mAP50_detection'],
            marker='o', linewidth=2, label='mAP50 detección')
    ax.plot(df['learning_rate'], df['mAP50_segmentation'],
            marker='s', linewidth=2, label='mAP50 segmentación')

    ax.set_xscale('log')
    ax.set_xlabel('Learning rate (lr0) — escala logarítmica')
    ax.set_ylabel('mAP50')
    ax.set_title('Learning Rate vs mAP50')
    ax.set_xticks(df['learning_rate'])
    ax.set_xticklabels([str(lr) for lr in df['learning_rate']])
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    return _save(fig, target)


def convergence(df, target, lr0):
    """Curva de mAP50 época por época."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['metrics/mAP50(B)'], linewidth=2,
            label='mAP50 detección')
    ax.plot(df['epoch'], df['metrics/mAP50(M)'], linewidth=2,
            label='mAP50 segmentación')
    ax.set_xlabel('Época')
    ax.set_ylabel('mAP50')
    ax.set_title(f'Convergencia: épocas vs mAP50 (lr0 = {lr0})')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    return _save(fig, target)


def map_per_class(df, target, title='mAP50 por clase'):
    """Barras de mAP50 por clase, de mayor a menor."""
    class_names = config.load_class_names()
    color_map = dict(zip(class_names, config.CLASS_COLORS_HEX))
    ordered = df.sort_values('mAP50', ascending=False)
    bar_colors = [color_map.get(c, 'steelblue') for c in ordered['class']]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(ordered['class'], ordered['mAP50'], color=bar_colors)
    ax.set_xlabel('Clase')
    ax.set_ylabel('mAP50 (segmentación)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')

    for bar, value in zip(bars, ordered['mAP50']):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                f'{value:.2f}', ha='center', fontsize=9)

    return _save(fig, target)


def _draw_matrix(ax, cm, labels, title, font=8):
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font)
    ax.set_yticklabels(labels, fontsize=font)
    ax.set_title(title, fontweight='bold')

    highest = cm.max() if cm.max() > 0 else 1
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = cm[row, col]
            if value > 0:
                ax.text(col, row, str(value), ha='center', va='center',
                        fontsize=font - 2,
                        color='white' if value > highest * 0.5 else 'black')
    return im


def confusion_matrix(cm, labels, target, title):
    """Una sola matriz de confusión con barra de color."""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = _draw_matrix(ax, cm, labels, title, font=9)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, target)


def combined_matrices(matrices, labels, target, cols=5):
    """Todas las matrices lado a lado, para comparar de un vistazo."""
    n = len(matrices)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5.5 * cols, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (name, cm) in enumerate(matrices.items()):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        _draw_matrix(ax, cm, labels, name, font=8)

    fig.savefig(target, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Gráfica guardada: {target}')
    return target


def benchmark_comparison(class_names, gt_totals, pred_per_model, target):
    """
    Dos paneles: conteo absoluto GT vs predicción, y diferencia por clase.
    Positivo = sobredetección, negativo = subdetección.
    """
    model_names = list(pred_per_model)
    x = range(len(class_names))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel superior: conteos absolutos
    ax = axes[0]
    n_bars = len(model_names) + 1
    width  = 0.8 / n_bars
    base   = [xi - width * (n_bars - 1) / 2 for xi in x]

    ax.bar(base, gt_totals, width, label='Ground truth', color='#73726c')
    for j, model_name in enumerate(model_names):
        pos = [b + width * (j + 1) for b in base]
        ax.bar(pos, pred_per_model[model_name], width, label=model_name,
               color=COLORS[j % len(COLORS)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Cantidad de detecciones')
    ax.set_title('Conteo por clase: ground truth vs predicciones')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Panel inferior: diferencia contra ground truth
    ax = axes[1]
    diff_width = 0.8 / max(1, len(model_names))
    diff_base  = [xi - diff_width * (len(model_names) - 1) / 2 for xi in x]

    for j, model_name in enumerate(model_names):
        diffs = [p - g for p, g in zip(pred_per_model[model_name], gt_totals)]
        pos   = [b + diff_width * j for b in diff_base]
        bars  = ax.bar(pos, diffs, diff_width, label=model_name,
                       color=COLORS[j % len(COLORS)])
        for bar, value in zip(bars, diffs):
            if value != 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        value + (2 if value > 0 else -6),
                        f'+{value}' if value > 0 else str(value),
                        ha='center', fontsize=7, color='#444')

    ax.axhline(y=0, color='#444', linewidth=1.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Diferencia (pred − GT)')
    ax.set_title('Diferencia vs ground truth '
                 '(positivo = sobredetección, negativo = subdetección)')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    return _save(fig, target)


def predict_summary(class_names, gt_totals, pred_per_model, target):
    """
    Single panel: grouped bars of ground truth vs each model's count per class.

    Each new model adds one bar per class. Simpler than benchmark_comparison
    (no diff panel) — used by the predict summary.
    """
    model_names = list(pred_per_model)
    x = range(len(class_names))

    n_bars = len(model_names) + 1
    width  = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 2.5), 7))

    base = [xi - width * (n_bars - 1) / 2 for xi in x]
    ax.bar(base, gt_totals, width, label='Ground truth', color='#73726c')

    for j, model_name in enumerate(model_names):
        pos = [b + width * (j + 1) for b in base]
        bars = ax.bar(pos, pred_per_model[model_name], width,
                      label=model_name, color=COLORS[j % len(COLORS)])
        for bar, value in zip(bars, pred_per_model[model_name]):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                        str(value), ha='center', fontsize=7, color='#444')

    # GT values on top of their bars too
    for bar, value in zip(ax.patches[:len(class_names)], gt_totals):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                    str(value), ha='center', fontsize=7, color='#444')

    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Cantidad de instancias')
    ax.set_title('Conteo por clase: ground truth vs predicciones')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)

    return _save(fig, target)


def analyze_comparison(class_names, counts_per_model, target, title):
    """
    Barras agrupadas por clase, una barra por modelo.

    `counts_per_model` es un dict {model_name: [count_cls0, count_cls1, ...]},
    ya ordenado por ranking del leaderboard.

    Solo se grafican las clases que tengan al menos una detección en algún
    modelo, para no llenar la gráfica de barras vacías.
    """
    model_names = list(counts_per_model)

    # Filtrar clases sin detecciones en ningún modelo
    active = []
    active_names = []
    for i, name in enumerate(class_names):
        total = sum(counts_per_model[m][i] for m in model_names)
        if total > 0:
            active.append(i)
            active_names.append(name)

    if not active:
        return None

    x = range(len(active_names))
    n_bars = len(model_names)
    width = min(0.8 / max(n_bars, 1), 0.25)

    fig, ax = plt.subplots(figsize=(max(10, len(active_names) * 2), 6))

    for j, model_name in enumerate(model_names):
        values = [counts_per_model[model_name][i] for i in active]
        pos = [xi + width * (j - n_bars / 2 + 0.5) for xi in x]
        bars = ax.bar(pos, values, width, label=model_name,
                      color=COLORS[j % len(COLORS)])

        # Valor encima de cada barra
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1,
                        str(value), ha='center', fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(active_names, rotation=45, ha='right')
    ax.set_ylabel('Cantidad de detecciones')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)

    return _save(fig, target)