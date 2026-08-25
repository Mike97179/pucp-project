"""
`ranking` command — the leaderboard as it stands right now.

Read-only view of models/models.csv: no model is loaded and nothing is
evaluated, so it is instant. The order shown is the one that matters,
mask_mAP50 descending, and the best model is marked with a star.

It also reports the state of the dataset: if it grew since the metrics were
measured, the ranking is comparing figures from different test sets and says
so.
"""

import os

import pandas as pd

from .. import config, models


RANKING_COLUMNS = ['rank', 'name', 'architecture', 'imgsz', 'data',
                   'box_mAP50', 'mask_mAP50', 'notes']

BEST_MARK = '★'


def _print_table(df):
    """Print the registry ordered by mask_mAP50, starring the best model."""
    table = df[[c for c in RANKING_COLUMNS if c in df.columns]].copy()
    # Las celdas vacías del CSV llegan como NaN: en una tabla se leen peor que
    # un hueco.
    table = table.fillna('')
    table.insert(0, ' ', [BEST_MARK] + [''] * (len(table) - 1))
    print(table.to_string(index=False))


def _stored_order():
    """
    Model names in the order the CSV's own rank column claims.

    `registry()` already normalises the rank, so this is the only way left to
    notice that the file on disk disagrees with the real ranking.
    """
    if not os.path.isfile(config.MODELS_CSV):
        return None

    df = pd.read_csv(config.MODELS_CSV)
    if 'rank' not in df.columns:
        return None

    return list(df.sort_values('rank')['name'])


def _print_dataset_state():
    """Images in the dataset and whether the metrics are stale."""
    current = models.dataset_fingerprint()
    stored  = models.saved_fingerprint()

    print(f'\n  Dataset: {current} imágenes en {config.IMG_DIR}')

    if models.needs_reevaluation():
        print(f'  RE-EVALUACIÓN PENDIENTE: había {stored} imágenes cuando se '
              f'midieron estas métricas.')
        print('  Los mAP50 de arriba salen de un test set distinto al actual, '
              'así que')
        print('  el orden no es fiable. El próximo `entrenar` re-evalúa todo '
              'antes de rankear.')
    elif stored is None:
        print('  Sin marca del dataset todavía: se escribe la primera vez que '
              'se registre')
        print('  o se re-evalúe un modelo.')
    else:
        print('  Métricas al día: el dataset no ha cambiado desde que se '
              'midieron.')


def run(args):
    # registry() ya lanza FileNotFoundError con un mensaje útil si no hay CSV;
    # cli.main lo imprime como ERROR.
    df = models.registry()

    print('\n' + '=' * 78)
    print(f' RANKING DE MODELOS — top {models.MAX_MODELS} por mask_mAP50')
    print('=' * 78)

    if df.empty:
        print('\n  No hay ningún modelo registrado todavía.')
        print('  Entrena uno con `python pucp_segmentation.py train`.')
        return

    # El orden manda sobre la columna rank: si el CSV quedó con un rank viejo,
    # la tabla sigue saliendo bien ordenada.
    order = df['mask_mAP50'].astype(float)
    df    = df.assign(_order=order).sort_values(
        '_order', ascending=False).reset_index(drop=True)

    print()
    _print_table(df)

    best = df.iloc[0]
    print(f'\n  {BEST_MARK} Mejor modelo: "{best["name"]}" '
          f'(mask_mAP50 {float(best["mask_mAP50"]):.4f}, '
          f'box_mAP50 {float(best["box_mAP50"]):.4f})')

    if len(df) > 1:
        gap = float(best['mask_mAP50']) - float(df.iloc[-1]['mask_mAP50'])
        print(f'    Distancia hasta el último de la tabla: {gap:+.4f}')

    if _stored_order() not in (None, list(df['name'])):
        print('\n  AVISO: la columna rank de models.csv no coincide con el '
              'orden por mask_mAP50.')
        print('  La tabla de arriba usa el orden real; el CSV se corrige solo '
              'al registrar')
        print('  un modelo nuevo o al re-evaluar.')

    _print_dataset_state()

    free = models.MAX_MODELS - len(df)
    if free > 0:
        print(f'\n  Plazas libres en el leaderboard: {free} de '
              f'{models.MAX_MODELS}.')
    else:
        print(f'\n  Leaderboard completo: el próximo modelo tendrá que '
              f'superar a "{df.iloc[-1]["name"]}" '
              f'({float(df.iloc[-1]["mask_mAP50"]):.4f}).')

    print()


def register(subparsers):
    p = subparsers.add_parser(
        'ranking', help='Ranking de los modelos registrados por mask_mAP50')
    p.set_defaults(func=run)
    return p
