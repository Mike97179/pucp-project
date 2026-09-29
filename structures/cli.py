"""
Interfaz de línea de comandos.

Cada comando vive en su propio módulo bajo structures/commands/ y se
registra aquí con sus propias opciones. Los nombres de comando y flags
están en inglés; el texto de ayuda y todo lo impreso en español.
"""

import argparse
import sys

from . import __version__, config, data, models
from .commands import (analyze, benchmark, convergence, eval_model, final, lr,
                       matrices, predict, ranking, stats, update, validate)

COMMANDS = [lr, convergence, final, benchmark, matrices, predict, ranking,
            stats, validate, analyze, update, eval_model]


def _cmd_models(args):
    models.print_registry()


def _cmd_split(args):
    if args.restore:
        print('Restaurando split sin repeticiones...\n')
        data.restore_split()
    else:
        seed = args.seed if args.seed is not None else None
        print('Generando split train/val/test...\n')
        data.generate_split(seed=seed)

    class_names = config.load_class_names()
    stats.show_class_per_split(class_names, from_split=True)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python pucp_segmentation.py',
        description='Segmentación de elementos estructurales con YOLO.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Ejemplos:\n'
            '  python pucp_segmentation.py models\n'
            '  python pucp_segmentation.py train --model yolo11s-seg.pt --imgsz 800 --oversample 3\n'
            '  python pucp_segmentation.py benchmark --conf 0.3\n'
            '  python pucp_segmentation.py matrices --split test\n'
            '  python pucp_segmentation.py ranking     # leaderboard actual\n'
            '  python pucp_segmentation.py split             # busca la mejor semilla automáticamente\n'
            '  python pucp_segmentation.py split --seed 18  # usa una semilla específica\n'
            '  python pucp_segmentation.py stats       # qué hay en el dataset\n'
            '  python pucp_segmentation.py validate    # comprueba imágenes y labels\n'
            '  python pucp_segmentation.py analyze     # analiza imágenes de analyze/input/\n'
            '  python pucp_segmentation.py predict     # pregunta qué modelo usar\n'
            '  python pucp_segmentation.py update carpeta/  # registra modelo de Colab\n'
            '  python pucp_segmentation.py evaluate modelo.pt test.txt  # evalúa un modelo sobre un dataset\n'
        ))
    parser.add_argument('--version', action='version',
                        version=f'structures {__version__}')

    sub = parser.add_subparsers(dest='comando', metavar='<comando>')

    for module in COMMANDS:
        module.register(sub)

    p_models = sub.add_parser('models', help='Lista los modelos disponibles')
    p_models.set_defaults(func=_cmd_models)

    p_split = sub.add_parser(
        'split', help='Busca la mejor semilla y genera train/val/test')
    p_split.add_argument('--seed', type=int, default=None,
                         help='Semilla manual (si se omite, se busca la mejor)')
    p_split.add_argument('--restore', dest='restore', action='store_true',
                         help='Quita las repeticiones que deja el oversampling')
    p_split.set_defaults(func=_cmd_split)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, 'func', None):
        parser.print_help()
        return 1

    try:
        result = args.func(args)
        if isinstance(result, int):
            return result
    except FileNotFoundError as e:
        print(f'\nERROR: {e}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nInterrumpido por el usuario.', file=sys.stderr)
        return 130

    return 0