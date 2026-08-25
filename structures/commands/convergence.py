"""
`convergencia` command — one long training run to see where it plateaus.

Plots epoch vs mAP50 from the results.csv YOLO writes and reports at which
epoch the maximum was reached, to size the final model's epoch count.
"""

import os

import pandas as pd

from .. import config, data, plots, train


def run(args):
    # A fresh folder per run: repeating the experiment does not clobber the
    # last one.
    runs_path = train.next_run_dir(prefix='runs_experimento_convergencia')
    run_name  = f'convergencia_lr{args.lr0}'

    train.check_environment()

    print(f'\nLearning rate : {args.lr0}')
    print(f'Épocas        : {args.epochs}')
    print(f'Patience      : {args.epochs} (igual a épocas, para ver la curva '
          f'completa)')

    print('\nGenerando split train/val/test (semilla fija)...')
    data.generate_split(seed=args.seed)

    train.train(
        run_name  = run_name,
        runs_path = runs_path,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        lr0       = args.lr0,
        patience  = args.epochs,   # do not stop early: we want the whole curve
        seed      = args.seed,
    )

    results_csv = os.path.join(runs_path, run_name, 'results.csv')
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    print(f'\nÉpocas registradas: {len(df)}')

    plots.convergence(
        df, os.path.join(runs_path, 'convergence.png'), args.lr0)

    seg_column  = 'metrics/mAP50(M)'
    idx         = df[seg_column].idxmax()
    best_epoch  = int(df.loc[idx, 'epoch'])
    best_map    = df.loc[idx, seg_column]

    print('\n' + '=' * 50)
    print(' ANÁLISIS DE CONVERGENCIA')
    print('=' * 50)
    print(f'  Mejor mAP50 segmentación : {best_map:.4f}')
    print(f'  Alcanzado en la época    : {best_epoch} de {len(df)}')
    print('=' * 50)

    if best_epoch < len(df) * 0.8:
        print(f'\n  El modelo se estabilizó cerca de la época {best_epoch}.')
        print(f'  Entrenar ~{best_epoch + 10} épocas sería suficiente.')
    else:
        print('\n  El modelo seguía mejorando cerca del final.')
        print('  Prueba con más épocas para ver si sigue subiendo.')


def register(subparsers):
    p = subparsers.add_parser(
        'convergence', help='Curva de épocas vs mAP50 (experimento 2)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr0', type=float, default=config.LR0)
    p.add_argument('--imgsz', type=int, default=config.IMGSZ)
    p.add_argument('--batch', type=int, default=config.BATCH)
    p.add_argument('--seed', type=int, default=config.SEED)
    p.set_defaults(func=run)
    return p
