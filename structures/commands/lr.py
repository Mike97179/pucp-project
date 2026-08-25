"""
`lr` command — learning rate sweep.

Trains once per learning rate keeping everything else fixed and reports which
one gives the best segmentation mAP50.

Methodological note: the optimizer is pinned to AdamW on purpose. With
optimizer='auto' Ultralytics overrides lr0 with its own value and the sweep is
invalidated — different learning rates produce identical curves.
"""

import os

import pandas as pd

from .. import config, data, evaluate, plots, train

LEARNING_RATES = [0.0005, 0.0009, 0.002, 0.005]


def run(args):
    class_names = config.load_class_names()
    # A fresh folder per run: repeating the sweep does not clobber the last one.
    runs_path = train.next_run_dir(prefix='runs_experimento_lr')

    train.check_environment()

    print(f'\nClases         : {len(class_names)} -> {class_names}')
    print(f'Optimizador    : {config.OPTIMIZER} (fijo, para que respete el lr0)')
    print(f'Learning rates : {args.lrs}')
    print(f'Entrenamientos : {len(args.lrs)}   Épocas c/u: {args.epochs}')

    print('\nGenerando split train/val/test (semilla fija)...')
    data.generate_split(seed=args.seed)

    results = []
    for i, lr0 in enumerate(args.lrs, start=1):
        print('\n' + '=' * 60)
        print(f' CORRIDA {i}/{len(args.lrs)} — learning rate = {lr0}')
        print('=' * 60)

        weights = train.train(
            run_name  = f'lr_{lr0}',
            runs_path = runs_path,
            epochs    = args.epochs,
            imgsz     = args.imgsz,
            batch     = args.batch,
            lr0       = lr0,
            seed      = args.seed,
        )

        metrics = evaluate.validate(weights, split='val')
        row = {'learning_rate': lr0, **evaluate.summary(metrics)}
        results.append(row)
        evaluate.free_memory()

        print(f'\n  -> mAP50 detección    : {row["mAP50_detection"]:.4f}')
        print(f'  -> mAP50 segmentación : {row["mAP50_segmentation"]:.4f}')

    df = pd.DataFrame(results)
    print('\nResultados del experimento:\n')
    print(df.to_string(index=False))

    os.makedirs(runs_path, exist_ok=True)
    csv_path = os.path.join(runs_path, 'lr_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nTabla guardada en: {csv_path}')

    plots.lr_vs_map(df, os.path.join(runs_path, 'lr_vs_map50.png'))

    best = df.loc[df['mAP50_segmentation'].idxmax()]
    print('\n' + '=' * 50)
    print(' MEJOR LEARNING RATE ENCONTRADO')
    print('=' * 50)
    print(f'  learning rate      : {best["learning_rate"]}')
    print(f'  mAP50 detección    : {best["mAP50_detection"]}')
    print(f'  mAP50 segmentación : {best["mAP50_segmentation"]}')
    print('=' * 50)
    print(f'\n  Usa --lr0 {best["learning_rate"]} en los comandos '
          f'`convergencia` y `entrenar`.')


def register(subparsers):
    p = subparsers.add_parser(
        'lr', help='Barrido de learning rate (experimento 1)')
    p.add_argument('--lrs', type=float, nargs='+', default=LEARNING_RATES,
                   help=f'Learning rates a probar (default: {LEARNING_RATES})')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=config.IMGSZ)
    p.add_argument('--batch', type=int, default=config.BATCH)
    p.add_argument('--seed', type=int, default=config.SEED)
    p.set_defaults(func=run)
    return p
