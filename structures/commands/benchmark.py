"""
`benchmark` command — model comparison over a fixed set of images.

Runs the models in models/ over the images in benchmark.txt, saves the drawn
predictions grouped by model and compares the object count against the ground
truth.

The benchmark images are kept out of train/val/test, so no model saw them
during training.

If --models is not given, the trained models are listed and the user is asked
which ones to compare ("todos" is accepted).
"""

import os
import sys
from collections import Counter

import pandas as pd
from ultralytics import YOLO

from .. import config, data, draw, models, plots


def run(args):
    class_names = config.load_class_names()

    # ---------------- images ----------------
    # The benchmark set is fixed: the same 7 images for every model, excluded
    # from train/val/test. They are resolved by file name so the same
    # benchmark.txt works locally and on Colab.
    if not os.path.isfile(config.BENCHMARK_TXT):
        sys.exit(f'ERROR: no existe {config.BENCHMARK_TXT}\n'
                 f'Crea el archivo con una ruta de imagen por línea.')

    images, missing = data.benchmark_paths()

    if missing:
        sys.exit(f'ERROR: faltan imágenes de benchmark en {config.IMG_DIR}: '
                 f'{missing}\n'
                 f'El conjunto de benchmark debe estar completo: si falta '
                 f'alguna, los modelos no se comparan sobre las mismas '
                 f'imágenes y las cifras no son equivalentes.')

    if not images:
        sys.exit('ERROR: no se encontró ninguna imagen de benchmark válida.')

    print(f'Imágenes de benchmark: {len(images)} (conjunto fijo)')

    # ---------------- models ----------------
    selection = models.ask(
        args.models, multiple=True,
        title='¿Qué modelos quieres comparar en el benchmark?')
    print(f'\nModelos a comparar: {len(selection)}')
    for model in selection:
        print(f'  {model["name"]:<25} {model["path"]}')

    # ---------------- ground truth ----------------
    gt_per_image = {}
    print('\nGround truth de las imágenes de benchmark:')
    for path in images:
        img_name = os.path.basename(path)
        gt = data.read_ground_truth(path)
        gt_per_image[img_name] = gt
        detail = ', '.join(f'{class_names[k]}={v}'
                           for k, v in sorted(gt.items()) if v > 0)
        print(f'  {img_name:<25} [{sum(gt.values()):>3} obj] {detail}')

    # ---------------- prediction ----------------
    os.makedirs(config.BENCHMARK_OUTPUT, exist_ok=True)
    rows = []

    for model in selection:
        print('\n' + '=' * 60)
        print(f' {model["name"]}')
        print('=' * 60)

        folder = os.path.join(config.BENCHMARK_OUTPUT, model['name'])
        os.makedirs(folder, exist_ok=True)
        yolo = YOLO(model['path'])

        for path in images:
            img_name = os.path.basename(path)
            base     = os.path.splitext(img_name)[0]

            result = yolo.predict(path, conf=args.conf, verbose=False)[0]

            pred = Counter()
            if result.boxes is not None and len(result.boxes) > 0:
                for cls_id in result.boxes.cls.tolist():
                    pred[int(cls_id)] += 1

            gt = gt_per_image[img_name]

            row = {'model': model['name'], 'image': img_name}
            for i, name in enumerate(class_names):
                row[f'{name}_gt']   = gt.get(i, 0)
                row[f'{name}_pred'] = pred.get(i, 0)
                row[f'{name}_diff'] = pred.get(i, 0) - gt.get(i, 0)
            row['TOTAL_gt']   = sum(gt.values())
            row['TOTAL_pred'] = sum(pred.values())
            row['TOTAL_diff'] = sum(pred.values()) - sum(gt.values())
            rows.append(row)

            draw.draw_masks(
                result, os.path.join(folder, f'{base}_pred.jpg'),
                class_names, alpha=args.alpha)

            parts = []
            for i, name in enumerate(class_names):
                g, p = gt.get(i, 0), pred.get(i, 0)
                if g or p:
                    d = p - g
                    parts.append(f'{name}: {g}->{p} ({"+" if d > 0 else ""}{d})')
            print(f'  {img_name:<25} {" | ".join(parts)}')

    # ---------------- outputs ----------------
    df = pd.DataFrame(rows)
    excel = os.path.join(config.BENCHMARK_OUTPUT, 'benchmark_comparison.xlsx')
    df.to_excel(excel, index=False, sheet_name='Benchmark')
    print(f'\nExcel comparativo guardado en: {excel}')

    print('\n' + '=' * 70)
    print(' RESUMEN COMPARATIVO (GT vs Predicción)')
    print('=' * 70)

    gt_totals      = None
    pred_per_model = {}

    for model_name in df['model'].unique():
        df_model = df[df['model'] == model_name]
        pred_per_model[model_name] = [
            int(df_model[f'{n}_pred'].sum()) for n in class_names]
        if gt_totals is None:
            gt_totals = [int(df_model[f'{n}_gt'].sum()) for n in class_names]

        print(f'\n  {model_name}:')
        print(f'    {"clase":<18} {"GT":>5} {"Pred":>5} {"Diff":>6}')
        print(f'    {"-" * 37}')
        for i, name in enumerate(class_names):
            g, p = gt_totals[i], pred_per_model[model_name][i]
            if g or p:
                d = p - g
                print(f'    {name:<18} {g:>5} {p:>5} '
                      f'{("+" + str(d)) if d > 0 else d:>6}')
        gt_total   = int(df_model['TOTAL_gt'].sum())
        pred_total = int(df_model['TOTAL_pred'].sum())
        diff_total = pred_total - gt_total
        print(f'    {"-" * 37}')
        print(f'    {"TOTAL":<18} {gt_total:>5} {pred_total:>5} '
              f'{("+" + str(diff_total)) if diff_total > 0 else diff_total:>6}')

    plots.benchmark_comparison(
        class_names, gt_totals, pred_per_model,
        os.path.join(config.BENCHMARK_OUTPUT, 'benchmark_comparison.png'))

    print(f'\nResultados visuales en: {config.BENCHMARK_OUTPUT}')


def clean_model_benchmark(model_name, output_dir=None):
    """
    Remove a model's benchmark outputs and rebuild the comparative files.

    Called by models.archive_model() when a model leaves the leaderboard.
    """
    import shutil

    if output_dir is None:
        output_dir = config.BENCHMARK_OUTPUT

    if not os.path.isdir(output_dir):
        return

    folder = os.path.join(output_dir, model_name)
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    excel = os.path.join(output_dir, 'benchmark_comparison.xlsx')
    chart = os.path.join(output_dir, 'benchmark_comparison.png')

    if not os.path.isfile(excel):
        for path in (excel, chart):
            if os.path.isfile(path):
                os.remove(path)
        return

    df = pd.read_csv(excel) if excel.endswith('.csv') else pd.read_excel(excel)
    df = df[df['model'] != model_name]

    if df.empty:
        for path in (excel, chart):
            if os.path.isfile(path):
                os.remove(path)
        print(f'  Limpieza: benchmark de "{model_name}" eliminado '
              f'(no quedan modelos)')
        return

    df.to_excel(excel, index=False, sheet_name='Benchmark')

    class_names = config.load_class_names()
    gt_totals = None
    pred_per_model = {}
    for mn in df['model'].unique():
        df_model = df[df['model'] == mn]
        pred_per_model[mn] = [
            int(df_model[f'{n}_pred'].sum()) for n in class_names]
        if gt_totals is None:
            gt_totals = [int(df_model[f'{n}_gt'].sum()) for n in class_names]

    plots.benchmark_comparison(class_names, gt_totals, pred_per_model, chart)
    print(f'  Limpieza: benchmark de "{model_name}" eliminado, '
          f'comparativo regenerado')


def register(subparsers):
    p = subparsers.add_parser(
        'benchmark', help='Compara modelos sobre las imágenes de benchmark.txt')
    p.add_argument('--models', dest='models', nargs='+', default=None,
                   help='Nombres del registro a comparar '
                        '(si se omite, se pregunta listando los disponibles)')
    p.add_argument('--conf', type=float, default=0.25,
                   help='Umbral de confianza (default 0.25)')
    p.add_argument('--alpha', type=float, default=0.4,
                   help='Opacidad del relleno de las máscaras (default 0.4)')
    p.set_defaults(func=run)
    return p
