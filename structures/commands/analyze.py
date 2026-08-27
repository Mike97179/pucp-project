"""
Comando `analyze` — predicción rápida sobre imágenes del usuario.

El usuario coloca imágenes en analyze/input/, elige uno o varios modelos,
y obtiene una subcarpeta por imagen en analyze/output/ con la predicción
con máscaras y un CSV de objetos detectados.  Dentro de cada carpeta de
imagen hay una subcarpeta por modelo, así se puede comparar la misma imagen
entre modelos sin sobreescribir nada.

Cuando un modelo se archiva del leaderboard sus subcarpetas se limpian
automáticamente (ver models.archive_model).

Ejemplo de estructura::

    analyze/
    ├── input/
    │   ├── foto1.jpg
    │   └── foto2.jpg
    └── output/
        ├── foto1/
        │   ├── model_1/
        │   │   ├── foto1_pred.jpg
        │   │   └── detections.csv
        │   ├── model_3/
        │   │   ├── foto1_pred.jpg
        │   │   └── detections.csv
        │   └── summary.csv          ← comparativa de modelos para esta imagen
        └── foto2/
            ├── model_1/
            │   ├── foto2_pred.jpg
            │   └── detections.csv
            └── summary.csv
"""

import os
import sys

import pandas as pd
from ultralytics import YOLO

from .. import config, data, draw, models, plots


def _input_images():
    """
    Imágenes en analyze/input/, ordenadas alfabéticamente.

    Retorna (rutas, mensaje_error).  Cuando no hay imágenes el segundo
    elemento lleva el texto de error para que el llamador lo imprima y salga.
    """
    input_dir = config.ANALYZE_INPUT
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        return [], (f'No hay imágenes en {input_dir}\n'
                    f'Coloca las imágenes que quieras analizar ahí y vuelve '
                    f'a correr el comando.')

    images = data.list_images(input_dir)
    if not images:
        return [], (f'No hay imágenes en {input_dir}\n'
                    f'Coloca las imágenes que quieras analizar ahí y vuelve '
                    f'a correr el comando.')

    return images, None


def _process_image(image_path, model_info, class_names, conf, alpha):
    """
    Ejecuta un modelo sobre una imagen: guarda la predicción con máscaras
    y un CSV de detecciones en analyze/output/<imagen>/<modelo>/.

    Retorna el dict de fila para la tabla resumen.
    """
    img_base   = os.path.splitext(os.path.basename(image_path))[0]
    model_name = model_info['name']

    # Carpeta de salida: analyze/output/<imagen>/<modelo>/
    output_dir = os.path.join(config.ANALYZE_OUTPUT, img_base, model_name)
    os.makedirs(output_dir, exist_ok=True)

    yolo   = YOLO(model_info['path'])
    result = yolo.predict(image_path, conf=conf, verbose=False)[0]

    # Imagen con máscaras
    pred_path = os.path.join(output_dir, f'{img_base}_pred.jpg')
    total = draw.draw_masks(result, pred_path, class_names, alpha=alpha)

    # Conteo por clase
    counts = {name: 0 for name in class_names}
    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id in result.boxes.cls.tolist():
            counts[class_names[int(cls_id)]] += 1

    row = {'model': model_name}
    row.update(counts)
    row['TOTAL'] = total

    # CSV individual por modelo e imagen
    df = pd.DataFrame([row])
    df.to_csv(os.path.join(output_dir, 'detections.csv'), index=False)

    return row


def _rebuild_summary(img_base, class_names):
    """
    Lee todos los detections.csv que existan en las subcarpetas de modelos
    de una imagen y genera el summary.csv consolidado, ordenado por el
    ranking del modelo en el leaderboard (mask_mAP50).

    Así el summary siempre refleja todos los modelos que se han usado sobre
    esa imagen, no solo los de la última corrida.
    """
    img_dir = os.path.join(config.ANALYZE_OUTPUT, img_base)
    if not os.path.isdir(img_dir):
        return None

    # Obtener el ranking actual de los modelos registrados
    try:
        reg = models.registry()
        rank_map = dict(zip(reg['name'], reg['rank']))
    except FileNotFoundError:
        rank_map = {}

    rows = []
    for model_name in sorted(os.listdir(img_dir)):
        det_csv = os.path.join(img_dir, model_name, 'detections.csv')
        if not os.path.isfile(det_csv):
            continue

        det = pd.read_csv(det_csv)
        if det.empty:
            continue

        row = det.iloc[0].to_dict()
        row['model'] = model_name
        # Modelos que ya no están en el registro van al final
        row['rank'] = rank_map.get(model_name, 999)
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Asegurar que TOTAL exista
    if 'TOTAL' not in df.columns:
        df['TOTAL'] = df[[n for n in class_names if n in df.columns]].sum(axis=1)

    # Ordenar por ranking del leaderboard (rank 1 primero)
    df = df.sort_values('rank', ascending=True).reset_index(drop=True)

    # rank solo se usa para ordenar, no se guarda en el CSV
    df = df.drop(columns=['rank'])

    # Guardar el summary en la carpeta de la imagen
    summary_path = os.path.join(img_dir, 'summary.csv')
    df.to_csv(summary_path, index=False)

    return df


def run(args):
    class_names = config.load_class_names()

    # Imágenes de entrada
    images, error = _input_images()
    if error:
        print(f'\n{error}')
        return

    print(f'\nImágenes en {config.ANALYZE_INPUT}: {len(images)}')
    for img in images:
        print(f'  {os.path.basename(img)}')

    # Selección de modelo(s)
    selection = models.ask(
        args.models, multiple=True,
        title='¿Con qué modelo(s) quieres analizar?')

    print(f'\nModelos seleccionados: {len(selection)}')
    for model in selection:
        print(f'  {model["name"]:<25} {model["path"]}')

    # Procesar cada imagen con cada modelo
    for model in selection:
        print('\n' + '=' * 60)
        print(f' {model["name"]}')
        print('=' * 60)

        for image_path in images:
            row = _process_image(
                image_path, model, class_names, args.conf, args.alpha)

            detail = ', '.join(
                f'{n}={row[n]}' for n in class_names if row[n])
            print(f'  {os.path.basename(image_path):<25} '
                  f'[{row["TOTAL"]:>3} obj] {detail}')

    # Reconstruir summary.csv de cada imagen leyendo TODOS los detections.csv
    # que existan en sus subcarpetas, no solo los de esta corrida.
    print('\n' + '=' * 60)
    print(' RESUMEN POR IMAGEN')
    print('=' * 60)

    for image_path in images:
        img_base = os.path.splitext(os.path.basename(image_path))[0]
        df = _rebuild_summary(img_base, class_names)

        if df is None or df.empty:
            continue

        # Mostrar tabla comparativa en pantalla
        print(f'\n  {img_base}:')

        # Columnas con al menos una detección en esta imagen
        active_classes = [n for n in class_names if df[n].sum() > 0]

        print(f'    {"modelo":<25} ', end='')
        for name in active_classes:
            print(f'{name:<15} ', end='')
        print(f'{"TOTAL":>6}')
        print(f'    {"-" * (26 + 16 * len(active_classes) + 6)}')

        for _, row in df.iterrows():
            print(f'    {row["model"]:<25} ', end='')
            for name in active_classes:
                print(f'{int(row[name]):<15} ', end='')
            print(f'{int(row["TOTAL"]):>6}')

        summary_path = os.path.join(config.ANALYZE_OUTPUT, img_base,
                                    'summary.csv')
        print(f'    Guardado: {summary_path}')

        # Gráfica de barras agrupadas por clase si hay más de un modelo
        if len(df) > 1:
            counts_per_model = {}
            for _, row in df.iterrows():
                counts_per_model[row['model']] = [
                    int(row.get(n, 0)) for n in class_names]

            chart_path = os.path.join(config.ANALYZE_OUTPUT, img_base,
                                      'comparison.png')
            plots.analyze_comparison(
                class_names, counts_per_model, chart_path,
                title=f'Detecciones por clase — {img_base}')

    print(f'\nResultados visuales en: {config.ANALYZE_OUTPUT}')


def clean_model_outputs(model_name, output_dir=None):
    """
    Elimina todas las subcarpetas de un modelo en analyze/output/.

    Lo llama models.archive_model() cuando un modelo sale del leaderboard,
    para que no se acumulen predicciones obsoletas.
    """
    if output_dir is None:
        output_dir = config.ANALYZE_OUTPUT

    if not os.path.isdir(output_dir):
        return

    import shutil

    removed = 0
    for image_folder in os.listdir(output_dir):
        model_dir = os.path.join(output_dir, image_folder, model_name)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
            removed += 1

            # Si la carpeta de la imagen quedó vacía, eliminarla también
            parent = os.path.join(output_dir, image_folder)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)

    if removed:
        print(f'  Limpieza: {removed} carpeta(s) de "{model_name}" '
              f'eliminada(s) de {output_dir}')


def register(subparsers):
    p = subparsers.add_parser(
        'analyze',
        help='Analiza imágenes sueltas desde analyze/input/')
    p.add_argument('--models', dest='models', nargs='+', default=None,
                   help='Modelos a usar '
                        '(si se omite, se pregunta listando los disponibles)')
    p.add_argument('--conf', type=float, default=0.25,
                   help='Umbral de confianza (default 0.25)')
    p.add_argument('--alpha', type=float, default=0.4,
                   help='Opacidad de las máscaras (default 0.4)')
    p.set_defaults(func=run)
    return p