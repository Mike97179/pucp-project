"""
`update` command — registers a model trained in Colab.

When training happens in Colab, the notebook saves its outputs to a folder in
Drive: weights, training history, and a registrar.txt snippet with the metrics
and metadata.  The user downloads that folder, places it inside models/, and
runs this command.

The command scans models/ for folders that have a registrar.txt but are not yet
in models.csv, parses the metrics, lets the user confirm or edit the name, and
registers the model into the leaderboard — the same as if it had been trained
locally.
"""

import ast
import os
import re
import sys

import pandas as pd

from .. import config, models


def _parse_registrar(path):
    """
    Extract the keyword arguments from a registrar.txt snippet.

    The file contains a Python call to models.register_new(...).  This parser
    reads the keyword arguments (architecture, imgsz, box_mAP50, etc.) and
    returns them as a dict, plus the .pt filename found in the snippet.
    """
    with open(path) as f:
        text = f.read()

    match = re.search(r'register_new\((.*)\)', text, re.DOTALL)
    if not match:
        sys.exit(f'ERROR: no se encontró register_new() en {path}')

    body = match.group(1)

    fields = {}
    pt_file = None

    for line in body.split('\n'):
        line = line.strip().rstrip(',')

        kw = re.match(r"(\w+)\s*=\s*(.+)", line)
        if kw:
            key = kw.group(1)

            if key in ('run_dir',):
                continue

            try:
                fields[key] = ast.literal_eval(kw.group(2).strip())
            except (ValueError, SyntaxError):
                fields[key] = kw.group(2).strip().strip("'\"")
            continue

        cleaned = line.strip("'\"")
        if cleaned.endswith('.pt'):
            pt_file = os.path.basename(cleaned)

    return fields, pt_file


def _find_weights(folder, hint=None):
    """Locate the .pt weights inside the folder."""
    if hint:
        candidate = os.path.join(folder, hint)
        if os.path.isfile(candidate):
            return candidate

    pts = [f for f in os.listdir(folder) if f.endswith('.pt')]
    if len(pts) == 1:
        return os.path.join(folder, pts[0])
    if len(pts) > 1:
        print(f'  Archivos .pt encontrados: {pts}')
        sys.exit(f'ERROR: hay más de un .pt en {folder}. '
                 f'Indica cuál usar con --weights.')
    sys.exit(f'ERROR: no se encontró ningún .pt en {folder}.')


def _find_pending_folder():
    """
    Scan models/ for a folder with registrar.txt that is not in models.csv.

    Returns the folder path, or exits with an error if none or more than one
    are found.
    """
    registered = set()
    if os.path.isfile(config.MODELS_CSV):
        registered = set(pd.read_csv(config.MODELS_CSV)['name'].astype(str))

    skip = {'archive', '__pycache__'}
    pending = []

    for entry in os.listdir(config.MODELS_DIR):
        if entry in skip or entry in registered:
            continue
        full = os.path.join(config.MODELS_DIR, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, 'registrar.txt')):
            pending.append(full)

    if not pending:
        sys.exit('No se encontró ninguna carpeta nueva con registrar.txt en '
                 f'{config.MODELS_DIR}.\n'
                 f'Coloca la carpeta descargada de Colab dentro de models/ '
                 f'y vuelve a ejecutar este comando.')

    if len(pending) > 1:
        names = [os.path.basename(p) for p in pending]
        sys.exit(f'Hay más de una carpeta pendiente de registrar: {names}\n'
                 f'Registra una a la vez: mueve las demás fuera de models/ '
                 f'temporalmente.')

    return pending[0]


def _ask_name(suggestion):
    """Show the suggested name and let the user accept (Enter) or edit it."""
    if not sys.stdin.isatty():
        return suggestion

    try:
        import readline
        readline.set_startup_hook(lambda: readline.insert_text(suggestion))
        try:
            answer = input('\n  Nombre del modelo (edita o Enter para aceptar): ')
        finally:
            readline.set_startup_hook()
    except ImportError:
        answer = input(f'\n  Nombre del modelo [{suggestion}]: ')

    name = answer.strip()
    return name if name else suggestion


def _registered_names():
    if not os.path.isfile(config.MODELS_CSV):
        return set()
    return set(pd.read_csv(config.MODELS_CSV)['name'].astype(str))


def run(args):
    folder = os.path.abspath(args.folder) if args.folder else _find_pending_folder()

    if not os.path.isdir(folder):
        sys.exit(f'ERROR: no existe la carpeta {folder}')

    registrar = os.path.join(folder, 'registrar.txt')
    if not os.path.isfile(registrar):
        sys.exit(f'ERROR: no se encontró registrar.txt en {folder}\n'
                 f'El notebook de Colab genera este archivo con las métricas '
                 f'y metadatos del modelo.')

    folder_name = os.path.basename(folder)
    print(f'Carpeta detectada: {folder_name}/')

    fields, pt_hint = _parse_registrar(registrar)
    weights_path = _find_weights(folder, pt_hint)
    weights_name = os.path.basename(weights_path)

    for key in ('box_mAP50', 'mask_mAP50'):
        if key not in fields:
            sys.exit(f'ERROR: falta "{key}" en registrar.txt')

    print(f'\nMétricas del modelo:')
    print(f'  box_mAP50  : {fields["box_mAP50"]}')
    print(f'  mask_mAP50 : {fields["mask_mAP50"]}')
    if 'architecture' in fields:
        print(f'  Arquitectura: {fields["architecture"]}')
    if 'imgsz' in fields:
        print(f'  Resolución  : {fields["imgsz"]}')
    if 'data' in fields:
        print(f'  Data        : {fields["data"]}')

    origin = fields.get('origin', os.path.splitext(pt_hint or folder_name)[0])
    suggestion = args.name or origin
    model_name = _ask_name(suggestion)

    # Rename folder if the chosen name differs from the current folder name
    if model_name != folder_name:
        new_folder = os.path.join(config.MODELS_DIR, model_name)
        if os.path.exists(new_folder):
            sys.exit(f'ERROR: ya existe {new_folder}. Elige otro nombre.')
        os.rename(folder, new_folder)
        folder = new_folder
        print(f'  Carpeta renombrada: {folder_name}/ -> {model_name}/')

    # Rename weights to <name>.pt
    pt_target = f'{model_name}.pt'
    if weights_name != pt_target:
        old_pt = os.path.join(folder, weights_name)
        new_pt = os.path.join(folder, pt_target)
        if os.path.exists(new_pt):
            sys.exit(f'ERROR: ya existe {new_pt}.')
        os.rename(old_pt, new_pt)
        print(f'  Pesos renombrados: {weights_name} -> {pt_target}')

    # Re-evaluate existing models if the dataset changed
    if os.path.isfile(config.MODELS_CSV) and models.needs_reevaluation():
        print('\n' + '=' * 70)
        print(' EL DATASET CAMBIÓ DESDE LA ÚLTIMA EVALUACIÓN')
        print('=' * 70)
        print(f'  Imágenes ahora: {models.dataset_fingerprint()}   '
              f'(antes: {models.saved_fingerprint()})')
        print('  Las métricas guardadas se midieron sobre otro test set.')
        print('  Re-evaluando los modelos ya registrados...')
        models.reevaluate_all(split='test')

    # Add row to CSV and enforce leaderboard
    print('\nRegistrando el modelo...')
    registered_before = _registered_names()

    csv_fields = {k: v for k, v in fields.items()
                  if k in ('architecture', 'imgsz', 'data', 'box_mAP50',
                            'mask_mAP50', 'origin', 'notes')}

    row = {c: csv_fields.get(c, '') for c in models.CSV_COLUMNS}
    row['name'] = model_name
    row['file'] = os.path.join(model_name, pt_target)

    if os.path.isfile(config.MODELS_CSV):
        df = pd.read_csv(config.MODELS_CSV)
        for col in models.CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = models._recompute_rank(df)
    df[models.CSV_COLUMNS].to_csv(config.MODELS_CSV, index=False)
    models.save_fingerprint()

    print(f'  Registrado como "{model_name}" en {config.MODELS_CSV}')

    final_path = models._enforce_leaderboard(
        model_name, os.path.join(folder, pt_target))

    registered_after = _registered_names()

    # Leaderboard report
    print('\n' + '=' * 70)
    print(f' LEADERBOARD — top {models.MAX_MODELS} por mask_mAP50')
    print('=' * 70)

    try:
        df = models.registry()
        cols = [c for c in ('rank', 'name', 'architecture', 'imgsz',
                            'data', 'box_mAP50', 'mask_mAP50')
                if c in df.columns]
        table = df.sort_values('rank').reset_index(drop=True)
        display = table[cols].copy()
        display['   '] = ['<-- nuevo' if n == model_name else ''
                          for n in table['name']]
        print(display.to_string(index=False))
    except FileNotFoundError:
        pass

    displaced = sorted(registered_before - registered_after - {model_name})
    if model_name in registered_after:
        for name in displaced:
            print(f'\n  "{model_name}" entró al top {models.MAX_MODELS} y '
                  f'desplazó a "{name}".')
            print(f'  "{name}" quedó archivado en {config.ARCHIVE_DIR}.')
        print(f'\n  "{model_name}" ya disponible en `models`, `benchmark`, '
              f'`matrices` y `predict`.')
    else:
        print(f'\n  "{model_name}" NO entró al top {models.MAX_MODELS}.')
        if final_path:
            print(f'  Se archivó en {final_path}.')
        else:
            print('  Se descartó.')

    # Remove registrar.txt — already consumed
    reg_path = os.path.join(folder if model_name in registered_after
                            else (final_path or ''), 'registrar.txt')
    if os.path.isfile(reg_path):
        os.remove(reg_path)


def register(subparsers):
    p = subparsers.add_parser(
        'update',
        help='Registra un modelo entrenado en Colab desde su carpeta en models/')
    p.add_argument('folder', nargs='?', default=None,
                   help='Carpeta dentro de models/ con los pesos y '
                        'registrar.txt (si se omite, se detecta automáticamente)')
    p.add_argument('--name', default=None,
                   help='Nombre del modelo (si se omite, se sugiere desde '
                        'registrar.txt y se puede editar)')
    p.set_defaults(func=run)
    return p
