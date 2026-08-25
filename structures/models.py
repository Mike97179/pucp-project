"""
Registry of trained models.

Each script used to carry its own hardcoded list of runs_experimento_*/ paths.
Now the weights live in models/ and this module discovers them from
models.csv, so deleting the runs* folders breaks nothing.

Every training run produces a new, numbered model (model_1, model_2, ...):
an earlier one is never overwritten. Commands that need a model ask which one
to use, listing the available ones.

The registry is a fixed leaderboard of MAX_MODELS entries ordered by
mask_mAP50: when a new model pushes it over that size, the worst one is
archived (weights and history moved to models/archive/) instead of deleted.
"""

import os
import re
import shutil
import sys

import pandas as pd

from . import config, data


MODEL_PREFIX  = 'model'
MODEL_PATTERN = re.compile(rf'^{MODEL_PREFIX}_(\d+)$')

# Size of the leaderboard: only the best MAX_MODELS models stay registered.
MAX_MODELS = 5

CSV_COLUMNS = ['rank', 'name', 'file', 'architecture', 'imgsz', 'data',
               'box_mAP50', 'mask_mAP50', 'origin', 'notes']

# What is kept from each training run so it can be reviewed later without
# retraining. Ultralytics does not always write all of them: whichever exist
# get copied.
HISTORY_FILES = [
    'results.csv', 'results.png', 'args.yaml',
    'confusion_matrix.png', 'confusion_matrix_normalized.png',
    'MaskPR_curve.png', 'MaskF1_curve.png', 'BoxPR_curve.png',
]


def registry(models_csv=config.MODELS_CSV):
    """
    DataFrame of the models declared in models/models.csv, plus an absolute
    'path' column and only the rows whose weights actually exist.

    The 'rank' column is recomputed here from mask_mAP50 instead of being
    taken from the file: rank 1 always means the best model, even if the CSV
    was edited by hand or a row was added with the wrong rank. The file itself
    is not rewritten — that happens when a model is registered or re-evaluated.
    """
    if not os.path.isfile(models_csv):
        raise FileNotFoundError(
            f'No existe {models_csv}.\n'
            f'Entrena un modelo con `python pucp_segmentation.py train` o '
            f'coloca los pesos .pt en {config.MODELS_DIR} y descríbelos en '
            f'models.csv (columnas: rank,name,file,...).'
        )

    df = pd.read_csv(models_csv)
    df['path'] = df['file'].apply(
        lambda f: os.path.join(config.MODELS_DIR, f))

    missing = df[~df['path'].apply(os.path.isfile)]
    if not missing.empty:
        print(f'  AVISO: pesos declarados pero no encontrados: '
              f'{missing["file"].tolist()}')

    df = df[df['path'].apply(os.path.isfile)].reset_index(drop=True)
    return _recompute_rank(df)


def resolve(name_or_path):
    """
    Accept a registry name ('model_3'), a file name ('model_3.pt') or a loose
    path to a .pt, and return the path to the weights.
    """
    if os.path.isfile(name_or_path):
        return name_or_path

    try:
        df = registry()
    except FileNotFoundError:
        raise FileNotFoundError(f'No se encontró el modelo: {name_or_path}')

    for column in ('name', 'file'):
        row = df[df[column] == name_or_path]
        if not row.empty:
            return row.iloc[0]['path']

    available = ', '.join(df['name'].tolist())
    raise FileNotFoundError(
        f'No se encontró el modelo "{name_or_path}".\n'
        f'Disponibles: {available}'
    )


def select(names=None):
    """
    List of {name, path} dicts for batch work.

    With no arguments it returns the whole registry, sorted by rank.
    """
    df = registry()
    if names:
        df = df[df['name'].isin(names)]
        if df.empty:
            raise FileNotFoundError(f'Ninguno de esos modelos existe: {names}')

    if 'rank' in df.columns:
        df = df.sort_values('rank')

    return [{'name': row['name'], 'path': row['path']}
            for _, row in df.iterrows()]


# ------------------------------------------------------------------
# Registering new models
# ------------------------------------------------------------------
def next_model_name():
    """
    First free name in the series: model_1, model_2, ...

    Looks at the CSV, the loose .pt files and the history folders at once: if
    any of the three already used a number, that number is not reused. That
    way a new training run never clobbers an earlier one's results.
    """
    used = {0}

    def note(name):
        match = MODEL_PATTERN.match(name)
        if match:
            used.add(int(match.group(1)))

    if os.path.isfile(config.MODELS_CSV):
        try:
            for name in pd.read_csv(config.MODELS_CSV)['name']:
                note(str(name))
        except (KeyError, pd.errors.EmptyDataError):
            pass

    if os.path.isdir(config.MODELS_DIR):
        for file in os.listdir(config.MODELS_DIR):
            note(os.path.splitext(file)[0])

    if os.path.isdir(config.HISTORY_DIR):
        for folder in os.listdir(config.HISTORY_DIR):
            note(folder)

    # El archivo tambien cuenta: un modelo archivado sigue existiendo, asi que
    # su numero no se reutiliza aunque ya no este en el CSV.
    if os.path.isdir(config.ARCHIVE_DIR):
        for entry in os.listdir(config.ARCHIVE_DIR):
            note(os.path.splitext(entry)[0])

    return f'{MODEL_PREFIX}_{max(used) + 1}'


def _recompute_rank(df):
    """Rank 1 = best mask_mAP50. Recomputed every time a model is added."""
    if 'mask_mAP50' not in df.columns:
        return df
    order = pd.to_numeric(df['mask_mAP50'], errors='coerce').fillna(-1)
    df = df.assign(rank=order.rank(ascending=False, method='first').astype(int))
    return df.sort_values('rank').reset_index(drop=True)


def save_history(name, run_dir):
    """Copy what the training run left behind to models/history/<name>/."""
    target = os.path.join(config.HISTORY_DIR, name)
    os.makedirs(target, exist_ok=True)

    copied = 0
    for file in HISTORY_FILES:
        source = os.path.join(run_dir, file)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(target, file))
            copied += 1

    return target, copied


def register_new(name, weights, run_dir=None, **fields):
    """
    Register a freshly trained model: copy the .pt to models/<name>.pt, save
    its history and append the matching row to models.csv.

    `fields` accepts any of the CSV columns (architecture, imgsz, data,
    box_mAP50, mask_mAP50, origin, notes).

    The registry only holds MAX_MODELS models: once the new row is in, the
    worst one is archived. Returns where the new model's weights ended up —
    models/<name>.pt normally, the archive path if it did not make the
    leaderboard, or None if the user chose to discard it.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    file_name = f'{name}.pt'
    target    = os.path.join(config.MODELS_DIR, file_name)
    if os.path.exists(target):
        raise FileExistsError(
            f'Ya existe {target}. Los modelos no se sobrescriben: borra ese '
            f'archivo o renómbralo si de verdad quieres reemplazarlo.')

    shutil.copy2(weights, target)

    if run_dir:
        folder, copied = save_history(name, run_dir)
        print(f'  Historial ({copied} archivos): {folder}')

    row = {c: fields.get(c, '') for c in CSV_COLUMNS}
    row['name'] = name
    row['file'] = file_name

    if os.path.isfile(config.MODELS_CSV):
        df = pd.read_csv(config.MODELS_CSV)
        for column in CSV_COLUMNS:
            if column not in df.columns:
                df[column] = ''
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = _recompute_rank(df)
    df[CSV_COLUMNS].to_csv(config.MODELS_CSV, index=False)

    print(f'  Registrado como "{name}" en {config.MODELS_CSV}')

    # Las metricas recien calculadas corresponden al dataset actual: se deja
    # constancia para que `needs_reevaluation()` sepa contra que comparar.
    save_fingerprint()

    return _enforce_leaderboard(name, target)


# ------------------------------------------------------------------
# Leaderboard of MAX_MODELS: archiving what falls out
# ------------------------------------------------------------------
def _csv_row(name, models_csv=config.MODELS_CSV):
    """The model's row in models.csv as a dict, or None if it is not there."""
    if not os.path.isfile(models_csv):
        return None
    df  = pd.read_csv(models_csv)
    row = df[df['name'] == name]
    return None if row.empty else row.iloc[0].to_dict()


def _weights_file(name):
    """File name of the model's weights: what the CSV says, or <name>.pt."""
    row = _csv_row(name)
    if row and isinstance(row.get('file'), str) and row['file']:
        return row['file']
    return f'{name}.pt'


def _drop_from_csv(name, models_csv=config.MODELS_CSV):
    """Remove the model's row from models.csv and recompute the ranks."""
    if not os.path.isfile(models_csv):
        return
    df = pd.read_csv(models_csv)
    df = df[df['name'] != name]
    df = _recompute_rank(df)
    df[CSV_COLUMNS].to_csv(models_csv, index=False)


def _move_without_overwriting(source, target):
    """shutil.move that never clobbers: adds _2, _3... to the name if needed."""
    base, extension = os.path.splitext(target)
    index = 2
    while os.path.exists(target):
        target = f'{base}_{index}{extension}'
        index += 1
    shutil.move(source, target)
    return target


def archive_model(name):
    """
    Take a model out of the leaderboard without losing it: its weights move to
    models/archive/, its models/history/<name>/ folder to
    models/archive/<name>/, and its row leaves models.csv.

    Returns the path of the archived weights, or None if there were none.
    """
    os.makedirs(config.ARCHIVE_DIR, exist_ok=True)

    file_name = _weights_file(name)
    archived  = None

    weights = os.path.join(config.MODELS_DIR, file_name)
    if os.path.isfile(weights):
        archived = _move_without_overwriting(
            weights, os.path.join(config.ARCHIVE_DIR, file_name))

    history = os.path.join(config.HISTORY_DIR, name)
    if os.path.isdir(history):
        _move_without_overwriting(
            history, os.path.join(config.ARCHIVE_DIR, name))

    _drop_from_csv(name)

    print(f'  Archivado "{name}" en {config.ARCHIVE_DIR}')
    return archived


def discard_model(name):
    """
    Delete a model for good: weights, history folder and row in models.csv.

    Only used when the user explicitly says so — archiving is the default.
    """
    weights = os.path.join(config.MODELS_DIR, _weights_file(name))
    if os.path.isfile(weights):
        os.remove(weights)

    history = os.path.join(config.HISTORY_DIR, name)
    if os.path.isdir(history):
        shutil.rmtree(history)

    _drop_from_csv(name)

    print(f'  Descartado "{name}": no se guardó nada.')


def _archive_or_discard(name):
    """
    Ask what to do with the model just trained when it did not make the top
    MAX_MODELS.

    Without an interactive terminal there is nobody to ask, so it is archived:
    disk space is cheaper than a lost training run.

    Returns the path of the archived weights, or None if discarded.
    """
    print(f'\n  "{name}" no superó a ninguno de los {MAX_MODELS} modelos del '
          f'leaderboard.')

    if not sys.stdin.isatty():
        print('  Sin terminal interactiva: se archiva por defecto.')
        return archive_model(name)

    while True:
        try:
            answer = input('  ¿Archivar o descartar? '
                           '[a/d] (Enter = archivar): ').strip().lower()
        except EOFError:
            answer = ''

        if answer in ('', 'a', 'archivar'):
            return archive_model(name)
        if answer in ('d', 'descartar'):
            discard_model(name)
            return None
        print('  Responde "a" para archivar o "d" para descartar.')


def _enforce_leaderboard(new_name=None, new_path=None):
    """
    Trim models.csv down to MAX_MODELS rows, archiving whatever falls out.

    `new_name` is the model that was just registered: if it is the one falling
    out, the user is asked whether to archive or discard it, since it never
    made the leaderboard to begin with.

    Returns where `new_name`'s weights ended up (None if they were discarded).
    """
    if not os.path.isfile(config.MODELS_CSV):
        return new_path

    df = pd.read_csv(config.MODELS_CSV)
    if len(df) <= MAX_MODELS or 'rank' not in df.columns:
        return new_path

    for name in df[df['rank'] > MAX_MODELS]['name'].tolist():
        if name == new_name:
            new_path = _archive_or_discard(name)
        else:
            print(f'\n  El leaderboard solo guarda {MAX_MODELS} modelos y '
                  f'"{name}" es el de peor mask_mAP50.')
            archive_model(name)

    return new_path


# ------------------------------------------------------------------
# Dataset changes and re-evaluation
# ------------------------------------------------------------------
def dataset_fingerprint(img_dir=config.IMG_DIR):
    """
    Total number of images in the dataset.

    Cheap stand-in for a content hash: what matters is spotting that the
    dataset grew (newly annotated images), because from that moment on the
    metrics stored in models.csv came from a different test set.
    """
    return len(data.list_images(img_dir))


def save_fingerprint(value=None, state_file=config.DATASET_STATE):
    """
    Write the current fingerprint to models/dataset_state.txt.

    Called whenever the stored metrics are known to match the dataset on disk:
    after registering a new model and after `reevaluate_all()`.
    """
    if value is None:
        value = dataset_fingerprint()

    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        f.write(f'{value}\n')

    return value


def saved_fingerprint(state_file=config.DATASET_STATE):
    """Fingerprint written the last time the metrics were up to date."""
    if not os.path.isfile(state_file):
        return None
    try:
        with open(state_file) as f:
            return int(f.read().strip())
    except ValueError:
        return None


def needs_reevaluation(img_dir=config.IMG_DIR,
                       state_file=config.DATASET_STATE):
    """
    True when the dataset changed since the metrics in models.csv were
    computed, so the ranking cannot be trusted until `reevaluate_all()` runs.

    With no fingerprint stored yet there is nothing to compare against and it
    returns False: the first evaluation writes one.
    """
    stored = saved_fingerprint(state_file)
    return stored is not None and stored != dataset_fingerprint(img_dir)


def reevaluate_all(split='test'):
    """
    Re-run model.val() on every registered model and refresh box_mAP50 and
    mask_mAP50 in models.csv, recomputing the ranks and the fingerprint.

    Rows whose weights are missing are left untouched instead of dropped.
    Returns the updated DataFrame.
    """
    # Importado aquí y no arriba: evaluate carga torch y ultralytics, y este
    # módulo lo importa cada comando del CLI.
    from . import evaluate

    if not os.path.isfile(config.MODELS_CSV):
        raise FileNotFoundError(f'No existe {config.MODELS_CSV}.')

    df = pd.read_csv(config.MODELS_CSV)
    if df.empty:
        print('No hay modelos registrados que re-evaluar.')
        return df

    print(f'\nRe-evaluando {len(df)} modelos sobre el split "{split}"...')

    for position, (index, row) in enumerate(df.iterrows(), start=1):
        name    = row['name']
        weights = os.path.join(config.MODELS_DIR, str(row['file']))

        if not os.path.isfile(weights):
            print(f'\n  [{position}/{len(df)}] {name}: faltan los pesos '
                  f'({row["file"]}) — se deja como está.')
            continue

        print(f'\n  [{position}/{len(df)}] {name}')
        totals = evaluate.summary(evaluate.validate(weights, split=split))
        df.at[index, 'box_mAP50']  = totals['mAP50_detection']
        df.at[index, 'mask_mAP50'] = totals['mAP50_segmentation']
        print(f'      box mAP50: {totals["mAP50_detection"]:.4f}   '
              f'mask mAP50: {totals["mAP50_segmentation"]:.4f}')
        evaluate.free_memory()

    df = _recompute_rank(df)
    df[CSV_COLUMNS].to_csv(config.MODELS_CSV, index=False)
    save_fingerprint()

    print(f'\nMétricas actualizadas en {config.MODELS_CSV}\n')
    print(df[_visible_columns(df)].to_string(index=False))
    print()

    return df


# ------------------------------------------------------------------
# Interactive selection
# ------------------------------------------------------------------
def _visible_columns(df):
    return [c for c in ('rank', 'name', 'architecture', 'imgsz',
                        'data', 'box_mAP50', 'mask_mAP50', 'notes')
            if c in df.columns]


def _print_numbered_table(df):
    """Print the registry with a 1..N index so it can be picked by number."""
    table = df[_visible_columns(df)].copy()
    table.insert(0, '#', range(1, len(table) + 1))
    print(table.to_string(index=False))


def print_registry():
    """Readable table of the available models."""
    df = registry()
    if 'rank' in df.columns:
        df = df.sort_values('rank').reset_index(drop=True)
    print(f'\nModelos disponibles en {config.MODELS_DIR}:\n')
    print(df[_visible_columns(df)].to_string(index=False))
    print()


def ask(names=None, multiple=False, allow_all=None,
        title='Modelos disponibles'):
    """
    Return the selected models as a list of {name, path} dicts.

    If `names` comes from the command line it is honoured as is. Otherwise
    every model in the registry is listed and the user is asked which one to
    use — by number or by name. `allow_all` (defaults to `multiple`) enables
    answering "todos".

    Without an interactive terminal there is nobody to ask: all models are
    used when `allow_all`, and otherwise it aborts asking for the explicit
    argument.
    """
    if names:
        return select(names)

    if allow_all is None:
        allow_all = multiple

    df = registry()
    if df.empty:
        raise FileNotFoundError(
            f'No hay ningún modelo en {config.MODELS_DIR}.\n'
            f'Entrena uno con `python pucp_segmentation.py train`.')

    if 'rank' in df.columns:
        df = df.sort_values('rank').reset_index(drop=True)

    if not sys.stdin.isatty():
        if allow_all:
            print('Sin terminal interactiva: se usan todos los modelos.')
            return select()
        raise FileNotFoundError(
            'Sin terminal interactiva no se puede preguntar qué modelo usar. '
            'Pásalo con --model.')

    available = df['name'].tolist()
    print(f'\n{title}:\n')
    _print_numbered_table(df)

    if multiple:
        prompt = ('\nElige uno o varios (números o nombres, separados por '
                  'espacios)')
    else:
        prompt = '\nElige uno (número o nombre)'
    if allow_all:
        prompt += ', o "todos"'

    while True:
        try:
            answer = input(f'{prompt}: ').strip()
        except EOFError:
            raise KeyboardInterrupt

        if not answer:
            continue

        if allow_all and answer.lower() in ('todos', 'todas', 'all', '*'):
            return select()

        picks = answer.split()
        if not multiple and len(picks) > 1:
            print('  Solo se puede elegir un modelo.')
            continue

        chosen = []
        error  = None
        for entry in picks:
            if entry.isdigit():
                i = int(entry)
                if not 1 <= i <= len(available):
                    error = f'  "{i}" está fuera de rango (1..{len(available)}).'
                    break
                chosen.append(available[i - 1])
            elif entry in available:
                chosen.append(entry)
            else:
                error = f'  "{entry}" no está en la lista.'
                break

        if error:
            print(error)
            continue

        # dict.fromkeys: drop duplicates while keeping the order they were
        # picked in
        return select(list(dict.fromkeys(chosen)))


def ask_one(name=None, title='Modelos disponibles'):
    """Like `ask`, but returns the path of the single chosen model."""
    if name:
        return resolve(name)
    return ask(multiple=False, allow_all=False, title=title)[0]['path']
