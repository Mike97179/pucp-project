"""
Registro de modelos entrenados.

Cada modelo vive en su propia carpeta models/<nombre>/ con los pesos, el
historial de entrenamiento y cualquier salida que haya producido Ultralytics.
Archivar un modelo es mover esa carpeta entera a models/archive/.

Cada entrenamiento produce un modelo nuevo y numerado (model_1, model_2, ...):
uno anterior nunca se sobreescribe. Los comandos que necesitan un modelo
preguntan cuál usar, listando los disponibles.

El registro es un leaderboard fijo de MAX_MODELS entradas ordenadas por
mask_mAP50: cuando un modelo nuevo lo excede, el peor se archiva en vez de
borrarse.
"""

import os
import re
import shutil
import sys

import pandas as pd

from . import config, data


MODEL_PREFIX  = 'model'
MODEL_PATTERN = re.compile(rf'^{MODEL_PREFIX}_(\d+)$')

# Tamaño del leaderboard: solo los mejores MAX_MODELS modelos quedan registrados.
MAX_MODELS = 5

CSV_COLUMNS = ['rank', 'name', 'file', 'architecture', 'imgsz', 'data',
               'box_mAP50', 'mask_mAP50', 'origin', 'notes']

# Lo que se guarda de cada entrenamiento para poder revisarlo después sin
# reentrenar. Ultralytics no siempre escribe todos: se copian los que existan.
HISTORY_FILES = [
    'results.csv', 'results.png', 'args.yaml',
    'confusion_matrix.png', 'confusion_matrix_normalized.png',
    'MaskPR_curve.png', 'MaskF1_curve.png', 'BoxPR_curve.png',
]


def registry(models_csv=config.MODELS_CSV):
    """
    DataFrame de los modelos declarados en models/models.csv, con una columna
    'path' absoluta y solo las filas cuyos pesos existan en disco.

    La columna 'rank' se recalcula aquí desde mask_mAP50 en vez de tomarse
    del archivo: rank 1 siempre es el mejor modelo, aunque el CSV se haya
    editado a mano o una fila tenga el rank equivocado. El archivo no se
    reescribe — eso pasa al registrar o re-evaluar un modelo.
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
    Acepta un nombre del registro ('model_3'), un nombre de archivo
    ('model_3.pt') o una ruta suelta a un .pt, y retorna la ruta a los pesos.
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
    Lista de dicts {name, path} para trabajo en lote.

    Sin argumentos retorna todo el registro, ordenado por rank.
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
# Registro de modelos nuevos
# ------------------------------------------------------------------
def next_model_name():
    """
    Primer nombre libre de la serie: model_1, model_2, ...

    Revisa el CSV, los .pt sueltos y las carpetas de historial a la vez:
    si cualquiera de los tres ya usó un número, ese número no se reutiliza.
    Así un entrenamiento nuevo nunca pisa los resultados de uno anterior.
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

    if os.path.isdir(config.MODELS_DIR):
        for entry in os.listdir(config.MODELS_DIR):
            if os.path.isdir(os.path.join(config.MODELS_DIR, entry)):
                note(entry)

    if os.path.isdir(config.ARCHIVE_DIR):
        for entry in os.listdir(config.ARCHIVE_DIR):
            note(os.path.splitext(entry)[0])

    return f'{MODEL_PREFIX}_{max(used) + 1}'


def _recompute_rank(df):
    """Rank 1 = mejor mask_mAP50. Se recalcula cada vez que se agrega un modelo."""
    if 'mask_mAP50' not in df.columns:
        return df
    order = pd.to_numeric(df['mask_mAP50'], errors='coerce').fillna(-1)
    df = df.assign(rank=order.rank(ascending=False, method='first').astype(int))
    return df.sort_values('rank').reset_index(drop=True)


def model_dir(name):
    """Carpeta de un modelo: models/<nombre>/."""
    return os.path.join(config.MODELS_DIR, name)


def save_history(name, run_dir):
    """Copia los archivos clave del entrenamiento a models/<nombre>/."""
    target = model_dir(name)
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
    Registra un modelo recién entrenado: copia el .pt a models/<nombre>.pt,
    guarda su historial y agrega la fila correspondiente a models.csv.

    `fields` acepta cualquier columna del CSV (architecture, imgsz, data,
    box_mAP50, mask_mAP50, origin, notes).

    El registro solo guarda MAX_MODELS modelos: al agregar la fila nueva,
    el peor se archiva. Retorna dónde quedaron los pesos del modelo nuevo —
    models/<nombre>.pt normalmente, la ruta del archivo si no entró al
    leaderboard, o None si el usuario decidió descartarlo.
    """
    mdir = model_dir(name)
    os.makedirs(mdir, exist_ok=True)

    pt_name  = f'{name}.pt'
    target   = os.path.join(mdir, pt_name)
    if os.path.exists(target):
        raise FileExistsError(
            f'Ya existe {target}. Los modelos no se sobrescriben: borra ese '
            f'archivo o renómbralo si de verdad quieres reemplazarlo.')

    shutil.copy2(weights, target)

    if run_dir and os.path.normpath(run_dir) != os.path.normpath(mdir):
        folder, copied = save_history(name, run_dir)
        print(f'  Historial ({copied} archivos): {folder}')

    row = {c: fields.get(c, '') for c in CSV_COLUMNS}
    row['name'] = name
    row['file'] = os.path.join(name, pt_name)

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
# Leaderboard de MAX_MODELS: archivar lo que queda fuera
# ------------------------------------------------------------------
def _csv_row(name, models_csv=config.MODELS_CSV):
    """Fila del modelo en models.csv como dict, o None si no está."""
    if not os.path.isfile(models_csv):
        return None
    df  = pd.read_csv(models_csv)
    row = df[df['name'] == name]
    return None if row.empty else row.iloc[0].to_dict()


def _weights_file(name):
    """Nombre de archivo de los pesos: lo que dice el CSV, o <nombre>.pt."""
    row = _csv_row(name)
    if row and isinstance(row.get('file'), str) and row['file']:
        return row['file']
    return f'{name}.pt'


def _drop_from_csv(name, models_csv=config.MODELS_CSV):
    """Elimina la fila del modelo de models.csv y recalcula los ranks."""
    if not os.path.isfile(models_csv):
        return
    df = pd.read_csv(models_csv)
    df = df[df['name'] != name]
    df = _recompute_rank(df)
    df[CSV_COLUMNS].to_csv(models_csv, index=False)


def _move_without_overwriting(source, target):
    """shutil.move que nunca sobreescribe: agrega _2, _3... al nombre si hace falta."""
    base, extension = os.path.splitext(target)
    index = 2
    while os.path.exists(target):
        target = f'{base}_{index}{extension}'
        index += 1
    shutil.move(source, target)
    return target


def _clean_all_outputs(name):
    """Remove every trace of a model outside models/: predictions, benchmark,
    matrices and analyze outputs."""
    from .commands.analyze import clean_model_outputs
    from .commands.predict import clean_model_predictions
    from .commands.benchmark import clean_model_benchmark
    from .commands.matrices import clean_model_matrices

    clean_model_outputs(name)
    clean_model_predictions(name)
    clean_model_benchmark(name)
    clean_model_matrices(name)


def archive_model(name):
    """
    Saca un modelo del leaderboard sin perderlo: mueve models/<nombre>/
    entera a models/archive/<nombre>/ y su fila desaparece de models.csv.

    Retorna la ruta de la carpeta archivada, o None si no había.
    """
    os.makedirs(config.ARCHIVE_DIR, exist_ok=True)

    mdir     = model_dir(name)
    archived = None

    if os.path.isdir(mdir):
        archived = _move_without_overwriting(
            mdir, os.path.join(config.ARCHIVE_DIR, name))

    _drop_from_csv(name)
    _clean_all_outputs(name)

    print(f'  Archivado "{name}" en {config.ARCHIVE_DIR}')
    return archived


def discard_model(name):
    """
    Elimina un modelo definitivamente: toda su carpeta y su fila en
    models.csv.

    Solo se usa cuando el usuario lo pide explícitamente — archivar es lo
    que se hace por defecto.
    """
    mdir = model_dir(name)
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)

    _drop_from_csv(name)
    _clean_all_outputs(name)

    print(f'  Descartado "{name}": no se guardó nada.')


def _archive_or_discard(name):
    """
    Pregunta qué hacer con el modelo recién entrenado cuando no entró al
    top MAX_MODELS.

    Sin terminal interactiva no hay a quién preguntar, así que se archiva:
    el espacio en disco es más barato que un entrenamiento perdido.

    Retorna la ruta de los pesos archivados, o None si se descartó.
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
    Recorta models.csv a MAX_MODELS filas, archivando lo que sobra.

    `new_name` es el modelo recién registrado: si es el que sobra, se
    pregunta al usuario si archivarlo o descartarlo, ya que nunca llegó
    a estar en el leaderboard.

    Retorna dónde quedaron los pesos de `new_name` (None si se descartaron).
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
# Cambios en el dataset y re-evaluación
# ------------------------------------------------------------------
def dataset_fingerprint(img_dir=config.IMG_DIR):
    """
    Número total de imágenes en el dataset.

    Sustituto barato de un hash de contenido: lo que importa es detectar
    que el dataset creció (imágenes recién anotadas), porque desde ese
    momento las métricas de models.csv vienen de un test set diferente.
    """
    return len(data.list_images(img_dir))


def save_fingerprint(value=None, state_file=config.DATASET_STATE):
    """
    Escribe el fingerprint actual en models/dataset_state.txt.

    Se llama cuando las métricas almacenadas coinciden con el dataset en
    disco: después de registrar un modelo nuevo y después de reevaluate_all().
    """
    if value is None:
        value = dataset_fingerprint()

    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        f.write(f'{value}\n')

    return value


def saved_fingerprint(state_file=config.DATASET_STATE):
    """Fingerprint guardado la última vez que las métricas estuvieron al día."""
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
    True cuando el dataset cambió desde que se calcularon las métricas de
    models.csv, así que el ranking no es confiable hasta que corra
    reevaluate_all().

    Sin fingerprint guardado todavía no hay contra qué comparar y retorna
    False: la primera evaluación lo escribe.
    """
    stored = saved_fingerprint(state_file)
    return stored is not None and stored != dataset_fingerprint(img_dir)


def reevaluate_all(split='test'):
    """
    Re-ejecuta model.val() sobre cada modelo registrado y actualiza box_mAP50
    y mask_mAP50 en models.csv, recalculando los ranks y el fingerprint.

    Las filas cuyos pesos no estén en disco se dejan intactas en vez de
    eliminarse. Retorna el DataFrame actualizado.
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
# Selección interactiva
# ------------------------------------------------------------------
def _visible_columns(df):
    return [c for c in ('rank', 'name', 'architecture', 'imgsz',
                        'data', 'box_mAP50', 'mask_mAP50', 'notes')
            if c in df.columns]


def _print_numbered_table(df):
    """Imprime el registro con índice 1..N para elegir por número."""
    table = df[_visible_columns(df)].copy()
    table.insert(0, '#', range(1, len(table) + 1))
    print(table.to_string(index=False))


def print_registry():
    """Tabla legible de los modelos disponibles."""
    df = registry()
    if 'rank' in df.columns:
        df = df.sort_values('rank').reset_index(drop=True)
    print(f'\nModelos disponibles en {config.MODELS_DIR}:\n')
    print(df[_visible_columns(df)].to_string(index=False))
    print()


def ask(names=None, multiple=False, allow_all=None,
        title='Modelos disponibles'):
    """
    Retorna los modelos seleccionados como lista de dicts {name, path}.

    Si `names` viene de la línea de comandos se respeta tal cual. Si no,
    se lista todo el registro y se pregunta al usuario cuál usar — por
    número o por nombre. `allow_all` (por defecto igual a `multiple`)
    habilita responder "todos".

    Sin terminal interactiva no hay a quién preguntar: se usan todos los
    modelos cuando `allow_all`, y si no aborta pidiendo el argumento
    explícito.
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

        # dict.fromkeys: elimina duplicados manteniendo el orden de elección
        return select(list(dict.fromkeys(chosen)))


def ask_one(name=None, title='Modelos disponibles'):
    """Como `ask`, pero retorna la ruta del único modelo elegido."""
    if name:
        return resolve(name)
    return ask(multiple=False, allow_all=False, title=title)[0]['path']