"""
`entrenar` command — trains a new model.

Trains with the given configuration and produces the three report outputs:
per-class metrics (CSV + plot) and confusion matrix.

Every run registers a new, numbered model (model_1, model_2...) in
models/models.csv: training again never overwrites the previous one, so all
of them stay available to compare with `benchmark`, `matrices` and
`predecir`.

Supports oversampling and copy-paste, which gave the best figures in the
experiments when combined with imgsz 800.

The registry is a leaderboard of models.MAX_MODELS entries: after registering
the new model this command reports the updated ranking, and says which model
was archived to make room for it — or that the new one did not make the cut.

Two ways of setting the hyperparameters:

  --reproduce  copies them from the rank 1 model, to retrain the winning
                configuration on a dataset that has grown.
  default       asks for the ones not given on the command line, so
                `entrenar --imgsz 800` only asks about the rest.
"""

import os
import re
import sys

import pandas as pd
import yaml

from .. import config, data, evaluate, models, plots, train


LEADERBOARD_COLUMNS = ['rank', 'name', 'architecture', 'imgsz', 'data',
                       'box_mAP50', 'mask_mAP50']

# Menú del modo nuevo: opciones ofrecidas y valor que toma Enter. Son también
# los valores que se usan cuando no hay terminal para preguntar.
ARCHITECTURES  = ['yolov8n-seg.pt', 'yolov8s-seg.pt',
                  'yolo11n-seg.pt', 'yolo11s-seg.pt']
IMGSZ_CHOICES  = [640, 800]
# El factor 1 es "no repetir nada". En el menú se ofrece como 'no' para que la
# opción de no aplicarlo se vea, en vez de esconderse detrás de un 1.
OVERSAMPLE_OFF     = 1
OVERSAMPLE_OFF_WORDS = ('no', 'n', 'ninguno', 'none', 'off', '1')
OVERSAMPLE_CHOICES = ['no', 2, 3]
EPOCHS_CHOICES = [50, 70, 100]
LR0_CHOICES    = [0.001, 0.002, 0.005]
BATCH_CHOICES  = ['auto', 2, 4, 8]

DEFAULTS = {
    'model'     : 'yolov8s-seg.pt',
    'imgsz'     : 800,
    'oversample': 3,
    'epochs'    : 70,
    'lr0'       : 0.002,
    'batch'     : 8,
    'copy_paste': 0.0,
}

# Lo que Ultralytics entiende por "elige tú el batch".
AUTO_BATCH = -1

# 'oversample_3x' en la columna data del CSV: de ahí sale el factor, porque el
# oversampling se hace antes de entrenar y no queda en args.yaml.
OVERSAMPLE_PATTERN = re.compile(r'oversample_(\d+)x')


def _describe_data(args):
    """Short summary of the data preparation, for the CSV column."""
    parts = []
    if args.oversample > 1:
        parts.append(f'oversample_{args.oversample}x')
    if args.copy_paste:
        parts.append(f'copy_paste {args.copy_paste}')
    return ' + '.join(parts) if parts else 'base'


def _parse_oversample(value):
    """Read an oversampling factor: 'no' (or 1) means do not apply it."""
    if str(value).strip().lower() in OVERSAMPLE_OFF_WORDS:
        return OVERSAMPLE_OFF
    return int(value)


def _format_oversample(factor):
    """Oversampling factor for the screen: 1 is not '1x', it is off."""
    return 'desactivado' if factor <= OVERSAMPLE_OFF else f'{factor}x'


def _format_batch(batch):
    """Batch as the user typed it: -1 is Ultralytics' automatic mode."""
    return 'auto' if batch == AUTO_BATCH else batch


def _ask(question, choices, default, cast=None):
    """
    Ask for one hyperparameter, listing the choices, with Enter as default.

    Any of the listed values is accepted. A value outside the list is accepted
    too when `cast` can read it (e.g. 120 epochs): the list is a suggestion,
    not a limit. Without `cast` only the listed values pass.
    """
    labels = ' / '.join(str(c) for c in choices)

    while True:
        try:
            answer = input(f'  {question} [{labels}] '
                           f'(Enter = {default}): ').strip()
        except EOFError:
            raise KeyboardInterrupt

        if not answer:
            return default

        for choice in choices:
            if answer.lower() == str(choice).lower():
                return choice

        if cast is not None:
            try:
                return cast(answer)
            except ValueError:
                pass

        print(f'    Responde con uno de: {labels}')


def _ask_missing(args):
    """
    Ask for every hyperparameter left as None by the command line.

    Passing `--imgsz 800 --oversample 3` skips those two questions and asks
    the rest. Without an interactive terminal there is nobody to ask and the
    defaults are used.
    """
    pending = [field for field in ('model', 'imgsz', 'oversample', 'epochs',
                                   'lr0', 'batch')
               if getattr(args, field) is None]

    if not pending:
        return

    if not sys.stdin.isatty():
        print('\nSin terminal interactiva: se usan los valores por defecto '
              'para lo no indicado.')
        _fill_defaults(args)
        return

    print('\n' + '=' * 70)
    print(' CONFIGURACIÓN DEL MODELO NUEVO')
    print('=' * 70)
    print('  Enter acepta el valor entre paréntesis.\n')

    if args.model is None:
        args.model = _ask('Arquitectura', ARCHITECTURES, DEFAULTS['model'])
    if args.imgsz is None:
        args.imgsz = int(_ask('Resolución  ', IMGSZ_CHOICES,
                              DEFAULTS['imgsz'], cast=int))
    if args.oversample is None:
        args.oversample = _parse_oversample(
            _ask('Oversampling', OVERSAMPLE_CHOICES, DEFAULTS['oversample'],
                 cast=_parse_oversample))
    if args.epochs is None:
        args.epochs = int(_ask('Épocas      ', EPOCHS_CHOICES,
                               DEFAULTS['epochs'], cast=int))
    if args.lr0 is None:
        args.lr0 = float(_ask('Learning rate', LR0_CHOICES,
                              DEFAULTS['lr0'], cast=float))
    if args.batch is None:
        chosen = _ask('Batch       ', BATCH_CHOICES, DEFAULTS['batch'],
                      cast=int)
        args.batch = AUTO_BATCH if str(chosen).lower() == 'auto' else int(chosen)


def _fill_defaults(args):
    """Give a value to whatever is still None after the command line."""
    for field, value in DEFAULTS.items():
        if getattr(args, field) is None:
            setattr(args, field, value)


def _best_model():
    """
    Registry row of the rank 1 model.

    The rank is recomputed here from mask_mAP50 instead of trusting the CSV
    column, which may have been left stale by a hand edit.
    """
    df = models.registry()
    if df.empty:
        raise FileNotFoundError(
            'No hay ningún modelo registrado: no hay configuración que '
            'reproducir. Entrena uno primero sin --reproduce.')

    order = df['mask_mAP50'].astype(float)
    return df.assign(_order=order).sort_values(
        '_order', ascending=False).iloc[0]


def _reproduce_setup(args):
    """
    Take the hyperparameters of the rank 1 model from its saved args.yaml.

    Meant for retraining the winning configuration after adding images: same
    architecture, imgsz, batch, lr0, epochs and copy_paste. The oversampling
    factor is not in args.yaml (it is applied to train.txt before training),
    so it is read from the `data` column of the registry.

    Anything given on the command line wins over the reproduced value, so
    `--reproduce --epochs 100` reproduces the rest and trains longer.
    """
    best      = _best_model()
    name      = best['name']
    args_yaml = os.path.join(config.HISTORY_DIR, name, 'args.yaml')

    if not os.path.isfile(args_yaml):
        raise FileNotFoundError(
            f'No existe {args_yaml}: de "{name}" no se guardó el args.yaml, '
            f'así que no se puede reproducir su configuración.')

    with open(args_yaml) as f:
        saved = yaml.safe_load(f) or {}

    # El args.yaml puede traer una ruta absoluta a los pesos base (Colab):
    # basta con el nombre del archivo.
    architecture = os.path.basename(str(saved.get('model', DEFAULTS['model'])))

    match      = OVERSAMPLE_PATTERN.search(str(best.get('data', '')))
    oversample = int(match.group(1)) if match else 1

    reproduced = {
        'model'     : architecture,
        'imgsz'     : saved.get('imgsz'),
        'batch'     : saved.get('batch'),
        'lr0'       : saved.get('lr0'),
        'epochs'    : saved.get('epochs'),
        'copy_paste': saved.get('copy_paste', 0.0),
        'oversample': oversample,
    }

    print('\n' + '=' * 70)
    print(f' REPRODUCIENDO LA CONFIGURACIÓN DE "{name}" (rank 1)')
    print('=' * 70)
    print(f'  Origen: {args_yaml}')
    print(f'  mask_mAP50 de referencia: {float(best["mask_mAP50"]):.4f}')

    for field, value in reproduced.items():
        if value is None:
            continue
        if getattr(args, field) is None:
            setattr(args, field, value)
            print(f'    {field:<11}: {_format_batch(value)}')
        else:
            print(f'    {field:<11}: {_format_batch(getattr(args, field))} '
                  f'(de la línea de comandos, en vez de '
                  f'{_format_batch(value)})')

    # Si el args.yaml venía incompleto, lo que falte toma el valor por defecto.
    _fill_defaults(args)

    return name


def _registered_names():
    """Names currently listed in models.csv; empty set if there is none yet."""
    if not os.path.isfile(config.MODELS_CSV):
        return set()
    return set(pd.read_csv(config.MODELS_CSV)['name'].astype(str))


def _refresh_old_metrics():
    """
    Re-evaluate the already registered models when the dataset changed.

    Their mask_mAP50 in models.csv was measured on an older test set, so
    ranking the new model against them would compare figures that do not come
    from the same data.
    """
    if not os.path.isfile(config.MODELS_CSV):
        return
    if not models.needs_reevaluation():
        return

    print('\n' + '=' * 70)
    print(' EL DATASET CAMBIÓ DESDE LA ÚLTIMA EVALUACIÓN')
    print('=' * 70)
    print(f'  Imágenes ahora: {models.dataset_fingerprint()}   '
          f'(antes: {models.saved_fingerprint()})')
    print('  Las métricas guardadas se midieron sobre otro conjunto de test,')
    print('  así que no son comparables con las del modelo nuevo.')
    print('  Re-evaluando los modelos ya registrados...')

    models.reevaluate_all(split='test')


def _leaderboard_table(highlight=None):
    """
    Print the registry as a comparative table, best mask_mAP50 first.

    `highlight` marks one row so the model just trained is easy to spot.
    Returns the DataFrame behind the table, or None if there is no registry.
    """
    try:
        df = models.registry()
    except FileNotFoundError:
        return None

    if df.empty:
        return df

    df    = df.sort_values('rank').reset_index(drop=True)
    table = df[[c for c in LEADERBOARD_COLUMNS if c in df.columns]].copy()
    table['   '] = ['<-- nuevo' if name == highlight else ''
                    for name in df['name']]
    print(table.to_string(index=False))

    return df


def _report_leaderboard(model_name, mask_map, before, after, final_path):
    """
    Say what the new model did to the leaderboard: whether it got in, which
    model it displaced, or why it stayed out.

    `before` / `after` are the sets of registered names around register_new().
    """
    print('\n' + '=' * 70)
    print(f' LEADERBOARD — top {models.MAX_MODELS} por mask_mAP50')
    print('=' * 70)
    df = _leaderboard_table(highlight=model_name)

    # Los que salieron del CSV para hacerle sitio al nuevo.
    displaced = sorted(before - after - {model_name})

    if model_name in after:
        for name in displaced:
            print(f'\n  "{model_name}" entró al top {models.MAX_MODELS} y '
                  f'desplazó a "{name}".')
            print(f'  "{name}" quedó archivado en {config.ARCHIVE_DIR} '
                  f'(pesos e historial).')
        if not displaced:
            free = models.MAX_MODELS - (len(after) if df is None else len(df))
            print(f'\n  "{model_name}" entra al leaderboard sin desplazar a '
                  f'nadie: aún quedan {max(free, 0)} plazas libres.')
        return

    # No entró: se compara con el peor de los que sí están.
    print(f'\n  "{model_name}" NO entró al top {models.MAX_MODELS}.')
    if df is not None and not df.empty:
        worst = df.iloc[-1]
        print(f'  mask_mAP50 {mask_map:.4f} frente a '
              f'{float(worst["mask_mAP50"]):.4f} de "{worst["name"]}", '
              f'el peor del top {models.MAX_MODELS}.')

    if final_path:
        print(f'  Se archivó en {final_path}: no aparece en `modelos` pero '
              f'los pesos siguen ahí.')
    else:
        print('  Se descartó: no se guardaron ni los pesos ni el historial.')


def run(args):
    # --no-oversample es un atajo de --oversample no: se resuelve antes de
    # nada para que valga también en modo --reproduce, donde la línea de
    # comandos manda sobre lo que traiga el modelo rank 1.
    if args.no_oversample:
        if args.oversample is not None and args.oversample != OVERSAMPLE_OFF:
            print(f'--no-oversample y --oversample {args.oversample} se '
                  f'contradicen: elige uno de los dos.')
            return 1
        args.oversample = OVERSAMPLE_OFF

    # Modo reproducir o modo nuevo: los dos dejan args con todos los
    # hiperparámetros resueltos antes de tocar nada.
    reproduced = _reproduce_setup(args) if args.reproduce else None
    if not reproduced:
        _ask_missing(args)
    _fill_defaults(args)

    class_names = config.load_class_names()
    runs_path   = (os.path.join(config.BASE_PATH, args.output) if args.output
                   else train.next_run_dir())

    # Name reserved before training: if model_1..model_N already exist, this
    # is the next free number and it clobbers none of them.
    model_name = args.name or models.next_model_name()
    run_name   = model_name

    train.check_environment()

    print('\nConfiguración del entrenamiento:')
    print(f'  Modelo nuevo  : {model_name}')
    print(f'  Clases        : {len(class_names)}')
    print(f'  Modelo base   : {args.model}')
    print(f'  Learning rate : {args.lr0}')
    print(f'  Épocas        : {args.epochs}   imgsz: {args.imgsz}')
    print(f'  Batch         : {_format_batch(args.batch)}   '
          f'oversampling: {_format_oversample(args.oversample)}')
    print(f'  Carpeta       : {runs_path}')
    if reproduced:
        print(f'  Reproduce     : {reproduced}')

    print('\nGenerando split train/val/test (semilla fija)...')
    train_imgs, val_imgs, test_imgs = data.generate_split(seed=args.seed)

    # Oversampling only repeats lines in train.txt; val and test are untouched.
    if args.oversample > 1:
        print(f'\nAplicando oversampling (factor máx {args.oversample}x)...')
        train_over = data.apply_oversampling(
            train_imgs, class_names, max_factor=args.oversample)
        data.write_split(train_over, val_imgs, test_imgs)

    extra = {}
    if args.copy_paste:
        extra['copy_paste'] = args.copy_paste

    try:
        weights = train.train(
            run_name   = run_name,
            runs_path  = runs_path,
            base_model = args.model,
            epochs     = args.epochs,
            imgsz      = args.imgsz,
            batch      = args.batch,
            lr0        = args.lr0,
            seed       = args.seed,
            exist_ok   = False,
            **extra
        )
    finally:
        # Whatever happens, train.txt goes back to the split without
        # repetitions: otherwise the next run would silently use oversampling.
        if args.oversample > 1:
            print('\nRestaurando train.txt al split original...')
            data.write_split(train_imgs, val_imgs, test_imgs)

    # ---------------- evaluation ----------------
    metrics = evaluate.validate(weights, split='test', verbose=True)

    print('\n' + '=' * 70)
    print(' MÉTRICAS POR CLASE (segmentación)')
    print('=' * 70)
    df_classes = evaluate.per_class_metrics(metrics, class_names)
    print(df_classes.to_string(index=False))

    metrics_csv = os.path.join(runs_path, 'per_class_metrics.csv')
    df_classes.to_csv(metrics_csv, index=False)
    print(f'\nTabla guardada en: {metrics_csv}')

    plots.map_per_class(
        df_classes,
        os.path.join(runs_path, 'per_class_metrics.png'),
        title=f'mAP50 por clase — {model_name}')

    plots.confusion_matrix(
        evaluate.confusion_matrix(metrics),
        config.matrix_labels(class_names),
        os.path.join(runs_path, 'confusion_matrix.png'),
        title=f'Matriz de confusión — {model_name}')

    totals = evaluate.summary(metrics)

    # ---------------- leaderboard: metricas comparables ----------------
    # Antes de rankear al nuevo contra los viejos, los viejos tienen que estar
    # medidos sobre el mismo test set.
    _refresh_old_metrics()

    # ---------------- registration ----------------
    run_dir = os.path.join(runs_path, run_name)
    print('\nDando de alta el modelo en el registro...')
    registered_before = _registered_names()
    final_path = models.register_new(
        model_name, weights, run_dir,
        architecture = os.path.splitext(os.path.basename(args.model))[0],
        imgsz        = args.imgsz,
        data         = _describe_data(args),
        box_mAP50    = totals['mAP50_detection'],
        mask_mAP50   = totals['mAP50_segmentation'],
        origin       = os.path.relpath(run_dir, config.BASE_PATH),
        notes        = args.notes or '',
    )

    # ---------------- leaderboard: como quedo la tabla ----------------
    _report_leaderboard(
        model_name, totals['mAP50_segmentation'],
        registered_before, _registered_names(), final_path)

    # ---------------- summary ----------------
    print('\n' + '=' * 60)
    print(f' RESUMEN — {model_name}')
    print('=' * 60)
    print(f'  Learning rate       : {args.lr0}')
    print(f'  Épocas              : {args.epochs}   imgsz: {args.imgsz}')
    print(f'  mAP50 global (seg)  : {totals["mAP50_segmentation"]:.4f}')
    print(f'  mAP50 global (det)  : {totals["mAP50_detection"]:.4f}')
    print('=' * 60)

    with_data = df_classes[df_classes['mAP50'] > 0]
    if not with_data.empty:
        weakest = with_data.loc[with_data['mAP50'].idxmin()]
        print(f'\n  Clase con menor mAP50: {weakest["class"]} '
              f'({weakest["mAP50"]:.4f})')
        print('  -> Candidata a necesitar más imágenes anotadas.')

    if final_path:
        print(f'\n  Pesos: {final_path}')
    if model_name in _registered_names():
        print(f'  Ya disponible en `modelos`, `benchmark`, `matrices` y '
              f'`predecir` como "{model_name}".')
    else:
        print(f'  Fuera del leaderboard: `modelos`, `benchmark`, `matrices` y '
              f'`predecir` no lo listan.')


def register(subparsers):
    p = subparsers.add_parser(
        'train', help='Entrena un modelo nuevo y genera sus métricas')
    p.add_argument('--model', dest='model', default=None,
                   help=f'Pesos de partida (si se omite se pregunta; '
                        f'default {DEFAULTS["model"]})')
    p.add_argument('--name', dest='name', default=None,
                   help='Nombre del modelo resultante '
                        '(default: el siguiente libre de la serie model_N)')
    p.add_argument('--notes', dest='notes', default=None,
                   help='Comentario para la columna notes de models.csv')
    # Sin valor por defecto en argparse: None significa "no lo indicó el
    # usuario", que es lo que distingue lo preguntado de lo impuesto.
    p.add_argument('--epochs', type=int, default=None,
                   help=f'Épocas (si se omite se pregunta; '
                        f'default {DEFAULTS["epochs"]})')
    p.add_argument('--imgsz', type=int, default=None,
                   help=f'Resolución (si se omite se pregunta; '
                        f'default {DEFAULTS["imgsz"]})')
    p.add_argument('--batch', type=int, default=None,
                   help=f'Batch, -1 = auto (si se omite se pregunta; '
                        f'default {DEFAULTS["batch"]})')
    p.add_argument('--lr0', type=float, default=None,
                   help=f'Learning rate (si se omite se pregunta; '
                        f'default {DEFAULTS["lr0"]})')
    p.add_argument('--seed', type=int, default=config.SEED)
    p.add_argument('--oversample', type=_parse_oversample, default=None,
                   metavar='N',
                   help=f'Factor máximo de repetición de clases débiles; '
                        f'"no" (o 1) lo desactiva (si se omite se pregunta; '
                        f'default {DEFAULTS["oversample"]})')
    p.add_argument('--no-oversample', dest='no_oversample',
                   action='store_true',
                   help='Entrena sin oversampling, igual que --oversample no')
    p.add_argument('--copy-paste', type=float, default=None, dest='copy_paste',
                   help='Augmentación copy-paste de 0 a 1 '
                        '(0.3 fue lo mejor; 0.5 empeora)')
    p.add_argument('--reproduce', dest='reproduce', action='store_true',
                   help='Reentrena con la configuración del modelo rank 1 '
                        '(su args.yaml), útil al añadir imágenes nuevas')
    p.add_argument('--output', dest='output', default=None,
                   help='Carpeta de salida (default: runs_final_N incremental)')
    p.set_defaults(func=run)
    return p
