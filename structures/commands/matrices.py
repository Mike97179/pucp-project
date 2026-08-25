"""
`matrices` command — confusion matrices of the models in models/.

Runs model.val() over the chosen split and saves one image per model plus a
combined one with all of them side by side.

If --models is not given, the trained models are listed and the user is asked
which ones to evaluate ("todos" is accepted).
"""

import os

from .. import config, evaluate, models, plots


def run(args):
    class_names = config.load_class_names()
    labels      = config.matrix_labels(class_names)

    selection = models.ask(
        args.models, multiple=True,
        title=f'¿Qué modelos quieres evaluar sobre {args.split}?')
    print(f'\nModelos a evaluar: {len(selection)}')
    for model in selection:
        print(f'  {model["name"]:<25} {model["path"]}')

    os.makedirs(config.MATRICES_OUTPUT, exist_ok=True)
    matrices = {}

    for i, model in enumerate(selection, start=1):
        print('\n' + '=' * 60)
        print(f' [{i}/{len(selection)}] {model["name"]}')
        print('=' * 60)

        metrics = evaluate.validate(model['path'], split=args.split)
        cm = evaluate.confusion_matrix(metrics)
        matrices[model['name']] = cm

        plots.confusion_matrix(
            cm, labels,
            os.path.join(config.MATRICES_OUTPUT, f'matrix_{model["name"]}.png'),
            title=f'Matriz de confusión — {model["name"]}')

        del metrics
        evaluate.free_memory()

    if len(matrices) > 1:
        plots.combined_matrices(
            matrices, labels,
            os.path.join(config.MATRICES_OUTPUT, 'all_matrices.png'))

    print(f'\nTodas las matrices en: {config.MATRICES_OUTPUT}')


def register(subparsers):
    p = subparsers.add_parser(
        'matrices', help='Matrices de confusión de los modelos entrenados')
    p.add_argument('--models', dest='models', nargs='+', default=None,
                   help='Nombres del registro a evaluar '
                        '(si se omite, se pregunta listando los disponibles)')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                   help='Partición sobre la que evaluar (default: test)')
    p.set_defaults(func=run)
    return p
