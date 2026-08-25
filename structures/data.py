"""
Dataset preparation — reproducible split, oversampling and ground truth.

The split is always regenerated from the fixed seed: reproducible with the
same data, and newly added images join the split on their own.
"""

import glob
import os
import random
from collections import Counter

from . import config


def list_images(img_dir=config.IMG_DIR):
    """Every dataset image, sorted so the shuffle is stable."""
    return sorted(
        path
        for ext in config.IMG_EXTENSIONS
        for path in glob.glob(os.path.join(img_dir, ext))
    )


def read_benchmark_names(benchmark_txt=config.BENCHMARK_TXT):
    """
    File names of the images reserved for the benchmark.

    They are compared by basename, not by full path, so the set stays fixed
    even if the project moves to another folder or runs on Colab.
    """
    if not os.path.isfile(benchmark_txt):
        return set()
    with open(benchmark_txt) as f:
        return {os.path.basename(ln.strip()) for ln in f if ln.strip()}


def benchmark_paths(benchmark_txt=config.BENCHMARK_TXT, img_dir=config.IMG_DIR):
    """
    Usable paths of the benchmark images, in the order given by the file.

    benchmark.txt stores absolute paths from the machine where it was created,
    so each entry is resolved by file name against the current environment's
    images/ directory. That way the same benchmark.txt works locally and on
    Colab.

    Returns (found_paths, missing_names).
    """
    if not os.path.isfile(benchmark_txt):
        return [], []

    with open(benchmark_txt) as f:
        entries = [ln.strip() for ln in f if ln.strip()]

    paths, missing = [], []
    for entry in entries:
        candidate = os.path.join(img_dir, os.path.basename(entry))
        if os.path.isfile(candidate):
            paths.append(candidate)
        elif os.path.isfile(entry):
            paths.append(entry)
        else:
            missing.append(os.path.basename(entry))

    return paths, missing


def generate_split(seed=config.SEED, ratio=config.SPLIT_RATIO,
                   exclude_benchmark=True, verbose=True):
    """
    Split the images into train/val/test and write the three .txt files.

    The images in benchmark.txt are excluded so they never land in any split:
    they are only used for visual comparison between models.

    Returns (train, val, test) as lists of paths.
    """
    images = list_images()
    if verbose:
        print(f'Total de imágenes encontradas: {len(images)}')

    if exclude_benchmark:
        benchmark = read_benchmark_names()
        if benchmark:
            images = [i for i in images
                      if os.path.basename(i) not in benchmark]
            if verbose:
                print(f'  Imágenes de benchmark excluidas: {len(benchmark)}')

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * ratio[0])
    n_val   = int(n * ratio[1])

    train = images[:n_train]
    val   = images[n_train:n_train + n_val]
    test  = images[n_train + n_val:]

    write_split(train, val, test)

    if verbose:
        print(f'  train: {len(train)} imágenes')
        print(f'  val  : {len(val)} imágenes')
        print(f'  test : {len(test)} imágenes')

    return train, val, test


def write_split(train, val, test):
    """Dump the three lists to train.txt / val.txt / test.txt."""
    for txt_path, items in ((config.TRAIN_TXT, train),
                            (config.VAL_TXT,   val),
                            (config.TEST_TXT,  test)):
        with open(txt_path, 'w') as f:
            f.write('\n'.join(items))


def label_path(img_path, lbl_dir=config.LBL_DIR):
    """Path of the annotation .txt matching an image."""
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(lbl_dir, f'{base}.txt')


def read_ground_truth(img_path, lbl_dir=config.LBL_DIR):
    """Annotated object count per class: Counter {cls_id: amount}."""
    counts = Counter()
    path = label_path(img_path, lbl_dir)
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts


def classes_per_image(images, lbl_dir=config.LBL_DIR):
    """{image_path: set(cls_id)} — which classes show up in each image."""
    return {img: set(read_ground_truth(img, lbl_dir)) for img in images}


def count_instances(images, n_classes, lbl_dir=config.LBL_DIR):
    """Total instances per class over a set of images."""
    total = Counter()
    for img in images:
        total.update(read_ground_truth(img, lbl_dir))
    return [total.get(i, 0) for i in range(n_classes)]


def apply_oversampling(train_imgs, class_names, max_factor=3,
                       lbl_dir=config.LBL_DIR, verbose=True):
    """
    Repeat in train.txt the images holding under-represented classes.

    No files are duplicated on disk: only lines are repeated. Each image's
    factor is set by its weakest class, capped by max_factor.

    Returns the list of paths with repetitions.
    """
    n_classes = len(class_names)
    instances = count_instances(train_imgs, n_classes, lbl_dir)
    present   = [c for c in instances if c > 0]
    if not present:
        return list(train_imgs)

    target = max(present)

    factors = {}
    for i, amount in enumerate(instances):
        if amount <= 0:
            factors[i] = 1.0
        else:
            factors[i] = min(max_factor, target / amount)

    if verbose:
        print(f'\nFactores de repetición por clase (máx {max_factor}x):')
        for i, name in enumerate(class_names):
            if instances[i] > 0:
                print(f'  {name:<18} {instances[i]:>5} inst  ->  {factors[i]:.2f}x')

    by_image = classes_per_image(train_imgs, lbl_dir)

    result = []
    for img in train_imgs:
        classes = by_image.get(img, set())
        factor  = max((factors[c] for c in classes), default=1.0)
        result.extend([img] * max(1, round(factor)))

    if verbose:
        print(f'\ntrain.txt: {len(train_imgs)} -> {len(result)} líneas '
              f'({len(train_imgs)} imágenes únicas)')

    return result


def restore_split(seed=config.SEED, verbose=True):
    """
    Rewrite train.txt without repetitions.

    Use it after an oversampling experiment: if the script is interrupted
    halfway, train.txt is left with repeated lines and the next training run
    would silently use oversampling.
    """
    generate_split(seed=seed, verbose=verbose)
    if verbose:
        print('train.txt restaurado al split original.')
