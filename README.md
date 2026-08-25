# pucp-yolo

Segmentación de elementos estructurales de albañilería con YOLO — PUCP.

Detecta y segmenta siete clases sobre fotografías de construcciones:
`columna`, `viga`, `sobrecimiento`, `muro_kk`, `muro_pandereta`,
`muro_revestido` y `muro_adobe`.

## Instalación

```bash
python3 -m venv .pucp-project
source .pucp-project/bin/activate
pip install -r requirements.txt
```

Con GPU, instala primero el `torch` que corresponda a tu CUDA desde
[pytorch.org](https://pytorch.org); si no, se usa la versión CPU y todo
funciona igual pero lento.

El dataset no está en el repositorio (son 366 MB). Colócalo en `dataset/` con
esta forma —`images/`, `labels/` y `data.yaml` con los nombres de las siete
clases— y genera los splits:

```
dataset/
├── images/      las fotografías
├── labels/      un .txt por imagen, formato YOLO de segmentación
└── data.yaml    names: {0: columna, 1: viga, ...}
```

```bash
python pucp_segmentation.py validate   # comprueba que todo cuadre
python pucp_segmentation.py split      # genera train/val/test
```

Los pesos `.pt` tampoco están en el repositorio: se generan con `train`, o se
copian a `models/` y se describen en `models/models.csv`.

## Uso

```bash
source .pucp-project/bin/activate
python pucp_segmentation.py --help
```

| Comando | Qué hace |
|---|---|
| `models` | Lista los modelos disponibles con sus métricas |
| `ranking` | Leaderboard por `mask_mAP50` y estado del dataset |
| `train` | Entrena un modelo **nuevo** (`model_1`, `model_2`...) y lo da de alta en el registro |
| `predict` | Dibuja máscaras y cuenta objetos detectados |
| `matrices` | Matrices de confusión de los modelos elegidos |
| `benchmark` | Compara modelos sobre un conjunto fijo de imágenes |
| `stats` | Instancias por clase, reparto de los splits y anotaciones que faltan |
| `validate` | Comprueba que imágenes y labels estén completos y bien formados |
| `lr` | Barrido de learning rate |
| `convergence` | Curva de épocas vs mAP50 |
| `split` | Regenera `train/val/test` desde la semilla fija |

Documentación completa de cada comando y de los módulos:
[`structures/README.md`](structures/README.md).

> **Idioma.** El código y la línea de comandos están en inglés: módulos,
> funciones, variables, docstrings, comentarios, nombres de comando, opciones
> y las columnas de `models/models.csv`. Todo lo que el programa imprime está
> en español.

## Modelo recomendado

El recomendado es siempre el **rank 1** del registro, es decir el de mejor
`mask_mAP50`. Hoy es `v8s_800_colab` — YOLOv8s-seg a 800 px con oversampling
3×. `ranking` lo marca con ★ y `models` lo lista primero.

```bash
python pucp_segmentation.py ranking                                  # quién va primero
python pucp_segmentation.py predict --model v8s_800_colab --split test
```

| Métrica | Valor |
|---|---|
| mAP50 detección | 0.5674 |
| mAP50 segmentación | 0.5259 |
| Tamaño | 22.8 MB |

`yolo11s_800_colab` (rank 2, 0.5236) queda a 0.0023 de mask mAP50 —dentro del
margen de ruido— y pesa un 14 % menos: 19.6 MB. Si el tamaño del modelo pesa
más que esa diferencia, es la alternativa razonable; para todo lo demás, el
criterio es el ranking.

## Estructura

```
dataset/               400 imágenes, labels y splits  (fuera del repo)
models/                pesos entrenados + models.csv + history/ + archive/
structures/            el software
benchmark_results/     comparación visual entre modelos
confusion_matrices/    matrices de confusión
comparativa_modelos.html  informe de la auditoría de modelos
pucp_segmentation.py   punto de entrada
requirements.txt       dependencias
```

## Antes de entrenar: revisar el dataset

```bash
python pucp_segmentation.py validate   # ¿está todo completo y bien formado?
python pucp_segmentation.py stats      # ¿qué hay dentro?
```

`validate` cruza imágenes y labels en las dos direcciones y lee cada `.txt`
línea a línea: `class_id` fuera del rango de `data.yaml`, polígonos de menos
de tres puntos, coordenadas impares. Sale con código 1 si encuentra algo, así
que se puede encadenar. Ultralytics se salta en silencio lo que no entiende y
entrena igual, y el síntoma aparece mucho después como una clase que no
aprende.

`stats` cuenta instancias por clase, muestra el reparto train/val/test y avisa
de las clases con menos de 50 instancias, que casi con seguridad darán 0.0 de
mAP50 — que es exactamente lo que pasa con `muro_adobe` (19 instancias).

Cuando el dataset crece, las métricas guardadas en `models.csv` dejan de ser
comparables porque se midieron sobre otro `test`. El proyecto lo detecta solo
(cuenta las imágenes y lo guarda en `models/dataset_state.txt`): `ranking` lo
avisa y `train` re-evalúa todos los modelos antes de rankear el nuevo.

## Modelos: uno nuevo por entrenamiento

`train` **nunca sobrescribe un modelo anterior**. Cada corrida reserva el
siguiente nombre libre de la serie (`model_1`, `model_2`, ...), copia sus
pesos a `models/<nombre>.pt`, guarda su historial en
`models/history/<nombre>/` y añade su fila a `models/models.csv` con las
métricas obtenidas sobre `test`.

```bash
python pucp_segmentation.py train --imgsz 800 --oversample 3   # -> model_1
python pucp_segmentation.py train --imgsz 640                  # -> model_2, el 1 sigue intacto
python pucp_segmentation.py train --no-oversample              # -> model_3, sin repetir clases débiles
```

Sin opciones, `train` pregunta la configuración una a una (arquitectura,
resolución, oversampling, épocas, learning rate y batch) con valores por
defecto razonables. En la pregunta del oversampling, `no` lo desactiva
(equivale a `--no-oversample` o `--oversample no`). Lo que se pase por línea
de comandos no se pregunta.
`train --reproduce` repite la configuración exacta del modelo rank 1, leída de
su `args.yaml`: es lo que hay que usar para reentrenar tras anotar imágenes
nuevas.

El registro es un **leaderboard de 5 modelos**. Cuando entra uno nuevo, el de
peor `mask_mAP50` se archiva en `models/archive/` —pesos e historial— y sale
del CSV; no se borra nada. Si el que sobra es el recién entrenado, se pregunta
si archivarlo o descartarlo.

Los comandos que usan un modelo ya entrenado (`predict`, `benchmark`,
`matrices`) **preguntan cuál usar** listando todos los disponibles si no se
les pasa `--model` / `--models`:

```
$ python pucp_segmentation.py predict --split test

¿Con qué modelo quieres predecir?:

 #  rank     name architecture  imgsz           data  box_mAP50  mask_mAP50
 1     1  model_2  yolo11s-seg    800  oversample_3x     0.5660      0.5236
 2     2  model_1  yolov8s-seg    640           base     0.5424      0.4806

Elige uno (número o nombre):
```

`benchmark` y `matrices` admiten varios de una vez o `todos`. Pasar
`--model`/`--models` salta la pregunta, que es lo que hay que hacer en
Colab o en un script sin terminal.

## Resultados

De 28 entrenamientos sobre 12 configuraciones, lo que movió la aguja:

| Cambio | Efecto en box mAP50 |
|---|---|
| nano → small | **+8.8 %** |
| 640 px → 800 px | **+3.2 %** |
| copy-paste 0.3 | +1.0 % en nano, nulo en small |
| oversampling 3× | +0.9 % en nano, nulo en small |
| small → medium | **−5.7 %** (sobreajuste con 275 imágenes) |

`muro_adobe` da 0.0000 en los 28 modelos por falta de instancias anotadas:
en la práctica esto es un problema de seis clases.

El informe completo de la comparativa está en `comparativa_modelos.html`.

## Notas metodológicas

- El optimizador se fija en **AdamW**. Con `optimizer='auto'` Ultralytics
  sobrescribe `lr0` y el barrido de learning rate queda invalidado.
- Las métricas salen de `model.val()`, no de la última fila de `results.csv`
  — esa es la última época, no la mejor.
- El split se regenera desde la semilla en cada corrida, así que agregar
  imágenes al dataset las incorpora automáticamente al reparto.
- Las siete imágenes de `dataset/benchmark.txt` se excluyen de
  train/val/test: ningún modelo las ve al entrenar.
