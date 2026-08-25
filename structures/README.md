# structures — Segmentación de elementos estructurales

Software unificado que reemplaza los scripts sueltos del proyecto.
Todo se ejecuta desde `pucp_segmentation.py`.

```bash
source .pucp-project/bin/activate
python pucp_segmentation.py --help
```

> **Idioma.** El código y la línea de comandos están en inglés: nombres de
> módulo, funciones, variables, docstrings, comentarios, nombres de comando y
> opciones, además de las columnas de `models/models.csv` y de los CSV que
> generan los comandos. Todo lo que se imprime en pantalla está en español,
> que es el idioma de trabajo del proyecto.

## Comandos

| Comando | Reemplaza a | Qué hace |
|---|---|---|
| `lr` | `01_lr_determination.py` | Barrido de learning rate |
| `convergence` | `02_convergencia.py` | Curva de épocas vs mAP50 |
| `train` | `03_modelo_final.py` | Modelo nuevo (`model_N`) + métricas + matriz de confusión |
| `benchmark` | `evaluar_benchmark.py` | Compara los modelos elegidos sobre `benchmark.txt` |
| `matrices` | `generar_matrices.py` | Matrices de confusión de los modelos elegidos |
| `predict` | `generar_mascaras.py`, `probar_modelo.py` | Dibuja máscaras y cuenta objetos |
| `models` | — | Lista los modelos disponibles |
| `ranking` | — | Leaderboard por `mask_mAP50` y estado del dataset |
| `stats` | — | Instancias por clase, splits y anotaciones que faltan |
| `validate` | — | Integridad de imágenes y labels antes de entrenar |
| `split` | — | Regenera `train/val/test` desde la semilla |

## Ejemplos

```bash
# Ver qué modelos hay y cómo van
python pucp_segmentation.py models
python pucp_segmentation.py ranking

# Revisar el dataset antes de entrenar
python pucp_segmentation.py stats
python pucp_segmentation.py validate

# Entrenar preguntando la configuración (menú interactivo)
python pucp_segmentation.py train

# Entrenar sin preguntas, dando los valores
python pucp_segmentation.py train --model yolo11s-seg.pt --imgsz 800 --oversample 3

# Entrenar sin oversampling (también vale --oversample no, o responder "no" al menú)
python pucp_segmentation.py train --no-oversample

# Repetir la configuración del modelo rank 1 (por ejemplo tras añadir imágenes)
python pucp_segmentation.py train --reproduce

# Comparar modelos sobre las imágenes de benchmark
python pucp_segmentation.py benchmark --conf 0.3

# Matrices de confusión de todos los modelos
python pucp_segmentation.py matrices --split test

# Predecir sobre imágenes sueltas (pregunta qué modelo usar)
python pucp_segmentation.py predict --images foto1.jpg foto2.jpg

# Arreglar train.txt si quedó con oversampling
python pucp_segmentation.py split --restore
```

## Módulos

```
structures/
├── config.py        rutas, hiperparámetros, carga de clases desde data.yaml
├── data.py          split reproducible, oversampling, lectura de ground truth
├── models.py        registro de pesos, leaderboard y archivado
├── train.py         envoltura única sobre YOLO.train()
├── evaluate.py      model.val(), métricas por clase, matriz de confusión
├── draw.py          dibujado de máscaras sobre las imágenes
├── plots.py         todas las gráficas
├── cli.py           definición de comandos y opciones
└── commands/        un módulo por comando
    ├── lr.py            comando `lr`
    ├── convergence.py   comando `convergence`
    ├── final.py         comando `train`
    ├── benchmark.py     comando `benchmark`
    ├── matrices.py      comando `matrices`
    ├── predict.py       comando `predict`
    ├── ranking.py       comando `ranking`
    ├── stats.py         comando `stats`
    └── validate.py      comando `validate`
```

Cada módulo de `commands/` expone `register(subparsers)` y `run(args)`. El
nombre del archivo coincide con el del comando salvo en `final.py`, que
registra `train` por razones históricas (era `03_modelo_final.py`).

Un comando puede devolver un entero y `cli.main()` lo usa como código de
salida del proceso: `validate` devuelve 1 cuando encuentra problemas, para
poder encadenarlo en un script. Los demás devuelven `None` y salen con 0.

## Modelos

Los pesos viven en `models/` y se describen en `models/models.csv`.
Los comandos `benchmark`, `matrices` y `predict` los leen de ahí, no de las
carpetas `runs*`, así que esas se pueden borrar sin romper nada.

**Un entrenamiento = un modelo nuevo.** `train` llama a
`models.next_model_name()`, que devuelve el primer número libre de la serie
`model_N` mirando a la vez el CSV, los `.pt` de `models/`, las carpetas de
`history/` y las de `archive/`. Al terminar, `models.register_new()` copia los
pesos, guarda el historial y escribe la fila del CSV recalculando el `rank`
por `mask_mAP50`. Si el `.pt` de destino ya existiera, aborta en vez de
sobrescribir. `--name` permite forzar otro nombre.

**El registro es un leaderboard de 5.** `models.MAX_MODELS` fija el tamaño.
Cuando entra un modelo nuevo y el CSV pasa de 5 filas, el de peor
`mask_mAP50` se **archiva**: sus pesos van a `models/archive/`, su historial a
`models/archive/<nombre>/` y su fila desaparece del CSV. No se borra nada, y
su número de serie no se reutiliza. Si el que sobra es justo el recién
entrenado —no superó a ninguno— se pregunta si archivarlo o descartarlo
(borrarlo sin guardar); sin terminal interactiva se archiva por defecto.

**Elegir modelo se pregunta, no se adivina.** `models.ask()` lista el registro
numerado y lee la respuesta: número, nombre, varios separados por espacios
(`benchmark`, `matrices`) o `todos`. `models.ask_one()` es la variante de un
solo modelo que usa `predict`. Pasar `--model`/`--models` salta la pregunta.
Sin terminal interactiva (Colab, cron, pipes) los comandos en lote usan todos
los modelos y `predict` aborta pidiendo `--model` explícito, en vez de
quedarse colgado esperando una respuesta.

Para agregar un modelo entrenado fuera: copia el `.pt` a `models/` y añade una
fila al CSV con las columnas `rank,name,file,architecture,imgsz,data,
box_mAP50,mask_mAP50,origin,notes`.

`models/history/<nombre>/` guarda, por cada modelo, lo que hace falta para
revisarlo después sin volver a entrenar:

| Archivo | Qué es |
|---|---|
| `results.csv` | métricas época a época del entrenamiento |
| `confusion_matrix.png` | matriz de confusión (conteos) |
| `confusion_matrix_normalized.png` | la misma, normalizada por fila |
| `MaskPR_curve.png`, `MaskF1_curve.png` | curvas de segmentación |
| `BoxPR_curve.png` | curva de detección |
| `results.png` | panel de pérdidas y métricas |
| `args.yaml` | configuración exacta con la que se entrenó |

## Cuando el dataset cambia

Las métricas del CSV se midieron sobre un `test` concreto. Si se anotan
imágenes nuevas, el split se rehace y esas métricas dejan de ser comparables
entre sí.

`models.dataset_fingerprint()` cuenta las imágenes de `dataset/images` y
`models/dataset_state.txt` guarda el valor de la última vez que las métricas
estuvieron al día. `models.needs_reevaluation()` compara los dos números.
Cuando no coinciden:

- `ranking` lo avisa y marca el orden como no fiable;
- `train` ejecuta `models.reevaluate_all()` **antes** de registrar el modelo
  nuevo, así que el ranking compara todo sobre el mismo `test`.

`reevaluate_all(split='test')` vuelve a pasar `model.val()` por cada modelo
del registro, actualiza `box_mAP50` y `mask_mAP50`, recalcula los `rank` y
reescribe el fingerprint. Las filas cuyos pesos no estén en disco se dejan
intactas en vez de desaparecer del CSV.

## Los dos modos de `train`

| Modo | Cuándo | Qué hace |
|---|---|---|
| `--reproduce` | Añadiste imágenes y quieres repetir la configuración ganadora | Lee `models/history/<rank 1>/args.yaml` y reentrena con esos hiperparámetros |
| default | Modelo nuevo | Pregunta lo que no venga por línea de comandos |

En modo `--reproduce` el oversampling no sale de `args.yaml` —se aplica a
`train.txt` antes de entrenar, Ultralytics no lo ve— sino de la columna `data`
del registro (`oversample_3x`). Lo que se pase por línea de comandos gana
sobre lo reproducido: `--reproduce --epochs 100` repite el resto y entrena más
tiempo.

En modo default se pregunta **solo lo que falta**: `train --imgsz 800` salta
esa pregunta y hace las demás. El menú del oversampling ofrece `no / 2 / 3`:
`no` entrena sin repetir nada, y la columna `data` del registro queda en
`base` en vez de `oversample_Nx`. Enter acepta el valor por defecto
(`yolov8s-seg.pt`, 800, oversampling 3, 70 épocas, `lr0` 0.002, batch 8), que
es también lo que se usa cuando no hay terminal interactiva. Se aceptan
valores fuera de la lista si son del tipo correcto (120 épocas, por ejemplo);
`auto` en batch se traduce al `-1` de Ultralytics.

## Notas metodológicas

- **El optimizador se fija en AdamW.** Con `optimizer='auto'` Ultralytics
  sobrescribe `lr0` con su propio valor: distintos learning rates producen
  curvas idénticas y el barrido queda invalidado.
- **Las métricas salen de `model.val()`**, no de la última fila de
  `results.csv` — esa es la última época, no la mejor.
- **`per_class_metrics.csv` contiene métricas de máscara**, no de caja.
- **El split se regenera desde la semilla** en cada corrida: reproducible, y
  si agregas imágenes entran solas al reparto.
- **El conjunto de benchmark es fijo.** `benchmark.txt` lista 7 imágenes que
  se excluyen de train/val/test: ningún modelo las ve al entrenar, y todos se
  comparan sobre exactamente las mismas. Las entradas se resuelven por nombre
  de archivo contra `dataset/images/`, así que el archivo funciona igual en
  local y en Colab aunque guarde rutas absolutas. Si falta alguna, el comando
  aborta en vez de comparar sobre un conjunto incompleto.
- **Los entrenamientos no se pisan.** `train` numera cada modelo y aborta
  antes de sobrescribir; `lr` y `convergence` escriben en una carpeta
  `runs_experimento_*_N` nueva por corrida.
- **El oversampling se restaura siempre**, incluso si el entrenamiento falla
  a mitad — si no, el siguiente run usaría oversampling sin que se note.
- **`muro_adobe` da 0.0000 en todos los modelos**: sin instancias suficientes.
  En la práctica esto es un problema de 6 clases.
