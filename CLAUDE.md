# CLAUDE.md

Notas de trabajo para Claude Code. **Este archivo es local: está en
`.gitignore` y no se sube al repositorio.** La documentación de uso, la que ve
quien clona el proyecto, es `README.md` (visión general) y
`structures/README.md` (detalle de cada comando y módulo).

## Qué es esto

Segmentación de 7 clases de elementos estructurales de albañilería con YOLO
(tesis PUCP). 400 imágenes anotadas, 5 modelos entrenados en el registro.
Todo se ejecuta desde un único punto de entrada:

```bash
source .pucp-project/bin/activate
python pucp_segmentation.py <comando>
```

Comandos: `lr`, `convergence`, `train`, `benchmark`, `matrices`, `predict`,
`models`, `ranking`, `stats`, `validate`, `split`.

Hay GPU (RTX 3060 laptop, CUDA disponible). Un `model.val()` sobre las 40
imágenes de test tarda segundos; un entrenamiento completo, no.

## Convenciones que no se negocian

**Idioma.** Código y línea de comandos en inglés: módulos, funciones,
variables, docstrings, comentarios, nombres de comando y flags. Todo lo que se
imprime en pantalla, en español. Los comentarios pueden ir en español si
aclaran una decisión.

**Un entrenamiento = un modelo nuevo.** `train` reserva el siguiente `model_N`
libre mirando el CSV, los `.pt`, `history/` y `archive/`. Nunca se sobrescribe
un modelo anterior: si el `.pt` de destino existe, se aborta. Elegir modelo se
pregunta listando todos (`models.ask()`), no se adivina.

**El recomendado es siempre el rank 1**, es decir el de mejor `mask_mAP50`.
Nunca una elección manual. Si cambia, hay que actualizar a la vez el README,
el bloque destacado de `comparativa_modelos.html` y la columna `notes` del
CSV — y **recalcular las cifras del HTML** (instancias no detectadas, pérdida
por clase, pandereta→kk), porque salen de la matriz de confusión del modelo
destacado:

```python
from structures import evaluate
m = evaluate.validate('models/<pesos>.pt', split='test')
cm = evaluate.confusion_matrix(m)   # [predicho, real], último índice = fondo
```

**El registro es un leaderboard de 5** (`models.MAX_MODELS`). Al entrar uno
nuevo, el peor se archiva en `models/archive/` — pesos e historial — y sale
del CSV. No se borra nada salvo que el usuario lo pida explícitamente.

## Detalles que se olvidan

- `registry()` recalcula el `rank` desde `mask_mAP50` en memoria: rank 1 es
  siempre el mejor aunque el CSV esté editado a mano. El archivo solo se
  reescribe al registrar o re-evaluar.
- `final.py` registra el comando `train` (viene de `03_modelo_final.py`). Es
  el único módulo de `commands/` cuyo nombre no coincide con su comando.
- Un comando puede devolver un `int` y `cli.main()` lo usa como código de
  salida. `validate` devuelve 1 si encuentra problemas.
- Al añadir un comando: módulo en `commands/` con `register(subparsers)` y
  `run(args)`, más el import y la entrada en `COMMANDS` de `cli.py`.
- `models/dataset_state.txt` guarda cuántas imágenes había cuando las métricas
  estaban al día. Si el dataset crece, `ranking` avisa y `train` re-evalúa
  todos los modelos antes de rankear el nuevo.
- El optimizador se fija en AdamW a propósito: con `optimizer='auto'`
  Ultralytics sobrescribe `lr0` e invalida cualquier barrido de learning rate.
- Las métricas salen de `model.val()`, nunca de la última fila de
  `results.csv` (esa es la última época, no la mejor).
- `muro_adobe` tiene 19 instancias en todo el dataset y da 0.0000 en todos los
  modelos. En la práctica esto es un problema de 6 clases.

## Fuera de git

`.gitignore` tapa los pesos (`*.pt`), el dataset completo, el venv
(`.pucp-project/`), `runs/` y este archivo. Al proponer un commit, comprobar
que no se cuela ningún `.pt` ni imágenes del dataset.
