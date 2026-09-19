# Harness de evaluación

Todo lo que se mide en este proyecto pasa por aquí. La nota de desempeño es
**8 de 20 puntos** y se calcula de forma **relativa entre grupos**:

```
G_performance = 8 · (mAP_i − mAP_worst) / (mAP_best − mAP_worst)
```

El peor grupo saca 0. Por eso dos cosas importan más que el código bonito:

1. Que nuestro mAP esté **bien calculado** (un bug aquí nos hunde o nos infla,
   y el enunciado avisa que las inconsistencias en la evaluación se penalizan).
2. Que **todos los experimentos del grupo** se midan con el mismo código, para
   que la tabla comparativa del reporte signifique algo.

## Archivos

| Archivo | Qué hace |
|---|---|
| `data/download_groundtruth.sh` | Descarga el ground truth oficial de VGG |
| `evaluation.py` | Parser del ground truth + AP, mAP, P@k, R@k |
| `retrieval.py` | Carga de embeddings, normalizaciones, métricas, fusión tardía |
| `4_evaluate_retrieval.py` | CLI: evalúa cualquier representación y hace barridos |
| `qualitative.py` | Grids de top-k con aciertos y fallos marcados |
| `test_evaluation.py` | 33 tests, incluyendo AP calculado a mano |
| `run_evaluation.slurm` | Corrida completa en Khipu |

## Puesta en marcha

```bash
./data/download_groundtruth.sh   # 55 queries por dataset
python evaluation.py --check     # valida el parseo
python test_evaluation.py        # valida las métricas
```

Los tres deben salir en verde antes de reportar cualquier número.

## Uso

El contrato es un `.npz` con `image_ids` (formato `"<dataset>/<image_id>"`) y
una matriz `(N, D)`. Eso es exactamente lo que ya guarda
`2_visual_vocabulary.py`, así que funciona sin tocar nada:

```bash
python 4_evaluate_retrieval.py \
    --embeddings data/features/vocabulary/image_histograms.npz \
    --name bovw_k500
```

Barrido de normalización × métrica (esto genera directo una tabla del reporte):

```bash
python 4_evaluate_retrieval.py \
    --embeddings data/features/vocabulary/image_histograms.npz \
    --name bovw_k500 \
    --normalize none l1 l2 power hellinger \
    --metric cosine l2 chi2 hellinger intersection \
    --quiet
```

Análisis cualitativo (las 6 mejores y 6 peores queries, con top-5 cada una):

```bash
python 4_evaluate_retrieval.py --embeddings ... --qualitative 6
```

En el cluster:

```bash
sbatch run_evaluation.slurm data/features/vocabulary/image_histograms.npz bovw_k500
```

## Salidas

Todo cae en `results/`:

- `<run>__<norm>__<metric>_per_query.csv` — AP y P@k de cada una de las 110 queries
- `<run>__<norm>__<metric>_summary.json` — resumen con desglose por landmark
- `<run>_comparison.csv` — la tabla comparativa de todas las configuraciones
- `<run>__..._rankings.txt` — top-100 por query, para reproducibilidad
- `qualitative/<run>/*.png` — los grids de top-k

## Decisiones de protocolo que el grupo debe fijar

Estas tres cambian el mAP varios puntos. No importa tanto cuál se elija, sí
importa que sea **la misma en todos los experimentos** y que quede escrita en
el reporte.

**AP oficial vs. la fórmula del PDF.** El benchmark de VGG interpola la curva
precision-recall con trapecios; el enunciado escribe la suma simple
`AP = (1/N) Σ P(k)·rel(k)`. Dan valores distintos — para `positivos={b}` y
ranking `[a, b]`, la oficial da 0.25 y la del PDF 0.5. El harness reporta las
dos (`mAP` y `mAP_simple`). Sugerencia: optimizar contra la oficial, reportar
ambas, y preguntarle al profe cuál usará él para el ranking entre grupos.

**Las imágenes `junk`.** El protocolo estándar las **elimina del ranking**: no
cuentan como acierto ni como error. Tratarlas como negativas baja el mAP
artificialmente. El harness las descarta siempre; si otro grupo no lo hace, sus
números no son comparables con los nuestros.

**La imagen consulta.** Está dentro de la base de datos y normalmente sale en
el puesto 1, lo que regala precisión. El benchmark original **no la quita**, así
que ese es el default aquí; `--exclude-query` la saca. El runner avisa cuántas
queries aparecen en su propio conjunto de positivos.

Una cuarta, menor: `--database same` (default) rankea las queries de Oxford
solo contra imágenes de Oxford, que es el protocolo publicado. `--database all`
mete el otro dataset como distractores — más difícil, y es el escenario que
menciona el enunciado. Vale como experimento extra en el reporte.

## Lo que falta y es fácil de olvidar

El ground truth trae un **bounding box** por query: el objeto de interés es un
recorte, no la foto completa. El protocolo publicado extrae los descriptores
**solo dentro del bbox**. Aquí, al partir de embeddings ya calculados sobre la
imagen entera, no se aplica — y eso cuesta mAP. Si queremos ese punto extra hay
que re-extraer features de los 110 recortes y pasarlos aparte. El harness ya
guarda el bbox en `Query.bbox` y `qualitative.py` lo dibuja, así que el gancho
está puesto.

## Query expansion

El harness puede aplicar **alpha-weighted Query Expansion** a cualquier
representación, con `--query-expansion N --qe-alpha A`:

```bash
python 4_evaluate_retrieval.py --embeddings ... --metric cosine --query-expansion 10
```

Promedia la consulta con sus N primeros resultados, pesando cada vecino por
su similitud elevada a alpha, y re-rankea. Con `--qe-alpha 0` se recupera la
Average QE clásica.

Solo funciona con métricas de producto punto (`cosine`, `dot`), porque la
expansión promedia vectores; con cualquier otra el runner aborta con un
mensaje explícito en vez de devolver un número sin sentido.

Dos advertencias que conviene reportar. La QE clásica verifica espacialmente
los vecinos antes de promediarlos; aquí se usa el top-N crudo, así que si la
búsqueda inicial es mala la expansión la empeora — es el fenómeno de *query
drift*, y se ve claro al subir N. Y no usa el ground truth en ningún momento,
solo los resultados del propio sistema, así que es una técnica legítima y no
una fuga de información.

## Re-ranking: DBA y difusión

Dos técnicas más, ambas *training-free* y agnósticas al descriptor, así que
sirven para BoVW, para los globales y para VLAD por igual.

**Database-side augmentation** (`--dba N`) reemplaza cada vector de la base
por una mezcla ponderada consigo mismo y sus N vecinos. Es la QE aplicada del
otro lado, se calcula una sola vez sin ver las consultas, y no cuesta nada en
tiempo de búsqueda.

**Difusión** (`--diffusion K`) construye un grafo k-NN de la base y propaga la
similitud por él. La idea es que las imágenes de un mismo landmark no forman
una bola en el espacio de descriptores sino una *variedad* alargada: una
cadena de vistas donde cada una se parece a la siguiente pero los extremos no
se parecen entre sí. La distancia directa no puede recorrer esa cadena; la
difusión sí.

```bash
python 4_evaluate_retrieval.py --embeddings ... --metric cosine --dba 3
python 4_evaluate_retrieval.py --embeddings ... --metric cosine --diffusion 50
```

La difusión solo rinde si la base es grande y tiene esa estructura de
variedad. En bases chicas el grafo conecta clases distintas de inmediato y
empeora el resultado — vale la pena reportarlo como resultado negativo si
ocurre.

## Extender esto

Para evaluar un descriptor nuevo basta con guardar el `.npz` con el contrato de
arriba. Para fusionar dos representaciones, `retrieval.fuse()` normaliza los
scores por query a [0,1] y los suma con pesos — necesario porque un coseno de
BoVW y una distancia χ² de HOG no viven en la misma escala.
