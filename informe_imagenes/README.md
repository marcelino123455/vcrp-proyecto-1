# Imagenes para el informe

Generado a partir de los resultados en `results/` y `data/features/`. Cada
carpeta corresponde a un punto de la lista pedida.

| Carpeta | Contenido | Estado |
|---|---|---|
| `01_sift_keypoints/` | Keypoints SIFT sobre imagenes de ejemplo (deteccion simple y con escala/orientacion) | Listo |
| `02_bovw/` | Palabras visuales asignadas + histograma BoVW de una imagen; parches de la misma palabra visual en varias imagenes | Listo |
| `03_diagrama_flujo/` | Nota con la estructura sugerida para el diagrama SIFT->HDF5 | Pendiente (a mano, draw.io) |
| `04_tabla_mAP_representacion/` | Tabla curada (mAP por representacion, Oxford 55 queries) + tabla completa de los 75+ experimentos corridos | Listo |
| `05_mAP_vs_K_bovw/` | mAP vs tamano de vocabulario K para BoVW (128/256/512/1000/2048/4096) | Listo |
| `06_mAP_vs_K/` | mAP vs K para VLAD (32/64/128/256) | Listo |
| `07_tabla_metricas/` | mAP por normalizacion x metrica de distancia, BoVW K=1000 (grid 5x5) | Listo |
| `08_pca_whitening/` | mAP vs dimension PCA y whiten-power, descriptor global combinado | Listo |
| `09_fusion_sweep/` | Barrido de pesos en la fusion tardia VLAD + descriptor global | Listo |
| `10_mAP_por_landmark/` | AP por landmark, VLAD base vs. mejor pipeline (DBA + difusion) | Listo |
| `11_qualitative/` | Grids de mejores/peores queries por representacion (color, BoVW, VLAD base, mejor pipeline) | Listo |
| `ppt_comparativas/` | Seleccion para diapositivas: comparativa PCA whitening + ejemplos VLAD (best case / failure case) del mejor pipeline | Listo |
| `12_oxford_vs_paris/` | mAP del mejor pipeline evaluado por separado en Oxford (55 queries) y Paris (55 queries) | Listo |
| `13_top4_oxford_vs_paris/` | Las 4 mejores representaciones, cada una evaluada por separado en Oxford y en Paris | Listo |
| `14_vlad_ablacion/` | Ablacion de componentes de VLAD: RootSIFT, intra-norm, mass-norm, power (K=64, sin PCA) | Listo |
| `15_reranking_progresion/` | Progresion QE -> DBA -> Difusion sobre VLAD, hasta el mejor pipeline | Listo |

## Resumen de resultados (mAP, Oxford 55 queries salvo que se indique otra cosa)

1. **VLAD (K=256, PCA512) + DBA + Difusion** — 0.7053 (mejor pipeline del proyecto, Oxford)
2. Fusion tardia VLAD + descriptor global — 0.5337
3. VLAD (K=256, PCA512) sin re-ranking — 0.5282
4. BoVW (K=4096, TF-IDF) — 0.3136 (mejor K del barrido; K=1000 da 0.2871)
5. Descriptor global combinado (PCA512, whiten=0.5) — 0.1644
6. HOG / GIST / LBP / Color — 0.10-0.15

## Oxford vs. Paris (evaluados por separado, 55 queries cada uno)

El mismo pipeline (VLAD K=256/PCA512 + DBA + difusion) se evaluo tambien sobre
las 55 queries oficiales de Paris, sin mezclarlas con Oxford:

| Dataset | VLAD base | + DBA + Difusion |
|---|---|---|
| Oxford | 0.5282 | 0.7053 |
| Paris  | 0.5514 | **0.8119** |
| Oxford + Paris juntos (110 queries) | — | 0.7586 |

"Juntos" = las 110 queries evaluadas en una sola corrida, cada una buscando
solo dentro de su propio dataset (protocolo estandar); como ambos datasets
tienen 55 queries, el mAP conjunto es el promedio simple de los dos (0.7586).

## Top-4 representaciones, Oxford vs. Paris (por separado)

| Representacion | mAP Oxford | mAP Paris |
|---|---|---|
| VLAD + DBA + Difusion | 0.7053 | 0.8119 |
| Fusion (VLAD + Global) | 0.5337 | 0.5581 |
| VLAD (K=256, PCA512) | 0.5282 | 0.5514 |
| BoVW (K=4096) | 0.3136 | 0.3641 |

El orden entre representaciones es identico en ambos datasets, y Paris rinde
mejor en las 4 — buena evidencia de que la ganancia de VLAD y del re-ranking
no es una casualidad de Oxford, sino que generaliza. Ver `13_top4_oxford_vs_paris/`.

Paris rinde mejor en ambos casos, y el salto del re-ranking es incluso mayor
(+0.260 vs +0.177 en Oxford). El caso de fallo mas marcado en Paris es
`eiffel` (AP=0.23 en su peor query) — la Torre Eiffel se fotografia desde
angulos y distancias muy distintas (de dia, de noche, de lejos, de cerca),
mucho mas variable que los edificios de Oxford. Ver
`12_oxford_vs_paris/` y `11_qualitative/paris_mejor_pipeline/`.

El caso de fallo persistente en todas las representaciones es el landmark
`magdalen` (ver `10_mAP_por_landmark/` y los grids `peor_*` en `11_qualitative/`):
las consultas de Magdalen recuperan escenas de arboles/estanque que comparten
textura oscura y de alto contraste con la fachada bajo ciertas condiciones de
iluminacion, y ni el re-ranking lo corrige.

## Notas sobre la tabla de mAP por representacion

La tabla curada usa siempre el numero **solo-Oxford** (55 queries) de cada
representacion, tomado de su `_summary.json` (campo `by_dataset.oxford`), para
que la comparacion sea justa incluso cuando algun experimento se corrio
tambien sobre Paris. La tabla completa (`tabla_mAP_TODOS_los_experimentos.csv`)
es un volcado literal del maximo de cada `*_comparison.csv` en `results/`, tal
como pide el enunciado, sin ese ajuste.
