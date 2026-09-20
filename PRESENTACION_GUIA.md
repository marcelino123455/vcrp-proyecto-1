# Guia de presentacion - Proyecto 1 de Vision por Computador

## Objetivo de la presentacion

Presentar un sistema clasico de recuperacion de imagenes de edificios. Dada una imagen consulta, el sistema devuelve las imagenes mas parecidas del dataset Oxford/Paris Buildings.

Idea central para repetir durante la exposicion:

> La representacion local SIFT + BoVW es un buen baseline, pero pierde informacion sobre la distribucion de los descriptores. VLAD conserva esa informacion y el re-ranking mejora el orden final.

Duracion sugerida: 10-12 minutos, mas preguntas.

---

## Diapositiva 1 - Titulo

**Titulo:** Recuperacion de imagenes de edificios con metodos clasicos

**Subtitulo:** SIFT, BoVW, VLAD y re-ranking sobre Oxford Buildings y Paris Buildings

**Mostrar:** una imagen de consulta y una tira de resultados recuperados.

**Mensaje oral:**

- El problema es recuperar imagenes del mismo edificio aunque cambien el punto de vista, escala, iluminacion u oclusion.
- No usamos redes neuronales ni descriptores preentrenados.

---

## Diapositiva 2 - Problema y dataset

**Contenido:**

- Oxford Buildings: 5,062 imagenes, 11 landmarks y 55 queries.
- Paris Buildings: 6,412 imagenes, 11 landmarks y 55 queries.
- Cada query tiene imagen, bounding box y listas `good`, `ok` y `junk`.

**Mostrar:** ejemplos de landmarks y un bounding box de una query.

**Mensaje oral:**

La dificultad no es solo encontrar la misma fachada. Las imagenes pueden contener vegetacion, cielo, otras fachadas y cambios importantes de perspectiva.

**Referencia visual:**

- `informe_imagenes/01_sift_keypoints/`
- `informe_imagenes/11_qualitative/`

---

## Diapositiva 3 - Pipeline general

**Diagrama recomendado:**

```text
Imagenes
   |
   v
SIFT: keypoints + descriptores de 128 dimensiones
   |
   +--> BoVW: K-means -> histograma de palabras visuales
   |
   +--> VLAD: K-means -> residuos agregados
   |
   +--> HOG/color/LBP/GIST: descriptores globales
             |
             v
Normalizacion y similitud
             |
             v
Ranking de imagenes
             |
             v
AP, mAP, Precision@k y analisis cualitativo
```

**Archivos principales:**

- `1_feature_extraction_sift.py`
- `2_visual_vocabulary.py`
- `3_feature_extraction_global.py`
- `3c_vlad.py`
- `4_evaluate_retrieval.py`

**Mensaje oral:**

Todos los metodos terminan en un vector de longitud fija. Eso permite comparar la consulta contra toda la base y producir un ranking.

---

## Diapositiva 4 - SIFT y RootSIFT

**Contenido:**

- SIFT detecta puntos estables en escala y orientacion.
- Cada keypoint se describe con un vector de 128 valores.
- Los descriptores se guardan en HDF5 para no recalcularlos.
- RootSIFT aplica normalizacion L1 y raiz cuadrada.

Formula de RootSIFT:

```text
x_root = sqrt(x / ||x||_1)
```

**Mostrar:** keypoints dibujados sobre 2 o 3 imagenes.

**Referencia visual:**

- `informe_imagenes/01_sift_keypoints/`

**Preguntas posibles:**

- Por que SIFT: es robusto a escala, rotacion y cambios moderados de iluminacion.
- Limitacion: puede detectar mucha textura irrelevante, como hojas y ramas.

---

## Diapositiva 5 - BoVW y vocabulario visual

**Contenido:**

1. Se toma una muestra de descriptores SIFT.
2. K-means aprende `K` centroides o palabras visuales.
3. Cada descriptor se asigna al centroide mas cercano.
4. Cada imagen se representa con un histograma de `K` posiciones.

**Ejemplo:**

```text
descriptores SIFT -> palabras visuales -> [12, 4, 0, 31, ...]
```

**Configuracion recomendada:**

```bash
python 2_visual_vocabulary.py \
    --vocab-size 4096 \
    --sample-size 1000000 \
    --rootsift \
    --tfidf \
    --normalize l1 \
    --suffix k4096
```

**Punto importante:** BoVW descarta la posicion espacial. Dos edificios distintos pueden tener histogramas parecidos si contienen texturas similares.

**Referencia visual:**

- `informe_imagenes/02_bovw/`
- `informe_imagenes/05_mAP_vs_K_bovw/`

---

## Diapositiva 6 - Representaciones comparadas

**Tabla sugerida:**

| Representacion | Que conserva | mAP Oxford |
|---|---|---:|
| Color | Distribucion HSV | 0.1049 |
| LBP | Textura | 0.1072 |
| GIST | Estructura global | 0.1427 |
| HOG | Bordes y forma | 0.1467 |
| BoVW K=4096 | Frecuencia de patrones locales | 0.3136 |
| VLAD K=256 + PCA | Residuos locales | 0.5282 |
| VLAD + DBA + difusion | Residuos + re-ranking | **0.7053** |

**Mensaje oral:**

Los descriptores globales son utiles como baselines, pero el cambio importante ocurre al modelar patrones locales. VLAD supera a BoVW porque conserva mas informacion que un simple conteo.

**Fuente:**

- `informe_imagenes/04_tabla_mAP_representacion/tabla_mAP_por_representacion.csv`

---

## Diapositiva 7 - VLAD y PCA whitening

**Contenido:**

VLAD asigna cada descriptor a un centroide y acumula el residuo:

```text
v_k = sum(x - c_k)
```

A diferencia de BoVW:

- BoVW guarda cuantos descriptores caen en cada centroide.
- VLAD guarda tambien en que direccion se alejan del centroide.

La configuracion principal fue:

```text
K = 256
RootSIFT = si
intra-normalization = si
PCA = 512 dimensiones
whiten-power = 0.25
```

**Preguntas posibles:**

- PCA reduce dimension y correlacion entre componentes.
- La normalizacion intra-bloque evita que una textura repetitiva domine el vector.

**Referencia visual:**

- `informe_imagenes/ppt_comparativas/pca_whitening_comparativa.png`

---

## Diapositiva 8 - Similitud y re-ranking

**Base:** similitud coseno entre vectores normalizados.

**Re-ranking evaluado:**

- Query Expansion: incorpora vecinos de la consulta.
- DBA, `n=5`: incorpora vecinos a cada vector de la base.
- Difusion, `k=100`: propaga similitud por un grafo de vecinos.

**Resultados de la progresion:**

| Etapa | mAP |
|---|---:|
| VLAD base | 0.5282 |
| + Query Expansion | 0.5545 |
| + DBA | 0.5817 |
| + Difusion | 0.6780 |
| + DBA(5) + Difusion(100), seeds=5 | **0.7053** |

**Mensaje oral:**

La difusion ayuda porque imagenes del mismo edificio pueden formar una cadena de vistas parecidas. Una imagen puede no parecerse mucho directamente a otra, pero ambas pueden conectarse mediante vecinos intermedios.

**Fuente:**

- `informe_imagenes/15_reranking_progresion/reranking_progresion.csv`

---

## Diapositiva 9 - Evaluacion

**Definiciones:**

- `Precision@k`: fraccion de imagenes relevantes dentro de las primeras `k`.
- `AP`: calidad del ranking para una query.
- `mAP`: promedio del AP sobre todas las queries.

Formula de Precision@k:

```text
Precision@k = relevantes en los primeros k / k
```

El proyecto reporta tanto la AP oficial de VGG como `mAP_simple`, la formula del enunciado. Para comparar experimentos se debe usar siempre el mismo protocolo.

**Detalles del protocolo:**

- Las imagenes `junk` se eliminan del ranking.
- Las queries se evalúan contra su propio dataset.
- Se utiliza la base completa, no solo el top-5, para calcular AP.

**Comando:**

```bash
python 4_evaluate_retrieval.py \
    --embeddings data/features/vlad/vlad_k256_pca512.npz \
    --name vlad_best \
    --normalize none \
    --metric cosine \
    --datasets oxford \
    --dba 5 \
    --diffusion 100 \
    --diffusion-seeds 5
```

---

## Diapositiva 10 - Analisis Oxford vs Paris

**Tabla:**

| Dataset | VLAD base | VLAD + DBA + difusion |
|---|---:|---:|
| Oxford | 0.5282 | 0.7053 |
| Paris | 0.5514 | **0.8119** |
| Oxford + Paris | - | 0.7586 |

**Mensaje oral:**

La mejora no aparece solo en Oxford. Tambien se observa en Paris, lo que indica que el re-ranking generaliza dentro de este protocolo.

**Fuente:**

- `informe_imagenes/12_oxford_vs_paris/oxford_vs_paris.csv`

---

## Diapositiva 11 - Casos exitosos y fallos

**Caso exitoso:**

- `vlad_best_case_1_bodleian_AP0.975.png`
- `vlad_best_case_2_keble_AP0.959.png`

Explicar que las vistas comparten estructura de fachada y que los patrones locales se mantienen pese al cambio de captura.

**Caso de fallo:**

- `vlad_failure_case_1_christchurch_AP0.022.png`
- `vlad_failure_case_2_magdalen_AP0.036.png`

Explicar que la vegetacion, fondos con textura y cambios de vista producen descriptores parecidos a los de otros landmarks.

**Conclusiones de error:**

- Magdalen es dificil en casi todas las representaciones.
- El bbox puede ayudar cuando el edificio ocupa una parte pequena de la imagen.
- El bbox no siempre mejora: eliminar demasiado contexto tambien puede perjudicar.

---

## Diapositiva 12 - Conclusiones y trabajo futuro

**Conclusiones:**

1. SIFT es una buena base para edificios porque representa patrones locales.
2. BoVW es eficiente, pero pierde informacion espacial y de residuos.
3. VLAD mejora claramente a BoVW.
4. DBA y difusion mejoran el ranking sin usar aprendizaje profundo.
5. La evaluacion debe combinar mAP, Precision@k y ejemplos visuales.

**Trabajo futuro:**

- Verificacion geometrica con Lowe ratio test y RANSAC.
- Uso sistematico del bounding box en las queries.
- ASMK o Fisher Vectors.
- Indexacion invertida para escalar el retrieval.

**Cierre recomendado:**

> El resultado mas importante no es solo que VLAD tenga mayor mAP, sino entender por que: conserva informacion que BoVW descarta y el re-ranking explota la estructura de vecinos del conjunto.

---

## Checklist de la rubrica

- [x] Problema y dataset explicados.
- [x] Representacion local: SIFT.
- [x] Representaciones globales: HOG, color, LBP y GIST.
- [x] Vocabulario visual construido con K-means.
- [x] Histograma fijo por imagen.
- [x] Metricas de similitud comparadas.
- [x] Efecto de K analizado.
- [x] Ranking top-5 disponible.
- [x] mAP, AP y Precision@k reportados.
- [x] Comparacion cuantitativa entre representaciones.
- [x] Casos de exito y fallo.
- [x] Restriccion de no usar deep learning respetada.
- [x] Codigo reproducible disponible.

---

## Preguntas probables del profesor

### Por que BoVW no es suficiente?

Porque representa cada imagen como un histograma de frecuencias y pierde donde aparecen los patrones. VLAD conserva el residuo de cada descriptor respecto a su centroide.

### Por que no usar solo el descriptor con mayor mAP?

Porque hay que demostrar el analisis experimental pedido: comparar representaciones, estudiar parametros y justificar las decisiones.

### Que significa K en BoVW o VLAD?

Es el numero de centroides del vocabulario. En BoVW define el numero de bins del histograma. En VLAD define el numero de bloques de residuos.

### Por que no siempre un K mayor es mejor?

Un K mayor puede ser mas discriminativo, pero necesita mas memoria y mas datos por cluster. Con pocos descriptores por cluster el vocabulario se vuelve inestable.

### Que diferencia hay entre AP y mAP?

AP mide una query. mAP es el promedio de AP sobre todas las queries.

### Por que el resultado del informe puede diferir de otra implementacion?

Porque cambiar el conjunto de queries, el uso del bounding box, el tratamiento de `junk`, la formula de AP o el ranking altera el mAP. Todos los modelos deben evaluarse con el mismo protocolo.

### El re-ranking usa deep learning?

No. DBA y difusion son operaciones sobre vectores y grafos construidos con similitudes; son metodos clasicos y training-free.

---

## Orden de demostracion en vivo

Si hay tiempo para una demostracion:

```bash
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate vcrp-proyecto-1

python 4_evaluate_retrieval.py \
    --embeddings data/features/vlad/vlad_k256_pca512.npz \
    --name demo_vlad \
    --normalize none \
    --metric cosine \
    --datasets oxford \
    --dba 5 \
    --diffusion 100 \
    --diffusion-seeds 5
```

Para mostrar solo una query visualmente, usa los resultados cualitativos ya generados en `informe_imagenes/11_qualitative/` o las imágenes de `informe_imagenes/ppt_comparativas/`.
