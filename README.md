# VCRP - Proyecto 1

Extracción de descriptores SIFT sobre el dataset Oxford & Paris Buildings, pensado
para correr en el cluster Khipu (Miniconda + SLURM).

## Pasos

### 1) Configurar el ambiente

```bash
./setup.sh
```

Crea (si no existe) el ambiente conda `vcrp-proyecto-1` con Python 3.11 e
instala las dependencias de `requirements.txt`.

### 2) Descargar el dataset

```bash
./data/download_dataset.sh
```

Descarga el dataset `skylord/oxbuildings` desde Kaggle vía la API REST y lo
descomprime en `data/oxbuildings/`.

Requiere credenciales de Kaggle: exporta `KAGGLE_USERNAME` y `KAGGLE_KEY`, o
coloca tu token en `~/.kaggle/kaggle.json` (Kaggle > Account > Create New Token).

### 3) Descomprimir y organizar las imágenes

```bash
./data/unzip_dataset.sh
```

Descomprime los `.tgz` de `data/oxbuildings/` y separa las imágenes en
`data/oxford/*.jpg` y `data/paris/*.jpg` (aplanando la estructura interna de
Paris).

### 4) Extraer descriptores SIFT

Prueba rápida (pocas imágenes, para verificar que todo funciona):

```bash
python 1_feature_extraction_sift.py --limit 20
```

Extracción completa (local):

```bash
python 1_feature_extraction_sift.py --workers 16
```

Extracción completa en Khipu vía SLURM:

```bash
sbatch run_sift_extraction.slurm
```

Genera:
- `data/features/sift/sift_features.h5` — keypoints y descriptores SIFT por imagen.
- `data/features/sift/sift_index.csv` — índice (`image_id, dataset, filename, num_keypoints`).

Ambos archivos también están subidos como dataset público de Kaggle:
https://www.kaggle.com/datasets/marcelinomaitavargas/vcrp-sift-features

## Imágenes problemáticas conocidas
**Corruptas en el dataset original** (20, todas en Paris): en vez del `.jpg` real,
el archivo contiene una página HTML de error ("302 Found") — no se pudo leer con
`cv2.imread` y no quedan en `sift_features.h5` ni en `sift_index.csv`:

```
paris_louvre_000136.jpg
paris_louvre_000146.jpg
paris_moulinrouge_000422.jpg
paris_museedorsay_001059.jpg
paris_notredame_000188.jpg
paris_pantheon_000284.jpg
paris_pantheon_000960.jpg
paris_pantheon_000974.jpg
paris_pompidou_000195.jpg
paris_pompidou_000196.jpg
paris_pompidou_000201.jpg
paris_pompidou_000467.jpg
paris_pompidou_000640.jpg
paris_sacrecoeur_000299.jpg
paris_sacrecoeur_000330.jpg
paris_sacrecoeur_000353.jpg
paris_triomphe_000662.jpg
paris_triomphe_000833.jpg
paris_triomphe_000863.jpg
paris_triomphe_000867.jpg
```

**Se leyeron bien pero SIFT no detectó ningún keypoint** (0 descriptores): sí quedan
en el HDF5/CSV, con `num_keypoints == 0`:

```
oxford/ashmolean_000214.jpg
```

## Estructura de datos

```
data/
  oxbuildings/            # .tgz descargados de Kaggle
  oxford/*.jpg
  paris/*.jpg
  features/sift/
    sift_features.h5
    sift_index.csv
```
