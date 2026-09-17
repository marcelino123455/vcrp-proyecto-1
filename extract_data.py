"""Extrae el dataset Oxford & Paris Buildings descargado de Kaggle.

Estructura esperada de entrada:
    data/archive (1).zip
        oxbuild_images.tgz
        paris_1.tgz
        paris_2.tgz

Estructura de salida:
    data/
        oxford/*.jpg
        paris/<landmark>/*.jpg

Los archivos intermedios (.zip y .tgz) NO se eliminan.
"""

import tarfile
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ZIP_GLOB_PATTERN = "archive*.zip"

TGZ_TO_OUTPUT_DIR = {
    "oxbuild_images.tgz": "oxford",
    "paris_1.tgz": "paris",
    "paris_2.tgz": "paris",
}


def find_archive_zip() -> Path:
    matches = sorted(DATA_DIR.glob(ZIP_GLOB_PATTERN))
    if not matches:
        raise FileNotFoundError(
            f"No se encontro ningun zip que coincida con '{ZIP_GLOB_PATTERN}' en {DATA_DIR}"
        )
    return matches[0]


def extract_tgz_members(zip_path: Path, tgz_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(tgz_name) as tgz_stream:
            with tarfile.open(fileobj=tgz_stream, mode="r|gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    # Aplana la jerarquia interna del tar (p.ej. "paris/defense/x.jpg")
                    # y conserva solo el nombre de archivo dentro de output_dir.
                    dest_name = Path(member.name).name
                    dest_path = output_dir / dest_name
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with open(dest_path, "wb") as dst:
                        dst.write(src.read())


def main() -> None:
    zip_path = find_archive_zip()
    print(f"Usando archivo zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        available = set(zf.namelist())

    for tgz_name, output_subdir in TGZ_TO_OUTPUT_DIR.items():
        if tgz_name not in available:
            print(f"Aviso: '{tgz_name}' no esta presente en {zip_path.name}, se omite.")
            continue

        output_dir = DATA_DIR / output_subdir
        print(f"Extrayendo {tgz_name} -> {output_dir}/ ...")
        extract_tgz_members(zip_path, tgz_name, output_dir)
        n_files = sum(1 for _ in output_dir.iterdir())
        print(f"  listo ({n_files} archivos en {output_dir})")

    print("Extraccion completa. Los archivos .zip y .tgz originales se conservaron.")


if __name__ == "__main__":
    main()
