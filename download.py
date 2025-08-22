import os
import requests
import zipfile
import tarfile

urls = {
    "train": "https://drive.google.com/file/d/1Z_O7eEl1k5zUsOKbNDBBV4YQK8Sj2f6k",
    "test": "https://drive.google.com/file/d/1NLLlFJ11ENCGvkXz6cBOsmcuGl-3uSQ6",
    "sample_submission": "https://drive.google.com/file/d/1wkRXDtRkgr6fdiqPQyuBwvWX8cC2ALWp"
}

os.makedirs("data", exist_ok=True)

def download_file(url, dest):
    print(f"Скачиваем {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Скачано: {dest}")

def extract_archive(filepath, extract_to):
    print(f"Распаковываем {filepath} ...")
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif filepath.endswith((".tar.gz", ".tgz")):
        with tarfile.open(filepath, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        print(f"Неизвестный формат архива: {filepath}")
    print(f"Распаковано в: {extract_to}")

for name, url in urls.items():
    archive_path = f"data/{name}.zip"
    if not os.path.exists(archive_path):
        download_file(url, archive_path)
    else:
        print(f"{archive_path} уже скачан.")
    extract_archive(archive_path, "data")

print("Все датасеты скачаны и распакованы!")
