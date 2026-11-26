"""Empaqueta exclusivamente los archivos que GitHub Pages debe servir."""
from pathlib import Path
import argparse,json,shutil

def empaquetar(destination):
    root=Path(__file__).resolve().parents[1]
    dest=destination.resolve()
    if dest == root or root in dest.parents and dest.name != "dist":
        raise ValueError("Use dist/ en el proyecto o un directorio externo vacío.")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("El directorio de salida debe estar vacío.")
    dest.mkdir(parents=True,exist_ok=True)
    files=["index.html",".nojekyll","python-files.json"]
    files.extend(json.loads((root/"python-files.json").read_text(encoding="utf8")))
    files.extend(p.relative_to(root).as_posix() for p in (root/"assets").rglob("*") if p.is_file())
    for name in files:
        target=dest/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    print(f"{len(files)} archivos de producción en {dest}")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("dist"))
    empaquetar(parser.parse_args().output)
