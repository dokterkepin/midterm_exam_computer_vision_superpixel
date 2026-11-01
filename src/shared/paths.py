"""Project folder layout, so scripts work from any working directory.

dataset/<wall>/image.png    main-camera view (all walls; wall.png also accepted)
dataset/<wall>/scene.json   simulator ground truth (synthetic walls)
dataset/<wall>/stereo.png   side-by-side stereo view, left eye | right eye (optional)
dataset/<wall>/camera.json  stereo camera height / distance (optional)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
OUT = ROOT / "outputs"


def _file(name, filename):
    p = DATASET / name / filename
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return str(p)


def image_path(name):
    """Main view; called image.png or wall.png."""
    if (DATASET / name / "wall.png").exists():
        return _file(name, "wall.png")
    return _file(name, "image.png")


def gt_path(name):
    return _file(name, "scene.json")


def stereo_path(name):
    return _file(name, "stereo.png")


def camera_path(name):
    return _file(name, "camera.json")


def synthetic_walls():
    """Names of the dataset folders that have a scene.json and a main image, in order."""
    return sorted(p.name for p in DATASET.iterdir()
                  if (p / "scene.json").exists() and ((p / "image.png").exists() or (p / "wall.png").exists()))


def real_walls():
    """Names of the real wall photos (dataset folders called real_wall*), in order."""
    return sorted(p.name for p in DATASET.iterdir() if p.name.startswith("real_wall") and (p / "image.png").exists())


def out_path(wall, *parts):
    """Path inside outputs/<wall>/ (mirrors dataset/<wall>/), creating the folder if needed.

    out_path(wall, "sweep.csv") is shared by the wall; out_path(wall, algo, "output.json") belongs to
    one superpixel algorithm (outputs/<wall>/<algo>/).
    """
    p = OUT.joinpath(wall, *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)
