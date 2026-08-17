from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
EXTERNAL = ROOT / "external"
STRUCTURES = DATA / "structures"

for d in (RESULTS, FIGURES, STRUCTURES):
    d.mkdir(parents=True, exist_ok=True)
