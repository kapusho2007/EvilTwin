"""Step-7 dependency rule: stages 2-5 never reference a raw dataset field or label value, and never
import config (where the paths live).
Scans the string literals in the code of every downstream module (docstrings and comments excluded)."""
import ast
import re
from pathlib import Path

import config as C

DOWNSTREAM = ["preprocess.py", "features.py", "models.py", "evaluate.py", "train.py"]
RAW = (set(C.RAW_TO_UNIFIED) | {C.LABEL_COL} | set(C.AWID3_LABELS) | set(C.RIDS_LABELS)
       | {C.RIDS_FOREIGN_TZ})


def _literals(path):
    tree = ast.parse(path.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_downstream_modules_are_dataset_agnostic():
    pkg = Path(__file__).resolve().parents[1] / "pipeline"
    checked = 0
    for name in DOWNSTREAM:
        f = pkg / name
        if not f.exists():
            continue
        checked += 1
        bad = [s for s in _literals(f) if any(r in s for r in RAW)]
        assert not bad, f"{name} references raw dataset items: {bad}"
        assert not re.search(r"^\s*(from|import)\s+config\b", f.read_text(), re.M), f"{name} imports config"
    assert checked >= 2
