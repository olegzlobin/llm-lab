import subprocess
import sys
from pathlib import Path

from llmlab.params import compare

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "baseline.yaml"


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(f"extends: {BASE}\nname: {name}\nmodel:\n{body}\n")
    return str(p)


def test_compare_ok_and_fail(tmp_path):
    ok = write(tmp_path, "ok.yaml", "  norm: rmsnorm")
    big = write(tmp_path, "big.yaml", "  n_layer: 8")
    assert compare(str(BASE), [ok])[1]
    rows, good = compare(str(BASE), [ok, big])
    assert not good and rows[-1][3] > 0.05


def test_script_exit_codes(tmp_path):
    script = str(ROOT / "scripts" / "compare_params.py")
    big = write(tmp_path, "big.yaml", "  n_layer: 8")
    ok = write(tmp_path, "ok.yaml", "  ffn: swiglu")
    r = subprocess.run([sys.executable, script, "--base", str(BASE), big], capture_output=True, text=True)
    assert r.returncode == 1 and "FAIL" in r.stdout
    r = subprocess.run([sys.executable, script, "--base", str(BASE), ok], capture_output=True, text=True)
    assert r.returncode == 0


def test_all_variants_within_5_percent():
    variants = sorted(str(p) for p in (ROOT / "configs" / "variants").glob("*.yaml"))
    assert variants
    rows, ok = compare(str(BASE), variants)
    assert ok, rows
