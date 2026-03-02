import gzip
import json
from pathlib import Path
import zipfile

from rl_hybrid.data.loaders import load_microstructure, load_cycle_summary
from rl_hybrid.data.alignment import align_datasets, validate_alignment


def _write(path: Path, rows):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')


def test_jsonl_and_gz_parse(tmp_path):
    rows = [
        {"type":"tick","ts":1,"asset":"BTC","cycle":"c1","quoteRate":0.2,"UP":{"bid":0.4,"ask":0.5},"DOWN":{"bid":0.5,"ask":0.6}},
        {"type":"end","ts":2,"asset":"BTC","cycle":"c1","quoteRate":0.2,"UP":{},"DOWN":{}},
    ]
    p1 = tmp_path/'m.jsonl'; _write(p1, rows)
    p2 = tmp_path/'m.jsonl.gz'
    with gzip.open(p2, 'wt', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')
    df = load_microstructure([p1, p2])
    assert len(df) >= 2


def test_zip_jsonl_parse(tmp_path):
    rows = [
        {"type":"tick","ts":1,"asset":"BTC","cycle":"c1","quoteRate":0.2,"UP":{"bid":0.4,"ask":0.5},"DOWN":{"bid":0.5,"ask":0.6}},
        {"type":"tick","ts":2,"asset":"BTC","cycle":"c1","quoteRate":0.3,"UP":{"bid":0.41,"ask":0.51},"DOWN":{"bid":0.49,"ask":0.59}},
    ]
    p = tmp_path / "m.jsonl"
    _write(p, rows)
    z = tmp_path / "m.jsonl.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(p, arcname="microstructure-5m.jsonl")
    df = load_microstructure([z])
    assert len(df) == 2


def test_alignment_and_checks(tmp_path):
    m = tmp_path/'m.jsonl'
    s = tmp_path/'s.jsonl'
    _write(m, [{"type":"tick","ts":1,"asset":"BTC","cycle":"c1","quoteRate":0.2,"UP":{"bid":0.4,"ask":0.5},"DOWN":{"bid":0.5,"ask":0.6}}])
    _write(s, [{"type":"cycle","asset":"BTC","cycle":"c1","winner":"UP","final":{}}])
    md = load_microstructure([m])
    sd = load_cycle_summary(s)
    checks = validate_alignment(md, sd, expected_assets=['BTC'])
    ad = align_datasets(md, sd)
    assert ad['winner'].iloc[0] == 'UP'
    assert checks['unexpected_assets'] == 0
