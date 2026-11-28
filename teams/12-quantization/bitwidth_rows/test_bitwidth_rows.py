"""Tests of bitwidth_rows.py with a fake llama-bench (no model, no llama.cpp).

    python -m pytest test_bitwidth_rows.py -q
"""
import argparse, hashlib, json, stat, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bitwidth_rows as b  # noqa: E402

TIME_L = """        0.60 real         0.19 user         0.07 sys
           512868352  maximum resident set size
                   0  average shared memory size
           346473576  peak memory footprint
"""


def test_parse_time_l():
    assert b.parse_time_l(TIME_L) == {"max_rss_bytes": 512868352, "peak_footprint_bytes": 346473576}
    assert b.parse_time_l("nothing") == {"max_rss_bytes": None, "peak_footprint_bytes": None}


def _row(n_prompt, n_gen, ts, depth=0, build=11669):
    return {"n_prompt": n_prompt, "n_gen": n_gen, "n_depth": depth, "avg_ts": ts, "stddev_ts": 1.0, "build_number": build,
            "build_commit": "620d222", "backends": "MTL", "cpu_info": "test cpu", "model_size": 300_000_000, "model_n_params": 400_000_000}


def test_pick_matches_prompt_gen_and_depth():
    rows = [_row(512, 0, 1000.0), _row(0, 128, 200.0), _row(0, 64, 150.0, depth=8192)]
    assert b.pick(rows, 512, 0)["avg_ts"] == 1000.0 and b.pick(rows, 0, 128)["avg_ts"] == 200.0
    assert b.pick(rows, 0, 64, 8192)["avg_ts"] == 150.0 and b.pick(rows, 0, 64, 0) is None


@pytest.fixture()
def fake(tmp_path):
    """A fake llama-bench that prints the JSON a real one would, depending on the arguments."""
    gguf = tmp_path / "m.gguf"; gguf.write_bytes(b"x" * 1000)
    bindir = tmp_path / "bin"; bindir.mkdir()
    script = bindir / "llama-bench"
    script.write_text('#!/usr/bin/env python3\nimport json, sys\na = sys.argv\n'
                      'deep = "-d" in a\nrows = [dict(n_prompt=0, n_gen=64, n_depth=int(a[a.index("-d")+1]), avg_ts=150.0, stddev_ts=2.0)] if deep else '
                      '[dict(n_prompt=512, n_gen=0, n_depth=0, avg_ts=1000.0, stddev_ts=5.0), dict(n_prompt=0, n_gen=128, n_depth=0, avg_ts=200.0, stddev_ts=3.0)]\n'
                      'for r in rows: r.update(build_number=11669, build_commit="620d222", backends="MTL", cpu_info="test cpu", model_size=300000000, model_n_params=400000000)\n'
                      'print(json.dumps(rows))\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    args = argparse.Namespace(llama_bin=str(bindir), ngl=99, repeats=1, depth=8192, spec_bandwidth_gbps=400.0, expect_build=11669,
                              allow_other_build=False, skip_sha=False)
    return gguf, args


def test_measure_row_reads_speed_size_and_derived_numbers(fake):
    gguf, args = fake
    res = b.measure_row({"model": "toy", "format": "Q8_0", "path": str(gguf), "sha256": hashlib.sha256(b"x" * 1000).hexdigest()}, args)
    assert res["pp512_tok_s"] == 1000.0 and res["tg128_tok_s"] == 200.0 and res["tg64_at_8192_tok_s"] == 150.0
    assert res["bits_per_weight"] == round(8 * 1000 / 400_000_000, 3) and res["file_bytes"] == 1000
    assert res["tg_times_file_gb_s"] == round(200.0 * 1000 / 1e9, 1)
    assert res["share_of_spec_bandwidth"] == round(200.0 * 1000 / 1e9 / 400.0, 3)


def test_sha_mismatch_stops_the_run(fake):
    gguf, args = fake
    with pytest.raises(SystemExit):
        b.measure_row({"model": "toy", "format": "Q8_0", "path": str(gguf), "sha256": "0" * 64}, args)
    args.skip_sha = True
    assert b.measure_row({"model": "toy", "format": "Q8_0", "path": str(gguf), "sha256": "0" * 64}, args)["file"] == "m.gguf"


def test_other_build_is_refused_unless_allowed(fake):
    gguf, args = fake
    args.expect_build = 10050
    with pytest.raises(SystemExit):
        b.measure_row({"model": "toy", "format": "Q8_0", "path": str(gguf)}, args)
    args.allow_other_build = True
    assert b.measure_row({"model": "toy", "format": "Q8_0", "path": str(gguf)}, args)["build_number"] == 11669


def test_main_writes_rows_json_and_markdown(fake, tmp_path):
    gguf, args = fake
    manifest = tmp_path / "rows.json"
    manifest.write_text(json.dumps({"machine_label": "Test Mac 64 GB", "rows": [
        {"model": "toy", "format": "Q8_0", "path": str(gguf)}, {"model": "toy", "format": "Q4_K_M", "path": str(gguf)}]}))
    out = tmp_path / "out"
    sys.argv = ["bitwidth_rows.py", "--manifest", str(manifest), "--llama-bin", args.llama_bin, "--out-dir", str(out),
                "--repeats", "1", "--spec-bandwidth-gbps", "400"]
    assert b.main() == 0
    md = (out / "rows.md").read_text()
    assert "Test Mac 64 GB" in md and "| toy | Q8_0 |" in md and "| toy | Q4_K_M |" in md and "cannot stand in" in md
    data = json.loads((out / "rows.json").read_text())
    assert len(data["rows"]) == 2 and data["machine"]["label"] == "Test Mac 64 GB"
