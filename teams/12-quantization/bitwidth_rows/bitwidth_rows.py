#!/usr/bin/env python3
"""Size, memory and speed per bit-width on one Mac (G12 tasks 1 and 7): one row per GGUF, same method for every machine.

    python bitwidth_rows.py --manifest rows.json --llama-bin <dir with llama-bench> --out-dir out [--ngl 99] [--repeats 3]
        [--depth 8192] [--spec-bandwidth-gbps 400] [--expect-build 11669]

rows.json (the same file list for every machine, so rows compare):
    {"machine_label": "Apple M3 Max 64 GB",
     "rows": [{"model": "granite-3.0-3b-a800m-base", "format": "Q8_0", "path": "/x/model-Q8_0.gguf", "sha256": "<hex>"}, ...]}

Per row it runs llama-bench (build 11669 unless --allow-other-build) and records:
  size          bytes of the GGUF, and bits per weight = 8 x bytes / parameters (llama-bench's own count)
  speed         prompt processing (pp512) and token generation (tg128) in tokens/s, mean and standard deviation of --repeats
  at depth      token generation (tg64) with the KV cache already holding --depth tokens (8k by default)
  memory        maximum resident set size of the llama-bench process (macOS `/usr/bin/time -l`), for a short run and for
                the run at depth; the difference is the cost of the long context (KV cache and a larger compute buffer).
                The table uses the resident set, which includes the memory-mapped weights. `peak memory footprint` is kept in
                rows.json only: it leaves the mapped weights out (0.26 GB for a 0.61 GB bf16 model in a test), so it understates.
  bandwidth     tg x file size, in GB/s: an UPPER bound on the memory traffic of decoding (a mixture of experts reads
                only the active experts, so it can use far less); next to --spec-bandwidth-gbps if you give it.
The file's sha256 is checked against the manifest first and the run stops on a mismatch (use --skip-sha to measure other files).
Writes rows.json and rows.md in --out-dir, each with the machine, macOS version, llama.cpp build and settings.

The numbers are for the machine they ran on: a 64 GB machine cannot stand in for a 16 GB or 32 GB one (memory bandwidth, GPU
memory limit), so label every row with the machine and report speed only where it was measured.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, platform, re, subprocess, sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parse_time_l(stderr: str) -> dict:
    """macOS `/usr/bin/time -l`: {"max_rss_bytes", "peak_footprint_bytes"} (None where a line is missing)."""
    def grab(label):
        m = re.search(r"^\s*(\d+)\s+" + label + r"\s*$", stderr, re.M)
        return int(m.group(1)) if m else None
    return {"max_rss_bytes": grab("maximum resident set size"), "peak_footprint_bytes": grab("peak memory footprint")}


def pick(rows: list[dict], n_prompt: int, n_gen: int, n_depth: int = 0) -> dict | None:
    for r in rows:
        if r["n_prompt"] == n_prompt and r["n_gen"] == n_gen and r.get("n_depth", 0) == n_depth:
            return r
    return None


def run_bench(binary: Path, gguf: Path, extra: list[str], with_memory: bool) -> tuple[list[dict], dict | None]:
    cmd = [str(binary), "-m", str(gguf), "-o", "json", *extra]
    mem = with_memory and platform.system() == "Darwin"
    out = subprocess.run((["/usr/bin/time", "-l"] if mem else []) + cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"llama-bench failed ({out.returncode}): {out.stderr[-600:]}")
    start = out.stdout.find("[")
    if start < 0:
        raise RuntimeError(f"no JSON from llama-bench: {out.stdout[:300]!r}")
    return json.loads(out.stdout[start:]), (parse_time_l(out.stderr) if mem else None)


def measure_row(row: dict, a) -> dict:
    gguf = Path(row["path"])
    if not gguf.is_file():
        raise FileNotFoundError(gguf)
    if row.get("sha256") and not a.skip_sha:
        got = sha256_file(gguf)
        if got != row["sha256"]:
            raise SystemExit(f"{gguf.name}: sha256 {got} does not match the manifest ({row['sha256']}); rows would not compare")
    bench = Path(a.llama_bin) / "llama-bench"
    common = ["-ngl", str(a.ngl), "-r", str(a.repeats)]
    short, mem_short = run_bench(bench, gguf, ["-p", "512", "-n", "128", *common], True)
    deep, mem_deep = run_bench(bench, gguf, ["-p", "0", "-n", "64", "-d", str(a.depth), *common], True)
    build = short[0]["build_number"]
    if a.expect_build and build != a.expect_build and not a.allow_other_build:
        raise SystemExit(f"llama-bench is build {build}, rows must come from build {a.expect_build} (--allow-other-build to override)")
    pp, tg, tgd = pick(short, 512, 0), pick(short, 0, 128), pick(deep, 0, 64, a.depth)
    size, params = short[0]["model_size"], short[0]["model_n_params"]
    file_bytes = gguf.stat().st_size
    res = {"model": row["model"], "format": row["format"], "file": gguf.name, "file_bytes": file_bytes,
           "model_size_bytes": size, "parameters": params, "bits_per_weight": round(8 * file_bytes / params, 3),
           "pp512_tok_s": pp and round(pp["avg_ts"], 2), "pp512_std": pp and round(pp["stddev_ts"], 2),
           "tg128_tok_s": tg and round(tg["avg_ts"], 2), "tg128_std": tg and round(tg["stddev_ts"], 2),
           f"tg64_at_{a.depth}_tok_s": tgd and round(tgd["avg_ts"], 2), "depth": a.depth,
           "peak_footprint_short_bytes": mem_short and mem_short["peak_footprint_bytes"],
           "peak_footprint_depth_bytes": mem_deep and mem_deep["peak_footprint_bytes"],
           "max_rss_short_bytes": mem_short and mem_short["max_rss_bytes"], "max_rss_depth_bytes": mem_deep and mem_deep["max_rss_bytes"],
           "n_gpu_layers": a.ngl, "repeats": a.repeats, "build_number": build, "build_commit": short[0]["build_commit"],
           "backends": short[0]["backends"], "cpu_info": short[0]["cpu_info"]}
    bw = tg["avg_ts"] * file_bytes / 1e9 if tg else None            # GB/s: tokens per second x bytes per token (the whole file)
    res["tg_times_file_gb_s"] = None if bw is None else round(bw, 1)
    if a.spec_bandwidth_gbps and bw is not None:
        res["share_of_spec_bandwidth"] = round(bw / a.spec_bandwidth_gbps, 3)
    return res


def machine_info(label: str) -> dict:
    mem = None
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize", "hw.model"], capture_output=True, text=True).stdout.split()
        mem = int(out[0]) if out else None
    return {"label": label, "system": platform.system(), "release": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(), "memory_bytes": mem, "date": datetime.datetime.now().isoformat(timespec="seconds")}


def render(machine: dict, results: list[dict], a) -> str:
    gb = lambda b: "-" if b is None else f"{b / 1e9:.2f}"
    key = f"tg64_at_{a.depth}_tok_s"
    lines = [f"# Size, memory and speed per bit-width: {machine['label']}", "",
             f"{machine['system']} {machine['release']} {machine['machine']}, llama.cpp build {results[0]['build_number']} "
             f"({results[0]['build_commit']}), backends {results[0]['backends']}, -ngl {a.ngl}, {a.repeats} repeats; "
             f"measured {machine['date']}. Memory = maximum resident set size of the llama-bench process (macOS), weights included; depth = {a.depth} tokens in the KV cache.", "",
             f"| Model | Format | Bits/weight | File GB | Mem GB (short) | Mem GB (depth {a.depth}) | pp512 tok/s | tg128 tok/s | tg64 at depth tok/s | tg x file GB/s (upper bound) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        pm = lambda v, s: "-" if v is None else f"{v:.1f} +- {s:.1f}"
        lines.append(f"| {r['model']} | {r['format']} | {r['bits_per_weight']:.2f} | {gb(r['file_bytes'])} | {gb(r['max_rss_short_bytes'])} | "
                     f"{gb(r['max_rss_depth_bytes'])} | {pm(r['pp512_tok_s'], r['pp512_std'] or 0)} | {pm(r['tg128_tok_s'], r['tg128_std'] or 0)} | "
                     f"{'-' if r.get(key) is None else format(r[key], '.1f')} | {r['tg_times_file_gb_s']} |")
    lines += ["", "tg x file GB/s is the most memory traffic decoding could cause; a mixture of experts reads only its active experts, "
              "so the real figure can be far lower. This machine cannot stand in for a 16 GB or 32 GB one."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path); ap.add_argument("--llama-bin", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--ngl", type=int, default=99); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--depth", type=int, default=8192); ap.add_argument("--spec-bandwidth-gbps", type=float, default=None)
    ap.add_argument("--expect-build", type=int, default=11669); ap.add_argument("--allow-other-build", action="store_true")
    ap.add_argument("--skip-sha", action="store_true")
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in manifest["rows"]:
        print("measuring", row["model"], row["format"], flush=True)
        results.append(measure_row(row, a))
        print("  ", {k: results[-1][k] for k in ("bits_per_weight", "tg128_tok_s", "pp512_tok_s")}, flush=True)
    machine = machine_info(manifest.get("machine_label", "unlabelled machine"))
    (a.out_dir / "rows.json").write_text(json.dumps({"machine": machine, "settings": {"ngl": a.ngl, "repeats": a.repeats, "depth": a.depth},
                                                      "rows": results}, indent=1))
    (a.out_dir / "rows.md").write_text(render(machine, results, a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
