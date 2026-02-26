#!/usr/bin/env python3
"""Run SMT handshake estimation and MG-window VCD validation for benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BenchPaths:
    bench: str
    repo: Path

    @property
    def bench_dir(self) -> Path:
        return self.repo / "integration-test" / self.bench

    @property
    def comp_dir(self) -> Path:
        return self.bench_dir / "out" / "comp"

    @property
    def hdl_dir(self) -> Path:
        return self.bench_dir / "out" / "hdl"

    @property
    def sim_vcd(self) -> Path:
        return self.bench_dir / "out" / "sim" / "HLS_VERIFY" / "trace.vcd"

    @property
    def top_verilog(self) -> Path:
        return self.hdl_dir / f"{self.bench}.v"

    @property
    def profiler_inputs(self) -> Path:
        return self.comp_dir / "profiler-inputs.txt"

    @property
    def cf_transformed(self) -> Path:
        return self.comp_dir / "cf_transformed_mem_interface_marked.mlir"

    @property
    def hs_transformed(self) -> Path:
        return self.comp_dir / "handshake_transformed.mlir"

    @property
    def trace_log(self) -> Path:
        return self.comp_dir / "data_trace.log"

    @property
    def frequencies(self) -> Path:
        return self.comp_dir / "frequencies.csv"

    @property
    def switching_csv(self) -> Path:
        return self.comp_dir / "switching_estimation.csv"

    @property
    def smt_dump(self) -> Path:
        return self.comp_dir / "smt_handshake_dump.json"

    @property
    def pass_log(self) -> Path:
        return self.comp_dir / "handshake_switch_test.mlir"

    @property
    def validation_json(self) -> Path:
        return self.comp_dir / "smt_mg_window_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMT handshake estimation and MG-window validation"
    )
    parser.add_argument(
        "--bench",
        action="append",
        required=True,
        help="Benchmark name under integration-test (repeatable)",
    )
    parser.add_argument(
        "--run-flow",
        action="store_true",
        help="Run compile/write-hdl/simulate + profiler + switching-estimation",
    )
    parser.add_argument(
        "--use-smt",
        action="store_true",
        help="Enable use-smt-handshake in switching-estimation",
    )
    parser.add_argument(
        "--target-period",
        type=float,
        default=8.0,
        help="Target clock period passed to flow and switching-estimation",
    )
    parser.add_argument(
        "--milp-solver",
        default="gurobi",
        help="Solver passed to handshake-place-buffers",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to dynamatic-mlir repository root",
    )
    parser.add_argument(
        "--max-worst",
        type=int,
        default=10,
        help="Max worst edges per MG in VCD comparison report",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path, log_path: Optional[Path] = None) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    if log_path is None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        f.write(proc.stdout)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def find_c_source(bench_dir: Path, bench: str) -> Path:
    preferred = bench_dir / f"{bench}.c"
    if preferred.exists():
        return preferred
    c_files = sorted(bench_dir.glob("*.c"))
    if not c_files:
        raise FileNotFoundError(f"No C source found in {bench_dir}")
    return c_files[0]


def run_dynamatic_flow(paths: BenchPaths, target_period: float) -> None:
    source = find_c_source(paths.bench_dir, paths.bench)
    dynamatic_bin = paths.repo / "bin" / "dynamatic"
    cmd_script = (
        f"set-dynamatic-path {paths.repo}; "
        f"set-src {source}; "
        f"set-clock-period {target_period}; "
        "compile --buffer-algorithm fpga20; "
        "write-hdl --hdl verilog; "
        "simulate; "
        "exit"
    )
    print(f"[CMD] {dynamatic_bin} --exit-on-failure")
    subprocess.run(
        [str(dynamatic_bin), "--exit-on-failure"],
        cwd=str(paths.repo),
        input=cmd_script + "\n",
        text=True,
        check=True,
    )


def run_profiler(paths: BenchPaths) -> None:
    profiler_bin = paths.repo / "bin" / "exp-frequency-profiler"
    cmd = [
        str(profiler_bin),
        str(paths.cf_transformed),
        f"--top-level-function={paths.bench}",
        f"--input-args-file={paths.profiler_inputs}",
        f"--trace-log-file={paths.trace_log}",
        "--mode=both",
    ]
    run_cmd(cmd, paths.repo, log_path=paths.frequencies)


def run_switching_estimation(
    paths: BenchPaths,
    target_period: float,
    milp_solver: str,
    use_smt: bool,
) -> None:
    opt_bin = paths.repo / "bin" / "dynamatic-opt"
    components = paths.repo / "data" / "components.json"
    blif_dir = paths.repo / "data" / "aig"

    place_buffers = (
        f"algorithm=fpga20 solver={milp_solver} frequencies={paths.frequencies} "
        f"timing-models={components} target-period={target_period} timeout=300 "
        f"dump-logs blif-files={blif_dir}/ lut-delay=0.55 lut-size=6 acyclic-type"
    )

    switch_opts = (
        f"data-trace={paths.trace_log} timing-models={components} "
        f"target-period={target_period} dump-file={paths.switching_csv} "
        f"use-smt-handshake={int(use_smt)} smt-handshake-dump={paths.smt_dump}"
    )

    cmd = [
        str(opt_bin),
        str(paths.hs_transformed),
        "--handshake-mark-fpu-impl=impl=vivado",
        "--handshake-set-buffering-properties=version=fpga20",
        f"--handshake-place-buffers={place_buffers}",
        f"--switching-estimation={switch_opts}",
    ]
    run_cmd(cmd, paths.comp_dir, log_path=paths.pass_log)


def run_vcd_compare(paths: BenchPaths, max_worst: int) -> Dict[str, object]:
    compare_py = (
        paths.repo
        / "experimental"
        / "lib"
        / "Analysis"
        / "SwitchingEstimation"
        / "SMTHandshake"
        / "VCDWindowExtract.py"
    )

    cmd = [
        sys.executable,
        str(compare_py),
        "--smt-json",
        str(paths.smt_dump),
        "--vcd",
        str(paths.sim_vcd),
        "--top-verilog",
        str(paths.top_verilog),
        "--max-worst",
        str(max_worst),
        "--output-json",
        str(paths.validation_json),
    ]
    run_cmd(cmd, paths.repo, log_path=None)

    return json.loads(paths.validation_json.read_text(encoding="utf-8"))


def require_paths(paths: BenchPaths, use_smt: bool) -> None:
    required = [
        paths.cf_transformed,
        paths.hs_transformed,
        paths.trace_log,
        paths.top_verilog,
        paths.sim_vcd,
    ]
    if use_smt:
        required.append(paths.smt_dump)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))


def summarize_bench(bench: str, report: Dict[str, object]) -> None:
    print(f"\n=== {bench} ===")
    results = report.get("results", [])
    ok = 0
    skipped = 0
    for res in results:
        status = res.get("status", "")
        mg = res.get("mg_label", "")
        compared = res.get("compared_edge_count", 0)
        rotation = res.get("rotation", 0)
        if status == "OK":
            ok += 1
        else:
            skipped += 1
        print(
            f"MG {mg}: status={status} compared_edges={compared} rotation={rotation} "
            f"reason={res.get('reason', '')}"
        )
    print(f"summary: ok={ok} skipped={skipped} total={len(results)}")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()

    benches = args.bench
    reports: Dict[str, Dict[str, object]] = {}

    for bench in benches:
        paths = BenchPaths(bench=bench, repo=repo)
        print(f"\n[INFO] Processing benchmark: {bench}")

        if args.run_flow:
            paths.comp_dir.mkdir(parents=True, exist_ok=True)
            run_dynamatic_flow(paths, args.target_period)
            run_profiler(paths)
            run_switching_estimation(
                paths,
                target_period=args.target_period,
                milp_solver=args.milp_solver,
                use_smt=args.use_smt,
            )

        if args.use_smt:
            require_paths(paths, use_smt=True)
            report = run_vcd_compare(paths, max_worst=args.max_worst)
            reports[bench] = report
            summarize_bench(bench, report)
        else:
            require_paths(paths, use_smt=False)
            print("[INFO] --use-smt is disabled; skipping VCD SMT comparison.")

    print("\n[INFO] Completed SMT MG-window validation.")


if __name__ == "__main__":
    main()
