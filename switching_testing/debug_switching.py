#!/usr/bin/env python3
"""
Switching-estimation debug analyzer.

This script compares estimator CSV against VCD golden (steady-state window
supported) and emits a debugging-oriented report:
  - top failing node/channel groups
  - over/under-estimation patterns
  - likely source-code hotspots to inspect
  - per-kernel violation CSV + markdown summary
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from test import (  # local switching_testing/test.py
    channel_ignored_small,
    channel_violation,
    get_vcd_switching,
    get_vcd_window,
    has_node_signal,
    read_estimator_csv,
)
from vcd_parser import VcdParser

CHANNELS: Tuple[str, str, str] = ("data", "valid", "ready")
DEFAULT_KERNELS: Tuple[str, ...] = (
    "kernel_3mm",
    "kernel_2mm",
    "gsum",
    "bicg",
    "gemver",
    "cnn",
    "matvec",
    "stencil_2d",
    "iir",
    "fir",
)

# Keep longer prefixes before shorter ones to avoid partial matches.
NODE_TYPE_PREFIXES: Tuple[str, ...] = (
    "control_merge",
    "cond_br",
    "mc_load",
    "mc_store",
    "lsq_load",
    "lsq_store",
    "buffer",
    "fork",
    "mux",
    "select",
    "merge",
    "br",
    "cmpi",
    "addi",
    "subi",
    "muli",
    "andi",
    "ori",
    "xori",
    "shli",
    "shrsi",
    "shrui",
    "extsi",
    "extui",
    "trunci",
    "load",
    "store",
    "constant",
    "source",
    "sink",
    "end",
)

# (channel, node_type) -> (file_hint, reason)
HOTSPOT_HINTS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("data", "mux"): (
        "experimental/lib/Analysis/SwitchingEstimation/DataChannelCal.cpp",
        "Mux source selection / fallback / glitch accumulation.",
    ),
    ("data", "cond_br"): (
        "experimental/lib/Analysis/SwitchingEstimation/DataChannelCal.cpp",
        "Conditional-branch data propagation and selected output handling.",
    ),
    ("data", "control_merge"): (
        "experimental/lib/Analysis/SwitchingEstimation/DataChannelCal.cpp",
        "Control-merge index/data relay and transition handling.",
    ),
    ("valid", "cond_br"): (
        "experimental/lib/Analysis/SwitchingEstimation/HandShakeChannelCal.cpp",
        "Conditional-branch valid propagation for true/false outputs.",
    ),
    ("ready", "cond_br"): (
        "experimental/lib/Analysis/SwitchingEstimation/HandShakeChannelCal.cpp",
        "Conditional-branch ready set/switching model.",
    ),
    ("valid", "fork"): (
        "experimental/lib/Analysis/SwitchingEstimation/NodeModels.cpp",
        "Fork valid switching/set derivation from successor readiness.",
    ),
    ("ready", "fork"): (
        "experimental/lib/Analysis/SwitchingEstimation/NodeModels.cpp",
        "Fork ready set union and edge-case handling.",
    ),
    ("ready", "buffer"): (
        "experimental/lib/Analysis/SwitchingEstimation/HandShakeChannelCal.cpp",
        "Buffer ready-set model from occupancy/type/transparent behavior.",
    ),
    ("ready", "source"): (
        "experimental/lib/Analysis/SwitchingEstimation/SwitchingEstimation.cpp",
        "Exported ready convention (node output-ready vs predecessor-ready).",
    ),
    ("ready", "load"): (
        "experimental/lib/Analysis/SwitchingEstimation/HandShakeChannelCal.cpp",
        "Load-unit ready model and buffer-influenced behavior.",
    ),
}


@dataclass(frozen=True)
class ChannelDelta:
    kernel: str
    node: str
    node_type: str
    channel: str
    est: int
    golden: int
    diff: int
    rel_error: Optional[float]
    violation_reason: Optional[str]

    @property
    def sign(self) -> str:
        if self.diff > 0:
            return "over"
        if self.diff < 0:
            return "under"
        return "match"

    @property
    def abs_error(self) -> int:
        return abs(self.diff)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate debugging-oriented switching-estimation reports.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing integration-test/")
    parser.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="Kernel name to analyze. Repeat for multiple kernels. If omitted, analyze the default benchmark list.",
    )
    parser.add_argument("--window", default="post-reset", choices=("full", "post-reset"))
    parser.add_argument("--reset-signal", default="", help="Optional explicit reset signal name.")
    parser.add_argument("--rel-error-threshold", type=float, default=0.10)
    parser.add_argument("--zero-abs-threshold", type=int, default=5)
    parser.add_argument("--ignore-both-below", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20, help="Top failing entries to print/report.")
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero when any violation is found.",
    )
    return parser.parse_args()


def node_type_of(node_name: str) -> str:
    for prefix in NODE_TYPE_PREFIXES:
        if node_name.startswith(prefix):
            return prefix
    # Fallback: strip trailing digits.
    idx = len(node_name)
    while idx > 0 and node_name[idx - 1].isdigit():
        idx -= 1
    return node_name[:idx] if idx > 0 else node_name


def fmt_rel(rel: Optional[float]) -> str:
    if rel is None:
        return "N/A"
    return f"{rel * 100:.2f}%"


def hint_for(channel: str, node_type: str) -> Tuple[str, str]:
    if (channel, node_type) in HOTSPOT_HINTS:
        return HOTSPOT_HINTS[(channel, node_type)]
    if channel == "data":
        return (
            "experimental/lib/Analysis/SwitchingEstimation/DataChannelCal.cpp",
            "Data propagation/glitch handling for this node class.",
        )
    if channel in ("valid", "ready"):
        return (
            "experimental/lib/Analysis/SwitchingEstimation/HandShakeChannelCal.cpp",
            "Handshake propagation/update path for this node class.",
        )
    return (
        "experimental/lib/Analysis/SwitchingEstimation/SwitchingEstimation.cpp",
        "Aggregation/export stage for channel totals.",
    )


def iter_kernel_deltas(
    kernel: str,
    repo_root: Path,
    window: str,
    reset_signal: str,
    rel_error_threshold: float,
    zero_abs_threshold: int,
    ignore_both_below: int,
) -> Tuple[List[ChannelDelta], int, int, int, str]:
    est_csv = repo_root / "integration-test" / kernel / "out" / "comp" / "switching_estimation.csv"
    vcd_file = repo_root / "integration-test" / kernel / "out" / "sim" / "HLS_VERIFY" / "trace.vcd"

    if not est_csv.is_file():
        raise FileNotFoundError(f"Estimator CSV not found: {est_csv}")
    if not vcd_file.is_file():
        raise FileNotFoundError(f"Golden VCD not found: {vcd_file}")

    estimator = read_estimator_csv(str(est_csv))
    vcd = VcdParser(str(vcd_file))

    # Reuse window logic from compare script.
    class WindowArgs:
        pass

    win_args = WindowArgs()
    win_args.window = window
    win_args.reset_signal = reset_signal
    start_time, end_time, start_exclusive, end_inclusive, window_desc = get_vcd_window(vcd, win_args)

    deltas: List[ChannelDelta] = []
    compared = 0
    ignored_small = 0
    missing_vcd_nodes = 0

    for node, (est_data, est_valid, est_ready) in sorted(estimator.items()):
        if not has_node_signal(node, vcd.unique_signal_names):
            missing_vcd_nodes += 1
            continue

        golden_data, golden_valid, golden_ready = get_vcd_switching(
            vcd,
            node,
            start_time=start_time,
            end_time=end_time,
            start_exclusive=start_exclusive,
            end_inclusive=end_inclusive,
        )
        node_type = node_type_of(node)

        for channel, est, golden in (
            ("data", est_data, golden_data),
            ("valid", est_valid, golden_valid),
            ("ready", est_ready, golden_ready),
        ):
            if channel_ignored_small(est, golden, ignore_both_below):
                ignored_small += 1
                continue

            compared += 1
            reason = channel_violation(est, golden, rel_error_threshold, zero_abs_threshold)
            rel_error = None if golden == 0 else abs(est - golden) / golden
            deltas.append(
                ChannelDelta(
                    kernel=kernel,
                    node=node,
                    node_type=node_type,
                    channel=channel,
                    est=est,
                    golden=golden,
                    diff=est - golden,
                    rel_error=rel_error,
                    violation_reason=reason,
                )
            )

    return deltas, compared, ignored_small, missing_vcd_nodes, window_desc


def write_violation_csv(path: Path, deltas: Iterable[ChannelDelta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "kernel",
                "node",
                "node_type",
                "channel",
                "est",
                "golden",
                "diff",
                "abs_error",
                "rel_error",
                "sign",
                "reason",
                "hint_file",
                "hint_note",
            ]
        )
        for d in deltas:
            hint_file, hint_note = hint_for(d.channel, d.node_type)
            writer.writerow(
                [
                    d.kernel,
                    d.node,
                    d.node_type,
                    d.channel,
                    d.est,
                    d.golden,
                    d.diff,
                    d.abs_error,
                    "" if d.rel_error is None else f"{d.rel_error:.6f}",
                    d.sign,
                    d.violation_reason or "",
                    hint_file,
                    hint_note,
                ]
            )


def build_group_summary(deltas: List[ChannelDelta]) -> List[Tuple[str, str, str, int, float, int]]:
    # channel, node_type, sign, fail_count, mean_rel%, max_abs
    grouped: Dict[Tuple[str, str, str], List[ChannelDelta]] = {}
    for d in deltas:
        if d.violation_reason is None:
            continue
        key = (d.channel, d.node_type, d.sign)
        grouped.setdefault(key, []).append(d)

    rows: List[Tuple[str, str, str, int, float, int]] = []
    for (channel, node_type, sign), items in grouped.items():
        rel_list = [x.rel_error for x in items if x.rel_error is not None]
        mean_rel_pct = statistics.fmean(rel_list) * 100 if rel_list else 0.0
        max_abs = max(x.abs_error for x in items)
        rows.append((channel, node_type, sign, len(items), mean_rel_pct, max_abs))

    rows.sort(key=lambda x: (-x[3], -x[4], -x[5], x[0], x[1], x[2]))
    return rows


def write_markdown_report(
    path: Path,
    kernel: str,
    window_desc: str,
    compared: int,
    ignored_small: int,
    missing_vcd_nodes: int,
    deltas: List[ChannelDelta],
    top_k: int,
) -> None:
    violations = [d for d in deltas if d.violation_reason is not None]
    passed = [d for d in deltas if d.violation_reason is None]
    group_rows = build_group_summary(deltas)
    top_violations = sorted(
        violations,
        key=lambda d: (d.abs_error if d.golden == 0 else (d.rel_error or 0.0), d.abs_error),
        reverse=True,
    )[:top_k]

    lines: List[str] = []
    lines.append(f"# Switching Debug Report: {kernel}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Window: `{window_desc}`")
    lines.append(f"- Compared channels: `{compared}`")
    lines.append(f"- Ignored small channels: `{ignored_small}`")
    lines.append(f"- Nodes missing in VCD (from estimator list): `{missing_vcd_nodes}`")
    lines.append(f"- Violations: `{len(violations)}`")
    lines.append(f"- Passed channels: `{len(passed)}`")
    lines.append("")

    lines.append("## Top Failing Groups")
    if not group_rows:
        lines.append("- No failing groups.")
    else:
        lines.append("| channel | node_type | sign | fail_count | mean_rel_error | max_abs_error | hint |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for channel, node_type, sign, fail_count, mean_rel_pct, max_abs in group_rows[:top_k]:
            hint_file, hint_note = hint_for(channel, node_type)
            hint = f"`{hint_file}`: {hint_note}"
            lines.append(
                f"| {channel} | {node_type} | {sign} | {fail_count} | "
                f"{mean_rel_pct:.2f}% | {max_abs} | {hint} |"
            )
    lines.append("")

    lines.append("## Top Violations")
    if not top_violations:
        lines.append("- No threshold violations.")
    else:
        lines.append("| node | channel | est | golden | diff | rel_error | reason | hint |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for d in top_violations:
            hint_file, hint_note = hint_for(d.channel, d.node_type)
            lines.append(
                f"| {d.node} | {d.channel} | {d.est} | {d.golden} | {d.diff} | "
                f"{fmt_rel(d.rel_error)} | {d.violation_reason} | `{hint_file}`: {hint_note} |"
            )
    lines.append("")

    lines.append("## Suggested Next Checks")
    lines.append("1. Inspect the top failing group and patch the hinted file first.")
    lines.append("2. Re-run `test_sa.sh <kernel>` then re-run this report.")
    lines.append("3. Compare per-node channel deltas in `switching_debug_violations.csv` after each patch.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    kernels = args.kernel if args.kernel else list(DEFAULT_KERNELS)

    any_violation = False
    for kernel in kernels:
        try:
            deltas, compared, ignored_small, missing_vcd_nodes, window_desc = iter_kernel_deltas(
                kernel=kernel,
                repo_root=repo_root,
                window=args.window,
                reset_signal=args.reset_signal,
                rel_error_threshold=args.rel_error_threshold,
                zero_abs_threshold=args.zero_abs_threshold,
                ignore_both_below=args.ignore_both_below,
            )
        except FileNotFoundError as e:
            print(f"[WARN] {kernel}: {e}")
            continue

        violations = [d for d in deltas if d.violation_reason is not None]
        any_violation = any_violation or bool(violations)

        out_dir = repo_root / "integration-test" / kernel / "out" / "comp"
        violation_csv = out_dir / "switching_debug_violations.csv"
        report_md = out_dir / "switching_debug_report.md"

        write_violation_csv(violation_csv, violations)
        write_markdown_report(
            path=report_md,
            kernel=kernel,
            window_desc=window_desc,
            compared=compared,
            ignored_small=ignored_small,
            missing_vcd_nodes=missing_vcd_nodes,
            deltas=deltas,
            top_k=args.top_k,
        )

        print(
            f"[DEBUG] {kernel}: compared={compared}, ignored_small={ignored_small}, "
            f"missing_vcd_nodes={missing_vcd_nodes}, violations={len(violations)}"
        )
        print(f"[DEBUG]   report: {report_md}")
        print(f"[DEBUG]   csv   : {violation_csv}")

    if args.fail_on_violation and any_violation:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
