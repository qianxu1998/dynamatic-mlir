import argparse
import csv
import re
import statistics
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from vcd_parser import VcdParser

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    tabulate = None

SwitchTriple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare switching-estimation CSV against VCD-derived switching (VCD is golden)."
    )
    parser.add_argument(
        "--vcd",
        default="/home/jianliu/TCAD26/dynamatic-mlir/integration-test/fir/out/sim/HLS_VERIFY/trace.vcd",
        help="Path to VCD file.",
    )
    parser.add_argument(
        "--est-csv",
        default="/home/jianliu/TCAD26/dynamatic-mlir/integration-test/fir/out/comp/switching_estimation.csv",
        help="Path to estimator CSV file (columns: node,data,valid,ready).",
    )
    parser.add_argument(
        "--include-vcd-only",
        action="store_true",
        help="Also include nodes found in VCD but missing in estimator CSV.",
    )
    parser.add_argument(
        "--window",
        default="full",
        choices=("full", "post-reset"),
        help="Golden extraction window for VCD toggles.",
    )
    parser.add_argument(
        "--reset-signal",
        default="",
        help="Optional explicit reset signal name. If empty, detect automatically.",
    )
    parser.add_argument(
        "--rel-error-threshold",
        type=float,
        default=0.10,
        help="Relative error threshold for channels with golden > 0.",
    )
    parser.add_argument(
        "--zero-abs-threshold",
        type=int,
        default=5,
        help="Absolute error threshold for channels with golden = 0.",
    )
    parser.add_argument(
        "--ignore-both-below",
        type=int,
        default=50,
        help=(
            "Ignore channel when both estimator and golden values are strictly "
            "below this threshold."
        ),
    )
    parser.add_argument(
        "--no-enforce-thresholds",
        action="store_true",
        help="Do not fail (exit code 1) when threshold violations are found.",
    )
    parser.add_argument(
        "--max-violation-report",
        type=int,
        default=40,
        help="Maximum number of channel violations to print in detail.",
    )
    return parser.parse_args()


def parse_int_field(raw: str) -> int:
    return int(raw.strip())


def read_estimator_csv(csv_path: str) -> Dict[str, SwitchTriple]:
    node_to_switching: Dict[str, SwitchTriple] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return node_to_switching

        required = ("node", "data", "valid", "ready")
        normalized = [h.strip().lower() for h in reader.fieldnames]
        missing = [name for name in required if name not in normalized]
        if missing:
            raise ValueError(f"Estimator CSV missing required column(s): {', '.join(missing)}")

        field_lookup = {h.strip().lower(): h for h in reader.fieldnames}
        for row in reader:
            if not row:
                continue

            node = row[field_lookup["node"]].strip()
            if not node:
                continue

            data = parse_int_field(row[field_lookup["data"]])
            valid = parse_int_field(row[field_lookup["valid"]])
            ready = parse_int_field(row[field_lookup["ready"]])
            node_to_switching[node] = (data, valid, ready)

    return node_to_switching


def extract_node_names(signals: Iterable[str], node_types: Iterable[str]) -> List[str]:
    sorted_types = sorted(node_types, key=len, reverse=True)
    pattern = re.compile(
        r"(?:^|[.])((?:"
        + "|".join(re.escape(node_type) for node_type in sorted_types)
        + r")\d+)(?=$|[._\[])",
    )

    matched_nodes = set()
    for signal in signals:
        match = pattern.search(signal)
        if match:
            matched_nodes.add(match.group(1))
    return sorted(matched_nodes)


def has_node_signal(node_name: str, signals: Iterable[str]) -> bool:
    pattern = re.compile(r"(?:^|[.])" + re.escape(node_name) + r"(?=$|[._\[])")
    return any(pattern.search(signal) for signal in signals)


def fmt_value(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    return str(value)


def format_diff(est: Optional[int], golden: Optional[int]) -> str:
    if est is None or golden is None:
        return "N/A"
    return str(est - golden)


def format_error_rate(est: Optional[int], golden: Optional[int]) -> str:
    if est is None or golden is None:
        return "N/A"
    if golden == 0:
        return "0.00%" if est == 0 else "inf"
    return f"{abs(est - golden) / golden * 100:.2f}%"


def fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_pretty_table(headers: List[str], rows: List[List[str]]) -> str:
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="grid")

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(cols: List[str]) -> str:
        padded = [cols[i].ljust(widths[i]) for i in range(len(cols))]
        return " | ".join(padded)

    sep = "-+-".join("-" * w for w in widths)
    out: List[str] = [fmt_row(headers), sep]
    out.extend(fmt_row(row) for row in rows)
    return "\n".join(out)


def _logic_char(value: str) -> Optional[str]:
    value = str(value).strip()
    if not value:
        return None
    first = value[0]
    if first in ("0", "1", "x", "X", "z", "Z"):
        return first
    return None


def _signal_width_is_one(vcd: VcdParser, signal_name: str) -> bool:
    if signal_name not in vcd.name_to_id:
        return False
    identifier = vcd.name_to_id[signal_name]
    signal = vcd.signals.get(identifier)
    if signal is None:
        return False
    return str(signal.bit_width) == "1"


def _reset_candidate_score(vcd: VcdParser, signal_name: str) -> int:
    name = signal_name.lower()
    score = 0

    if name == "tb.duv_inst.rst":
        score += 1000
    if name.endswith(".rst"):
        score += 300
    if name.endswith("_rst"):
        score += 280
    if name.endswith(".reset"):
        score += 260
    if name.endswith("_reset"):
        score += 240
    if re.search(r"(?:^|[._])rst(?:$|[._\[])", name):
        score += 120
    if re.search(r"(?:^|[._])reset(?:$|[._\[])", name):
        score += 100

    score -= signal_name.count(".")
    if _signal_width_is_one(vcd, signal_name):
        score += 20
    else:
        score -= 500
    return score


def detect_reset_signal(vcd: VcdParser, reset_signal_hint: str) -> str:
    if reset_signal_hint:
        if reset_signal_hint in vcd.name_to_id:
            return reset_signal_hint

        suffix_matches = [
            signal_name
            for signal_name in vcd.unique_signal_names
            if signal_name.endswith(reset_signal_hint)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise ValueError("Reset signal hint is ambiguous. Matches: " + ", ".join(sorted(suffix_matches)))
        raise ValueError(f"Reset signal '{reset_signal_hint}' not found in VCD signal list.")

    candidates = [
        signal_name
        for signal_name in vcd.unique_signal_names
        if re.search(r"(?:^|[._])(rst|reset)(?:$|[._\[])", signal_name.lower())
    ]
    if not candidates:
        raise ValueError("Could not auto-detect reset signal in VCD.")

    scored = sorted(
        ((signal_name, _reset_candidate_score(vcd, signal_name)) for signal_name in candidates),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if scored[0][1] <= 0:
        raise ValueError("Reset candidates detected, but all have non-positive confidence.")
    return scored[0][0]


def find_reset_deassert_time(vcd: VcdParser, reset_signal: str) -> int:
    identifier = vcd.name_to_id.get(reset_signal)
    if identifier is None:
        raise ValueError(f"Reset signal '{reset_signal}' not found in VCD mapping.")

    signal = vcd.signals.get(identifier)
    if signal is None or not signal.time_value_pair_list:
        raise ValueError(f"Reset signal '{reset_signal}' has no value-change events.")

    value_list = signal.time_value_pair_list
    prev_logic = _logic_char(value_list[0][1])
    first_known_zero_time: Optional[int] = value_list[0][0] if prev_logic == "0" else None

    for event_time, raw_value in value_list[1:]:
        logic = _logic_char(raw_value)
        if logic == "0" and first_known_zero_time is None:
            first_known_zero_time = event_time

        if prev_logic == "1" and logic == "0":
            return event_time

        if logic is not None:
            prev_logic = logic

    if first_known_zero_time is not None:
        return first_known_zero_time

    raise ValueError(f"Reset signal '{reset_signal}' never deasserts to 0 in the VCD trace.")


def get_vcd_window(vcd: VcdParser, args: argparse.Namespace) -> Tuple[Optional[int], Optional[int], bool, bool, str]:
    if args.window == "full":
        return None, None, False, True, "full-trace"

    if args.window == "post-reset":
        reset_signal = detect_reset_signal(vcd, args.reset_signal)
        deassert_time = find_reset_deassert_time(vcd, reset_signal)
        desc = f"post-reset (reset={reset_signal}, deassert_time={deassert_time})"
        return deassert_time, None, True, True, desc

    raise ValueError(f"Unsupported window mode: {args.window}")


def get_vcd_switching(
    vcd: VcdParser,
    node_name: str,
    start_time: Optional[int],
    end_time: Optional[int],
    start_exclusive: bool,
    end_inclusive: bool,
) -> SwitchTriple:
    vcd.get_node_toggle_count(
        node_name,
        start_time=start_time,
        end_time=end_time,
        start_exclusive=start_exclusive,
        end_inclusive=end_inclusive,
    )
    node_info = vcd.node_switching_info[node_name]
    return (
        node_info.total_dataout_channel_switching,
        node_info.total_valid_switching,
        node_info.total_ready_switching,
    )


def channel_violation(
    est: Optional[int],
    golden: Optional[int],
    rel_threshold: float,
    zero_abs_threshold: int,
) -> Optional[str]:
    if est is None or golden is None:
        return None

    diff = abs(est - golden)
    if golden == 0:
        if diff > zero_abs_threshold:
            return f"abs={diff} > {zero_abs_threshold}"
        return None

    rel = diff / golden
    if rel > rel_threshold:
        return f"rel={rel * 100:.2f}% > {rel_threshold * 100:.2f}%"
    return None


def channel_ignored_small(
    est: Optional[int], golden: Optional[int], ignore_both_below: int
) -> bool:
    if est is None or golden is None:
        return False
    return est < ignore_both_below and golden < ignore_both_below


def main() -> None:
    args = parse_args()

    estimator = read_estimator_csv(args.est_csv)
    vcd = VcdParser(args.vcd)
    start_time, end_time, start_exclusive, end_inclusive, window_desc = get_vcd_window(vcd, args)

    node_types = {
        "cmpi",
        "addi",
        "subi",
        "muli",
        "extsi",
        "extui",
        "trunci",
        "buffer",
        "mc_load",
        "mc_store",
        "lsq_load",
        "lsq_store",
        "merge",
        "control_merge",
        "fork",
        "d_return",
        "cond_br",
        "br",
        "end",
        "andi",
        "ori",
        "xori",
        "shli",
        "shrsi",
        "shrui",
        "select",
        "mux",
        "source",
        "sink",
        "constant",
        "load",
        "store",
    }

    nodes = set(estimator.keys())
    if args.include_vcd_only:
        nodes.update(extract_node_names(vcd.unique_signal_names, node_types))
    nodes_sorted = sorted(nodes)

    headers = [
        "Node",
        "Data Est",
        "Data VCD",
        "Data Diff (E-G)",
        "Data Error",
        "Valid Est",
        "Valid VCD",
        "Valid Diff (E-G)",
        "Valid Error",
        "Ready Est",
        "Ready VCD",
        "Ready Diff (E-G)",
        "Ready Error",
        "Status",
    ]

    rows: List[List[str]] = []
    violations: List[Tuple[str, str, int, int, str]] = []
    compared_channels = 0
    ignored_small_channels = 0
    channel_stats = {
        "data": {
            "compared": 0,
            "passed": 0,
            "rel_errors": [],
            "abs_errors": [],
            "zero_abs_errors": [],
        },
        "valid": {
            "compared": 0,
            "passed": 0,
            "rel_errors": [],
            "abs_errors": [],
            "zero_abs_errors": [],
        },
        "ready": {
            "compared": 0,
            "passed": 0,
            "rel_errors": [],
            "abs_errors": [],
            "zero_abs_errors": [],
        },
    }

    for node_name in nodes_sorted:
        est_switching = estimator.get(node_name)
        if est_switching is None:
            continue
        est_data, est_valid, est_ready = est_switching

        node_visible_in_vcd = has_node_signal(node_name, vcd.unique_signal_names)
        if node_visible_in_vcd:
            vcd_data, vcd_valid, vcd_ready = get_vcd_switching(
                vcd,
                node_name,
                start_time=start_time,
                end_time=end_time,
                start_exclusive=start_exclusive,
                end_inclusive=end_inclusive,
            )
        else:
            vcd_data, vcd_valid, vcd_ready = (None, None, None)

        node_status = "N/A"
        node_has_compared = False
        node_failed = False

        for channel_name, est_val, vcd_val in (
            ("data", est_data, vcd_data),
            ("valid", est_valid, vcd_valid),
            ("ready", est_ready, vcd_ready),
        ):
            if vcd_val is None:
                continue
            if channel_ignored_small(est_val, vcd_val, args.ignore_both_below):
                ignored_small_channels += 1
                continue

            node_has_compared = True
            compared_channels += 1
            stats = channel_stats[channel_name]
            stats["compared"] += 1
            abs_err = abs(est_val - vcd_val)
            stats["abs_errors"].append(abs_err)
            if vcd_val == 0:
                stats["zero_abs_errors"].append(abs_err)
            else:
                stats["rel_errors"].append(abs_err / vcd_val)
            violation = channel_violation(
                est=est_val,
                golden=vcd_val,
                rel_threshold=args.rel_error_threshold,
                zero_abs_threshold=args.zero_abs_threshold,
            )
            if violation is not None:
                node_failed = True
                violations.append((node_name, channel_name, est_val, vcd_val, violation))
            else:
                stats["passed"] += 1

        if node_has_compared:
            node_status = "FAIL" if node_failed else "PASS"

        rows.append(
            [
                node_name,
                fmt_value(est_data),
                fmt_value(vcd_data),
                format_diff(est_data, vcd_data),
                format_error_rate(est_data, vcd_data),
                fmt_value(est_valid),
                fmt_value(vcd_valid),
                format_diff(est_valid, vcd_valid),
                format_error_rate(est_valid, vcd_valid),
                fmt_value(est_ready),
                fmt_value(vcd_ready),
                format_diff(est_ready, vcd_ready),
                format_error_rate(est_ready, vcd_ready),
                node_status,
            ]
        )

    print(f"[INFO] Window: {window_desc}")
    print(
        f"[INFO] Thresholds: rel<={args.rel_error_threshold:.4f} for golden>0, "
        f"abs<={args.zero_abs_threshold} for golden=0"
    )
    print(
        "[INFO] Ignore rule: skip channel when both est and golden are < "
        f"{args.ignore_both_below}"
    )
    print(build_pretty_table(headers, rows))

    total_nodes = len(rows)
    print(f"[SUMMARY] Compared nodes: {total_nodes}")
    print(f"[SUMMARY] Compared channels: {compared_channels}")
    print(f"[SUMMARY] Ignored small channels: {ignored_small_channels}")
    print(f"[SUMMARY] Violating channels: {len(violations)}")

    channel_headers = [
        "Channel",
        "Compared",
        "Passed",
        "Pass Rate",
        "Mean Rel Err",
        "Median Rel Err",
        "Max Rel Err",
        "Mean Abs Err",
        "Max Abs Err",
        "Mean Abs Err (golden=0)",
        "Max Abs Err (golden=0)",
    ]
    channel_rows: List[List[str]] = []
    for channel in ("data", "valid", "ready"):
        stats = channel_stats[channel]
        compared = int(stats["compared"])
        passed = int(stats["passed"])
        pass_rate = (passed / compared) if compared > 0 else 1.0
        rel_errors = stats["rel_errors"]
        abs_errors = stats["abs_errors"]
        zero_abs_errors = stats["zero_abs_errors"]

        mean_rel = statistics.fmean(rel_errors) if rel_errors else 0.0
        median_rel = statistics.median(rel_errors) if rel_errors else 0.0
        max_rel = max(rel_errors) if rel_errors else 0.0
        mean_abs = statistics.fmean(abs_errors) if abs_errors else 0.0
        max_abs = max(abs_errors) if abs_errors else 0
        mean_zero_abs = (
            statistics.fmean(zero_abs_errors) if zero_abs_errors else 0.0
        )
        max_zero_abs = max(zero_abs_errors) if zero_abs_errors else 0

        channel_rows.append(
            [
                channel,
                str(compared),
                str(passed),
                fmt_percent(pass_rate),
                fmt_percent(mean_rel),
                fmt_percent(median_rel),
                fmt_percent(max_rel),
                f"{mean_abs:.2f}",
                str(max_abs),
                f"{mean_zero_abs:.2f}",
                str(max_zero_abs),
            ]
        )
    print("[SUMMARY] Per-channel accuracy metrics:")
    print(build_pretty_table(channel_headers, channel_rows))

    if violations:
        print("[SUMMARY] Top violations:")
        for idx, (node_name, channel_name, est_val, vcd_val, reason) in enumerate(
            violations[: args.max_violation_report], start=1
        ):
            print(
                f"  {idx:2d}. {node_name}.{channel_name}: "
                f"est={est_val}, golden={vcd_val}, {reason}"
            )
        if len(violations) > args.max_violation_report:
            print(f"  ... and {len(violations) - args.max_violation_report} more violations")

    if not args.no_enforce_thresholds and violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
