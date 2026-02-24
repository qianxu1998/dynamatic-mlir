#!/usr/bin/env python3
"""
Dump data-channel values from a VCD in human-readable decimal form.

The script reconstructs vector values over time for channel families used by the
switching-estimation pass (e.g., dataOut/outs/result/addrOut/index/...).
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

from test import get_vcd_window
from vcd_parser import VcdParser


CHANNEL_SUFFIX_RE = re.compile(
    r"^(?P<node>.+)_(?P<channel>"
    r"result|outs(?:_[0-9]+)?|dataOut(?:Array)?(?:_[0-9]+)?|"
    r"addrOut|dataToMem|index|trueOut|falseOut)$"
)
INDEXED_SIGNAL_RE = re.compile(r"^(?P<base>.+)\[(?P<bit>[0-9]+)\]$")


@dataclass
class DataVector:
    full_name: str
    node: str
    channel: str
    width: int = 0
    bit_signals: Dict[int, str] = field(default_factory=dict)
    packed_signal: Optional[str] = None


@dataclass
class DumpRow:
    time: int
    binary: str
    decimal: Optional[int]


@dataclass
class PropagationEvent:
    time: int
    channel_id: str
    vector: DataVector
    row: DumpRow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract data-channel vector values from a VCD and dump decimal traces."
    )
    parser.add_argument("--vcd", required=True, help="Path to VCD file (e.g., trace.vcd).")
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        help="Exact node name filter (repeat for multiple nodes). Example: --node load0",
    )
    parser.add_argument(
        "--vector-regex",
        default="",
        help="Optional regex filter applied to full vector names.",
    )
    parser.add_argument(
        "--window",
        default="full",
        choices=("full", "post-reset"),
        help="Window selection when --start-time/--end-time are not used.",
    )
    parser.add_argument(
        "--reset-signal",
        default="",
        help="Optional reset signal hint for --window post-reset.",
    )
    parser.add_argument(
        "--start-time",
        type=int,
        default=None,
        help="Optional explicit window start time (inclusive). Overrides --window logic.",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=None,
        help="Optional explicit window end time (inclusive). Overrides --window logic.",
    )
    parser.add_argument(
        "--include-initial",
        action="store_true",
        help="Also dump one initial row at --start-time (or auto-window start) when available.",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Interpret decimal value as signed two's complement.",
    )
    parser.add_argument(
        "--show-binary",
        action="store_true",
        help="Include binary values in text output.",
    )
    parser.add_argument(
        "--view",
        default="per-vector",
        choices=("per-vector", "propagation", "channel-sequential"),
        help=(
            "Output layout: 'per-vector' prints one table per channel; "
            "'propagation' prints channel index + time-ordered global event stream; "
            "'channel-sequential' prints channel index + per-channel sequential values."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional text output path. Default: print to stdout.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path (always includes both binary and decimal columns).",
    )
    return parser.parse_args()


def _time_in_window(
    time: int,
    start_time: Optional[int],
    end_time: Optional[int],
    start_exclusive: bool,
    end_inclusive: bool,
) -> bool:
    if start_time is not None:
        if start_exclusive:
            if not (time > start_time):
                return False
        elif not (time >= start_time):
            return False

    if end_time is not None:
        if end_inclusive:
            if not (time <= end_time):
                return False
        elif not (time < end_time):
            return False

    return True


def _normalize_bit(value: str) -> str:
    raw = str(value).strip().lower()
    if not raw:
        return "x"
    bit = raw[0]
    if bit in ("0", "1", "x", "z"):
        return bit
    return "x"


def _normalize_vector_value(value: str, width: int) -> str:
    raw = str(value).strip().lower()
    if not raw:
        return "x" * width

    if raw[0] in ("b", "r") and len(raw) > 1:
        raw = raw[1:]

    raw = "".join(ch if ch in ("0", "1", "x", "z") else "x" for ch in raw)

    if len(raw) < width:
        pad = raw[0] if raw and raw[0] in ("x", "z") else "0"
        raw = pad * (width - len(raw)) + raw
    elif len(raw) > width:
        raw = raw[-width:]

    return raw


def _bits_to_binary(bits: Dict[int, str], width: int) -> str:
    return "".join(bits.get(bit_idx, "x") for bit_idx in range(width - 1, -1, -1))


def _binary_to_decimal(binary: str, signed: bool) -> Optional[int]:
    if any(ch not in ("0", "1") for ch in binary):
        return None

    value = int(binary, 2)
    if signed and binary and binary[0] == "1":
        value -= 1 << len(binary)
    return value


def _decimal_str(value: Optional[int]) -> str:
    return "X" if value is None else str(value)


def _signal_width(vcd: VcdParser, signal_name: str) -> int:
    signal_id = vcd.name_to_id.get(signal_name)
    if signal_id is None:
        return 1
    signal = vcd.signals.get(signal_id)
    if signal is None:
        return 1
    try:
        return int(signal.bit_width)
    except ValueError:
        return 1


def _split_channel(base_name: str) -> Optional[Tuple[str, str]]:
    leaf = base_name.split(".")[-1]
    match = CHANNEL_SUFFIX_RE.match(leaf)
    if not match:
        return None
    return match.group("node"), match.group("channel")


def _node_selected(node: str, node_filter: set[str]) -> bool:
    if not node_filter:
        return True
    return node in node_filter


def _vector_selected(full_name: str, vector_regex: Optional[Pattern[str]]) -> bool:
    if vector_regex is None:
        return True
    return vector_regex.search(full_name) is not None


def collect_data_vectors(
    vcd: VcdParser, node_filter: set[str], vector_regex: Optional[Pattern[str]]
) -> List[DataVector]:
    vectors: Dict[str, DataVector] = {}

    for signal_name in vcd.unique_signal_names:
        indexed_match = INDEXED_SIGNAL_RE.match(signal_name)
        if indexed_match:
            base_name = indexed_match.group("base")
            channel_info = _split_channel(base_name)
            if channel_info is None:
                continue

            node_name, channel_name = channel_info
            if not _node_selected(node_name, node_filter):
                continue
            if not _vector_selected(base_name, vector_regex):
                continue

            bit_index = int(indexed_match.group("bit"))
            vector = vectors.setdefault(
                base_name,
                DataVector(full_name=base_name, node=node_name, channel=channel_name),
            )
            vector.bit_signals[bit_index] = signal_name
            continue

        # Support packed-vector VCDs where each vector is dumped as one signal.
        channel_info = _split_channel(signal_name)
        if channel_info is None:
            continue

        node_name, channel_name = channel_info
        if not _node_selected(node_name, node_filter):
            continue
        if not _vector_selected(signal_name, vector_regex):
            continue

        width = _signal_width(vcd, signal_name)
        if width <= 0:
            continue

        if signal_name in vectors and vectors[signal_name].bit_signals:
            continue

        vectors[signal_name] = DataVector(
            full_name=signal_name,
            node=node_name,
            channel=channel_name,
            width=width,
            packed_signal=signal_name,
        )

    out = []
    for vector in vectors.values():
        if vector.bit_signals:
            vector.width = max(vector.bit_signals) + 1
            vector.packed_signal = None
        if vector.width > 0:
            out.append(vector)

    out.sort(key=lambda v: (v.node, v.channel, v.full_name))
    return out


def build_rows_for_indexed_vector(
    vcd: VcdParser,
    vector: DataVector,
    start_time: Optional[int],
    end_time: Optional[int],
    start_exclusive: bool,
    end_inclusive: bool,
    signed: bool,
    include_initial: bool,
) -> List[DumpRow]:
    bits: Dict[int, str] = {bit: "x" for bit in range(vector.width)}

    if start_time is not None:
        for bit_index, signal_name in vector.bit_signals.items():
            value = vcd[signal_name][start_time]
            if value is not None:
                bits[bit_index] = _normalize_bit(value)

    events_by_time: Dict[int, Dict[int, str]] = {}
    for bit_index, signal_name in vector.bit_signals.items():
        value_list = vcd[signal_name].time_value_pair_list
        for event_time, raw_value in value_list:
            if not _time_in_window(
                event_time,
                start_time=start_time,
                end_time=end_time,
                start_exclusive=start_exclusive,
                end_inclusive=end_inclusive,
            ):
                continue
            events_by_time.setdefault(event_time, {})[bit_index] = _normalize_bit(raw_value)

    rows: List[DumpRow] = []

    if include_initial and start_time is not None:
        initial_binary = _bits_to_binary(bits, vector.width)
        rows.append(
            DumpRow(
                time=start_time,
                binary=initial_binary,
                decimal=_binary_to_decimal(initial_binary, signed),
            )
        )

    for event_time in sorted(events_by_time):
        before = _bits_to_binary(bits, vector.width)
        for bit_index, new_value in events_by_time[event_time].items():
            bits[bit_index] = new_value
        after = _bits_to_binary(bits, vector.width)
        if after == before:
            continue
        rows.append(
            DumpRow(
                time=event_time,
                binary=after,
                decimal=_binary_to_decimal(after, signed),
            )
        )

    return rows


def build_rows_for_packed_vector(
    vcd: VcdParser,
    vector: DataVector,
    start_time: Optional[int],
    end_time: Optional[int],
    start_exclusive: bool,
    end_inclusive: bool,
    signed: bool,
    include_initial: bool,
) -> List[DumpRow]:
    if vector.packed_signal is None:
        return []

    signal = vcd[vector.packed_signal]
    current = "x" * vector.width
    if start_time is not None:
        value_at_start = signal[start_time]
        if value_at_start is not None:
            current = _normalize_vector_value(value_at_start, vector.width)

    rows: List[DumpRow] = []
    if include_initial and start_time is not None:
        rows.append(DumpRow(time=start_time, binary=current, decimal=_binary_to_decimal(current, signed)))

    for event_time, raw_value in signal.time_value_pair_list:
        if not _time_in_window(
            event_time,
            start_time=start_time,
            end_time=end_time,
            start_exclusive=start_exclusive,
            end_inclusive=end_inclusive,
        ):
            continue
        normalized = _normalize_vector_value(raw_value, vector.width)
        if normalized == current:
            continue
        current = normalized
        rows.append(
            DumpRow(
                time=event_time,
                binary=current,
                decimal=_binary_to_decimal(current, signed),
            )
        )

    return rows


def build_rows(
    vcd: VcdParser,
    vector: DataVector,
    start_time: Optional[int],
    end_time: Optional[int],
    start_exclusive: bool,
    end_inclusive: bool,
    signed: bool,
    include_initial: bool,
) -> List[DumpRow]:
    if vector.bit_signals:
        return build_rows_for_indexed_vector(
            vcd,
            vector,
            start_time=start_time,
            end_time=end_time,
            start_exclusive=start_exclusive,
            end_inclusive=end_inclusive,
            signed=signed,
            include_initial=include_initial,
        )
    return build_rows_for_packed_vector(
        vcd,
        vector,
        start_time=start_time,
        end_time=end_time,
        start_exclusive=start_exclusive,
        end_inclusive=end_inclusive,
        signed=signed,
        include_initial=include_initial,
    )


def build_text_report(
    vectors: List[DataVector],
    rows_per_vector: Dict[str, List[DumpRow]],
    vcd_path: str,
    window_desc: str,
    signed: bool,
    show_binary: bool,
) -> str:
    mode = "signed" if signed else "unsigned"
    lines = [
        "# Data-channel value dump",
        f"# vcd: {vcd_path}",
        f"# window: {window_desc}",
        f"# decimal mode: {mode}",
        f"# vectors matched: {len(vectors)}",
        "",
    ]

    for vector in vectors:
        lines.append(
            "=== "
            + f"{vector.full_name} (node={vector.node}, channel={vector.channel}, width={vector.width})"
        )
        rows = rows_per_vector[vector.full_name]

        if not rows:
            lines.append("  (no value changes in selected window)")
            lines.append("")
            continue

        if show_binary:
            lines.append(f"{'time':>14}  {'decimal':>18}  binary")
            lines.append(f"{'-' * 14}  {'-' * 18}  {'-' * max(6, vector.width)}")
            for row in rows:
                lines.append(
                    f"{row.time:>14}  {_decimal_str(row.decimal):>18}  {row.binary}"
                )
        else:
            lines.append(f"{'time':>14}  {'decimal':>18}")
            lines.append(f"{'-' * 14}  {'-' * 18}")
            for row in rows:
                lines.append(f"{row.time:>14}  {_decimal_str(row.decimal):>18}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_channel_ids(vectors: List[DataVector]) -> Dict[str, str]:
    return {vector.full_name: f"CH{index:04d}" for index, vector in enumerate(vectors, start=1)}


def collect_propagation_events(
    vectors: List[DataVector], rows_per_vector: Dict[str, List[DumpRow]]
) -> List[PropagationEvent]:
    channel_ids = build_channel_ids(vectors)
    events: List[PropagationEvent] = []
    for vector in vectors:
        channel_id = channel_ids[vector.full_name]
        for row in rows_per_vector[vector.full_name]:
            events.append(
                PropagationEvent(
                    time=row.time,
                    channel_id=channel_id,
                    vector=vector,
                    row=row,
                )
            )

    events.sort(key=lambda event: (event.time, event.channel_id))
    return events


def build_propagation_report(
    vectors: List[DataVector],
    rows_per_vector: Dict[str, List[DumpRow]],
    vcd_path: str,
    window_desc: str,
    signed: bool,
    show_binary: bool,
) -> str:
    mode = "signed" if signed else "unsigned"
    channel_ids = build_channel_ids(vectors)
    events = collect_propagation_events(vectors, rows_per_vector)

    lines = [
        "# Data-channel propagation dump",
        "# format: channel-index + global time-ordered events",
        f"# vcd: {vcd_path}",
        f"# window: {window_desc}",
        f"# decimal mode: {mode}",
        f"# vectors matched: {len(vectors)}",
        f"# total events: {len(events)}",
        "",
        "## Channel Index",
        f"{'id':>7}  {'node':<20}  {'channel':<16}  {'width':>5}  {'events':>8}  vector",
        f"{'-' * 7}  {'-' * 20}  {'-' * 16}  {'-' * 5}  {'-' * 8}  {'-' * 40}",
    ]

    for vector in vectors:
        lines.append(
            f"{channel_ids[vector.full_name]:>7}  "
            f"{vector.node:<20}  "
            f"{vector.channel:<16}  "
            f"{vector.width:>5}  "
            f"{len(rows_per_vector[vector.full_name]):>8}  "
            f"{vector.full_name}"
        )

    lines.extend(
        [
            "",
            "## Time-Ordered Events",
            "# Use channel id to trace a value through the circuit over time.",
        ]
    )

    if not events:
        lines.append("(no value changes in selected window)")
        return "\n".join(lines).rstrip() + "\n"

    current_time: Optional[int] = None
    if show_binary:
        lines.append("# event row: <channel-id>  <node.channel>  dec=<value>  bin=<value>")
    else:
        lines.append("# event row: <channel-id>  <node.channel>  dec=<value>")

    for event in events:
        if event.time != current_time:
            current_time = event.time
            lines.append("")
            lines.append(f"@{current_time}")

        label = f"{event.vector.node}.{event.vector.channel}"
        dec = _decimal_str(event.row.decimal)
        if show_binary:
            lines.append(
                f"  {event.channel_id:>7}  {label:<36}  dec={dec:>12}  bin={event.row.binary}"
            )
        else:
            lines.append(f"  {event.channel_id:>7}  {label:<36}  dec={dec:>12}")

    return "\n".join(lines).rstrip() + "\n"


def build_channel_sequential_report(
    vectors: List[DataVector],
    rows_per_vector: Dict[str, List[DumpRow]],
    vcd_path: str,
    window_desc: str,
    signed: bool,
    show_binary: bool,
) -> str:
    mode = "signed" if signed else "unsigned"
    channel_ids = build_channel_ids(vectors)
    total_events = sum(len(rows_per_vector[vector.full_name]) for vector in vectors)

    lines = [
        "# Data-channel sequential dump",
        "# format: channel-index + per-channel sequential value trace",
        f"# vcd: {vcd_path}",
        f"# window: {window_desc}",
        f"# decimal mode: {mode}",
        f"# vectors matched: {len(vectors)}",
        f"# total events: {total_events}",
        "",
        "## Channel Index",
        f"{'id':>7}  {'node':<20}  {'channel':<16}  {'width':>5}  {'events':>8}  vector",
        f"{'-' * 7}  {'-' * 20}  {'-' * 16}  {'-' * 5}  {'-' * 8}  {'-' * 40}",
    ]

    for vector in vectors:
        lines.append(
            f"{channel_ids[vector.full_name]:>7}  "
            f"{vector.node:<20}  "
            f"{vector.channel:<16}  "
            f"{vector.width:>5}  "
            f"{len(rows_per_vector[vector.full_name]):>8}  "
            f"{vector.full_name}"
        )

    lines.append("")
    lines.append("## Per-Channel Sequences")
    lines.append("# Each section is one data channel; rows are in channel-local time order.")

    for vector in vectors:
        channel_id = channel_ids[vector.full_name]
        rows = rows_per_vector[vector.full_name]
        lines.append("")
        lines.append(
            f"=== {channel_id} {vector.full_name} "
            f"(node={vector.node}, channel={vector.channel}, width={vector.width}, events={len(rows)})"
        )
        if not rows:
            lines.append("  (no value changes in selected window)")
            continue

        if show_binary:
            lines.append(f"{'seq':>6}  {'time':>14}  {'decimal':>18}  binary")
            lines.append(f"{'-' * 6}  {'-' * 14}  {'-' * 18}  {'-' * max(6, vector.width)}")
            for idx, row in enumerate(rows, start=1):
                lines.append(
                    f"{idx:>6}  {row.time:>14}  {_decimal_str(row.decimal):>18}  {row.binary}"
                )
        else:
            lines.append(f"{'seq':>6}  {'time':>14}  {'decimal':>18}")
            lines.append(f"{'-' * 6}  {'-' * 14}  {'-' * 18}")
            for idx, row in enumerate(rows, start=1):
                lines.append(f"{idx:>6}  {row.time:>14}  {_decimal_str(row.decimal):>18}")

    return "\n".join(lines).rstrip() + "\n"


def write_csv_dump(
    csv_path: Path,
    vectors: List[DataVector],
    rows_per_vector: Dict[str, List[DumpRow]],
    signed: bool,
    view: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "signed" if signed else "unsigned"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if view == "propagation":
            writer.writerow(
                [
                    "time",
                    "channel_id",
                    "node",
                    "channel",
                    "vector",
                    "width",
                    "decimal",
                    "binary",
                    "decimal_mode",
                ]
            )
            for event in collect_propagation_events(vectors, rows_per_vector):
                writer.writerow(
                    [
                        event.time,
                        event.channel_id,
                        event.vector.node,
                        event.vector.channel,
                        event.vector.full_name,
                        event.vector.width,
                        _decimal_str(event.row.decimal),
                        event.row.binary,
                        mode,
                    ]
                )
            return

        if view == "channel-sequential":
            writer.writerow(
                [
                    "channel_id",
                    "seq",
                    "time",
                    "node",
                    "channel",
                    "vector",
                    "width",
                    "decimal",
                    "binary",
                    "decimal_mode",
                ]
            )
            channel_ids = build_channel_ids(vectors)
            for vector in vectors:
                channel_id = channel_ids[vector.full_name]
                rows = rows_per_vector[vector.full_name]
                for seq, row in enumerate(rows, start=1):
                    writer.writerow(
                        [
                            channel_id,
                            seq,
                            row.time,
                            vector.node,
                            vector.channel,
                            vector.full_name,
                            vector.width,
                            _decimal_str(row.decimal),
                            row.binary,
                            mode,
                        ]
                    )
            return

        writer.writerow(
            [
                "vector",
                "node",
                "channel",
                "width",
                "time",
                "decimal",
                "binary",
                "decimal_mode",
            ]
        )
        for vector in vectors:
            for row in rows_per_vector[vector.full_name]:
                writer.writerow(
                    [
                        vector.full_name,
                        vector.node,
                        vector.channel,
                        vector.width,
                        row.time,
                        _decimal_str(row.decimal),
                        row.binary,
                        mode,
                    ]
                )


def resolve_window(
    vcd: VcdParser, args: argparse.Namespace
) -> Tuple[Optional[int], Optional[int], bool, bool, str]:
    if args.start_time is not None or args.end_time is not None:
        start = args.start_time
        end = args.end_time
        desc = f"custom (start={start}, end={end}, inclusive)"
        return start, end, False, True, desc

    window_args = argparse.Namespace(window=args.window, reset_signal=args.reset_signal)
    return get_vcd_window(vcd, window_args)


def main() -> None:
    args = parse_args()
    vcd = VcdParser(args.vcd)

    start_time, end_time, start_exclusive, end_inclusive, window_desc = resolve_window(vcd, args)

    node_filter = set(args.node)
    vector_regex = re.compile(args.vector_regex) if args.vector_regex else None
    vectors = collect_data_vectors(vcd, node_filter=node_filter, vector_regex=vector_regex)

    rows_per_vector: Dict[str, List[DumpRow]] = {}
    for vector in vectors:
        rows_per_vector[vector.full_name] = build_rows(
            vcd,
            vector,
            start_time=start_time,
            end_time=end_time,
            start_exclusive=start_exclusive,
            end_inclusive=end_inclusive,
            signed=args.signed,
            include_initial=args.include_initial,
        )

    if args.view == "propagation":
        report = build_propagation_report(
            vectors=vectors,
            rows_per_vector=rows_per_vector,
            vcd_path=args.vcd,
            window_desc=window_desc,
            signed=args.signed,
            show_binary=args.show_binary,
        )
    elif args.view == "channel-sequential":
        report = build_channel_sequential_report(
            vectors=vectors,
            rows_per_vector=rows_per_vector,
            vcd_path=args.vcd,
            window_desc=window_desc,
            signed=args.signed,
            show_binary=args.show_binary,
        )
    else:
        report = build_text_report(
            vectors=vectors,
            rows_per_vector=rows_per_vector,
            vcd_path=args.vcd,
            window_desc=window_desc,
            signed=args.signed,
            show_binary=args.show_binary,
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Wrote text dump: {output_path}")
    else:
        print(report, end="")

    if args.csv:
        csv_path = Path(args.csv)
        write_csv_dump(
            csv_path,
            vectors=vectors,
            rows_per_vector=rows_per_vector,
            signed=args.signed,
            view=args.view,
        )
        print(f"Wrote CSV dump: {csv_path}")


if __name__ == "__main__":
    main()
