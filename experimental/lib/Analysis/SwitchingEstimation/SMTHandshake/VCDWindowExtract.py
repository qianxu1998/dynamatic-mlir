#!/usr/bin/env python3
"""Extract one steady-state II window per MG from VCD and compare against SMT waveforms.

This script compares the SMT-estimated valid/ready/transfer waveforms from
`smt-handshake-dump` JSON against ModelSim VCD traces, restricted to MG
steady-state windows (runs with at least 3 consecutive occurrences).
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "switching_testing" / "vcd_parser.py").exists():
            return parent
    raise RuntimeError("Could not locate repo root from script location")


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "switching_testing"))
from vcd_parser import VcdParser  # pylint: disable=wrong-import-position


EdgeKey = Tuple[str, str]


@dataclass
class EdgeMismatch:
    edge: EdgeKey
    valid_mismatches: List[int]
    ready_mismatches: List[int]
    transfer_mismatches: List[int]
    valid_toggle_smt: int
    valid_toggle_vcd: int
    ready_toggle_smt: int
    ready_toggle_vcd: int

    @property
    def mismatch_score(self) -> int:
        return (
            len(self.valid_mismatches)
            + len(self.ready_mismatches)
            + len(self.transfer_mismatches)
        )


@dataclass
class MGComparison:
    mg_label: str
    status: str
    reason: str
    ii: int
    rotation: int
    selected_phase: Optional[int]
    selected_run_count: Optional[int]
    selected_segment_index: Optional[int]
    selected_occurrence_index: Optional[int]
    selected_anchor_transfer_index: Optional[int]
    selected_start_cycle: Optional[int]
    selected_window_shift: Optional[int]
    anchor_edge: Optional[EdgeKey]
    mapped_edge_count: int
    compared_edge_count: int
    boundary_ignored_edge_count: int
    worst_edges: List[EdgeMismatch]
    note: str = ""
    valid_mismatch_points: int = 0
    ready_mismatch_points: int = 0
    transfer_mismatch_points: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SMT MG waveforms against one steady-state II VCD window per MG"
    )
    parser.add_argument("--smt-json", required=True, help="SMT JSON dump path")
    parser.add_argument("--vcd", required=True, help="ModelSim VCD trace path")
    parser.add_argument("--top-verilog", required=True, help="Top-level generated Verilog path")
    parser.add_argument(
        "--mg-label",
        action="append",
        default=[],
        help="Restrict comparison to selected MG label(s). Can be repeated.",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=3,
        help="Minimum consecutive executions to consider an MG steady-state run",
    )
    parser.add_argument(
        "--max-worst",
        type=int,
        default=10,
        help="Max number of worst edges to print/store per MG",
    )
    parser.add_argument(
        "--steady-occurrence",
        default="auto",
        help=(
            "Occurrence index (1-based) selected inside a steady-state run. "
            "Use 'auto' (default) to prefer 4th/3rd occurrence and avoid run-entry skew."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path for detailed comparison results",
    )
    parser.add_argument(
        "--include-boundary-channels",
        action="store_true",
        help=(
            "Include boundary channels in mismatch comparison. "
            "Default behavior ignores boundary channels."
        ),
    )
    return parser.parse_args()


def parse_bitstring(bits: str, expected_len: int) -> List[int]:
    out = [1 if c == "1" else 0 for c in bits.strip()]
    if len(out) < expected_len:
        out.extend([0] * (expected_len - len(out)))
    return out[:expected_len]


def rotate_left(bits: Sequence[int], shift: int) -> List[int]:
    if not bits:
        return []
    shift %= len(bits)
    if shift == 0:
        return list(bits)
    return list(bits[shift:]) + list(bits[:shift])


def hamming(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def cyclic_toggles(bits: Sequence[int]) -> int:
    if not bits:
        return 0
    return sum(1 for i in range(len(bits)) if bits[i] != bits[(i + 1) % len(bits)])


def is_mg_label(seg: str) -> bool:
    return seg.isdigit()


def parse_preferred_occurrence(raw: str) -> Optional[int]:
    if raw.strip().lower() == "auto":
        return None
    occ = int(raw)
    if occ <= 0:
        raise ValueError("--steady-occurrence must be >= 1 or 'auto'")
    return occ


def choose_local_occurrence(run_count: int, preferred_occurrence: Optional[int]) -> int:
    """Return 0-based index inside one run.

    Auto policy prefers later interior points to avoid warm-up behavior:
    - run >= 5: pick 4th occurrence (index 3)
    - run >= 3: pick 3rd occurrence (index 2)
    - run == 2: pick 2nd occurrence (index 1)
    - run == 1: pick 1st occurrence (index 0)
    """

    if run_count <= 0:
        return 0
    if preferred_occurrence is not None:
        return min(preferred_occurrence - 1, run_count - 1)
    if run_count >= 5:
        return 3
    if run_count >= 3:
        return 2
    if run_count == 2:
        return 1
    return 0


def choose_steady_occurrence(
    mg_label: str,
    phases: Sequence[dict],
    executed_segments: Sequence[str],
    min_consecutive: int,
    preferred_occurrence: Optional[int],
) -> Optional[Tuple[int, int, int, int, int]]:
    """Return (phase, run_count, seg_index, mg_occurrence_index, global_mg_occurrence)."""

    if phases:
        seg_pos = 0
        for phase_obj in phases:
            seg = str(phase_obj.get("segment", ""))
            count = int(phase_obj.get("count", 0))
            phase_idx = int(phase_obj.get("phase", -1))
            if seg == mg_label and count >= min_consecutive:
                local_occ = choose_local_occurrence(count, preferred_occurrence)
                selected_seg_index = seg_pos + local_occ
                if selected_seg_index >= len(executed_segments):
                    return None
                mg_occurrence_index = sum(
                    1
                    for s in executed_segments[: selected_seg_index + 1]
                    if str(s) == mg_label
                ) - 1
                return (
                    phase_idx,
                    count,
                    selected_seg_index,
                    local_occ,
                    mg_occurrence_index,
                )
            seg_pos += count

    # Metadata fallback: derive runs from executed segment trace.
    if not executed_segments:
        return None

    phase_idx = 0
    i = 0
    mg_seen = 0
    while i < len(executed_segments):
        seg = str(executed_segments[i])
        j = i
        while j < len(executed_segments) and str(executed_segments[j]) == seg:
            j += 1
        run_count = j - i
        if seg == mg_label and run_count >= min_consecutive:
            local_occ = choose_local_occurrence(run_count, preferred_occurrence)
            selected_seg_index = i + local_occ
            mg_occurrence_index = mg_seen + local_occ
            return (
                phase_idx,
                run_count,
                selected_seg_index,
                local_occ,
                mg_occurrence_index,
            )
        if seg == mg_label:
            mg_seen += run_count
        i = j
        phase_idx += 1

    return None


def parse_instances(top_verilog: str) -> Dict[str, Dict[str, str]]:
    text = Path(top_verilog).read_text(encoding="utf-8", errors="ignore")
    inst_map: Dict[str, Dict[str, str]] = {}

    # Parse module instantiations in the top module body.
    inst_re = re.compile(
        r"(?ms)^\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:#\s*\(.*?\))?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\);"
    )
    port_re = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)")

    for match in inst_re.finditer(text):
        inst_name = match.group(1)
        port_blob = match.group(2)
        ports: Dict[str, str] = {}
        for pm in port_re.finditer(port_blob):
            ports[pm.group(1)] = pm.group(2).strip()
        if ports:
            inst_map[inst_name] = ports

    return inst_map


def expr_signals(expr: str) -> List[str]:
    cleaned = expr.replace("{", " ").replace("}", " ")
    out: List[str] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        if "'" in token:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?", token):
            out.append(token)
    return out


def signal_belongs_to_src(wire: str, src: str) -> bool:
    base = re.sub(r"_(valid|ready)$", "", wire)
    return base == src or base.startswith(src + "_")


def choose_channel_wire(
    candidates_valid: Sequence[str], candidates_ready: Sequence[str]
) -> Optional[Tuple[str, str]]:
    valid_by_base = {re.sub(r"_valid$", "", w): w for w in candidates_valid}
    ready_by_base = {re.sub(r"_ready$", "", w): w for w in candidates_ready}
    common = sorted(set(valid_by_base) & set(ready_by_base))
    if common:
        base = common[0]
        return (valid_by_base[base], ready_by_base[base])
    if len(candidates_valid) == 1 and len(candidates_ready) == 1:
        return (candidates_valid[0], candidates_ready[0])
    return None


def map_edges_to_wires(
    edges: Iterable[EdgeKey], instances: Dict[str, Dict[str, str]]
) -> Dict[EdgeKey, Tuple[str, str]]:
    mapping: Dict[EdgeKey, Tuple[str, str]] = {}
    for src, dst in edges:
        ports = instances.get(dst)
        if not ports:
            continue
        all_signals: List[str] = []
        for expr in ports.values():
            all_signals.extend(expr_signals(expr))

        valid_candidates = sorted(
            {
                s
                for s in all_signals
                if s.endswith("_valid") and signal_belongs_to_src(s, src)
            }
        )
        ready_candidates = sorted(
            {
                s
                for s in all_signals
                if s.endswith("_ready") and signal_belongs_to_src(s, src)
            }
        )
        pair = choose_channel_wire(valid_candidates, ready_candidates)
        if pair:
            mapping[(src, dst)] = pair
    return mapping


def logic_to_bit(raw: str) -> int:
    s = str(raw).strip()
    if not s:
        return 0
    c = s[0]
    if c == "1":
        return 1
    if c == "0":
        return 0
    if c in ("b", "B") and len(s) > 1:
        return 1 if "1" in s[1:] else 0
    return 0


def collapse_events(signal) -> List[Tuple[int, str]]:
    events: List[Tuple[int, str]] = []
    for t, v in signal.time_value_pair_list:
        if events and events[-1][0] == t:
            events[-1] = (t, v)
        else:
            events.append((t, v))
    return events


def sample_events(events: Sequence[Tuple[int, str]], sample_times: Sequence[int]) -> List[int]:
    out: List[int] = []
    idx = 0
    cur = 0
    for t in sample_times:
        while idx < len(events) and events[idx][0] <= t:
            cur = logic_to_bit(events[idx][1])
            idx += 1
        out.append(cur)
    return out


def detect_signal_by_suffix(all_names: Sequence[str], wire_name: str) -> Optional[str]:
    exact = [n for n in all_names if n == wire_name]
    if len(exact) == 1:
        return exact[0]
    suffix = [n for n in all_names if n.endswith("." + wire_name)]
    if not suffix:
        return None
    # Prefer the shortest hierarchical path (usually top DUT scope).
    return min(suffix, key=len)


def detect_clock_signal(all_names: Sequence[str]) -> str:
    cands = [
        n
        for n in all_names
        if re.search(r"(?:^|[._])clk(?:$|[._\[])", n.lower())
    ]
    if not cands:
        raise RuntimeError("No clock-like signal found in VCD")

    def score(name: str) -> Tuple[int, int]:
        s = 0
        lname = name.lower()
        if lname.endswith(".clk"):
            s += 50
        if ".duv" in lname or ".dut" in lname:
            s += 20
        return (s, -name.count("."))

    return max(cands, key=score)


def detect_reset_signal(all_names: Sequence[str]) -> Optional[str]:
    cands = [
        n
        for n in all_names
        if re.search(r"(?:^|[._])(rst|reset)(?:$|[._\[])", n.lower())
    ]
    if not cands:
        return None

    def score(name: str) -> Tuple[int, int]:
        s = 0
        lname = name.lower()
        if lname.endswith(".rst"):
            s += 40
        if lname.endswith(".reset"):
            s += 30
        return (s, -name.count("."))

    return max(cands, key=score)


def find_rising_edges(events: Sequence[Tuple[int, str]]) -> List[int]:
    rises: List[int] = []
    prev = 0
    for t, v in events:
        cur = logic_to_bit(v)
        if prev == 0 and cur == 1:
            rises.append(t)
        prev = cur
    return rises


def find_reset_deassert(events: Sequence[Tuple[int, str]]) -> Optional[int]:
    prev = None
    first_zero = None
    for t, v in events:
        cur = logic_to_bit(v)
        if cur == 0 and first_zero is None:
            first_zero = t
        if prev == 1 and cur == 0:
            return t
        prev = cur
    return first_zero


def map_mg_occurrence_to_anchor_transfer(
    global_mg_occ: int,
    mg_total_occurrences: int,
    anchor_transfer_count: int,
) -> Optional[int]:
    """Map MG occurrence index to anchor-transfer index.

    When anchor transfers are one-per-MG, this is identity. If an anchor emits
    multiple transfers per MG occurrence (or vice versa), we map by stride or
    proportional interpolation.
    """

    if anchor_transfer_count <= 0:
        return None
    if global_mg_occ < 0:
        return None

    if mg_total_occurrences <= 0:
        return min(global_mg_occ, anchor_transfer_count - 1)

    if anchor_transfer_count == mg_total_occurrences:
        return min(global_mg_occ, anchor_transfer_count - 1)

    if anchor_transfer_count % mg_total_occurrences == 0:
        stride = anchor_transfer_count // mg_total_occurrences
        return min(global_mg_occ * stride, anchor_transfer_count - 1)

    if mg_total_occurrences > 1:
        ratio = float(anchor_transfer_count - 1) / float(mg_total_occurrences - 1)
        mapped = int(round(global_mg_occ * ratio))
        return max(0, min(mapped, anchor_transfer_count - 1))

    return min(global_mg_occ, anchor_transfer_count - 1)


def edge_vectors_from_smt(mg_obj: dict) -> Dict[EdgeKey, Dict[str, List[int]]]:
    ii = int(mg_obj.get("ii", 0))
    out: Dict[EdgeKey, Dict[str, List[int]]] = {}
    for ch in mg_obj.get("channels", []):
        key = (str(ch.get("src", "")), str(ch.get("dst", "")))
        valid = parse_bitstring(str(ch.get("valid", "")), ii)
        ready = parse_bitstring(str(ch.get("ready", "")), ii)
        transfer = parse_bitstring(str(ch.get("transfer", "")), ii)
        out[key] = {"valid": valid, "ready": ready, "transfer": transfer}
    return out


def edge_presence_by_mg_label(mg_objs: Sequence[dict]) -> Dict[str, Set[EdgeKey]]:
    presence: Dict[str, Set[EdgeKey]] = {}
    for mg_obj in mg_objs:
        label = str(mg_obj.get("mg_label", ""))
        edges: Set[EdgeKey] = set()
        for ch in mg_obj.get("channels", []):
            edges.add((str(ch.get("src", "")), str(ch.get("dst", ""))))
        presence[label] = edges
    return presence


def is_boundary_edge(edge: EdgeKey) -> bool:
    """Return True when edge likely crosses MG/local-model boundaries.

    We intentionally ignore channels touching known memory-interface units since
    their handshake decomposition depends on LSQ/memory logic outside the MG
    cut, even when transfer matches.
    """

    src, dst = edge
    names = (src.lower(), dst.lower())
    boundary_tokens = (
        "load",
        "store",
        "lsq",
        "mem_controller",
        "mc_",
    )
    return any(any(tok in name for tok in boundary_tokens) for name in names)


def pick_best_rotation(
    ii: int,
    smt_vectors: Dict[EdgeKey, Dict[str, List[int]]],
    vcd_vectors: Dict[EdgeKey, Dict[str, List[int]]],
    anchor_edge: Optional[EdgeKey],
) -> int:
    common_edges = sorted(set(smt_vectors) & set(vcd_vectors))
    if not common_edges or ii <= 0:
        return 0

    anchor = anchor_edge if anchor_edge in common_edges else common_edges[0]
    best = (10**9, 10**9, 0)
    for k in range(ii):
        anchor_dist = hamming(
            rotate_left(smt_vectors[anchor]["transfer"], k),
            vcd_vectors[anchor]["transfer"],
        )
        total_transfer = 0
        for edge in common_edges:
            total_transfer += hamming(
                rotate_left(smt_vectors[edge]["transfer"], k),
                vcd_vectors[edge]["transfer"],
            )
        candidate = (anchor_dist, total_transfer, k)
        if candidate < best:
            best = candidate
    return best[2]


def compare_mg(
    mg_obj: dict,
    vcd: VcdParser,
    instances: Dict[str, Dict[str, str]],
    edge_presence_map: Dict[str, Set[EdgeKey]],
    phases: Sequence[dict],
    executed_segments: Sequence[str],
    min_consecutive: int,
    preferred_occurrence: Optional[int],
    max_worst: int,
    include_boundary_channels: bool,
) -> MGComparison:
    mg_label = str(mg_obj.get("mg_label", ""))
    status = str(mg_obj.get("status", ""))
    ii = int(mg_obj.get("ii", 0))

    if status != "SAT":
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason=f"SMT status is {status}",
            ii=ii,
            rotation=0,
            selected_phase=None,
            selected_run_count=None,
            selected_segment_index=None,
            selected_occurrence_index=None,
            selected_anchor_transfer_index=None,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=None,
            mapped_edge_count=0,
            compared_edge_count=0,
            boundary_ignored_edge_count=0,
            worst_edges=[],
        )

    selection = choose_steady_occurrence(
        mg_label,
        phases,
        executed_segments,
        min_consecutive,
        preferred_occurrence,
    )
    if selection is None:
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason=f"No steady-state run with count>={min_consecutive}",
            ii=ii,
            rotation=0,
            selected_phase=None,
            selected_run_count=None,
            selected_segment_index=None,
            selected_occurrence_index=None,
            selected_anchor_transfer_index=None,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=None,
            mapped_edge_count=0,
            compared_edge_count=0,
            boundary_ignored_edge_count=0,
            worst_edges=[],
        )

    phase_idx, run_count, seg_index, interior_occ, global_mg_occ = selection

    smt_vectors = edge_vectors_from_smt(mg_obj)
    edge_map = map_edges_to_wires(smt_vectors.keys(), instances)

    all_names = list(vcd.unique_signal_names)
    clock_name = detect_clock_signal(all_names)
    clock_events = collapse_events(vcd.signals[vcd.name_to_id[clock_name]])
    rising_edges = find_rising_edges(clock_events)

    reset_name = detect_reset_signal(all_names)
    if reset_name is not None:
        reset_events = collapse_events(vcd.signals[vcd.name_to_id[reset_name]])
        deassert_time = find_reset_deassert(reset_events)
        if deassert_time is not None:
            rising_edges = [t for t in rising_edges if t >= deassert_time]

    anchor_edge: Optional[EdgeKey] = None
    if mg_obj.get("has_vcd_anchor", False):
        anchor_edge = (
            str(mg_obj.get("vcd_anchor_src", "")),
            str(mg_obj.get("vcd_anchor_dst", "")),
        )
    elif mg_obj.get("has_anchor", False):
        anchor_edge = (str(mg_obj.get("anchor_src", "")), str(mg_obj.get("anchor_dst", "")))
    if anchor_edge not in edge_map:
        backedges = [
            (str(be.get("src", "")), str(be.get("dst", "")))
            for be in mg_obj.get("backedges", [])
            if isinstance(be, dict)
        ]
        for be in backedges:
            if be in edge_map:
                anchor_edge = be
                break
    if anchor_edge not in edge_map:
        common_sorted = sorted(edge_map.keys())
        anchor_edge = common_sorted[0] if common_sorted else None

    if anchor_edge is None:
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason="No edge-to-wire mapping available",
            ii=ii,
            rotation=0,
            selected_phase=phase_idx,
            selected_run_count=run_count,
            selected_segment_index=seg_index,
            selected_occurrence_index=interior_occ,
            selected_anchor_transfer_index=None,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=None,
            mapped_edge_count=0,
            compared_edge_count=0,
            boundary_ignored_edge_count=0,
            worst_edges=[],
        )

    anchor_valid_wire, anchor_ready_wire = edge_map[anchor_edge]
    anchor_valid_sig = detect_signal_by_suffix(all_names, anchor_valid_wire)
    anchor_ready_sig = detect_signal_by_suffix(all_names, anchor_ready_wire)
    if anchor_valid_sig is None or anchor_ready_sig is None:
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason="Anchor wire not found in VCD",
            ii=ii,
            rotation=0,
            selected_phase=phase_idx,
            selected_run_count=run_count,
            selected_segment_index=seg_index,
            selected_occurrence_index=interior_occ,
            selected_anchor_transfer_index=None,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=anchor_edge,
            mapped_edge_count=len(edge_map),
            compared_edge_count=0,
            boundary_ignored_edge_count=0,
            worst_edges=[],
        )

    cache_samples: Dict[str, List[int]] = {}

    def sampled_bits(signal_name: str) -> List[int]:
        if signal_name in cache_samples:
            return cache_samples[signal_name]
        signal = vcd.signals[vcd.name_to_id[signal_name]]
        events = collapse_events(signal)
        bits = sample_events(events, rising_edges)
        cache_samples[signal_name] = bits
        return bits

    anchor_v = sampled_bits(anchor_valid_sig)
    anchor_r = sampled_bits(anchor_ready_sig)
    anchor_x = [a & b for a, b in zip(anchor_v, anchor_r)]
    anchor_transfer_cycles = [idx for idx, bit in enumerate(anchor_x) if bit == 1]
    anchor_eligible_labels = {
        label for label, edges in edge_presence_map.items() if anchor_edge in edges
    }
    if not anchor_eligible_labels:
        anchor_eligible_labels = {mg_label}
    eligible_seg_indices = [
        idx for idx, seg in enumerate(executed_segments) if seg in anchor_eligible_labels
    ]
    if not eligible_seg_indices:
        eligible_seg_indices = [
            idx for idx, seg in enumerate(executed_segments) if str(seg) == mg_label
        ]
    if not eligible_seg_indices:
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason="No anchor-eligible segment occurrences in execution trace",
            ii=ii,
            rotation=0,
            selected_phase=phase_idx,
            selected_run_count=run_count,
            selected_segment_index=seg_index,
            selected_occurrence_index=interior_occ,
            selected_anchor_transfer_index=None,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=anchor_edge,
            mapped_edge_count=len(edge_map),
            compared_edge_count=0,
            boundary_ignored_edge_count=0,
            worst_edges=[],
        )

    if seg_index in eligible_seg_indices:
        global_anchor_occ = eligible_seg_indices.index(seg_index)
    else:
        global_anchor_occ = max(0, bisect.bisect_right(eligible_seg_indices, seg_index) - 1)
    mapped_anchor_occ = map_mg_occurrence_to_anchor_transfer(
        global_anchor_occ, len(eligible_seg_indices), len(anchor_transfer_cycles)
    )

    if mapped_anchor_occ is None:
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason="Anchor has no transfers in VCD",
            ii=ii,
            rotation=0,
            selected_phase=phase_idx,
            selected_run_count=run_count,
            selected_segment_index=seg_index,
            selected_occurrence_index=interior_occ,
            selected_anchor_transfer_index=None,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=anchor_edge,
            mapped_edge_count=len(edge_map),
            compared_edge_count=0,
            boundary_ignored_edge_count=0,
            worst_edges=[],
        )

    if mapped_anchor_occ >= len(anchor_transfer_cycles):
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason=(
                f"Anchor transfer index {mapped_anchor_occ} out of range "
                f"(found {len(anchor_transfer_cycles)} transfers)"
            ),
            ii=ii,
            rotation=0,
            selected_phase=phase_idx,
            selected_run_count=run_count,
            selected_segment_index=seg_index,
            selected_occurrence_index=interior_occ,
            selected_anchor_transfer_index=None,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=anchor_edge,
            mapped_edge_count=len(edge_map),
            compared_edge_count=0,
            boundary_ignored_edge_count=0,
            worst_edges=[],
        )

    anchor_cycle_base = anchor_transfer_cycles[mapped_anchor_occ]
    candidate_starts = []
    for delta in range(-(ii - 1), ii):
        start = anchor_cycle_base + delta
        end = start + ii
        if start < 0 or end > len(rising_edges):
            continue
        candidate_starts.append(start)
    candidate_starts = sorted(set(candidate_starts))

    def build_vcd_vectors(start_cycle_index: int) -> Dict[EdgeKey, Dict[str, List[int]]]:
        end_cycle_index = start_cycle_index + ii
        vectors: Dict[EdgeKey, Dict[str, List[int]]] = {}
        for edge, (valid_wire, ready_wire) in edge_map.items():
            valid_sig = detect_signal_by_suffix(all_names, valid_wire)
            ready_sig = detect_signal_by_suffix(all_names, ready_wire)
            if valid_sig is None or ready_sig is None:
                continue
            valid_bits = sampled_bits(valid_sig)[start_cycle_index:end_cycle_index]
            ready_bits = sampled_bits(ready_sig)[start_cycle_index:end_cycle_index]
            transfer_bits = [v & r for v, r in zip(valid_bits, ready_bits)]
            vectors[edge] = {
                "valid": valid_bits,
                "ready": ready_bits,
                "transfer": transfer_bits,
            }
        return vectors

    best_alignment = None
    best_vectors: Dict[EdgeKey, Dict[str, List[int]]] = {}
    best_rotation = 0
    best_start: Optional[int] = None
    best_shift: Optional[int] = None
    boundary_ignored_edges: Set[EdgeKey] = set()

    for start in candidate_starts:
        vcd_vectors = build_vcd_vectors(start)
        if not vcd_vectors:
            continue

        rotation = pick_best_rotation(ii, smt_vectors, vcd_vectors, anchor_edge)
        raw_compared_edges = sorted(set(smt_vectors) & set(vcd_vectors))
        if include_boundary_channels:
            compared_edges = raw_compared_edges
        else:
            compared_edges = [e for e in raw_compared_edges if not is_boundary_edge(e)]
            boundary_ignored_edges.update(set(raw_compared_edges) - set(compared_edges))
        if not compared_edges:
            continue

        total_valid = 0
        total_ready = 0
        total_transfer = 0
        for edge in compared_edges:
            total_valid += hamming(
                rotate_left(smt_vectors[edge]["valid"], rotation),
                vcd_vectors[edge]["valid"],
            )
            total_ready += hamming(
                rotate_left(smt_vectors[edge]["ready"], rotation),
                vcd_vectors[edge]["ready"],
            )
            total_transfer += hamming(
                rotate_left(smt_vectors[edge]["transfer"], rotation),
                vcd_vectors[edge]["transfer"],
            )

        anchor_dist = 0
        if anchor_edge in smt_vectors and anchor_edge in vcd_vectors:
            anchor_dist = hamming(
                rotate_left(smt_vectors[anchor_edge]["transfer"], rotation),
                vcd_vectors[anchor_edge]["transfer"],
            )
        candidate = (
            anchor_dist,
            total_transfer,
            total_valid + total_ready,
            abs(start - anchor_cycle_base),
        )
        if best_alignment is None or candidate < best_alignment:
            best_alignment = candidate
            best_vectors = vcd_vectors
            best_rotation = rotation
            best_start = start
            best_shift = start - anchor_cycle_base

    if best_alignment is None or not best_vectors:
        return MGComparison(
            mg_label=mg_label,
            status="SKIPPED",
            reason="No mapped edge signals found in VCD",
            ii=ii,
            rotation=0,
            selected_phase=phase_idx,
            selected_run_count=run_count,
            selected_segment_index=seg_index,
            selected_occurrence_index=interior_occ,
            selected_anchor_transfer_index=mapped_anchor_occ,
            selected_start_cycle=None,
            selected_window_shift=None,
            anchor_edge=anchor_edge,
            mapped_edge_count=len(edge_map),
            compared_edge_count=0,
            boundary_ignored_edge_count=len(boundary_ignored_edges),
            worst_edges=[],
        )

    vcd_vectors = best_vectors
    rotation = best_rotation

    mismatches: List[EdgeMismatch] = []
    raw_compared_edges = sorted(set(smt_vectors) & set(vcd_vectors))
    if include_boundary_channels:
        compared_edges = raw_compared_edges
    else:
        compared_edges = [e for e in raw_compared_edges if not is_boundary_edge(e)]
        boundary_ignored_edges.update(set(raw_compared_edges) - set(compared_edges))
    for edge in compared_edges:
        smt_valid = rotate_left(smt_vectors[edge]["valid"], rotation)
        smt_ready = rotate_left(smt_vectors[edge]["ready"], rotation)
        smt_transfer = rotate_left(smt_vectors[edge]["transfer"], rotation)

        vcd_valid = vcd_vectors[edge]["valid"]
        vcd_ready = vcd_vectors[edge]["ready"]
        vcd_transfer = vcd_vectors[edge]["transfer"]

        valid_m = [i for i in range(ii) if smt_valid[i] != vcd_valid[i]]
        ready_m = [i for i in range(ii) if smt_ready[i] != vcd_ready[i]]
        transfer_m = [i for i in range(ii) if smt_transfer[i] != vcd_transfer[i]]

        mismatches.append(
            EdgeMismatch(
                edge=edge,
                valid_mismatches=valid_m,
                ready_mismatches=ready_m,
                transfer_mismatches=transfer_m,
                valid_toggle_smt=cyclic_toggles(smt_valid),
                valid_toggle_vcd=cyclic_toggles(vcd_valid),
                ready_toggle_smt=cyclic_toggles(smt_ready),
                ready_toggle_vcd=cyclic_toggles(vcd_ready),
            )
        )

    mismatches.sort(key=lambda m: m.mismatch_score, reverse=True)
    worst = [m for m in mismatches if m.mismatch_score > 0][:max_worst]

    total_valid_m = sum(len(m.valid_mismatches) for m in mismatches)
    total_ready_m = sum(len(m.ready_mismatches) for m in mismatches)
    total_transfer_m = sum(len(m.transfer_mismatches) for m in mismatches)
    note = ""
    if total_ready_m > (2 * max(1, total_valid_m)):
        note = "Ready mismatches dominate valid mismatches; likely backpressure phase skew."
    elif total_transfer_m == 0 and (total_valid_m + total_ready_m) > 0:
        note = "Transfer matches but valid/ready split differs; handshake decomposition mismatch."

    return MGComparison(
        mg_label=mg_label,
        status="OK",
        reason="",
        ii=ii,
        rotation=rotation,
        selected_phase=phase_idx,
        selected_run_count=run_count,
        selected_segment_index=seg_index,
        selected_occurrence_index=interior_occ,
        selected_anchor_transfer_index=mapped_anchor_occ,
        selected_start_cycle=best_start,
        selected_window_shift=best_shift,
        anchor_edge=anchor_edge,
        mapped_edge_count=len(edge_map),
        compared_edge_count=len(compared_edges),
        boundary_ignored_edge_count=len(boundary_ignored_edges),
        worst_edges=worst,
        note=note,
        valid_mismatch_points=total_valid_m,
        ready_mismatch_points=total_ready_m,
        transfer_mismatch_points=total_transfer_m,
    )


def print_report(results: Sequence[MGComparison]) -> None:
    for res in results:
        print(
            f"[MG {res.mg_label}] status={res.status} ii={res.ii} "
            f"mapped_edges={res.mapped_edge_count} compared_edges={res.compared_edge_count} "
            f"rotation={res.rotation}"
        )
        if res.boundary_ignored_edge_count:
            print(f"  boundary_edges_ignored={res.boundary_ignored_edge_count}")
        if res.reason:
            print(f"  reason: {res.reason}")
        if res.selected_phase is not None:
            print(
                "  selected_run: "
                f"phase={res.selected_phase} count={res.selected_run_count} "
                f"segment_idx={res.selected_segment_index} interior_occ={res.selected_occurrence_index}"
            )
        if res.anchor_edge is not None:
            print(f"  anchor_edge: {res.anchor_edge[0]} -> {res.anchor_edge[1]}")
        if res.selected_anchor_transfer_index is not None:
            print(
                "  anchor_window: "
                f"transfer_idx={res.selected_anchor_transfer_index} "
                f"start_cycle={res.selected_start_cycle} "
                f"shift={res.selected_window_shift}"
            )
        if res.note:
            print(f"  note: {res.note}")
        if res.status == "OK":
            total_points = max(1, res.compared_edge_count * max(1, res.ii))
            print(
                "  mismatch_points: "
                f"valid={res.valid_mismatch_points}/{total_points}, "
                f"ready={res.ready_mismatch_points}/{total_points}, "
                f"transfer={res.transfer_mismatch_points}/{total_points}"
            )

        if res.worst_edges:
            print("  worst_edges:")
            for m in res.worst_edges:
                print(
                    f"    - {m.edge[0]}->{m.edge[1]} "
                    f"mismatch(v/r/x)=({len(m.valid_mismatches)}/{len(m.ready_mismatches)}/{len(m.transfer_mismatches)}) "
                    f"toggle(v {m.valid_toggle_smt}/{m.valid_toggle_vcd}, "
                    f"r {m.ready_toggle_smt}/{m.ready_toggle_vcd})"
                )
                if m.valid_mismatches:
                    print(f"      valid_idx={m.valid_mismatches}")
                if m.ready_mismatches:
                    print(f"      ready_idx={m.ready_mismatches}")
                if m.transfer_mismatches:
                    print(f"      xfer_idx={m.transfer_mismatches}")


def main() -> None:
    args = parse_args()
    preferred_occurrence = parse_preferred_occurrence(args.steady_occurrence)

    smt_dump = json.loads(Path(args.smt_json).read_text(encoding="utf-8"))
    mg_objs = smt_dump.get("mg_results", [])
    requested = set(args.mg_label)
    if requested:
        mg_objs = [m for m in mg_objs if str(m.get("mg_label", "")) in requested]
    edge_presence_map = edge_presence_by_mg_label(smt_dump.get("mg_results", []))

    vcd = VcdParser(args.vcd)
    instances = parse_instances(args.top_verilog)

    phases = list(smt_dump.get("execution_phase_runs", []))
    executed_segments = [str(s) for s in smt_dump.get("executed_segment_trace", [])]

    results: List[MGComparison] = []
    for mg_obj in mg_objs:
        results.append(
            compare_mg(
                mg_obj,
                vcd,
                instances,
                edge_presence_map,
                phases,
                executed_segments,
                args.min_consecutive,
                preferred_occurrence,
                args.max_worst,
                args.include_boundary_channels,
            )
        )

    print_report(results)

    if args.output_json:
        serializable = []
        for res in results:
            serializable.append(
                {
                    "mg_label": res.mg_label,
                    "status": res.status,
                    "reason": res.reason,
                    "ii": res.ii,
                    "rotation": res.rotation,
                    "selected_phase": res.selected_phase,
                    "selected_run_count": res.selected_run_count,
                    "selected_segment_index": res.selected_segment_index,
                    "selected_occurrence_index": res.selected_occurrence_index,
                    "selected_anchor_transfer_index": res.selected_anchor_transfer_index,
                    "selected_start_cycle": res.selected_start_cycle,
                    "selected_window_shift": res.selected_window_shift,
                    "anchor_edge": list(res.anchor_edge) if res.anchor_edge else None,
                    "mapped_edge_count": res.mapped_edge_count,
                    "compared_edge_count": res.compared_edge_count,
                    "boundary_ignored_edge_count": res.boundary_ignored_edge_count,
                    "valid_mismatch_points": res.valid_mismatch_points,
                    "ready_mismatch_points": res.ready_mismatch_points,
                    "transfer_mismatch_points": res.transfer_mismatch_points,
                    "note": res.note,
                    "worst_edges": [
                        {
                            "edge": [m.edge[0], m.edge[1]],
                            "valid_mismatches": m.valid_mismatches,
                            "ready_mismatches": m.ready_mismatches,
                            "transfer_mismatches": m.transfer_mismatches,
                            "valid_toggle_smt": m.valid_toggle_smt,
                            "valid_toggle_vcd": m.valid_toggle_vcd,
                            "ready_toggle_smt": m.ready_toggle_smt,
                            "ready_toggle_vcd": m.ready_toggle_vcd,
                        }
                        for m in res.worst_edges
                    ],
                }
            )

        out = {
            "smt_json": str(Path(args.smt_json).resolve()),
            "vcd": str(Path(args.vcd).resolve()),
            "top_verilog": str(Path(args.top_verilog).resolve()),
            "results": serializable,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
