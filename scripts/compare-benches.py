#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
compare-benches.py -- read criterion JSON from multiple architectures and
produce a markdown comparison table.

Usage: python3 compare-benches.py <results-dir>

The results directory is expected to contain subdirectories named by arch
label (e.g. intel_x86, amd_x86, graviton3), each with a criterion/ tree
containing estimates.json files.

Output: markdown table to stdout with median and MAD per benchmark per arch.
Benchmarks more than 10% slower than the fastest arch are marked with (*).
"""

import json
import os
import sys
from pathlib import Path


def find_estimates(criterion_dir: Path) -> dict[str, dict]:
    """Walk criterion output and collect estimates.json files.

    Returns {benchmark_name: parsed_json}.
    """
    results = {}
    for root, _dirs, files in os.walk(criterion_dir):
        if "estimates.json" in files:
            est_path = Path(root) / "estimates.json"
            # Benchmark name: path relative to criterion dir, minus /new/estimates.json
            rel = est_path.relative_to(criterion_dir)
            parts = list(rel.parts)
            # criterion stores results as <group>/<bench>/new/estimates.json
            # or <bench>/new/estimates.json
            if "new" in parts:
                parts.remove("new")
            parts.remove("estimates.json")
            bench_name = "/".join(parts)
            if not bench_name:
                continue
            with open(est_path) as f:
                results[bench_name] = json.load(f)
    return results


def extract_median_mad(estimates: dict) -> tuple[float, float]:
    """Extract median (ns) and median absolute deviation (ns) from estimates."""
    median_ns = estimates.get("median", {}).get("point_estimate", 0.0)
    mad_ns = estimates.get("median_abs_dev", {}).get("point_estimate", 0.0)
    return median_ns, mad_ns


def format_time(ns: float) -> str:
    """Format nanoseconds into a human-readable string."""
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} us"
    return f"{ns:.0f} ns"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results-dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect per-arch benchmark data.
    # arch_label -> {bench_name -> (median_ns, mad_ns)}
    arch_data: dict[str, dict[str, tuple[float, float]]] = {}
    arch_labels = sorted(
        d.name
        for d in results_dir.iterdir()
        if d.is_dir() and (d / "criterion").is_dir()
    )

    if not arch_labels:
        print("No criterion results found in subdirectories.", file=sys.stderr)
        sys.exit(1)

    for label in arch_labels:
        criterion_dir = results_dir / label / "criterion"
        estimates = find_estimates(criterion_dir)
        arch_data[label] = {}
        for bench, est in estimates.items():
            median_ns, mad_ns = extract_median_mad(est)
            arch_data[label][bench] = (median_ns, mad_ns)

    # Union of all benchmark names, sorted.
    all_benches = sorted(set().union(*(d.keys() for d in arch_data.values())))

    if not all_benches:
        print("No benchmarks found in estimates.json files.", file=sys.stderr)
        sys.exit(1)

    # Build table.
    header = "| Benchmark | " + " | ".join(arch_labels) + " |"
    sep = "|---|" + "|".join("---" for _ in arch_labels) + "|"

    print(header)
    print(sep)

    for bench in all_benches:
        # Collect medians for regression detection.
        medians = []
        for label in arch_labels:
            entry = arch_data[label].get(bench)
            medians.append(entry[0] if entry else None)

        valid_medians = [m for m in medians if m is not None and m > 0]
        best = min(valid_medians) if valid_medians else None

        cells = []
        for i, label in enumerate(arch_labels):
            entry = arch_data[label].get(bench)
            if entry is None:
                cells.append("--")
                continue
            median_ns, mad_ns = entry
            cell = f"{format_time(median_ns)} +/- {format_time(mad_ns)}"
            # Mark regressions: >10% slower than best arch.
            if best and median_ns > best * 1.10:
                pct = (median_ns - best) / best * 100
                cell += f" (*+{pct:.0f}%*)"
            cells.append(cell)

        row = f"| {bench} | " + " | ".join(cells) + " |"
        print(row)

    print()
    print(
        "(*) = more than 10% slower than the fastest architecture for that benchmark."
    )


if __name__ == "__main__":
    main()
