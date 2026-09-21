#!/usr/bin/env python
"""Measure the real VRAM cost of a pipeline stage, then size a card for it.

    # one stage at a time, in whichever env that stage needs
    python scripts/alaya/measure_vram.py --label stage10 -- \
        python scripts/pipeline/10_generate_views.py --reference <img> --count 1

    python scripts/alaya/measure_vram.py --label stage30 -- \
        python scripts/pipeline/30_tryon_batch.py --in-dir work/variants \
            --garment <img> --desc "..." --limit 1

    # then, once both have been recorded
    python scripts/alaya/measure_vram.py --report

Samples nvidia-smi from OUTSIDE the measured process, so it needs nothing
importable in that process and no edits to the stage scripts. That also means
it sees allocations torch cannot report: the CUDA context itself, and the
ONNX Runtime and detectron2 allocations the human-parsing and DensePose steps
make. torch.cuda.max_memory_allocated() misses all of those.

Results land in work/vram/<label>.json so stages measured in different
environments, hours apart, can still be added up.
"""

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

MIB = 1024 ** 2
OUT_DIR = Path("work/vram")

# A stage is "active" once it is this far above the idle baseline. Below it we
# are looking at someone else's noise, not at our model.
ACTIVE_MIB = 512


def nvidia_smi(args):
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def total_used_mib(gpu):
    out = nvidia_smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits",
                      f"--id={gpu}"])
    if not out:
        return None
    try:
        return int(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def gpu_info(gpu):
    out = nvidia_smi(["--query-gpu=name,memory.total",
                      "--format=csv,noheader,nounits", f"--id={gpu}"])
    if not out:
        return "unknown", None
    try:
        name, total = out.strip().splitlines()[0].split(",")
        return name.strip(), int(total)
    except (ValueError, IndexError):
        return "unknown", None


def pgid_of(pid):
    """Process group of a pid, from /proc - no psutil dependency.

    Field 5 of /proc/<pid>/stat is pgrp, but field 2 (comm) can contain spaces
    and parentheses, so split after the last ')'.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        return int(data[data.rindex(")") + 2:].split()[2])
    except (OSError, ValueError, IndexError):
        return None


def ours_mib(gpu, pgid):
    """VRAM used by our process group, or None if the container hides it.

    Inside a container nvidia-smi often cannot map compute-apps to PIDs (it
    reports "Insufficient Permission", or the PIDs belong to another
    namespace). Callers fall back to total-minus-baseline.
    """
    out = nvidia_smi(["--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits", f"--id={gpu}"])
    if not out:
        return None
    total, matched = 0, False
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, used = int(parts[0]), int(parts[1])
        except ValueError:
            continue           # "[N/A]" / "Insufficient Permission"
        if pgid_of(pid) == pgid:
            total += used
            matched = True
    return total if matched else None


class Sampler(threading.Thread):
    def __init__(self, gpu, interval):
        super().__init__(daemon=True)
        self.gpu, self.interval = gpu, interval
        self.pgid = None
        self.stop_flag = threading.Event()
        self.samples = []          # (t, total_mib, ours_mib_or_None)
        self.per_proc_ok = None

    def run(self):
        t0 = time.time()
        while not self.stop_flag.is_set():
            total = total_used_mib(self.gpu)
            mine = ours_mib(self.gpu, self.pgid) if self.pgid else None
            if self.per_proc_ok is None and total is not None:
                self.per_proc_ok = mine is not None
            if total is not None:
                self.samples.append((time.time() - t0, total, mine))
            self.stop_flag.wait(self.interval)


def summarise(samples, baseline, per_proc_ok):
    """Split the run into resident weights and the activation spike on top.

    Peak alone overstates what a second model would have to coexist with: it
    includes a transient that only exists while that stage is stepping. Taking
    the median of the active window as the resident level and the peak above it
    as the spike gives a combined estimate that is not the sum of two
    transients that never overlap.
    """
    if per_proc_ok:
        series = [m for _, _, m in samples if m is not None]
    else:
        series = [max(0, t - baseline) for _, t, _ in samples]
    active = [v for v in series if v > ACTIVE_MIB]
    if not active:
        return {"peak_mib": max(series, default=0), "resident_mib": 0,
                "spike_mib": 0, "samples": len(samples), "active_samples": 0}
    peak = max(active)
    resident = int(statistics.median(active))
    return {"peak_mib": peak, "resident_mib": resident,
            "spike_mib": peak - resident, "samples": len(samples),
            "active_samples": len(active)}


def gb(mib):
    return f"{mib / 1024:.1f} GB"


def measure(args):
    if not shutil.which("nvidia-smi"):
        print("ERROR: nvidia-smi not found - nothing to measure.", file=sys.stderr)
        return 1
    if not args.command:
        print("ERROR: no command. Put it after --, e.g.\n"
              "  python scripts/alaya/measure_vram.py --label stage10 -- "
              "python scripts/pipeline/10_generate_views.py ...", file=sys.stderr)
        return 2

    name, capacity = gpu_info(args.gpu)
    baseline = total_used_mib(args.gpu)
    if baseline is None:
        print(f"ERROR: could not read GPU {args.gpu}.", file=sys.stderr)
        return 1
    print(f"gpu       {args.gpu}: {name}"
          + (f", {gb(capacity)} total" if capacity else ""))
    print(f"baseline  {gb(baseline)} already in use before we start")
    if baseline > ACTIVE_MIB:
        print("          (something else is on this GPU; the measurement is "
              "still\n           valid but you have that much less headroom)")
    print(f"running   {' '.join(args.command)}\n")

    sampler = Sampler(args.gpu, args.interval)
    sampler.start()
    t0 = time.time()
    # New session so the whole tree shares one pgid and a Ctrl-C reaches it.
    proc = subprocess.Popen(args.command, start_new_session=True)
    sampler.pgid = os.getpgid(proc.pid)

    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        os.killpg(sampler.pgid, signal.SIGINT)
        rc = proc.wait()
    finally:
        sampler.stop_flag.set()
        sampler.join(timeout=args.interval * 4)
    elapsed = time.time() - t0

    if not sampler.samples:
        print("\nERROR: no samples. Did the command exit immediately?",
              file=sys.stderr)
        return 1

    per_proc_ok = bool(sampler.per_proc_ok)
    stats = summarise(sampler.samples, baseline, per_proc_ok)
    stats.update(label=args.label, gpu_name=name, gpu_total_mib=capacity,
                 baseline_mib=baseline, seconds=round(elapsed, 1),
                 exit_code=rc, per_process=per_proc_ok,
                 command=args.command, when=time.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"\n── {args.label} ── {elapsed:.0f}s, exit {rc}, "
          f"{stats['samples']} samples")
    if not per_proc_ok:
        print("    per-process accounting unavailable in this container; "
              "using\n    total-minus-baseline, which also counts anything "
              "else that grew.")
    print(f"    resident  {gb(stats['resident_mib']):>9}   weights held "
          "for the whole run")
    print(f"    spike     {gb(stats['spike_mib']):>9}   activations on top")
    print(f"    PEAK      {gb(stats['peak_mib']):>9}")
    if capacity:
        print(f"    headroom  {gb(capacity - baseline - stats['peak_mib']):>9}"
              f"   left on this card")
    if rc != 0:
        print("\n    NOTE: the command failed, so this may not be a full run.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{args.label}.json"
    dest.write_text(json.dumps(stats, indent=2))
    print(f"\n    saved {dest}")

    if args.csv:
        Path(args.csv).write_text(
            "seconds,total_mib,ours_mib\n" + "".join(
                f"{t:.2f},{tot},{'' if m is None else m}\n"
                for t, tot, m in sampler.samples))
        print(f"    timeline {args.csv}")

    others = sorted(p for p in OUT_DIR.glob("*.json") if p != dest)
    if others:
        print("\n    --report will combine this with: "
              + ", ".join(p.stem for p in others))
    return rc


def report(args):
    runs = []
    for path in sorted(OUT_DIR.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            print(f"skipping unreadable {path}", file=sys.stderr)
    if not runs:
        print(f"No measurements in {OUT_DIR}. Record one first:\n"
              "  python scripts/alaya/measure_vram.py --label stage10 -- "
              "<command>", file=sys.stderr)
        return 1

    print(f"{'stage':<12} {'resident':>10} {'spike':>10} {'peak':>10}   run")
    for r in runs:
        print(f"{r['label']:<12} {gb(r['resident_mib']):>10} "
              f"{gb(r['spike_mib']):>10} {gb(r['peak_mib']):>10}   "
              f"{r.get('when', '?')}"
              + ("  (FAILED)" if r.get("exit_code") else ""))

    if len(runs) < 2:
        print("\nOne stage measured. Record another and re-run --report to "
              "size\na card that holds both at once.")
        return 0

    resident = sum(r["resident_mib"] for r in runs)
    spike = max(r["spike_mib"] for r in runs)
    # One CUDA context per process, and the stages are separate processes
    # because the diffusers pin forces separate environments.
    ctx = 512 * len(runs)
    estimate = resident + spike + ctx
    conservative = sum(r["peak_mib"] for r in runs) + ctx

    print(f"\nAll {len(runs)} stages resident at once:")
    rows = [("weights held by every stage", "", resident),
            ("largest single activation spike", "+", spike),
            (f"CUDA context, {len(runs)} processes", "+", ctx),
            ("ESTIMATE", "", estimate),
            ("upper bound (every peak at once)", "", conservative)]
    for label, sign, value in rows:
        print(f"  {label:<34}{sign:>2} {gb(value):>9}")
    print("\n  Only one stage steps at a time, so the estimate adds one spike "
          "rather\n  than all of them. The upper bound assumes they coincide, "
          "which needs\n  them running concurrently - not what this pipeline "
          "does.")

    need = estimate / 1024
    print(f"\n  Card sizes, against the {need:.0f} GB estimate:")
    for size in (24, 40, 48, 80, 96, 141):
        if size < need * 1.05:
            verdict = "no"
        elif size < need * 1.25:
            verdict = "marginal - fine until a longer prompt or a bigger image"
        else:
            verdict = "comfortable"
        print(f"    {size:>4} GB   {verdict}")
    print("\n  Sequential instead: each stage needs only its own peak, so the "
          "largest\n  single stage sets the floor "
          f"({gb(max(r['peak_mib'] for r in runs))}), at the cost of a reload "
          "between stages.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="run",
                    help="name for this measurement, e.g. stage10")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.25,
                    help="seconds between samples (default 0.25)")
    ap.add_argument("--csv", help="also write the full timeline here")
    ap.add_argument("--report", action="store_true",
                    help="combine the saved measurements and size a card")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="the command to measure, after --")
    args = ap.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return report(args) if args.report else measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
