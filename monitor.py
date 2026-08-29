"""
Live training monitor. Run in a second terminal while train.py writes to a log file.

Usage:
    # terminal 1 — training
    python3 train.py --epochs 300 2>/dev/null | tee logs/run.log

    # terminal 2 — monitor
    python3 monitor.py logs/run.log
"""

import sys
import time
import re
import os

LOG_REFRESH = 2   # seconds between re-reads


def parse_line(line):
    m = re.search(
        r"epoch\s+(\d+)/(\d+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)"
        r"(?:\s+pval\s+([\d.]+))?"
        r"\s+mr\s+([\d.]+)\s+pf\s+([\d.]+)\s+lr\s+([\d.e+-]+)",
        line,
    )
    if not m:
        return None
    return {
        "epoch":  int(m.group(1)),
        "total":  int(m.group(2)),
        "train":  float(m.group(3)),
        "val":    float(m.group(4)),
        "pval":   float(m.group(5)) if m.group(5) else None,
        "mr":     float(m.group(6)),
        "pf":     float(m.group(7)),
        "lr":     float(m.group(8)),
    }


def render(rows, log_path):
    os.system("clear")
    if not rows:
        print("waiting for training to start...")
        return

    last   = rows[-1]
    has_pval = any(r["pval"] is not None for r in rows)
    ckpt_key = "pval" if has_pval else "val"
    best   = min((r for r in rows if r[ckpt_key] is not None), key=lambda r: r[ckpt_key])
    recent = rows[-10:]

    gap = last["train"] - last["val"]
    gap_str = f"{gap:+.4f}"

    print(f"  log: {log_path}")
    print(f"  epoch  {last['epoch']:>4} / {last['total']}   "
          f"lr {last['lr']:.2e}   mr {last['mr']:.2f}   pf {last['pf']:.2f}")
    print()
    hdr = f"  {'epoch':>6}  {'train':>8}  {'val':>8}  {'pval':>8}  {'gap':>8}  {'best?':>6}"
    print(hdr)
    print("  " + "-" * 54)
    for r in recent:
        is_best = r["epoch"] == best["epoch"]
        g       = r["train"] - r["val"]
        pval_s  = f"{r['pval']:8.4f}" if r["pval"] is not None else f"{'—':>8}"
        print(f"  {r['epoch']:>6}  {r['train']:>8.4f}  {r['val']:>8.4f}  "
              f"{pval_s}  {g:>+8.4f}  {'***' if is_best else '':>6}")

    print()
    ckpt_val = best[ckpt_key]
    print(f"  best {ckpt_key:<4}  {ckpt_val:.4f}  at epoch {best['epoch']}")
    print(f"  current gap (train - val)  {gap_str}")

    # simple gap trend
    if len(rows) >= 20:
        early_gap = rows[len(rows)//2]["train"] - rows[len(rows)//2]["val"]
        late_gap  = gap
        trend = "widening" if late_gap > early_gap + 0.005 else (
                "narrowing" if late_gap < early_gap - 0.005 else "stable")
        print(f"  generalisation gap trend  {trend}")

    print()
    print(f"  refreshing every {LOG_REFRESH}s — ctrl-c to quit")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/run.log"
    rows = []

    while True:
        try:
            if os.path.exists(log_path):
                with open(log_path) as f:
                    rows = [r for r in (parse_line(l) for l in f) if r]
            render(rows, log_path)
            time.sleep(LOG_REFRESH)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
