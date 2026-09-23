"""One timing format for both segmentation paths, so they can be compared.

Load time is reported apart from per-image time on purpose. The two methods
differ far more in what they cost to start than in what they cost per image -
one loads a few hundred MB of ONNX, the other 54 GB off a shared mount - so a
single seconds-per-image figure that folds the load in says whatever the batch
size happened to be, not which method is faster.
"""

import time


class Timer:
    def __init__(self, method: str):
        self.method = method
        self.load = 0.0
        self.per_image = []
        self._t0 = None

    def start_load(self):
        self._t0 = time.time()

    def end_load(self):
        self.load = time.time() - self._t0
        print(f"load      {self.load:.1f}s")

    def start_image(self):
        self._t0 = time.time()

    def end_image(self) -> float:
        dt = time.time() - self._t0
        self.per_image.append(dt)
        return dt

    def report(self):
        n = len(self.per_image)
        if not n:
            print("\nNo images timed.")
            return
        total = sum(self.per_image)
        mean = total / n
        print(f"\n── timing: {self.method} ──")
        row = lambda label, value, note: print(
            f"  {label:<14}{value:>8}   {note}")
        row("load", f"{self.load:.1f}s", "once per run")
        row("per image", f"{mean:.2f}s",
            f"mean of {n}  (min {min(self.per_image):.2f}, "
            f"max {max(self.per_image):.2f})")
        row(f"{n} image{'' if n == 1 else 's'}", f"{total:.1f}s", "excluding load")
        row("wall clock", f"{self.load + total:.1f}s", "including load")
        if self.load > total:
            print(f"\n  The load dominates at this batch size: {n} images cost "
                  f"{total:.0f}s\n  against {self.load:.0f}s to start. Compare "
                  "per-image figures for the\n  method, and wall clock for "
                  "this particular run.")
