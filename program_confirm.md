# Phase-2 Override: Confirm

Inherits `program.md`. Overrides:

- **Time budget:** 60 minutes per experiment.
- **Seeds:** 3 per experiment; report mean ± std.
- **Selection rule:** keep if mean improvement is real AND the 95% confidence interval does not overlap baseline's 95% CI.
- **Preferred inputs:** only experiments that the explore phase already promoted.
- **Stop after:** 5 consecutive non-improving experiments OR user-defined compute cap.

The goal of this phase is statistical rigor. Only confirmed wins go into the paper's main results table.
