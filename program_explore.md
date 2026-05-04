# Phase-1 Override: Explore

Inherits `program.md`. Overrides:

- **Time budget:** 15 minutes per experiment.
- **Selection rule:** keep if `new_macro_f1 > current_best + 0.001` (single seed, noisy OK).
- **Preferred directions:** fast-signal knobs — learning rate, batch size, dropout, augmentation toggles, optimizer choice.
- **Stop after:** 20 consecutive non-improving experiments OR user-defined compute cap.

The goal of this phase is breadth, not certainty. False positives are expected and filtered by the confirm phase.
