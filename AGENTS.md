# Verification and completion

Before reporting a change as complete, run actual inference on the available GPU
with local pretrained weights and verify the behavior affected by that change.
A Metal/CUDA build or CPU fixtures alone do not satisfy this completion gate.
Use an explicit device selection so a CPU fallback cannot count as GPU evidence.

Run the relevant CPU regressions as well. Record what changed, exact commands,
model/artifact identity, actual device, sampling settings, observed results, and
remaining limitations in the documentation. Prefer repeatable regression checks
over a one-off prompt. If GPU access or weights are unavailable, document the
blocker and leave GPU verification explicitly pending; never report it as passed.
