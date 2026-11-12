# Engine integration changes

The base engine comes from the official Ultralytics8.3.9 wheel identified in upstream_wheel.json. Selected source class bodies are recorded in extracted_operators.json and verified by docs/source_fidelity.json.

The local tasks.py integration adds explicit imports for the selected classes, detection stride/bias initialization and device application for Detect_LSCD_LQE, parser channel/repeat registration for the custom feature blocks, and width scaling of the shared head's hidden channels. Stride probing retains the selected research source's eval-backbone/training-head2x640 path. Submodules return to train mode after probing for conventional API behavior; explicit train/eval commands control subsequent mode. YAML loading respects an explicit scale when no size suffix is inferred.

All other engine files are retained from that pinned official wheel. There is no automatic replacement with a newer installed Ultralytics version. Original checkpoints and historical data are absent, so this preparation is not asserted to be a bitwise recreation of an unknown training run.
