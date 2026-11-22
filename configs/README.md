# Baseline and best-known configurations (M01)

Every file here pins one bench configuration so a later comparison never
mixes two revisions or two states of the machine.

- `<model>.kinetic.json` — the `kinetic bench` configuration used for the
  model's baseline entry in `../ledger.jsonl` (device, repetitions,
  test sizes).
- `<model>.llama.json` — the matching `llama-bench` configuration
  (device, `-ngl`, repetitions).
- The source of truth for what "the configuration of an essai" covers is
  the `command` field of each ledger entry: it lists the exact binary,
  flags and repetitions, plus the commit and machine snapshot taken at
  that moment.

Adding an entry: run `python scripts/bench_baseline.py <model> <ngl>`,
which appends the kinetic and llama-bench rows and writes both config
files. One GPU job at a time; never append by hand.

The best-known configuration per model lives in `../best-config.json`,
written by `scripts/regression_gate.py --record` after a campaign, and
read back by the same script (and the GPU CI) as the regression
reference.
