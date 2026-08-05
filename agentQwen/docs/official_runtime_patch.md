# Official upstream runtime patch provenance

The industrial controller vendors `work/upstream/data_synth_and_rl` at commit
`760825ccab31c6383ea6bc51e0594141a27905ce` and applies three narrowly scoped
runtime fixes before the cloud smoke:

1. `call_llms.py` reads provider timeout/retry settings from environment
   variables and records the configured endpoint without persisting secrets.
2. `virtual_tools.py` treats absent `checked_tools` as an ordinary failed
   episode and writes the failure row instead of raising while finalizing.
3. `run_data_gen.py` returns failure for a broken final state, so the batch
   controller can resume and count failures rather than silently reporting
   success.

These are failure-recovery changes, not algorithm changes. The upstream commit,
local tree hash, patch rationale, and every stage command are recorded by
`industrial_pipeline.py` in the run-root manifests.
