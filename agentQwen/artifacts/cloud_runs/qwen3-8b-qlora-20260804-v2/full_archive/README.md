# Full AutoDL archive

The complete V2 run archive is intentionally excluded from Git because it is
1,197,755,565 bytes. Its verified metadata is kept in
`agenticqwen-qwen3-8b-qlora-v2-results.tar.gz.json`.

- Expected SHA-256: `5bdd2a5a701d707e2fcefb6847565e0793b795308ab854616b34d67954a908d5`
- Expected bytes: `1197755565`
- Contents: Stage-1/Stage-2 adapters, Trainer checkpoints, traces, frozen task
  splits, manifests, logs, and fresh-process replay artifacts.
- Git-tracked compact evidence: `../evidence/`

Local verification completed successfully: the byte count and SHA-256 match the
remote metadata, `tar -tzf` returned 140 entries, and both adapter weight files,
`verification.json`, `fresh_replay/summary.json`, and
`artifact_inventory.json` are present. See `local_verification.json`.
