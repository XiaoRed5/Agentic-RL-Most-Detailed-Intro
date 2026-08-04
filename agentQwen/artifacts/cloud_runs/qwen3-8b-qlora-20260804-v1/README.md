# Cloud run V1: reward-saturation failure

This run completed both 12-step QLoRA-GRPO stages and the independent
fresh-process replay, but the final verifier correctly returned **FAIL**.

- Stage 1 and Stage 2 each produced 48/48 successful training trajectories.
- Every training reward was exactly `1.3`, so within-group reward variance was zero.
- Both stages reported `train_loss = 0.0`.
- Stage 1 and Stage 2 adapter weight hashes were identical.
- Independent holdout replay still executed successfully, with 4/6 task success;
  that proves reloadability, not learning.

The complete cloud archive is 324,633,965 bytes with SHA-256
`26997b4907bc5f109b07b1a90e5c7c3a30dcf1e836dd2c88d2187f53af628354`.
It is intentionally excluded from Git because it exceeds GitHub's ordinary
single-file limit. Its machine-generated package manifest is kept beside this
README.
