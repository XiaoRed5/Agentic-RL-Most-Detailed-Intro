# AutoDL browser operations

## Official references

- Hugging Face cache and acceleration: <https://www.autodl.com/docs/huggingface/>
- Academic network acceleration: <https://www.autodl.com/docs/network_turbo/>
- AutoDL console: <https://www.autodl.com/console/homepage/personal>

## Create/reuse checklist

1. Inspect existing instances before creating another billable resource.
2. On the create page, verify GPU name, VRAM, card count, region, hourly price, image, and data-disk capacity.
3. Prefer one sufficiently large card for QLoRA unless the project explicitly implements distributed training.
4. Stop before the final create action when the displayed cost exceeds the user's approved bound.
5. Wait for `running`, then open JupyterLab.

## Jupyter upload contract

- Upload generated artifacts to `/root`; keep project/model/run data under `/root/autodl-tmp`.
- Use the browser file chooser rather than pasting local file contents into a terminal.
- A successful file chooser event is not enough: verify the remote filename, size, and SHA-256.
- Never copy or disclose the Jupyter token from the address bar.

## Terminal reliability

Computer-use text entry can lose underscores, commas, quoting, or IME state. Therefore:

- generate all complex commands locally;
- use hyphen-only launch/status/collect filenames;
- type only `bash /root/<job>-autodl-launch.sh`, `...status.sh`, or `...collect.sh`;
- use `ctrl+c`/`ctrl+z` only after refreshing UI state;
- use a second terminal for diagnostics while a foreground download runs;
- rely on `nohup` for long jobs so browser disconnects do not kill training.

## Network choice

1. Prefer ModelScope for an exact public Qwen snapshot on mainland AutoDL.
2. For Hugging Face, set cache on the data disk and source the official accelerator:

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
```

3. Measure growth twice before switching sources. Stop the slower duplicate once the replacement is proven faster.
4. Record model source ID, revision, local path, file sizes, and hashes.

## Shutdown gate

Before shutdown, verify the downloaded local archive against its sidecar and confirm the run's verifier/evaluator artifacts. Use the AutoDL instance list to shut down the exact instance. Shutdown should stop billing but preserve the data disk; do not release/delete the instance unless explicitly requested.
