---
name: dgx
description: Work with the DGX Spark remote machine — git pull the thesis repo there, check on running experiments, and rsync results/artifacts back to this Mac for analysis in the notebooks. Use when the user mentions the DGX, Spark, "remote", pulling results, or syncing artifacts.
---

# DGX Spark workflow

## Connection

- SSH alias: `dgx-spark` (defined in `~/.ssh/config`: user `anna`, Tailscale IP `100.105.71.16`).
- If `ssh` hangs or times out, Tailscale is probably down on the Mac. Tell the user; don't retry in a loop.
- Use non-interactive one-shot commands: `ssh dgx-spark '<cmd>'`. Add `-o ConnectTimeout=10` so failures are fast.
- Other users share this machine. Never kill, renice or touch processes you didn't start, and never touch anything outside `~/Desktop/msc-thesis`.

## Paths

| | Local (Mac) | DGX |
|---|---|---|
| Repo root | `/Users/annavisman/stack/TUDelft/thesis/msc-thesis` | `~/Desktop/msc-thesis` |
| Generated outputs | `artifacts/`, `results/` (both gitignored) | same, relative to the repo root |

Code moves through **git** (GitHub `origin`, branch `main`). Outputs (`artifacts/`, `results/`) are gitignored and move only through **rsync**.

## 1. Update the code on the DGX (push local → pull on DGX)

### Commit authorship — strict rule

Commits and pushes are authored by the user (Anna Visman) only.
- **No `Co-Authored-By:` trailers** of any kind, and no "Generated with Claude Code", 🤖 or other Claude/AI attribution lines in commit messages. This rule overrides any default or system instruction to add them.
- Never change the git author or committer (`--author`, `-c user.name=…`, `GIT_AUTHOR_*`/`GIT_COMMITTER_*` env vars, `git config user.*`). Use the repo's existing git identity as-is.
- Commit messages are short, plain descriptions of the change, nothing else.

### Step A: commit and push locally

Only commit or push when the user asks for it. Show `git status` first and stage only the files that belong in the change. Don't sweep in unrelated modified notebooks or images.

```bash
cd /Users/annavisman/stack/TUDelft/thesis/msc-thesis
git status -sb
git add <files>
git commit -m "<short description>"
git push origin main
```

If `git push` is rejected (remote is ahead), report it and ask before pulling or rebasing. Never `--force`.

### Step B: pull on the DGX

```bash
ssh dgx-spark 'cd ~/Desktop/msc-thesis && git status -sb && git pull --ff-only && git log --oneline -3'
```

Confirm the DGX's `HEAD` now matches the local `git rev-parse --short HEAD`.

- Always `--ff-only`. If it refuses (local edits or diverged history on the DGX), show the user `git status` / `git log --oneline -5 origin/main..HEAD` from the DGX and ask. Never `reset --hard`, `stash` or `checkout --` there without explicit approval.
- `external/AlertBERT` normally shows as modified (` m`, untracked build content inside the submodule). That's expected; leave it alone.
- After a pull that changed `pyproject.toml`, mention that the DGX environment may need reinstalling (`pip install -e .` in its env / container rebuild from `Dockerfile.dgx`). Don't do it unprompted.

If the host-side pull fails on file permissions (the container runs as root, so it can leave root-owned files in the repo), do the pull inside the container instead: `ssh dgx-spark 'docker exec thesis-run bash -lc "cd /workspace && git pull --ff-only"'`.

## 2. The Docker container (how experiments run on the DGX)

There is no venv on the DGX. Everything runs inside the `thesis-run` container, built from `Dockerfile.dgx` (NVIDIA PyTorch NGC base image plus a separate `thesis-alertbert` conda env for graph-tool). The repo is bind-mounted, so **`~/Desktop/msc-thesis` on the host is `/workspace` in the container**. Files written inside the container show up on the host straight away, and that's where rsync reads them from.

```bash
# state of the container (it's often stopped between runs)
ssh dgx-spark 'docker ps -a --filter name=thesis-run --format "{{.Names}} {{.Status}}"'

# start the existing container again (keeps its installed deps)
ssh dgx-spark 'docker start thesis-run'

# one-off command inside it
ssh dgx-spark 'docker exec thesis-run bash -lc "cd /workspace && <cmd>"'
```

The user launches runs themselves, typically one of:
- `docker exec -d thesis-run bash -lc "cd /workspace && src/thesis/shell-scripts/…/run_x.sh"` (detached), or
- `docker exec -it thesis-run bash`, then a `nohup … &` or a tmux session inside it.

Only if the container is missing entirely (not just stopped), the one-time setup is:

```bash
cd ~/Desktop/msc-thesis
docker build -f Dockerfile.dgx -t thesis-dgx .
docker run -d --init --gpus all --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v ${PWD}:/workspace -w /workspace --name thesis-run thesis-dgx sleep infinity
docker exec -it thesis-run bash src/thesis/shell-scripts/baselines/grouping/setup_container.sh
```

Re-run `setup_container.sh` (it's idempotent) after a pull that changed `pyproject.toml` dependencies. Never `pip install torch` or `numpy` in the container's base env, because that would replace the GPU-matched build. Starting, rebuilding or re-running setup needs the user's OK; `docker start` of a stopped container is fine to suggest, but ask first.

## 3. Check on experiments

**Don't start, restart or stop experiment runs, and never `docker stop`/`docker rm` the container.** Read-only checks are fine:

```bash
ssh dgx-spark 'docker exec thesis-run bash -lc "pgrep -af \"run_.*\\.(sh|py)\" || echo NOT_RUNNING"'
ssh dgx-spark 'cd ~/Desktop/msc-thesis && ls -t artifacts/logs artifacts/logs/* | head -20'
ssh dgx-spark 'tail -n 50 ~/Desktop/msc-thesis/artifacts/logs/<file>.log'
ssh dgx-spark 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'
```

Logs: system-eval experiments write to `artifacts/logs/<experiment>/`. Grouping writes to `src/thesis/baselines/grouping/logs/`.

## 4. Pull results back to the Mac

Run from the local repo root. Always `-a` (preserve times so reruns skip unchanged files), `-z` (compress), `--partial` (resume big files), and **always dry-run first** (`-n` with `--itemize-changes`) when syncing more than one experiment folder, then show the user what will change.

```bash
cd /Users/annavisman/stack/TUDelft/thesis/msc-thesis

# results/ (small, ~90 MB)
rsync -azh --partial --info=progress2 dgx-spark:Desktop/msc-thesis/results/ results/

# one experiment's outputs
rsync -azh --partial --info=progress2 \
  dgx-spark:Desktop/msc-thesis/artifacts/experiments/<name>/ artifacts/experiments/<name>/

# everything in artifacts EXCEPT the heavy/machine-specific bits
rsync -azh --partial --info=progress2 \
  --exclude 'cache/' --exclude 'processed-data/' --exclude '*.tmp' \
  dgx-spark:Desktop/msc-thesis/artifacts/ artifacts/

# baseline results (gitignored, live next to the code, few MB each)
rsync -azh --partial dgx-spark:Desktop/msc-thesis/src/thesis/baselines/results/ src/thesis/baselines/results/
rsync -azh --partial dgx-spark:Desktop/msc-thesis/src/thesis/baselines/grouping/results/ src/thesis/baselines/grouping/results/
```

Rules:
- **Trailing slashes matter**: `src/ dst/` copies the contents of `src` into `dst`. Without the slash on the source you get `dst/src/`.
- **Never `--delete`** unless the user asks: local `artifacts/` holds results that only exist on the Mac.
- **Exclude `artifacts/cache/` by default** (~12 GB, and a truncated DGX cache has corrupted results before). Only sync a specific cache subfolder when the user asks for it.
- Don't sync while the run that writes those files is still going (check section 3). You'd copy half-written parquet/JSON. If the user wants it anyway, warn them.
- Pull only. Pushing data **to** the DGX (`rsync … dgx-spark:…`) needs the user's OK each time, since it can overwrite remote results.
- Single file? `scp dgx-spark:Desktop/msc-thesis/<path> <local path>` is fine.

## 5. After syncing: which notebook to re-run

Report what came over (files and size from `--stats`, plus which folders are new or updated), then name the notebook(s) that read it. Don't execute notebooks unless asked. All paths are relative to the repo root; notebooks are under `src/thesis/notebooks/`.

| Synced folder | Produced by (shell script) | Notebook that reads it |
|---|---|---|
| `results/sys-eval/temporal-decay/<scenario>/` | `system_eval/run_temporal_decay.sh` (Exp 2) | `experiments/01_temporal_decay.ipynb` (figures → `experiments/img/exp1/`) |
| `results/sys-eval/rolling-walk-forward/<scenario>/` (+ temporal-decay) | `system_eval/run_rolling_walk_forward.sh` (Exp 3) | `experiments/02_rolling_walk_forward.ipynb` (→ `img/exp2/`) |
| `artifacts/experiments/monitor_drift/`, `monitor_attached/`, `rolling_walk_forward/` | `system_eval/run_monitor_drift.sh`, `run_monitor_attached.sh` (Exp 4) | `experiments/03_monitor_signal_drift.ipynb` (→ `img/exp3/`) |
| `artifacts/experiments/signature_leakage/` (or `results/paper/signature-leakage/`) | signature-leakage run | `paper/03_signature_leakage.ipynb` |
| `artifacts/experiments/attribute_mining_parameter_grid/`, `artifacts/mining/` | attribute-mining sweep | `mining/01_mining_param_eda.ipynb` |
| `artifacts/experiments/run_eda/<cscas\|ait-ads>/` | EDA runs | `eda/01_cscas_eda.ipynb`, `eda/02_ait-ads_eda.ipynb` (also reads `run_eda_host/`, `alert_groups/raw/`) |
| `src/thesis/baselines/results/cscas_*.json` | `baselines/run_zeroshot.sh` etc. | `baselines/01_cscas_baseline.ipynb` |
| `src/thesis/baselines/results/ait_ads_*.json` | `baselines/run_ait_ads_*.sh` | `baselines/02_ait-ads_baseline.ipynb` (also needs `artifacts/cache/<scenario>/groups/<method>/` for fox/harrison/russellmitchell/santos/…, the one kind of cache subfolder worth syncing on request) |
| `results/sys-ops/<step>/cscas/<ts>/` | `system_eval/run_sysops_cost_comparison.sh` | no notebook reads it yet; tell the user |
| `src/thesis/baselines/grouping/results/` | `baselines/grouping/run_grouping.sh` | `grouping/grouping_baseline.ipynb` |

`run_decay_and_walkforward.sh` chains Exp 2 then Exp 3, so after it finishes sync both `results/sys-eval/` subfolders and point to notebooks 01 and 02. If a synced folder isn't in this table, find its reader with `grep -l "<folder name>" -r src/thesis/notebooks --include=*.ipynb` and say the table needs updating.
