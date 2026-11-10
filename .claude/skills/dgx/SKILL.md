---
name: dgx
description: Work with the DGX Spark remote machine — git pull the thesis repo there, launch requested runs, watch or catch up on running/earlier experiments (status, crashes, whether a run finished), and rsync results/artifacts back to this Mac for analysis in the notebooks. Use when the user mentions the DGX, Spark, "remote", a run or experiment's status, pulling results, or syncing artifacts.
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

## 3. Launching and watching a run (only when the user asks)

By default the user launches runs themselves. Only launch one when the user explicitly asks for that specific script in this conversation. Before launching:

1. Make sure the DGX is on the latest pushed code (section 1) and the container is up (`docker start thesis-run` if it's stopped).
2. Check nothing else of the user's is already running (`pgrep` below). If something is, ask before starting a second run.
3. Confirm the exact command (script and any env overrides like `MAX_STEPS`, `SEEDS`, `FORCE`) back to the user in one line, then launch. Offer a smoke test first when the script has one.

Launch detached inside the container, with the launcher's own stdout in a timestamped log so it survives the SSH session closing:

```bash
ssh dgx-spark 'docker exec -d thesis-run bash -lc "cd /workspace && mkdir -p artifacts/logs/launcher && <ENV=val …> <script> > artifacts/logs/launcher/<name>_$(date -u +%Y%m%d_%H%M%S).log 2>&1"'
```

Never kill, restart or re-launch a run without being asked, even if it looks stuck or crashed. Report it and let the user decide. Never `docker stop`/`docker rm` the container.

### Known scripts

| Script | Python process to look for | Log | Success marker | Smoke test |
|---|---|---|---|---|
| `src/thesis/shell-scripts/paper/run_signature_leakage.sh` | `run_signature_leakage.py` | `artifacts/logs/signature_leakage/<UTC ts>.log` | `Saved →` / `Mirrored →` lines; output in `artifacts/experiments/signature_leakage/<ts>/` | `MAX_STEPS=1 SEEDS="0"` |
| `src/thesis/shell-scripts/paper/run_mining_setting_selection.sh` | `run_mining_setting_selection.py` | `artifacts/logs/mining_setting_selection/<UTC ts>.log` | `Saved →` / `Mirrored →` lines; output in `artifacts/experiments/mining_setting_selection/<ts>/` (`stage2_grid.csv` is rewritten after every grid point, so its row count shows progress out of 54) | `SMOKE=1 SEEDS="0"` |
| `src/thesis/shell-scripts/paper/run_rule_precision_gate.sh` | `run_rule_precision_gate.py` | `artifacts/logs/rule_precision_gate/<UTC ts>.log` | `Saved →` / `Mirrored →` lines; output in `artifacts/experiments/rule_precision_gate/<ts>/` | `MAX_STEPS=1 SEEDS="0"` |

(The script tries `conda activate thesis`. That env doesn't exist in the container, so a `[warn] 'conda activate thesis' failed` line is expected and harmless: it falls back to the image's Python. A `FATAL: … can't import the core deps` line is a real failure.)

### Watching for crashes

When asked to keep an eye on a run, use the **Monitor** tool (load it via ToolSearch) with a polling loop over SSH, not repeated manual checks. Poll every ~5 minutes; print a line only when something changes, so each line is worth a notification:

```bash
LOGDIR=artifacts/logs/signature_leakage   # from the table
PROC=run_signature_leakage.py
while true; do
  out=$(ssh -o ConnectTimeout=15 dgx-spark "cd ~/Desktop/msc-thesis; L=\$(ls -t $LOGDIR/*.log | head -1); \
    docker exec thesis-run pgrep -f $PROC >/dev/null && echo ALIVE || echo GONE; \
    echo \$L; grep -cE 'Traceback|Error|FATAL|FAILED|Killed|out of memory' \$L; grep -c 'Saved →' \$L; tail -n 1 \$L" 2>&1) \
    || { echo "SSH_FAIL $(date +%H:%M)"; sleep 300; continue; }
  state=$(echo "$out" | sed -n 1p); errs=$(echo "$out" | sed -n 3p); saved=$(echo "$out" | sed -n 4p)
  [[ "$errs" != "0" ]] && echo "ERROR_IN_LOG: $(echo "$out" | sed -n 2p)"
  if [[ "$state" == GONE ]]; then
    [[ "$saved" != "0" ]] && echo "FINISHED_OK" || echo "PROCESS_GONE_WITHOUT_SAVE (crash or OOM kill)"
    break
  fi
  sleep 300
done
```

How to interpret it:
- `ERROR_IN_LOG`: show the user the traceback (`grep -n -B5 -A20 Traceback <log>`), say what failed, and don't fix or restart without their OK. Some scripts log handled warnings that match `Error`, so read the context before calling it a crash.
- `PROCESS_GONE_WITHOUT_SAVE` with no traceback is usually an OOM kill. Check with `ssh dgx-spark 'free -h; dmesg 2>/dev/null | tail -5'` (dmesg may need root; if denied, say so).
- `FINISHED_OK`: offer to rsync the outputs (section 4) and name the notebook (section 5).
- Repeated `SSH_FAIL`: Tailscale or the Mac's network dropped. The run on the DGX is unaffected.

Monitoring only lasts as long as this Claude session is open and the Mac is awake. For an overnight run, tell the user that, and suggest they keep the Mac awake or just check back later with a one-off status check.

## 3b. Read-only status checks

These are always fine:

```bash
ssh dgx-spark 'docker exec thesis-run bash -lc "pgrep -af \"run_.*\\.(sh|py)\" || echo NOT_RUNNING"'
ssh dgx-spark 'cd ~/Desktop/msc-thesis && ls -t artifacts/logs artifacts/logs/* | head -20'
ssh dgx-spark 'tail -n 50 ~/Desktop/msc-thesis/artifacts/logs/<file>.log'
ssh dgx-spark 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'
```

Logs: system-eval experiments write to `artifacts/logs/<experiment>/`. Grouping writes to `src/thesis/baselines/grouping/logs/`.

## 3c. Catching up on an earlier run

A run keeps going on the DGX when the Mac sleeps or the Claude session closes; only the monitoring stops. When the user asks "how's the run going", "did it finish" or "check on X" for a run started earlier (possibly in another session), don't assume anything from memory. Reconstruct the state from the DGX:

1. **Which run:** if the user didn't say, list recent logs across experiments and the launcher logs, newest first, and pick the most recent one (say which you picked):
   ```bash
   ssh dgx-spark 'cd ~/Desktop/msc-thesis && ls -lt --time-style=+%m-%d_%H:%M artifacts/logs/*/*.log src/thesis/baselines/grouping/logs/*.log 2>/dev/null | head -10'
   ```
2. **Still running?** `pgrep -af` the process from the known-scripts table inside the container. Also check the container is up at all (`docker ps -a --filter name=thesis-run`). If the container exited, the run died with it; report the exit time from `docker ps -a`.
3. **What the log says:** last modification time (a log silent for much longer than its usual step time while the process is alive may be hung), `tail -n 30`, and `grep -nE 'Traceback|Error|FATAL|FAILED|Killed|out of memory'`. Look for the script's success marker too (e.g. `Saved →`).
4. **Outputs:** list the newest output folder for that experiment (e.g. `ls -lt artifacts/experiments/signature_leakage | head -3`) and whether it already exists locally.

Then report one of:
- **Still running**: how far it is (last progress line, elapsed time from the log's start), and offer to resume watching with the Monitor loop from section 3.
- **Finished OK**: the output folder, whether it's already synced, and an offer to rsync it (section 4) plus the notebook to re-run (section 5).
- **Crashed or killed**: the traceback or the OOM evidence and roughly when it stopped. Don't restart it; ask the user.

Read-only throughout, so no permission needed for any of these checks.

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
| `artifacts/experiments/mining_setting_selection/` (or `results/paper/mining-setting-selection/`) | `paper/run_mining_setting_selection.sh` (Check 1) | `paper/01_mining_setting_selection.ipynb` |
| `artifacts/experiments/rule_precision_gate/` (or `results/paper/rule-precision-gate/`) | `paper/run_rule_precision_gate.sh` (Check 2) | `paper/02_rule_precision_gate.ipynb` |
| `artifacts/experiments/attribute_mining_parameter_grid/`, `artifacts/mining/` | attribute-mining sweep | `mining/01_mining_param_eda.ipynb` |
| `artifacts/experiments/run_eda/<cscas\|ait-ads>/` | EDA runs | `eda/01_cscas_eda.ipynb`, `eda/02_ait-ads_eda.ipynb` (also reads `run_eda_host/`, `alert_groups/raw/`) |
| `src/thesis/baselines/results/cscas_*.json` | `baselines/run_zeroshot.sh` etc. | `baselines/01_cscas_baseline.ipynb` |
| `src/thesis/baselines/results/ait_ads_*.json` | `baselines/run_ait_ads_*.sh` | `baselines/02_ait-ads_baseline.ipynb` (also needs `artifacts/cache/<scenario>/groups/<method>/` for fox/harrison/russellmitchell/santos/…, the one kind of cache subfolder worth syncing on request) |
| `results/sys-ops/<step>/cscas/<ts>/` | `system_eval/run_sysops_cost_comparison.sh` | no notebook reads it yet; tell the user |
| `src/thesis/baselines/grouping/results/` | `baselines/grouping/run_grouping.sh` | `grouping/grouping_baseline.ipynb` |

`run_decay_and_walkforward.sh` chains Exp 2 then Exp 3, so after it finishes sync both `results/sys-eval/` subfolders and point to notebooks 01 and 02. If a synced folder isn't in this table, find its reader with `grep -l "<folder name>" -r src/thesis/notebooks --include=*.ipynb` and say the table needs updating.

## 6. "Give me the command for …" (cheat sheet)

When the user asks for **the command** to do something on the DGX ("what's the command to…", "give me the command for…", "how do I…"), **answer with the command; don't run it**. Only run it if they say "do it" / "run it" / "check it for me".

How to answer:
- Put each command in its own ```` ```bash ```` block (the app adds a Run button to single-line blocks), with one short sentence above it saying what it does.
- Default to the **from-the-Mac one-liner** (`ssh dgx-spark '…'`). If the user says they're already on the DGX or inside the container, drop the `ssh dgx-spark` / `docker exec` wrapper accordingly.
- Fill in real values (script names, log folders, experiment names) from this skill instead of leaving placeholders, when the request makes them clear. Check the actual name on the DGX first if unsure (a read-only `ls` is fine).
- For destructive commands (kill, stop, rm, reset), give the command but add a one-line warning about what it ends or deletes.
- If the request isn't covered below, compose it from the same building blocks and say so.

Three layers to keep straight: the **Mac**, the **DGX host** (`~/Desktop/msc-thesis`), and the **`thesis-run` container** (`/workspace`, same files). Python processes and the GPU env live in the container.

### Connecting
```bash
# shell on the DGX host
ssh dgx-spark
# straight into a shell inside the container
ssh -t dgx-spark 'docker exec -it thesis-run bash'
# list / attach tmux sessions on the host (detach again with Ctrl-b d)
ssh -t dgx-spark 'tmux ls'
ssh -t dgx-spark 'tmux attach -t <name>'
```

### Container
```bash
# is it running?
ssh dgx-spark 'docker ps -a --filter name=thesis-run --format "{{.Names}}  {{.Status}}"'
# start it (after a reboot or `docker stop`)
ssh dgx-spark 'docker start thesis-run'
# CPU / memory use of the container right now
ssh dgx-spark 'docker stats --no-stream thesis-run'
# stop it — WARNING: kills every run inside it
ssh dgx-spark 'docker stop thesis-run'
# re-run the dependency setup (after pyproject.toml changes; idempotent)
ssh dgx-spark 'docker exec thesis-run bash src/thesis/shell-scripts/baselines/grouping/setup_container.sh'
# check torch sees the GPU inside the container
ssh dgx-spark 'docker exec thesis-run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
```

### Processes inside the container
```bash
# which experiment scripts are running
ssh dgx-spark 'docker exec thesis-run pgrep -af "run_.*\.(sh|py)" || echo NOT_RUNNING'
# with elapsed time, CPU and memory
ssh dgx-spark 'docker exec thesis-run ps -eo pid,etime,%cpu,%mem,cmd --sort=-%cpu | head -15'
# live view (q to quit)
ssh -t dgx-spark 'docker exec -it thesis-run top'
# stop one run — WARNING: the run is lost, partial outputs may be half-written
ssh dgx-spark 'docker exec thesis-run pkill -f run_signature_leakage.py'
```

### GPU and machine
```bash
ssh dgx-spark 'nvidia-smi'
# refresh every 2 s (Ctrl-c to quit)
ssh -t dgx-spark 'watch -n 2 nvidia-smi'
# memory, disk, load, who else is on the machine
ssh dgx-spark 'free -h; df -h ~; uptime; who'
# what's taking space in the repo
ssh dgx-spark 'cd ~/Desktop/msc-thesis && du -sh artifacts/* results src/thesis/baselines/*/results 2>/dev/null | sort -h'
```

### Logs
```bash
# newest logs across all experiments
ssh dgx-spark 'cd ~/Desktop/msc-thesis && ls -lt artifacts/logs/*/*.log | head -10'
# follow the newest log of one experiment live (Ctrl-c to stop; the run keeps going)
ssh -t dgx-spark 'cd ~/Desktop/msc-thesis && tail -f $(ls -t artifacts/logs/signature_leakage/*.log | head -1)'
# last 50 lines
ssh dgx-spark 'cd ~/Desktop/msc-thesis && tail -n 50 $(ls -t artifacts/logs/signature_leakage/*.log | head -1)'
# any errors in it
ssh dgx-spark 'cd ~/Desktop/msc-thesis && grep -nE "Traceback|Error|FATAL|FAILED|Killed|out of memory" $(ls -t artifacts/logs/signature_leakage/*.log | head -1)'
```

### Git on the DGX
```bash
# where is the DGX's checkout, and is it clean
ssh dgx-spark 'cd ~/Desktop/msc-thesis && git status -sb && git log --oneline -3'
# update to the latest pushed main
ssh dgx-spark 'cd ~/Desktop/msc-thesis && git pull --ff-only'
# same, from inside the container (if the host pull hits root-owned files)
ssh dgx-spark 'docker exec thesis-run bash -lc "cd /workspace && git pull --ff-only"'
```

### Launching a run
```bash
# detached inside the container, launcher output logged (survives closing SSH)
ssh dgx-spark 'docker exec -d thesis-run bash -lc "cd /workspace && mkdir -p artifacts/logs/launcher && src/thesis/shell-scripts/paper/run_signature_leakage.sh > artifacts/logs/launcher/signature_leakage_\$(date -u +%Y%m%d_%H%M%S).log 2>&1"'
# smoke-test variant: put env overrides before the script, e.g.
#   MAX_STEPS=1 SEEDS=0 src/thesis/shell-scripts/paper/run_signature_leakage.sh
```
Interactive alternative the user also uses: tmux exists on the DGX **host only** (not in the container). So `ssh -t dgx-spark 'tmux new -s <name>'`, then inside that tmux session `docker exec -it thesis-run bash` and run the script; detach with Ctrl-b d, re-attach later with `tmux attach -t <name>`. Or, inside a container shell, `nohup <script> > <log> 2>&1 &`.

### Results
```bash
# newest output folders of one experiment
ssh dgx-spark 'ls -lt ~/Desktop/msc-thesis/artifacts/experiments/signature_leakage | head -5'
# preview what rsync would copy (dry run), then the real copy — run from the local repo root
rsync -azhn --itemize-changes dgx-spark:Desktop/msc-thesis/artifacts/experiments/signature_leakage/ artifacts/experiments/signature_leakage/
rsync -azh --partial --info=progress2 dgx-spark:Desktop/msc-thesis/artifacts/experiments/signature_leakage/ artifacts/experiments/signature_leakage/
# a single file
scp dgx-spark:Desktop/msc-thesis/<path> <local path>
```
(For which folders to sync and what to exclude, see section 4.)
