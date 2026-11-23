#!/usr/bin/env bash
# Read-only snapshot of what OTHER users are doing on the DGX Spark, plus
# free resources. Runs on the DGX; call it from the Mac with:
#   ssh dgx-spark 'bash -s' < .claude/skills/dgx/check_shared_usage.sh
# It never changes anything (no kill/renice/docker writes).

ME=$(id -un)
# PIDs (host view) of processes inside our own container, so they aren't
# mistaken for another user's root-owned container workload.
OURS=$(docker top thesis-run -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ')

echo "== logged in (sessions per user)"
who | awk '{print $1}' | sort | uniq -c

echo "== other users' busy processes (>=10% CPU or >=1 GB RAM, running >1 min)"
echo "   (root-owned rows outside thesis-run are usually someone else's container or a system service)"
ps -eo user:12,pid,etime,%cpu,rss,args --sort=-rss | awk -v me="$ME" -v ours=" $OURS " '
  NR==1 {print substr($0,1,150); next}
  $3 ~ /^00:/ {next}                       # started <1 min ago: transient
  index(ours, " "$2" ") {next}             # ours (inside thesis-run)
  $1!=me && ($4>=10 || $5>=1048576) {print substr($0,1,150)}' | head -15

echo "== containers (thesis-run is ours; any other belongs to someone else)"
docker ps --format '{{.Names}}  {{.Image}}  up {{.RunningFor}}'
docker stats --no-stream --format '{{.Name}}  cpu {{.CPUPerc}}  mem {{.MemUsage}}'

echo "== GPU (single GB10, shared)"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | sed 's/^/util: /'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
  while IFS=, read -r p n m; do echo "pid $p ($(ps -o user= -p "$p")) $n $m"; done

echo "== ollama (shared system service; models currently loaded)"
ollama ps 2>&1 | tail -n +2

echo "== machine"
free -g | awk 'NR==2{print "RAM (unified CPU+GPU): " $7 " GB available of " $2}'
echo "load: $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) cores"
df -h ~ | awk 'NR==2{print "disk: " $4 " free (" $5 " used)"}'
