#!/usr/bin/env bash
# Take the app down: the dashboard, the model services (Laya, Decider) and Label Studio.
#
#   ./stop.sh                  # everything
#   ./stop.sh --keep-models    # leave Laya / Decider loaded (a restart with the same model is then quick)
#   ./stop.sh --keep-labelstudio
#   ./stop.sh --force          # even while a run or recording is in progress (it is cut off)
#
# Containers are stopped and removed; volumes are kept (model weights cache, Label Studio projects and logins),
# as are runs/, datasets/ and data/.
set -euo pipefail
cd "$(dirname "$0")"

keep_models=0 keep_ls=0 force=0
for arg in "$@"; do
  case "$arg" in
    --keep-models) keep_models=1 ;;
    --keep-labelstudio) keep_ls=1 ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

busy() {  # true while the app is up and running or recording something
  local rec run
  rec=$(curl -s -m 3 localhost:8000/recording 2>/dev/null) || return 1
  run=$(curl -s -m 3 localhost:8000/runs/current 2>/dev/null) || return 1
  [[ ( -n $rec && $rec != null ) || ( -n $run && $run != null ) ]]
}
if (( ! force )) && busy; then
  echo "A run or recording is in progress: stop it in the dashboard first (or pass --force)." >&2
  exit 1
fi

services=(sopvision)
(( keep_models )) || services+=(laya decider)
(( keep_ls )) || services+=(labelstudio)
compose=(docker compose --profile laya --profile decider --profile labelling)

"${compose[@]}" stop "${services[@]}"
"${compose[@]}" rm -f "${services[@]}" >/dev/null
echo "Stopped: ${services[*]}"
