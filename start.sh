#!/usr/bin/env bash
# Bring the app up in Docker with a chosen decision model, optionally with Label Studio.
#
#   ./start.sh                                  # asks (model, Laya checkpoint, Label Studio)
#   ./start.sh --model laya [--checkpoint laya-sop-v2-e12] [--labelstudio]
#   ./start.sh --model jev | decider
#   ./start.sh --model compare --shadow laya    # Jev decides, the shadow model's answers are logged next to it
#
# The model only lasts for this start: .env's DECISION_MODE / LAYA_CHECKPOINT are the defaults and are not edited.
# Model services the choice doesn't need are stopped, to free the GPU. The app is not restarted while a run or a
# recording is in progress (--force to do it anyway). Take everything down with ./stop.sh.
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR=data/training/models
env_default() { grep -E "^$1=" .env 2>/dev/null | tail -n1 | cut -d= -f2- || true; }

model="" checkpoint="" shadow="" labelstudio="" force=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --shadow) shadow="$2"; shift 2 ;;
    --labelstudio) labelstudio=1; shift ;;
    --no-labelstudio) labelstudio=0; shift ;;
    --force) force=1; shift ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

ask() {  # ask <prompt> <default> <choice>...  -> echoes the choice
  local prompt=$1 def=$2; shift 2
  local i=1 c
  for c in "$@"; do printf '  %d) %s%s\n' "$i" "$c" "$([[ $c == "$def" ]] && echo '  (default)')" >&2; i=$((i + 1)); done
  read -rp "$prompt " reply
  if [[ -z $reply ]]; then echo "$def"
  elif [[ $reply =~ ^[0-9]+$ ]] && (( reply >= 1 && reply <= $# )); then echo "${!reply}"
  else echo "$reply"; fi
}

checkpoints() { find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort; }

interactive=0
[[ -z $model && -t 0 ]] && interactive=1
if [[ -z $model ]]; then
  def=$(env_default DECISION_MODE); def=${def:-jev}
  if (( interactive )); then
    echo "Decision model:" >&2
    model=$(ask "Choose [default $def]:" "$def" jev laya decider compare)
  else
    model=$def
  fi
fi
case "$model" in jev|laya|decider|compare) ;; *) echo "unknown model: $model (jev | laya | decider | compare)" >&2; exit 2 ;; esac

if [[ $model == compare && -z $shadow ]]; then
  if (( interactive )); then
    echo "Shadow model (answers logged next to Jev's):" >&2
    shadow=$(ask "Choose [default laya]:" laya laya decider)
  else
    shadow=laya
  fi
fi
[[ -n $shadow && $shadow != laya && $shadow != decider ]] && { echo "unknown shadow: $shadow (laya | decider)" >&2; exit 2; }

needs_laya=0; [[ $model == laya || $shadow == laya ]] && needs_laya=1
needs_decider=0; [[ $model == decider || $shadow == decider ]] && needs_decider=1

if (( needs_laya )) && [[ -z $checkpoint ]]; then
  def=$(basename "$(env_default LAYA_CHECKPOINT)" 2>/dev/null || true)
  mapfile -t found < <(checkpoints)
  if (( interactive )); then
    echo "Laya checkpoint ('shipped' = the base English model):" >&2
    checkpoint=$(ask "Choose [default ${def:-shipped}]:" "${def:-shipped}" "${found[@]}" shipped)
  else
    checkpoint=${def:-shipped}
  fi
fi
if (( needs_laya )); then
  if [[ $checkpoint == shipped ]]; then
    laya_ckpt=""
  else
    checkpoint=$(basename "$checkpoint")
    [[ -d $MODELS_DIR/$checkpoint ]] || { echo "no checkpoint $MODELS_DIR/$checkpoint; have: $(checkpoints | tr '\n' ' ')" >&2; exit 2; }
    laya_ckpt=/data/models/$checkpoint
  fi
fi

if [[ -z $labelstudio ]]; then
  if (( interactive )); then
    read -rp "Start Label Studio too? [y/N] " reply
    [[ $reply =~ ^[Yy] ]] && labelstudio=1 || labelstudio=0
  else
    labelstudio=0
  fi
fi

# Don't pull the app out from under a run or a recording.
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

echo
echo "Starting: model $model${shadow:+ (shadow $shadow)}$( (( needs_laya )) && echo ", Laya checkpoint ${checkpoint}")$( (( labelstudio )) && echo ", Label Studio")"

# Free the GPU from model services this choice doesn't use.
stop=()
(( needs_laya )) || stop+=(laya)
(( needs_decider )) || stop+=(decider)
(( ${#stop[@]} )) && docker compose --profile laya --profile decider stop "${stop[@]}" >/dev/null 2>&1 || true

wait_healthy() {  # wait_healthy <service> <timeout s>
  local svc=$1 limit=$2 waited=0 status
  printf 'Waiting for %s to load' "$svc"
  while :; do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q "$svc")" 2>/dev/null || echo starting)
    [[ $status == healthy ]] && { echo " ready"; return 0; }
    if (( waited >= limit )); then echo; echo "$svc not healthy after ${limit}s: docker compose logs $svc" >&2; return 1; fi
    printf '.'; sleep 5; waited=$((waited + 5))
  done
}

if (( needs_laya )); then
  # Recreate only when the checkpoint changed (compose compares the config), so an unchanged Laya stays loaded.
  LAYA_CHECKPOINT=$laya_ckpt docker compose --profile laya up -d laya
  wait_healthy laya 600
fi
if (( needs_decider )); then
  docker compose --profile decider up -d decider
  wait_healthy decider 900
fi
(( labelstudio )) && docker compose --profile labelling up -d labelstudio

DECISION_MODE=$model COMPARE_SHADOW=${shadow:-decider} LAYA_CHECKPOINT=${laya_ckpt:-} \
  docker compose up -d --force-recreate sopvision

printf 'Waiting for the app'
for _ in $(seq 60); do
  curl -sf -m 2 localhost:8000/sources >/dev/null 2>&1 && break
  printf '.'; sleep 2
done
echo
echo "Dashboard:     http://localhost:8000"
(( labelstudio )) && echo "Label Studio:  http://localhost:8080"
echo "Stop with:     ./stop.sh"
