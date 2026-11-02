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
# recording is in progress (--force to do it anyway). The worker (the SOP builder's background jobs) starts too.
# Take everything down with ./stop.sh.
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR=data/training/models
env_default() { grep -E "^$1=" .env 2>/dev/null | tail -n1 | cut -d= -f2- || true; }

# Compose needs .env; a fresh clone starts from the example.
[[ -f .env ]] || { cp .env.example .env; echo "Created .env from .env.example." >&2; }

set_env() {  # set_env KEY VALUE: replace KEY's line in .env, or append it
  local tmp
  tmp=$(mktemp)
  K=$1 V=$2 awk 'BEGIN { k = ENVIRON["K"]; v = ENVIRON["V"] }
    $0 ~ "^" k "=" { if (!done) print k "=" v; done = 1; next } { print }
    END { if (!done) print k "=" v }' .env > "$tmp"
  cat "$tmp" > .env && rm -f "$tmp"
}

require_secret() {  # require_secret KEY "what it is for": prompt (hidden) and save to .env when it is empty
  local key=$1 why=$2 value
  [[ -n $(env_default "$key") ]] && return 0
  if [[ ! -t 0 ]]; then echo "$key is not set in .env ($why)." >&2; exit 1; fi
  read -rsp "$key ($why): " value; echo >&2
  [[ -n $value ]] || { echo "$key is required for this; add it to .env." >&2; exit 1; }
  set_env "$key" "$value"
  echo "Saved $key to .env." >&2
}

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

checkpoints() { find "$MODELS_DIR" -mindepth 2 -maxdepth 2 -type f -name model.safetensors -printf '%h\n' 2>/dev/null | xargs -r -n1 basename | sort; }

# A fresh clone has no detectors or recordings (gitignored): offer the demo set from Hugging Face (scripts/demo_assets.py).
missing_assets() {
  local dataset w
  dataset=$(env_default SOP_DATASET_DIR); dataset=${dataset:-./sop-sample-training-data}
  [[ -d $dataset/server_fan/raw ]] || echo "$dataset/server_fan"
  for w in $(sed -n 's/^ *weights: *\([^ #]*\).*/\1/p' sops/*.yaml | sort -u); do [[ -f $w ]] || echo "$w"; done
}
mapfile -t missing < <(missing_assets)
if (( ${#missing[@]} )); then
  echo "Missing demo assets: ${missing[*]}" >&2
  fetch=(docker compose run --rm demo-assets fetch)
  if [[ -t 0 ]]; then
    read -rp "Download the demo assets from Hugging Face now (~3.4 GB; needs HF_TOKEN in .env)? [Y/n] " reply
    if [[ ! $reply =~ ^[Nn] ]]; then
      require_secret HF_TOKEN "Hugging Face access token, read access"
      HOST_USER="$(id -u):$(id -g)" "${fetch[@]}" || true
    fi
    mapfile -t missing < <(missing_assets)
  fi
  (( ${#missing[@]} )) && { echo "The app needs them to start. Fetch with: ${fetch[*]}" >&2; exit 1; }
fi

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
[[ $model == jev || $model == compare ]] && require_secret TYPESAFE_API_KEY "TypeSafe API key for Jev; or use --model laya"

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
# The worker runs the SOP builder's jobs (timeline drafts, slot auto-fit, training); idle, it holds no GPU memory.
docker compose up -d worker

printf 'Waiting for the app'
ready=0
for _ in $(seq 60); do
  curl -sf -m 2 localhost:8000/sources >/dev/null 2>&1 && { ready=1; break; }
  # The app exits at once when a startup check fails (key, GPU, weights, model service): show why.
  if [[ $(docker inspect -f '{{.State.Running}}' "$(docker compose ps -aq sopvision)" 2>/dev/null) != true ]]; then
    echo; echo "The app stopped during startup:" >&2
    docker compose logs --no-log-prefix --tail 20 sopvision >&2
    exit 1
  fi
  printf '.'; sleep 2
done
echo
(( ready )) || { echo "The app didn't answer within 2 minutes; see: docker compose logs sopvision" >&2; exit 1; }
echo "Dashboard:     http://localhost:8000"
(( labelstudio )) && echo "Label Studio:  http://localhost:8080"
echo "Stop with:     ./stop.sh"
