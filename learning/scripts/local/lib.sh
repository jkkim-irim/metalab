#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# lib.sh — shared config for LOCAL (workstation) MetaLab runs (metalab_train.sh / metalab_eval.sh).
# The trainer + the engine sim-service run over RPC, both in the engine's uv venv. Values are env-overridable.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Repo root: this file is learning/scripts/local/lib.sh -> ../../.. = <repo>.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

log(){ echo "[${LOG_TAG:-local}] $*"; }

# A uv freshly installed by sim/setup.sh (astral installer) lands in ~/.local/bin; a shell that hasn't
# picked up the installer's rc change won't see it. Add it so setup_env.sh/metalab_train.sh find uv anyway.
command -v uv >/dev/null 2>&1 || case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# uv-managed engine envs (replaces conda). Each engine is a uv project under sim/metalab/setup/<engine>/
# (pyproject.toml + uv.lock, committed); setup_env.sh `uv sync`s it into a venv under this root, and
# metalab_train.sh/metalab_eval.sh activate that venv. Override the root via env.
#
# PER WORKTREE, by the directory name. `uv sync` is EXCLUSIVE — it removes whatever the project's lock
# does not list — so one venv shared by several checkouts belongs to whichever synced last: working in
# one worktree silently strips the packages another needs (measured: rerun-sdk and tensorboard vanished
# three times in one session, and warp-lang got swapped for a dev build). It also makes each worktree's
# committed uv.lock a claim the environment cannot honour. Isolation costs ~18 MiB per extra venv, not
# gigabytes: uv hardlinks every package from its shared cache (measured on a second newton env).
export METALAB_VENV_ROOT="${METALAB_VENV_ROOT:-$HOME/.metalab/venvs/$(basename "$ROOT")}"
engine_uv_project(){ echo "$ROOT/sim/metalab/setup/$1"; }   # $1 = genesis|newton → uv project dir
engine_venv(){ echo "$METALAB_VENV_ROOT/$1"; }           # $1 = genesis|newton → its venv path

# Trainer log root (checkpoints land under $RL_LOG_ROOT/<exp>/<run_name>/model_*.pt).
export RL_LOG_ROOT="${RL_LOG_ROOT:-$ROOT/logs/rl}"

# ── task · recipe (two axes; mirrors sim/metalab/contract/loader.py) ─────────────────────────────
# A TASK is a family folder tasks/<task>/ (or a single-file tasks/<task>.py); a RECIPE is one
# tasks/<task>/<task>_<recipe>.py beside the shared _base.py. A family is not runnable by itself, so
# both axes are required for one — resolved here in bash too, to fail before the venv/engine boot.
_TASKS_DIR="$ROOT/sim/metalab/contract/tasks"
list_tasks(){                      # every runnable --task: family folders + top-level contracts
  local p n out=()
  for p in "$_TASKS_DIR"/*/; do
    n="$(basename "$p")"
    [ "$n" = standalone ] || [ ! -f "$p/__init__.py" ] || out+=("${n//_/-}")
  done
  for p in "$_TASKS_DIR"/*.py; do   # '_*.py' is a shared library (_assets), not a contract
    n="$(basename "$p" .py)"
    [ ! -f "$p" ] || [ "${n#_}" != "$n" ] || out+=("${n//_/-}")
  done
  [ ${#out[@]} -eq 0 ] || printf '%s\n' "${out[@]}" | sort -u
}
list_standalone_tasks(){           # tasks/standalone/[<group>/]*.py — the CONTRACT stems --task accepts.
  local p n out=()                 # the group folder is a shelf (the Launchpad's second combobox), not
  for p in "$_TASKS_DIR"/standalone/*.py "$_TASKS_DIR"/standalone/*/*.py; do   # part of the task name
    n="$(basename "$p" .py)"
    [ ! -f "$p" ] || [ "${n#_}" != "$n" ] || out+=("${n//_/-}")
  done
  [ ${#out[@]} -eq 0 ] || printf '%s\n' "${out[@]}" | sort
}
list_recipes(){                    # $1 = task (dash or underscore) → its recipes, empty if single-file
  local t="${1//-/_}" p n out=()
  for p in "$_TASKS_DIR/$t/${t}_"*.py; do
    [ -f "$p" ] || continue
    n="$(basename "$p" .py)"; out+=("${n#"${t}_"}")
  done
  [ ${#out[@]} -eq 0 ] || printf '%s\n' "${out[@]//_/-}" | sort
}
require_task_recipe(){             # $1 = tag for the message, $2 = task, $3 = recipe (may be empty)
  local tag="$1" t="${2//-/_}" r="$3" recipes
  recipes="$(list_recipes "$t")"
  if [ -d "$_TASKS_DIR/$t" ]; then
    [ -n "$recipes" ] || { echo "[$tag] task '$2' is a family with no recipe — add $_TASKS_DIR/$t/${t}_<recipe>.py" >&2; exit 2; }
    if [ -z "$r" ]; then
      echo "[$tag] --recipe is required for task '$2' (a task family). Available recipes:" >&2
      echo "$recipes" | sed 's/^/  - /' >&2
      exit 2
    fi
    grep -qx -- "${r//_/-}" <<<"$recipes" || {
      echo "[$tag] unknown recipe '$r' for task '$2'. Available:" >&2
      echo "$recipes" | sed 's/^/  - /' >&2
      exit 2
    }
  elif [ -n "$r" ]; then
    echo "[$tag] task '$2' is a single-file contract — it takes no --recipe (got '$r')" >&2
    exit 2
  fi
}
