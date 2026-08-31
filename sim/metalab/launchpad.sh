#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launchpad.sh — MetaLab Launchpad launcher + one-time desktop-icon self-install.
#
# First run from a terminal registers a per-user desktop entry (app icon) at
#   ~/.local/share/applications/metalab-launchpad.desktop
# so the Launchpad appears in the GNOME app grid (Ubuntu 24.04) and can be pinned. The icon
# launches THIS script with --bg, which starts the Launchpad server DETACHED (no terminal
# window) and just opens the browser; re-clicking reuses the running server instead of
# spawning a second one. Stop the detached server with `launchpad.sh --stop`.
#
# Run WITHOUT --bg from a terminal to keep the server in the FOREGROUND (logs on screen,
# Ctrl-C to stop) — the dev path. The Launchpad is stdlib-only python3 (no conda env needed);
# the launched training scripts activate their own env.
#
# Flags:
#   --bg              start the server detached (no terminal), open the browser, exit
#   --stop            stop the detached Launchpad server (via pidfile), then exit
#   --no-icon         skip the desktop-entry step
#   --reinstall-icon  force-rewrite the desktop entry
#   --install-only    write/refresh the desktop entry and exit (no launch)
# Any other flag falls through to the server (--host, --port, --no-browser).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SERVER="$SCRIPT_DIR/launchpad/server.py"
ICON_PATH="$SCRIPT_DIR/launchpad/assets/metalab_logo.png"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

# Branding — distinct per checkout so multiple worktrees install as SEPARATE app icons instead of
# clobbering one shared desktop entry / grouping into one taskbar window. Defaults derive from the
# repo dir: the `metalab-motor-to-joint-control` worktree registers as "M2J Launchpad" (own .desktop,
# WM class, browser profile, port) alongside a main "MetaLab Launchpad" install. Override any via env.
# Patterns are suffix globs so renaming a worktree's prefix does not silently drop it to the default
# branding — which would collide with the main install's .desktop, WM class and port.
#
# PORTS ARE 878x ON PURPOSE. An unrelated checkout on this machine can ship its own launchpad with its
# own defaults; sharing a port means clicking this icon opens THAT console (see _hub_probe). Keep every
# MetaLab port inside 878x, and give a new worktree its own number here rather than reusing one.
case "$(basename "$REPO")" in
  *motor-to-joint-control) _NAME="M2J Launchpad"; _DSLUG="m2j-launchpad"; _WM="m2j-hub"; _PORT=8781 ;;
  *)                       _NAME="MetaLab Launchpad"; _DSLUG="metalab-launchpad"; _WM="metalab-hub"; _PORT=8780 ;;
esac
LAUNCHPAD_NAME="${LAUNCHPAD_NAME:-$_NAME}"
LAUNCHPAD_DESKTOP_SLUG="${LAUNCHPAD_DESKTOP_SLUG:-$_DSLUG}"
LAUNCHPAD_WMCLASS="${LAUNCHPAD_WMCLASS:-$_WM}"

DESKTOP_FILE="$DESKTOP_DIR/${LAUNCHPAD_DESKTOP_SLUG}.desktop"
PORT="${HUB_PORT:-$_PORT}"
PIDFILE="$REPO/logs/launchpad/launchpad.pid"
LOGFILE="$REPO/logs/launchpad/launchpad_server.log"
PY="${PYTHON:-python3}"

# Dedicated, isolated Chrome profile for the Launchpad's app window(s). Giving Chrome its own
# --user-data-dir spawns a SEPARATE Chrome instance (own process + taskbar entry), fully
# decoupled from the user's normal Chrome: closing the Launchpad never touches their tabs, and
# quitting their Chrome never closes the Launchpad. Exported so the in-sim telemetry viewer opens
# its app window in the SAME isolated instance. Override to relocate.
export METALAB_HUB_BROWSER_PROFILE="${METALAB_HUB_BROWSER_PROFILE:-${XDG_DATA_HOME:-$HOME/.local/share}/${LAUNCHPAD_WMCLASS}/browser}"

NO_ICON=0; REINSTALL=0; INSTALL_ONLY=0; BG=0; STOP=0; PASS=()
for a in "$@"; do
  case "$a" in
    --bg)             BG=1 ;;
    --stop)           STOP=1 ;;
    --no-icon)        NO_ICON=1 ;;
    --reinstall-icon) REINSTALL=1 ;;
    --install-only)   INSTALL_ONLY=1 ;;
    *)                PASS+=("$a") ;;   # forwarded to the server (--host, --port, --no-browser)
  esac
done

log(){ printf '[launchpad] %s\n' "$*"; }

# Open the Launchpad as a standalone Chrome "app window" (--app): no tabs/address bar/toolbar, its own
# taskbar entry, and — via the dedicated --user-data-dir above — its own Chrome instance, so it
# behaves like a native app decoupled from the user's normal Chrome (the whole point: turn the app
# on/off without touching their tabs). telemetry (--viz) opens as its own app window in the SAME
# isolated instance. Best-effort; degrades to python webbrowser new-window.
_open_url(){
  [ -n "${HUB_NO_OPEN:-}" ] && return 0   # headless/test: start detached without opening a browser
  local url="$1" def bin=""
  def="$(xdg-settings get default-web-browser 2>/dev/null || true)"
  case "${BROWSER:-$def}" in
    *chrom*)   bin="$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chromium-browser || true)" ;;
    *firefox*) bin="$(command -v firefox || true)" ;;
  esac
  [ -n "$bin" ] || bin="$(command -v google-chrome || command -v chromium || command -v firefox || true)"
  case "$bin" in
    *chrom*)
      # --app=$url = standalone app window; own --user-data-dir = own isolated Chrome instance.
      # --class sets the window WM_CLASS so GNOME groups it under the MetaLab Launchpad taskbar icon
      # (matched by StartupWMClass in the .desktop entry) instead of absorbing it into Chrome's icon.
      local flags=("--app=$url" "--class=$LAUNCHPAD_WMCLASS" "--user-data-dir=$METALAB_HUB_BROWSER_PROFILE" --no-first-run --no-default-browser-check) dim=""
      command -v xrandr >/dev/null 2>&1 && dim="$(xrandr 2>/dev/null | awk '/\*/{print $1; exit}')"
      if [ -n "$dim" ]; then flags+=(--window-position=0,0 "--window-size=${dim/x/,}")   # fill the screen
      else flags+=(--start-maximized); fi
      "$bin" "${flags[@]}" >/dev/null 2>&1 & ;;
    *firefox*) "$bin" --new-window "$url" >/dev/null 2>&1 & ;;
    *)         "$PY" -c "import webbrowser,sys;webbrowser.open(sys.argv[1],new=1)" "$url" >/dev/null 2>&1 & ;;
  esac
}

# Who is on $PORT?  0 = OUR hub · 1 = nobody · 2 = someone else's server.
# "Something is listening" is NOT enough to call it ours: another checkout's launchpad can default to
# the same port, and attaching to it opens THAT repo's console under this icon — you then launch runs
# against the wrong tree without noticing. /api/discover reports the repo a hub serves, so only a
# matching repo counts as ours.
_hub_probe(){
  "$PY" - "$PORT" "$REPO" <<'PYEOF' 2>/dev/null
import json, sys, urllib.request
port, repo = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/discover", timeout=1.0) as r:
        served = json.load(r).get("repo", "")
except Exception:
    sys.exit(1)                                  # nothing there (or not an HTTP hub)
sys.exit(0 if served == repo else 2)
PYEOF
}

# Write (or overwrite) the per-user desktop entry with absolute paths, then refresh the
# desktop database and validate — fail loud on a malformed entry. Terminal=false + --bg so a
# click never opens a stray terminal window.
write_desktop_entry(){
  mkdir -p "$DESKTOP_DIR"
  cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=$LAUNCHPAD_NAME
GenericName=RL Training Console
Comment=원클릭 학습·검증 웹 콘솔 ($LAUNCHPAD_NAME)
Exec=bash "$SCRIPT_PATH" --bg
Icon=$ICON_PATH
Terminal=false
Categories=Development;
Keywords=MetaLab;M2J;RL;sim;newton;genesis;training;
StartupNotify=true
StartupWMClass=$LAUNCHPAD_WMCLASS
EOF
  chmod +x "$DESKTOP_FILE" 2>/dev/null || true
  command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
  if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$DESKTOP_FILE" || { log "desktop entry 검증 실패: $DESKTOP_FILE"; exit 1; }
  fi
}

# Create on first run; refresh only if the Exec drifted (repo moved/renamed or older entry).
ensure_desktop_entry(){
  [ "$NO_ICON" = 1 ] && return 0
  if [ "$REINSTALL" = 1 ]; then
    write_desktop_entry; log "desktop 아이콘 재설치 → $DESKTOP_FILE"; return 0
  fi
  if [ ! -f "$DESKTOP_FILE" ]; then
    write_desktop_entry
    log "desktop 아이콘 생성 → $DESKTOP_FILE"
    log "  GNOME 'Show Applications'에서 '$LAUNCHPAD_NAME' 검색 → 우클릭 'Add to Favorites'로 독에 고정."
    return 0
  fi
  # Refresh if the Exec drifted (repo moved) OR the entry predates StartupWMClass (needed so the app
  # window groups under this icon instead of Chrome) — self-heals on a normal --bg launch.
  if ! grep -qF "Exec=bash \"$SCRIPT_PATH\" --bg" "$DESKTOP_FILE" \
     || ! grep -q "^StartupWMClass=$LAUNCHPAD_WMCLASS" "$DESKTOP_FILE"; then
    write_desktop_entry; log "desktop 아이콘 갱신 → $DESKTOP_FILE"
  fi
}

stop_server(){
  if [ -f "$PIDFILE" ]; then
    local pid; pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true; log "Launchpad 종료 (pid $pid)"
    else
      log "실행 중인 Launchpad 없음 (pidfile stale)"
    fi
    rm -f "$PIDFILE"
  else
    log "실행 중인 Launchpad 없음 (pidfile 없음)"
  fi
}

# Foreground (dev/terminal): block on the server, logs on screen, Ctrl-C to stop.
run_hub_fg(){
  command -v "$PY" >/dev/null 2>&1 || { log "python3 없음 — Launchpad 는 stdlib-only(conda 불필요)"; exit 1; }
  # --port explicitly, or this path silently falls back to server.py's own default instead of this
  # checkout's branded port. A --port the caller passed through wins (it lands in PASS after ours).
  log "Launchpad 서버 시작 (foreground, port $PORT, Ctrl-C 종료)…"
  exec "$PY" "$SERVER" --port "$PORT" ${PASS[@]+"${PASS[@]}"}
}

# Detached (icon/--bg): reuse a running server if present, else start one in the background
# (output → $LOGFILE) and open the browser. No terminal window.
run_hub_bg(){
  command -v "$PY" >/dev/null 2>&1 || { log "python3 없음 — Launchpad 는 stdlib-only(conda 불필요)"; exit 1; }
  mkdir -p "$(dirname "$PIDFILE")"
  # `|| _st=$?`, not a bare call: this script runs under `set -e`, where a non-zero return outside a
  # condition context kills it — a free port (1) would exit silently instead of starting the Launchpad.
  _st=0; _hub_probe || _st=$?
  if [ "$_st" = 2 ]; then
    log "포트 $PORT 에 다른 런치패드가 떠 있습니다 — 그 콘솔을 이 아이콘으로 여는 것을 막기 위해 중단합니다."
    log "  다른 포트로 띄우려면:  HUB_PORT=<빈 포트> $SCRIPT_PATH"
    log "  또는 그 프로세스를 먼저 종료하세요:  ss -ltnp | grep :$PORT"
    exit 1
  fi
  if [ "$_st" = 0 ]; then
    log "Launchpad 가 이미 실행 중 — 브라우저만 엽니다 (http://127.0.0.1:$PORT)"
    _open_url "http://127.0.0.1:$PORT"; exit 0
  fi
  : > "$LOGFILE"
  nohup "$PY" "$SERVER" --no-browser --port "$PORT" >>"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  # wait until the server prints its bound URL (≤ ~3s), then open that exact port
  local url="" i=0
  while [ "$i" -lt 30 ]; do
    url="$(grep -oE 'http://127\.0\.0\.1:[0-9]+' "$LOGFILE" 2>/dev/null | head -1 || true)"
    [ -n "$url" ] && break
    sleep 0.1; i=$((i+1))
  done
  [ -n "$url" ] || url="http://127.0.0.1:$PORT"
  log "Launchpad 백그라운드 시작 (pid $(cat "$PIDFILE"), 로그 $LOGFILE) → $url"
  log "종료하려면: $SCRIPT_PATH --stop"
  _open_url "$url"
}

# --- dispatch ---------------------------------------------------------------
[ "$STOP" = 1 ] && { stop_server; exit 0; }
ensure_desktop_entry
[ "$INSTALL_ONLY" = 1 ] && { log "install-only: 아이콘만 설치하고 종료."; exit 0; }
if [ "$BG" = 1 ]; then run_hub_bg; else run_hub_fg; fi
