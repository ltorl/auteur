#!/bin/sh
# Auteur installer — an AI film director in your terminal.
#
#   curl -fsSL https://raw.githubusercontent.com/ltorl/auteur/main/install.sh | sh
#
# Environment knobs:
#   AUTEUR_REPO=https://github.com/ltorl/auteur  where to install from (git)
#   AUTEUR_REF=main                               branch / tag / commit
#   AUTEUR_SOURCE=<path|git+url|https tarball>    install from somewhere else entirely
#   AUTEUR_LITE=1                                 skip the local LTX-2.5 engine (torch etc.)
#   AUTEUR_NO_DEPS=1                              don't install ffmpeg / node
#   AUTEUR_NO_MODIFY_PATH=1                       don't touch your shell profile
#
# Uninstall:  uv tool uninstall auteur

set -eu

# where `curl | sh` installs from
AUTEUR_REPO="${AUTEUR_REPO:-https://github.com/ltorl/auteur}"
AUTEUR_REF="${AUTEUR_REF:-main}"
AUTEUR_SOURCE="${AUTEUR_SOURCE:-}"
AUTEUR_LITE="${AUTEUR_LITE:-0}"
AUTEUR_NO_DEPS="${AUTEUR_NO_DEPS:-0}"
AUTEUR_NO_MODIFY_PATH="${AUTEUR_NO_MODIFY_PATH:-0}"
PYTHON_VERSION="3.12"

# ---- output ------------------------------------------------------------------ #

if [ -t 1 ] && [ "${NO_COLOR:-}" = "" ]; then
  AMBER="$(printf '\033[1;38;2;255;176;0m')"
  ROSE="$(printf '\033[1;38;2;255;61;127m')"
  TEAL="$(printf '\033[38;2;61;219;217m')"
  DIM="$(printf '\033[38;2;138;133;124m')"
  BOLD="$(printf '\033[1m')"
  OFF="$(printf '\033[0m')"
else
  AMBER="" ROSE="" TEAL="" DIM="" BOLD="" OFF=""
fi

say()  { printf '%s\n' "$*"; }
step() { printf '\n%s◆%s %s%s%s\n' "$AMBER" "$OFF" "$BOLD" "$*" "$OFF"; }
ok()   { printf '  %s✓%s %s\n' "$AMBER" "$OFF" "$*"; }
info() { printf '  %s· %s%s\n' "$DIM" "$*" "$OFF"; }
warn() { printf '  %s!%s %s\n' "$AMBER" "$OFF" "$*"; }
die()  { printf '\n  %s✕ %s%s\n' "$ROSE" "$*" "$OFF" >&2; exit 1; }
has()  { command -v "$1" >/dev/null 2>&1; }

banner() {
  say ""
  say "  ${AMBER}▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆${OFF}"
  say "     ${BOLD}A U T E U R${OFF}   ${DIM}installer${OFF}"
  say "     ${DIM}an AI film director in your terminal${OFF}"
  say "  ${AMBER}▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆ ▆${OFF}"
}

# Run with sudo only when needed and possible.
as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"
  elif has sudo; then sudo "$@"
  else return 1
  fi
}

# ---- steps --------------------------------------------------------------------- #

check_platform() {
  step "Checking your machine"
  OS="$(uname -s)"
  ARCH="$(uname -m)"
  case "$OS" in
    Darwin) ok "macOS · $ARCH" ;;
    Linux)  ok "Linux · $ARCH" ;;
    *) die "Auteur supports macOS and Linux (found $OS). On Windows, use WSL." ;;
  esac
  has curl || has wget || die "need curl or wget"
}

ensure_uv() {
  step "Python toolchain (uv)"
  if has uv; then
    ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
    return
  fi
  info "installing uv from astral.sh"
  if has curl; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH="$AUTEUR_NO_MODIFY_PATH" sh >/dev/null
  else
    wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH="$AUTEUR_NO_MODIFY_PATH" sh >/dev/null
  fi
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  export PATH
  has uv || die "uv installed but not on PATH — open a new shell and re-run"
  ok "uv $(uv --version | awk '{print $2}')"
}

ensure_system_deps() {
  step "ffmpeg (the edit) and Node.js (ChatGPT sign-in)"
  missing=""
  has ffmpeg || missing="$missing ffmpeg"
  has npx || missing="$missing node"
  if [ -z "$missing" ]; then
    ok "ffmpeg · node $(node --version 2>/dev/null)"
    return
  fi
  if [ "$AUTEUR_NO_DEPS" = "1" ]; then
    warn "missing:$missing — install them before filming"
    return
  fi
  if [ "$OS" = "Darwin" ] && has brew; then
    info "brew install$missing"
    # shellcheck disable=SC2086
    brew install $missing >/dev/null || warn "brew couldn't install$missing"
  elif [ "$OS" = "Linux" ] && has apt-get; then
    pkgs=""
    case "$missing" in *ffmpeg*) pkgs="$pkgs ffmpeg" ;; esac
    case "$missing" in *node*) pkgs="$pkgs nodejs npm" ;; esac
    info "apt-get install$pkgs"
    # shellcheck disable=SC2086
    { as_root apt-get update -qq && as_root apt-get install -y -qq $pkgs; } >/dev/null 2>&1 \
      || warn "couldn't install$pkgs (no sudo?)"
  elif [ "$OS" = "Linux" ] && has dnf; then
    pkgs="$(echo "$missing" | sed 's/node/nodejs/')"
    info "dnf install$pkgs"
    # shellcheck disable=SC2086
    as_root dnf install -y $pkgs >/dev/null 2>&1 || warn "dnf failed"
  fi
  has ffmpeg || warn "ffmpeg still missing — https://ffmpeg.org/download.html"
  has npx || warn "Node.js still missing — https://nodejs.org (needed for /login)"
  if has ffmpeg && has npx; then ok "ffmpeg · node $(node --version)"; fi
}

resolve_source() {
  if [ -n "$AUTEUR_SOURCE" ]; then
    SRC="$AUTEUR_SOURCE"
  else
    # Run as a file from a checkout (./install.sh)? Install that. When piped,
    # $0 is the shell, so this never matches.
    here=""
    case "$0" in
      *install.sh) here="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)" ;;
    esac
    if [ -n "$here" ] && [ -f "$here/pyproject.toml" ] && grep -q '^name = "auteur"' "$here/pyproject.toml"; then
      SRC="$here"
    else
      SRC="git+${AUTEUR_REPO%.git}.git@${AUTEUR_REF}"
    fi
  fi
  case "$SRC" in
    git+*|http://*|https://*|file://*) URL="$SRC" ;;
    *)
      [ -e "$SRC" ] || die "AUTEUR_SOURCE not found: $SRC"
      URL="file://$(cd "$SRC" && pwd)"
      ;;
  esac
  if [ "$AUTEUR_LITE" = "1" ]; then SPEC="auteur @ $URL"; else SPEC="auteur[render] @ $URL"; fi
}

install_auteur() {
  step "Installing Auteur"
  resolve_source
  info "from $URL"
  if [ "$AUTEUR_LITE" = "1" ]; then
    info "lite: no local LTX engine — render with /backend animatic or ltx-pipelines"
  else
    info "with the LTX-2.5 engine (torch, diffusers) — first install takes a few minutes"
  fi
  log="${TMPDIR:-/tmp}/auteur-install.$$.log"
  if ! uv tool install --force --python "$PYTHON_VERSION" "$SPEC" >"$log" 2>&1; then
    tail -n 25 "$log" >&2
    die "uv tool install failed (full log: $log)"
  fi
  rm -f "$log"
  BIN_DIR="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
  [ -x "$BIN_DIR/auteur" ] || die "install finished but $BIN_DIR/auteur is missing"
  ok "$("$BIN_DIR/auteur" --version)  →  $BIN_DIR/auteur"
}

ensure_path() {
  case ":$PATH:" in
    *":$BIN_DIR:"*) return ;;
  esac
  if [ "$AUTEUR_NO_MODIFY_PATH" = "1" ]; then
    warn "$BIN_DIR is not on your PATH — add it to use \`auteur\`"
    return
  fi
  uv tool update-shell >/dev/null 2>&1 || true
  PATH_NOTE=1
}

finish() {
  say ""
  say "  ${AMBER}✻${OFF} ${BOLD}Auteur is installed.${OFF}"
  say ""
  if [ "${PATH_NOTE:-0}" = "1" ]; then
    say "  ${DIM}open a new terminal (PATH was updated), then:${OFF}"
  fi
  say "    ${TEAL}auteur${OFF}            ${DIM}start a session${OFF}"
  say "    ${TEAL}/login${OFF}            ${DIM}sign in with ChatGPT — your director${OFF}"
  say "    ${TEAL}/login hf${OFF}  ${TEAL}/fetch${OFF}  ${DIM}get LTX-2.5 for local rendering${OFF}"
  say ""
  say "  ${DIM}then just describe a film.${OFF}"
  say ""
}

main() {
  banner
  check_platform
  ensure_uv
  ensure_system_deps
  install_auteur
  ensure_path
  finish
}

main "$@"
