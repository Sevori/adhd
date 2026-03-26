#!/bin/sh
set -eu

REPO="${ICE_INSTALL_REPO:-Sevori/adhd}"
VERSION="${ICE_VERSION:-latest}"
PREFIX="${ICE_PREFIX:-${HOME}/.local}"
BIN_DIR="${ICE_BIN_DIR:-${PREFIX}/bin}"
FROM_SOURCE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Install the `ice` binary from a GitHub release asset or, if unavailable, from source with cargo.

Usage:
  install.sh [--version <tag|latest>] [--prefix <dir>] [--bin-dir <dir>] [--repo <owner/name>] [--from-source] [--dry-run]

Options:
  --version <tag|latest>  Release tag to install, defaults to `latest`
  --prefix <dir>          Install prefix for cargo fallback, defaults to ~/.local
  --bin-dir <dir>         Final binary directory, defaults to <prefix>/bin
  --repo <owner/name>     GitHub repository, defaults to Sevori/adhd
  --from-source           Skip release assets and force cargo install
  --dry-run               Print the planned actions without changing the machine
  --help                  Show this help text

Environment:
  ICE_VERSION             Same as --version
  ICE_PREFIX              Same as --prefix
  ICE_BIN_DIR             Same as --bin-dir
  ICE_INSTALL_REPO        Same as --repo

Requirements:
  - GitHub CLI (`gh`) must be installed
  - `gh auth login` must have access to the target repository
EOF
}

log() {
  printf '%s\n' "$*" >&2
}

die() {
  log "error: $*"
  exit 1
}

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] $*"
    return 0
  fi
  "$@"
}

require_gh() {
  if ! command -v gh >/dev/null 2>&1; then
    die "install requires GitHub CLI (`gh`); install it first and then run `gh auth login`"
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  if ! gh auth status >/dev/null 2>&1; then
    die "`gh` is not authenticated for ${REPO}; run `gh auth login` first"
  fi
}

resolve_release_tag() {
  if [ "$VERSION" = "latest" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
      printf '%s\n' "<latest-release-tag>"
      return 0
    fi
    gh api "repos/${REPO}/releases/latest" --jq .tag_name 2>/dev/null || die "could not resolve the latest release tag for ${REPO}"
    return 0
  fi

  case "$VERSION" in
    ice-v*)
      printf '%s\n' "$VERSION"
      ;;
    v*)
      printf 'ice-%s\n' "$VERSION"
      ;;
    *)
      printf '%s\n' "$VERSION"
      ;;
  esac
}

download_release_pattern() {
  release_tag="$1"
  pattern="$2"
  dest_dir="$3"

  if [ "$release_tag" = "latest" ]; then
    gh release download -R "$REPO" --pattern "$pattern" --dir "$dest_dir" --clobber >/dev/null 2>&1
    return $?
  fi

  gh release download "$release_tag" -R "$REPO" --pattern "$pattern" --dir "$dest_dir" --clobber >/dev/null 2>&1
}

detect_target() {
  os_name="$(uname -s)"
  arch_name="$(uname -m)"

  case "$os_name" in
    Darwin)
      os="apple-darwin"
      ;;
    Linux)
      os="unknown-linux-gnu"
      ;;
    *)
      die "unsupported operating system: $os_name"
      ;;
  esac

  case "$arch_name" in
    x86_64|amd64)
      arch="x86_64"
      ;;
    arm64|aarch64)
      arch="aarch64"
      ;;
    *)
      die "unsupported architecture: $arch_name"
      ;;
  esac

  printf '%s-%s\n' "$arch" "$os"
}

verify_checksum() {
  checksum_path="$1"
  asset_path="$2"
  checksum_dir="$(dirname "$checksum_path")"
  normalized_checksum_path="${checksum_path}.normalized"
  checksum_value="$(awk 'NR == 1 { print $1 }' "$checksum_path")"

  [ -n "$checksum_value" ] || die "checksum file ${checksum_path} did not contain a SHA-256 value"
  printf '%s  %s\n' "$checksum_value" "$(basename "$asset_path")" > "$normalized_checksum_path"

  if command -v shasum >/dev/null 2>&1; then
    if [ "$DRY_RUN" -eq 1 ]; then
      log "[dry-run] shasum -a 256 -c $(basename "$normalized_checksum_path")"
      return 0
    fi
    (
      cd "$checksum_dir"
      shasum -a 256 -c "$(basename "$normalized_checksum_path")"
    )
    return $?
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    if [ "$DRY_RUN" -eq 1 ]; then
      log "[dry-run] sha256sum -c $(basename "$normalized_checksum_path")"
      return 0
    fi
    (
      cd "$checksum_dir"
      sha256sum -c "$(basename "$normalized_checksum_path")"
    )
    return $?
  fi

  log "warning: no shasum/sha256sum found; skipping checksum verification for $asset_path"
}

install_from_release() {
  require_gh
  target="$(detect_target)"
  asset="ice-${target}.tar.gz"
  checksum="${asset}.sha256"
  release_tag="$(resolve_release_tag)"

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$release_tag" = "<latest-release-tag>" ]; then
      log "[dry-run] gh release download -R ${REPO} --pattern ${asset} --dir <tmp> --clobber"
      log "[dry-run] gh release download -R ${REPO} --pattern ${checksum} --dir <tmp> --clobber"
    else
      log "[dry-run] gh release download ${release_tag} -R ${REPO} --pattern ${asset} --dir <tmp> --clobber"
      log "[dry-run] gh release download ${release_tag} -R ${REPO} --pattern ${checksum} --dir <tmp> --clobber"
    fi
    log "[dry-run] install ice to ${BIN_DIR}/ice"
    return 0
  fi

  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT INT TERM HUP
  asset_path="${tmp_dir}/${asset}"
  checksum_path="${tmp_dir}/${checksum}"

  if ! download_release_pattern "$release_tag" "$asset" "$tmp_dir"; then
    log "warning: release asset not available for ${target} in ${REPO} (${release_tag})"
    return 1
  fi

  if download_release_pattern "$release_tag" "$checksum" "$tmp_dir"; then
    verify_checksum "$checksum_path" "$asset_path"
  else
    log "warning: checksum asset not available for ${asset}; continuing without checksum verification"
  fi

  tar -xzf "$asset_path" -C "$tmp_dir"
  binary_path="$(find "$tmp_dir" -type f -name ice -print -quit)"
  [ -n "$binary_path" ] || die "release asset did not contain an ice binary"

  run mkdir -p "$BIN_DIR"
  run install -m 0755 "$binary_path" "${BIN_DIR}/ice"
  log "installed ice to ${BIN_DIR}/ice from ${REPO} release ${release_tag}"
  return 0
}

install_from_source() {
  require_gh
  command -v cargo >/dev/null 2>&1 || die "cargo is required for source installation fallback; install a Rust toolchain or retry with a release that has prebuilt assets attached"

  ref="$(resolve_release_tag)"
  tmp_dir="$(mktemp -d)"
  clone_dir="${tmp_dir}/repo"
  trap 'rm -rf "$tmp_dir"' EXIT INT TERM HUP

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] gh repo clone ${REPO} ${clone_dir} -- --depth=1 --branch ${ref}"
    log "[dry-run] cargo install --locked --root ${PREFIX} --path ${clone_dir}"
    return 0
  fi

  run gh repo clone "$REPO" "$clone_dir" -- --depth=1 --branch "$ref"
  run cargo install --locked --root "$PREFIX" --path "$clone_dir"
  log "installed ice with cargo into ${PREFIX}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)
      [ "$#" -ge 2 ] || die "--version requires a value"
      VERSION="$2"
      shift 2
      ;;
    --prefix)
      [ "$#" -ge 2 ] || die "--prefix requires a value"
      PREFIX="$2"
      BIN_DIR="${PREFIX}/bin"
      shift 2
      ;;
    --bin-dir)
      [ "$#" -ge 2 ] || die "--bin-dir requires a value"
      BIN_DIR="$2"
      shift 2
      ;;
    --repo)
      [ "$#" -ge 2 ] || die "--repo requires a value"
      REPO="$2"
      shift 2
      ;;
    --from-source)
      FROM_SOURCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [ "$FROM_SOURCE" -eq 1 ]; then
  install_from_source
  exit 0
fi

if install_from_release; then
  exit 0
fi

log "falling back to cargo installation because no matching release asset is available for ${VERSION} on $(detect_target)"
log "hint: this usually means the selected release has no prebuilt binary for your platform yet"
install_from_source
