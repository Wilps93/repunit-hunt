#!/bin/bash
# Сборка repunit-hunt и независимая перепроверка префиксов последовательностей OEIS.
# Запускать из WSL:  bash paper/build_and_verify.sh
set -e

export RUSTUP_HOME=/opt/rust/rustup
export CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
export PATH=/opt/rust/cargo/bin:$RUSTUP_HOME/toolchains/*/bin:$PATH
[ -f /etc/profile.d/rust.sh ] && . /etc/profile.d/rust.sh
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/rh-target}

REPO=/mnt/c/Users/Dokuchaev_ts/Downloads/repunit-hunt
cd "$REPO"

echo "== rustc: $(rustc --version 2>&1)"
echo "== target dir: $CARGO_TARGET_DIR"

cargo build --release 2>&1 | tail -20
BIN="$CARGO_TARGET_DIR/release/repunit-hunt"
"$BIN" --devices || true
echo "BINARY=$BIN"
