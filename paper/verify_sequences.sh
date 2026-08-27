#!/bin/bash
# Независимая перепроверка префиксов последовательностей OEIS собственным
# поисковиком repunit-hunt: для каждого основания b пересчитываются ВСЕ
# индексы n < KMAX, и список PRP сверяется с записью OEIS.
#
# Смысл: OEIS-запись подтверждает только то, что перечисленные индексы дают
# простое; она не гарантирует, что между ними ничего не пропущено. Полный
# пересчёт префикса закрывает именно этот пробел.
#
# Запуск:  bash paper/verify_sequences.sh [KMAX] [БАЗЫ...]
set -u

# Блокировка: два одновременных прогона пишут в одни и те же каталоги
# ($HOME/rh-verify и paper/verify), и сверка может прочитать файлы в
# промежуточном состоянии — получится мнимый «пропущенный член». Ровно это
# и случилось однажды: убитый по таймауту прогон оставил живого потомка,
# который несколько часов продолжал переписывать результаты.
exec 9>"${TMPDIR:-/tmp}/rh-verify.lock"
if ! flock -n 9; then
  echo "ОШИБКА: пересчёт уже выполняется другим процессом." >&2
  echo "Дождитесь его окончания или снимите: pkill -f verify_sequences.sh" >&2
  exit 2
fi

KMAX=${1:-20000}
shift || true
BASES=${*:-"2 3 5 6 7 10 11 12 13 14 15 17 18 19 20 21 22 23 24 26"}

export RUSTUP_HOME=/opt/rust/rustup
[ -f /etc/profile.d/rust.sh ] && . /etc/profile.d/rust.sh
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/rh-target}
BIN="$CARGO_TARGET_DIR/release/repunit-hunt"

REPO=/mnt/c/Users/Dokuchaev_ts/Downloads/repunit-hunt
WORK=$HOME/rh-verify
mkdir -p "$WORK"
OUTDIR="$REPO/paper/verify"
mkdir -p "$OUTDIR"

for b in $BASES; do
  d="$WORK/b$b"
  # Каталог обязательно чистим: repunit-hunt возобновляет работу по
  # worklog.jsonl и пропускает уже закрытые k, из-за чего results.json
  # прошлого прогона содержал бы лишь ДОБАВКУ к предыдущему.
  rm -rf "$d"
  mkdir -p "$d"
  echo "=== b=$b  n < $KMAX  ($(date +%H:%M:%S))"
  t0=$(date +%s)
  ( cd "$d" && "$BIN" --base "$b" --kmin 2 --kmax "$KMAX" \
        --double-check "${DC:-0.05}" >run.log 2>&1 )
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    echo "    ОШИБКА (код $rc), последние строки лога:"
    tail -5 "$d/run.log"
    continue
  fi
  cp "$d/results.json" "$OUTDIR/b$b.json" 2>/dev/null
  n=$(grep -o '"prp_exponents"' -A0 "$d/results.json" >/dev/null 2>&1 && \
      python3 -c "import json,sys;print(len(json.load(open('$d/results.json'))['prp_exponents']))")
  echo "    готово за $((t1-t0)) с, PRP найдено: $n"
done
echo "результаты в $OUTDIR"
