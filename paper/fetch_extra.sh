#!/bin/bash
# b-файлы для оснований 21..26 (проверка устойчивости к границе отбора).
cd "$(dirname "$0")/data" || exit 1
for s in A127996 A127997 A204940 A127998 A127999; do
  num=${s#A}
  curl -s -m 120 "https://oeis.org/$s/b$num.txt" -o "$s.txt"
  echo "$s terms=$(grep -c '^[0-9]' "$s.txt")"
done
