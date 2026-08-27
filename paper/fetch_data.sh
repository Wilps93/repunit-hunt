#!/bin/bash
# Загрузка b-файлов OEIS для всех использованных последовательностей.
set -u
cd "$(dirname "$0")/data" || exit 1
for s in A000043 A028491 A004061 A004062 A004063 A004023 A005808 \
         A004064 A016054 A006032 A006033 A006034 A133857 A006035 A127995; do
  num=${s#A}
  curl -s -m 120 "https://oeis.org/$s/b$num.txt" -o "$s.txt"
  n=$(grep -c '^[0-9]' "$s.txt")
  echo "$s  terms=$n  bytes=$(wc -c < "$s.txt")"
done
