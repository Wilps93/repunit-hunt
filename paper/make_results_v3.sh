#!/bin/bash
# Сборка results_v3.txt из вывода скриптов версии 3 (теги [S13]-[S25]).
#
#   bash make_results_v3.sh          калибровка c_b [S19] берётся из кэша
#                                    results_S19.txt, если он есть
#   bash make_results_v3.sh --full   калибровка пересчитывается (около 10 минут)
#
# Все скрипты детерминированы (фиксированные зёрна), поэтому повторный запуск
# даёт побитово тот же файл.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python

if [ "${1:-}" = "--full" ] || [ ! -s results_S19.txt ]; then
  echo "cb_empirical.py: калибровка c_b (около 10 минут)..." >&2
  "$PY" cb_empirical.py > results_S19.txt
fi

{
  echo "# results_v3.txt — вывод lpw_second_order.py, cb_empirical.py, cb_theory.py,"
  echo "# sim_second_order.py, frontiers.py и stopping_rules.py (версия 3 статьи)."
  echo "# Собирается командой: bash make_results_v3.sh"
  echo
  "$PY" lpw_second_order.py
  echo
  cat results_S19.txt
  echo
  "$PY" cb_theory.py
  echo
  "$PY" sim_second_order.py 10000
  echo
  "$PY" frontiers.py
  echo
  "$PY" stopping_rules.py 20000
} > results_v3.txt
echo "results_v3.txt собран ($(wc -l < results_v3.txt) строк)" >&2
