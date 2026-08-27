#!/bin/bash
#==============================================================================
# Проверки нативного слоя, которые не покрываются `cargo test`.
#
#   1. Montgomery-арифметика трёх ширин против эталона, посчитанного на Python
#      (native/tests/mont_vectors.h), плюс запуск самих ядер rh_tf_k64/k96/k128
#      на НАСТОЯЩИХ делителях репьюнитов.
#   2. Компиляция GWNUM-бэкенда с заглушками API — ловит синтаксис и типы,
#      когда Prime95 SDK не установлен.
#
# Запуск из корня репозитория:  bash native/tests/run_tests.sh
# Векторы перегенерируются так: python native/tests/gen_vectors.py native/tests/mont_vectors.h
#==============================================================================
set -u
cd "$(dirname "$0")/../.." || exit 1

ARCH="${RH_TEST_ARCH:-sm_75}"
fail=0

echo "=== 1. Montgomery-арифметика и ядра TF (arch=$ARCH) ==="
if command -v nvcc >/dev/null 2>&1; then
    nvcc -std=c++17 -arch="$ARCH" -O2 \
         -I native/include -I native/cuda -I native/tests \
         native/tests/test_mont.cu native/cuda/tf_kernel.cu -o /tmp/rh_test_mont \
         -diag-suppress 550 2>&1 | grep -E "error" && fail=1
    if [ -x /tmp/rh_test_mont ]; then
        /tmp/rh_test_mont || fail=1
    fi
else
    echo "  nvcc не найден — проверка пропущена"
fi

echo
echo "=== 2. Компиляция GWNUM-бэкенда (заглушки API) ==="
# ВНИМАНИЕ: заглушки в native/tests/gwnum_stub восстановлены по вызовам в
# prp_gwnum.c, а не по официальному SDK. Проверка подтверждает синтаксис и
# типы внутри нашего файла, но НЕ соответствие реальному API GWNUM.
gcc -fsyntax-only -std=gnu11 -Wall -Wextra -Wno-unused-parameter \
    -DRH_HAVE_GWNUM \
    -I native/include -I native/tests/gwnum_stub \
    native/prp/prp_gwnum.c && echo "  синтаксис и типы: ОК" || { echo "  ПРОВАЛ"; fail=1; }

echo
[ $fail -eq 0 ] && echo "ИТОГ: все проверки пройдены" || echo "ИТОГ: ЕСТЬ ОШИБКИ"
exit $fail
