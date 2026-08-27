#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py -- самодостаточное воспроизведение всех числовых результатов работы
«Схема наблюдения в статистике обобщённых репьюнитных простых:
калибровка объединённой проверки константы Ленстры -- Померанса -- Вагстаффа».

Каждое печатаемое число помечено тегом [Табл. N] / [§N.N], совпадающим с
номером таблицы или раздела статьи.

Запуск:  python3 analysis.py            полный прогон
         python3 analysis.py --fast     сокращённое число реализаций Монте-Карло

Данные: b-файлы OEIS в каталоге data/ (скачиваются fetch_data.sh).
Зависимости: numpy, scipy, matplotlib.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats, optimize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"

EULER_GAMMA = 0.5772156649015328606
KAPPA_LPW = math.exp(EULER_GAMMA)          # e^gamma = 1.781072...
G_LPW = math.exp(-EULER_GAMMA)             # e^-gamma = 0.5614594...

SEED = 20260827

# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------
# Основания 2 <= b <= 20, не являющиеся точными степенями (исключены 4, 8, 9, 16).
SEQ = {
    2: "A000043", 3: "A028491", 5: "A004061", 6: "A004062", 7: "A004063",
    10: "A004023", 11: "A005808", 12: "A004064", 13: "A016054", 14: "A006032",
    15: "A006033", 17: "A006034", 18: "A133857", 19: "A006035", 20: "A127995",
}

# Раздел DATA записи A000043 обрывается на 50-м члене (77 232 917): порядок
# следующих показателей OEIS считает недоказанным. 51-й и 52-й известные
# показатели Мерсенна документированы GIMPS (M51 = 82 589 933, дек. 2018;
# M52 = 136 279 841, окт. 2024) и в комментариях к A000043; включаем их явно.
A000043_EXTRA = [82589933, 136279841]

# Основания за границей отбора b <= 20 -- для проверки устойчивости самой
# границы (см. функцию check_selection_rule).
SEQ_EXTRA = {21: "A127996", 22: "A127997", 23: "A204940",
             24: "A127998", 26: "A127999"}

# Фронты GIMPS (PrimeNet Milestones, снято 27.08.2026 04:30 UTC).
# ВНИМАНИЕ: X_ver движется. На 27.08.2026 отчёт даёт 81 648 221; более раннее
# значение 81 491 519 относится к весне 2026 г. Обе величины лежат между
# M50 = 77 232 917 и M51 = 82 589 933, поэтому число дважды проверенных
# показателей Мерсенна (50) от выбора не зависит.
GIMPS_XVER = 81648221      # все показатели ниже проверены дважды
GIMPS_XVER_OLD = 81491519  # значение, использованное в первой редакции
GIMPS_XFT = 141308443      # все показатели ниже проверены хотя бы однократно
GIMPS_ASOF = "2026-08-27"


def load_sequences(table=None):
    seqs = {}
    for b, sid in (table or SEQ).items():
        path = DATA / (sid + ".txt")
        if not path.exists():
            sys.exit("нет файла %s; запустите fetch_data.sh" % path)
        vals = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.append(int(line.split()[-1]))
        if b == 2:
            vals = sorted(set(vals) | set(A000043_EXTRA))
        seqs[b] = np.array(sorted(vals), dtype=np.int64)
    return seqs


class BaseData:
    """Достаточные статистики по одному основанию."""

    def __init__(self, b, idx):
        self.b = b
        self.idx = np.asarray(idx, dtype=np.float64)
        self.lnb = math.log(b)
        self.t = np.log(self.idx)
        self.n_primes = len(self.idx)
        self.N = self.n_primes - 1          # событий строго правее t0
        self.t0 = float(self.t[0])
        self.D = float(self.t[-1] - self.t[0])
        self.nmin = int(self.idx[0])
        self.nmax = int(self.idx[-1])

    @property
    def G_end(self):
        return self.D / (self.N * self.lnb)

    @property
    def G_ols(self):
        k = np.arange(self.N + 1, dtype=float)
        slope = np.polyfit(k, self.t, 1)[0]
        return slope / self.lnb

    @property
    def G_ols0(self):
        """Регрессия через начало координат: t_k - t_0 на k."""
        k = np.arange(self.N + 1, dtype=float)
        y = self.t - self.t0
        slope = float(np.dot(k, y) / np.dot(k, k))
        return slope / self.lnb

    @property
    def kappa_hat(self):
        return self.N * self.lnb / self.D

    @property
    def kappa_tilde(self):
        return (self.N - 1) * self.lnb / self.D

    @property
    def expo(self):
        """Вклад основания в объединённую экспозицию S."""
        return self.D / self.lnb

    def ci(self, level=0.95):
        a = (1 - level) / 2
        lo = stats.gamma.ppf(a, self.N) / self.expo
        hi = stats.gamma.ppf(1 - a, self.N) / self.expo
        return lo, hi

    def pit(self, kappa=KAPPA_LPW):
        return float(stats.gamma.cdf(kappa * self.expo, self.N))


def build(seqs, truncate=None, drop=(), cap=None, min_events=1):
    """min_events -- минимальное N_b, при котором основание остаётся в выборке.

    min_events=1 (по умолчанию): достаточно двух известных индексов, чтобы
    основание дало одно событие. min_events=2 отбрасывает основания, у которых
    после усечения остался ровно один интервал; выбор влияет на таблицу 6.
    """
    out = []
    for b in sorted(seqs):
        if b in drop:
            continue
        idx = seqs[b]
        if truncate is not None:
            idx = idx[idx > truncate]
        if cap is not None:
            idx = idx[idx <= cap]
        if len(idx) - 1 < min_events:
            continue
        out.append(BaseData(b, idx))
    return out


def pooled(bases):
    M = sum(bd.N for bd in bases)
    S = sum(bd.expo for bd in bases)
    return M, S, len(bases)


# --------------------------------------------------------------------------
# Печать
# --------------------------------------------------------------------------
OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    OUT.append(line)
    print(line)


def rule(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


# --------------------------------------------------------------------------
# Таблицы данных и оценок
# --------------------------------------------------------------------------
def table1(bases):
    rule("[Табл. 1] Использованные последовательности")
    say("%3s %8s %8s %5s %7s %13s %10s %10s" %
        ("b", "OEIS", "простых", "N_b", "n_min", "n_max", "D_b", "D_b/ln b"))
    for bd in bases:
        say("%3d %8s %8d %5d %7d %13d %10.4f %10.4f" %
            (bd.b, SEQ[bd.b], bd.n_primes, bd.N, bd.nmin, bd.nmax, bd.D, bd.expo))
    M, S, B = pooled(bases)
    say("%3s %8s %8d %5d %7s %13s %10s %10.4f" %
        ("всего", "", sum(bd.n_primes for bd in bases), M, "", "", "", S))
    say("  M = %d, S = %.4f, B = %d" % (M, S, B))
    return M, S, B


def table5(bases):
    rule("[Табл. 5] Оценки по основаниям")
    say("%3s %4s %8s %8s %8s %8s %8s %20s %6s" %
        ("b", "N_b", "G_end", "G_OLS", "разн.%", "kappa^", "kappa~", "95% ДИ", "PIT"))
    worst = (0.0, None)
    for bd in bases:
        d = 100 * (bd.G_ols - bd.G_end) / bd.G_end
        lo, hi = bd.ci()
        say("%3d %4d %8.4f %8.4f %+8.1f %8.3f %8.3f %20s %6.3f" %
            (bd.b, bd.N, bd.G_end, bd.G_ols, d, bd.kappa_hat, bd.kappa_tilde,
             "[%.3f; %.3f]" % (lo, hi), bd.pit()))
        if abs(d) > abs(worst[0]):
            worst = (d, bd.b)
    say("  максимальное расхождение форм оценки: %+.1f%% (b = %d)" % worst)
    top3 = sorted(bases, key=lambda x: -abs(100 * (x.G_ols - x.G_end) / x.G_end))[:3]
    say("  три наибольших расхождения: " + ", ".join(
        "b=%d %+.1f%%" % (x.b, 100 * (x.G_ols - x.G_end) / x.G_end) for x in top3))

    # [§3.4] Переворот ярлыка «удачливости». Ранг 1 = наибольшее G,
    # то есть самое «неудачливое» основание.
    by_end = sorted(bases, key=lambda x: -x.G_end)
    by_ols = sorted(bases, key=lambda x: -x.G_ols)
    med_ols = float(np.median([x.G_ols for x in bases]))
    say("  [§3.4] упорядочение по G_end (убыв.): " +
        ", ".join("%d:%.4f" % (x.b, x.G_end) for x in by_end[:5]) + " ...")
    say("  [§3.4] упорядочение по G_OLS (убыв.): " +
        ", ".join("%d:%.4f" % (x.b, x.G_ols) for x in by_ols[:5]) + " ...")
    say("  наибольшее G_end (самое «неудачливое» по концевой форме): b=%d, G=%.4f"
        % (by_end[0].b, by_end[0].G_end))
    for tgt in (18, 13, 11):
        bd = [x for x in bases if x.b == tgt]
        if not bd:
            continue
        bd = bd[0]
        say("  b=%d: G_end=%.4f (ранг %d из %d), G_OLS=%.4f (ранг %d из %d), "
            "медиана G_OLS=%.4f -> %s медианы"
            % (tgt, bd.G_end, 1 + by_end.index(bd), len(bases),
               bd.G_ols, 1 + by_ols.index(bd), len(bases), med_ols,
               "ниже" if bd.G_ols < med_ols else "выше"))
    # [§3.4] Сопоставление двух величин, которые в литературе смешивают:
    # расхождения между формами оценки и отклонений оценки от предсказания ЛПВ.
    dev = sorted(100 * abs(bd.G_end - G_LPW) / G_LPW for bd in bases)
    spr = sorted(100 * abs(bd.G_ols - bd.G_end) / bd.G_end for bd in bases)
    med_dev = float(np.median(dev))
    med_spr = float(np.median(spr))
    say("  [§3.4] отклонения |G_end - e^-gamma|/e^-gamma: медиана %.1f%%, макс %.1f%%"
        % (med_dev, max(dev)))
    say("  [§3.4] расхождение форм |G_OLS - G_end|/G_end: медиана %.1f%%, макс %.1f%%"
        % (med_spr, max(spr)))
    say("  [§3.4] макс. расхождение форм / медиана отклонений = %.2f" %
        (max(spr) / med_dev))
    n_over = sum(1 for bd in bases
                 if abs(bd.G_ols - bd.G_end) / bd.G_end
                 > abs(bd.G_end - G_LPW) / G_LPW)
    say("  [§3.4] оснований, где выбор формы весит больше собственного "
        "отклонения от ЛПВ: %d из %d" % (n_over, len(bases)))

    below = sum(1 for bd in bases if bd.G_end < G_LPW)
    below_ols = sum(1 for bd in bases if bd.G_ols < G_LPW)
    say("  ниже e^-gamma: концевая форма %d/%d, МНК %d/%d"
        % (below, len(bases), below_ols, len(bases)))
    pits = np.array([bd.pit() for bd in bases])
    say("  PIT: min %.3f (b=%d), max %.3f (b=%d)"
        % (pits.min(), bases[int(pits.argmin())].b,
           pits.max(), bases[int(pits.argmax())].b))
    ks = stats.kstest(pits, "uniform")
    say("  KS-тест равномерности PIT: D = %.4f, p = %.3f" % (ks.statistic, ks.pvalue))
    say("  все PIT внутри [0,025; 0,975]: %s" %
        bool(np.all((pits > 0.025) & (pits < 0.975))))
    b13 = [x for x in bases if x.b == 13]
    if b13:
        say("  b=13 («удачливое» в фольклоре): PIT = %.3f" % b13[0].pit())


def section52(bases):
    rule("[§5.2] Объединённые оценки")
    M, S, B = pooled(bases)
    kA = (M - 1) / S
    kB = (M - B) / S
    loA = stats.gamma.ppf(0.025, M) / S
    hiA = stats.gamma.ppf(0.975, M) / S
    x = KAPPA_LPW * S
    pA = 2 * min(stats.gamma.cdf(x, M), 1 - stats.gamma.cdf(x, M))
    say("  схема A: kappa~ = (M-1)/S = %.4f, 95%% ДИ = [%.4f; %.4f], p = %.3f"
        % (kA, loA, hiA, pA))
    say("  схема B: kappa^ = (M-B)/S = %.4f" % kB)
    say("  превышение над e^gamma: схема A %+.1f%%, схема B %+.1f%%"
        % (100 * (kA / KAPPA_LPW - 1), 100 * (kB / KAPPA_LPW - 1)))
    say("  разница между оценками: %.1f%%" % (100 * (kA / kB - 1)))
    # [§4.2] Диапазон, порождаемый неопределённостью допущения (A1):
    # delta_b = m/lambda_b даёт kappa^ = (M - B*m)/S.
    say("  [§4.2] диапазон по допущению (A1), kappa^ = (M - B*m)/S:")
    for m in (0.0, 0.5, 1.0):
        say("         m = %.1f -> %.4f" % (m, (M - B * m) / S))
    say("         ВНИМАНИЕ: при m = 0 моментная оценка даёт M/S = %.4f, тогда как"
        % (M / S))
    say("         точная несмещённая оценка схемы A -- (M-1)/S = %.4f. Разница в"
        % kA)
    say("         одно событие из %d (0,5%%); в тексте следует держаться одной"
        % M)
    say("         формы, иначе верхний конец отрезка меняется с %.3f на %.3f."
        % (M / S, kA))
    return kA, kB


# --------------------------------------------------------------------------
# Табл. 2: Монте-Карло проверка предложений 1-2
# --------------------------------------------------------------------------
def table2(reps):
    rule("[Табл. 2] Проверка предложений 1-2 методом Монте-Карло "
         "(%d реализаций на строку)" % reps)
    rng = np.random.default_rng(SEED)
    lam = 1.0
    say("%4s %10s %11s %10s %10s %10s %10s" %
        ("N", "E[lam^]/lam", "N/(N-1)", "E[lam~]/lam", "E[G^]lam", "E[G~]lam", "E[beta^]lam"))
    for N in (7, 10, 16, 22, 51):
        gaps = rng.exponential(1.0 / lam, size=(reps, N))
        t = np.cumsum(gaps, axis=1)          # t_k - t_0
        D = t[:, -1]
        lam_hat = N / D
        lam_til = (N - 1) / D
        G_hat = D / N                        # с ln b = 1
        G_til = (N / (N - 1)) * G_hat
        k = np.arange(N + 1, dtype=float)
        tt = np.concatenate([np.zeros((reps, 1)), t], axis=1)
        kbar = k.mean()
        w = (k - kbar) / np.sum((k - kbar) ** 2)
        beta = tt @ w
        say("%4d %10.4f %11.4f %10.4f %10.4f %10.4f %10.4f" %
            (N, lam_hat.mean() / lam, N / (N - 1.0), lam_til.mean() / lam,
             G_hat.mean() * lam, G_til.mean() * lam, beta.mean() * lam))
    say("  целевое значение всех столбцов, кроме второго, равно 1,0000")


# --------------------------------------------------------------------------
# Схема B: симуляция
# --------------------------------------------------------------------------
def simulate_scheme_B(bases, reps, rng, frontier_rule, kappa=KAPPA_LPW):
    """Возвращает массивы M, S по реализациям схемы B.

    frontier_rule(bd, lam_b) -> ширина окна W_b = ln L_b - t0.
    """
    Ms = np.zeros(reps)
    Ss = np.zeros(reps)
    for bd in bases:
        lam = kappa / bd.lnb
        W = frontier_rule(bd, lam, rng, reps)
        W = np.atleast_1d(W)
        if W.size == 1:
            W = np.full(reps, float(W[0]))
        mu = lam * W
        n = rng.poisson(mu)
        # последнее событие = максимум n равномерных на (0, W]
        u = rng.random(reps)
        D = np.where(n > 0, W * u ** (1.0 / np.maximum(n, 1)), 0.0)
        Ms += np.maximum(n - 1, 0)          # N_b = число событий правее первого... см. ниже
        Ss += D / bd.lnb
    return Ms, Ss


def simulate_scheme_B2(bases, reps, rng, frontier_rule, kappa=KAPPA_LPW):
    """Схема B в той параметризации, в какой её использует статья.

    t0 фиксирован (наименьший известный индекс), фронт ln L_b = t0 + W_b,
    N_b -- число событий процесса на (t0, ln L_b], D_b -- положение последнего.
    Реализации с N_b = 0 отбрасываются (ресемплинг), как и в данных, где
    у каждого основания есть хотя бы одно событие.
    """
    Ms = np.zeros(reps, dtype=np.int64)
    Ss = np.zeros(reps)
    for bd in bases:
        lam = kappa / bd.lnb
        W = frontier_rule(bd, lam, rng, reps)
        W = np.atleast_1d(np.asarray(W, dtype=float))
        if W.size == 1:
            W = np.full(reps, float(W[0]))
        n = rng.poisson(lam * W)
        bad = n < 1
        tries = 0
        while np.any(bad) and tries < 200:
            n[bad] = rng.poisson(lam * W[bad])
            bad = n < 1
            tries += 1
        n = np.maximum(n, 1)
        u = rng.random(reps)
        D = W * u ** (1.0 / n)              # максимум n равномерных на (0, W]
        Ms += n
        Ss += D / bd.lnb
    return Ms, Ss


def rule_expected_observed(bd, lam, rng, reps):
    """E[#событий] равно наблюдённому N_b."""
    return bd.N / lam


def make_rule_shift(shift):
    """Фронт отнесён на `shift` средних интервалов дальше ожидаемого
    последнего события."""
    def f(bd, lam, rng, reps):
        return (bd.N + shift) / lam
    return f


def make_rule_scale(factor):
    def f(bd, lam, rng, reps):
        return factor * bd.D
    return f


def rule_random(bd, lam, rng, reps):
    """Случайный фронт, свой у каждого основания и каждой реализации."""
    return (bd.N / lam) * rng.uniform(0.7, 1.8, size=reps)


def table3(bases, reps):
    rule("[Табл. 3] Несмещённость двух объединённых оценок при различном "
         "расположении фронта (%d реализаций на строку)" % reps)
    M0, S0, B = pooled(bases)
    rows = [
        ("E[#событий] = наблюдённое", rule_expected_observed),
        ("фронт +0,5 среднего интервала", make_rule_shift(0.5)),
        ("фронт +2 средних интервала", make_rule_shift(2.0)),
        ("фронт +5 средних интервалов", make_rule_shift(5.0)),
        ("фронт в 1,5 раза дальше по t", make_rule_scale(1.5)),
        ("фронт вдвое дальше по t", make_rule_scale(2.0)),
        ("фронты случайные, разные по основаниям", rule_random),
    ]
    say("%40s %13s %8s %13s %8s" %
        ("Правило для фронта", "E[(M-B)/S]", "откл.", "E[(M-1)/S]", "откл."))
    for name, fr in rows:
        rng = np.random.default_rng(SEED + 1)
        Ms, Ss = simulate_scheme_B2(bases, reps, rng, fr)
        eB = np.mean((Ms - B) / Ss)
        eA = np.mean((Ms - 1) / Ss)
        say("%40s %13.4f %+7.2f%% %13.4f %+7.2f%%" %
            (name, eB, 100 * (eB / KAPPA_LPW - 1), eA, 100 * (eA / KAPPA_LPW - 1)))
    say("  истинное значение e^gamma = %.4f" % KAPPA_LPW)


def table4(bases, reps):
    rule("[Табл. 4] Калибровка пивота kappa*S ~ Gamma(M,1) при схеме B "
         "(%d реализаций на строку)" % reps)
    M0, S0, B = pooled(bases)
    say("%18s %26s %14s" % ("Схема", "ошибка I рода (ном. 0,05)", "E[kappa~]/kappa"))
    for shift in (0.0, 0.5, 1.0, 2.0):
        rng = np.random.default_rng(SEED + 2)
        Ms, Ss = simulate_scheme_B2(bases, reps, rng, make_rule_shift(shift))
        x = KAPPA_LPW * Ss
        lo = stats.gamma.ppf(0.025, Ms)
        hi = stats.gamma.ppf(0.975, Ms)
        rej = np.mean((x < lo) | (x > hi))
        c = np.mean((Ms - 1) / Ss) / KAPPA_LPW
        say("%18s %26.4f %14.4f" % ("B, сдвиг %.1f" % shift, rej, c))
    # контроль: схема A
    rng = np.random.default_rng(SEED + 3)
    Ss = np.zeros(reps)
    for bd in bases:
        lam = KAPPA_LPW / bd.lnb
        D = rng.gamma(bd.N, 1.0 / lam, size=reps)
        Ss += D / bd.lnb
    x = KAPPA_LPW * Ss
    rej = np.mean((x < stats.gamma.ppf(0.025, M0)) | (x > stats.gamma.ppf(0.975, M0)))
    c = np.mean((M0 - 1) / Ss) / KAPPA_LPW
    say("%18s %26.4f %14.4f" % ("A (контроль)", rej, c))
    say("  точность Монте-Карло по уровню: примерно +-%.4f"
        % (1.96 * math.sqrt(0.05 * 0.95 / reps)))


def scheme_B_interval(bases, reps, level=0.95):
    """Доверительный интервал и p-значение для kappa инверсией симуляции схемы B.

    Для сетки значений kappa симулируем схему B (фронт = наблюдённый, сдвиг 0)
    и смотрим, попадает ли наблюдённая статистика (M-B)/S в центральную область.
    """
    M0, S0, B = pooled(bases)
    obs = (M0 - B) / S0

    def quantiles(kappa):
        rng = np.random.default_rng(SEED + 7)
        Ms, Ss = simulate_scheme_B2(bases, reps, rng, make_rule_shift(0.0), kappa=kappa)
        est = (Ms - B) / Ss
        return est

    def frac_below(kappa):
        return float(np.mean(quantiles(kappa) < obs))

    lo = optimize.brentq(lambda k: frac_below(k) - 0.975, 1.2, obs)
    hi = optimize.brentq(lambda k: frac_below(k) - 0.025, obs, 2.8)
    est_null = quantiles(KAPPA_LPW)
    p = 2 * min(np.mean(est_null <= obs), np.mean(est_null >= obs))
    return obs, lo, hi, float(p)


# --------------------------------------------------------------------------
# §4.3 -- взвешенное среднее по основаниям
# --------------------------------------------------------------------------
def section43(bases, reps):
    rule("[§4.3] Данные не выбирают схему за нас")
    M0, S0, B = pooled(bases)
    say("  (M-1)/S = %.4f, (M-B)/S = %.4f, разница %.1f%%"
        % ((M0 - 1) / S0, (M0 - B) / S0, 100 * ((M0 - 1) / (M0 - B) - 1)))
    # несмещённость kappa~_b по основаниям при схеме B
    rng = np.random.default_rng(SEED + 11)
    Ms, Ss = None, None
    per_base = {}
    for bd in bases:
        lam = KAPPA_LPW / bd.lnb
        W = (bd.N) / lam
        # реализации с N_b = 0 пересэмплируются: у каждого основания в данных
        # есть хотя бы одно событие. Обрезка n до 1 давала бы нулевые слагаемые
        # и занижала среднее.
        n = rng.poisson(lam * W, size=reps)
        bad = n < 1
        while np.any(bad):
            n[bad] = rng.poisson(lam * W, size=int(bad.sum()))
            bad = n < 1
        u = rng.random(reps)
        D = W * u ** (1.0 / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = (n - 1) * bd.lnb / D
        kt = kt[np.isfinite(kt)]
        per_base[bd.b] = float(kt.mean())
    vals = np.array(list(per_base.values()))
    say("  E[kappa~_b] при схеме B по основаниям: от %.3f до %.3f (истинное %.3f)"
        % (vals.min(), vals.max(), KAPPA_LPW))
    say("      (оценка имеет бесконечную дисперсию при N_b = 2, поэтому выборочное")
    say("       среднее сходится медленно; разброс по основаниям -- шум Монте-Карло)")
    # взвешенное среднее с обратно-дисперсионными весами
    for scheme in ("A", "B"):
        rng = np.random.default_rng(SEED + 12)
        num = np.zeros(reps)
        den = np.zeros(reps)
        for bd in bases:
            lam = KAPPA_LPW / bd.lnb
            if scheme == "A":
                n = np.full(reps, bd.N)
                D = rng.gamma(bd.N, 1.0 / lam, size=reps)
            else:
                W = bd.N / lam
                n = np.maximum(rng.poisson(lam * W, size=reps), 1)
                u = rng.random(reps)
                D = W * u ** (1.0 / n)
            kt = (n - 1) * bd.lnb / D
            w = np.maximum(n - 2, 0)
            num += w * kt
            den += w
        star = num / np.maximum(den, 1e-12)
        star = star[np.isfinite(star)]
        say("  взвешенное среднее kappa* при схеме %s: смещение %+.2f%%"
            % (scheme, 100 * (star.mean() / KAPPA_LPW - 1)))


# --------------------------------------------------------------------------
# §5.3 -- однородность и мощность
# --------------------------------------------------------------------------
def section53(bases, reps):
    rule("[§5.3] Однородность оснований и мощность критерия")
    M0, S0, B = pooled(bases)
    khat = M0 / S0
    Lam = 2 * sum(bd.N * math.log(bd.kappa_hat / khat) for bd in bases)
    df = B - 1
    p = stats.chi2.sf(Lam, df)
    say("  Lambda = %.2f при %d степенях свободы, p = %.2f" % (Lam, df, p))
    crit = stats.chi2.ppf(0.95, df)
    say("  критическое значение chi2_0.95(%d) = %.2f" % (df, crit))

    say("  мощность против логнормального разброса kappa_b:")
    rng = np.random.default_rng(SEED + 21)
    header, powers = [], []
    for cv in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.50):
        sigma = math.sqrt(math.log(1 + cv * cv))
        mu = math.log(KAPPA_LPW) - sigma * sigma / 2
        rej = 0
        for _ in range(reps):
            kb = rng.lognormal(mu, sigma, size=B)
            Ns, Ds = [], []
            for bd, k in zip(bases, kb):
                lam = k / bd.lnb
                D = rng.gamma(bd.N, 1.0 / lam)
                Ns.append(bd.N)
                Ds.append(D / bd.lnb)
            Ns = np.array(Ns, float)
            Ds = np.array(Ds)
            kh = Ns.sum() / Ds.sum()
            khb = Ns / Ds
            L = 2 * np.sum(Ns * np.log(khb / kh))
            rej += L > crit
        header.append(cv)
        powers.append(rej / reps)
    say("  " + " ".join("cv=%.2f" % c for c in header))
    say("  " + " ".join("   %.3f" % p for p in powers))
    cvs = np.array(header)
    pw = np.array(powers)
    if pw.max() >= 0.8:
        cv80 = float(np.interp(0.8, pw, cvs))
        se = math.sqrt(0.8 * 0.2 / reps)
        say("  мощность 80%% достигается при cv ~ %.2f (точность сетки/МК: "
            "+-%.02f по мощности)" % (cv80, 1.96 * se))


# --------------------------------------------------------------------------
# §5.4 -- устойчивость
# --------------------------------------------------------------------------
def table6(seqs):
    rule("[Табл. 6] Устойчивость к нижнему усечению")
    say("  Правило выбытия основания указывается явно: min_events -- наименьшее")
    say("  N_b, при котором основание остаётся в выборке после усечения.")
    cuts = (("нет", None), ("n > 10", 10), ("n > 10^2", 100),
            ("n > 10^3", 1000), ("n > 10^4", 10000))
    for me in (1, 2):
        say("")
        say("  -- min_events = %d --" % me)
        say("  %10s %4s %5s %12s %12s %12s %s" %
            ("усечение", "B", "M", "(M-1)/S", "(M-B)/S", "p (схема A)", "выбыли"))
        kbs = []
        for label, tr in cuts:
            bs = build(seqs, truncate=tr, min_events=me)
            M, S, B = pooled(bs)
            x = KAPPA_LPW * S
            p = 2 * min(stats.gamma.cdf(x, M), 1 - stats.gamma.cdf(x, M))
            gone = sorted(set(seqs) - {bd.b for bd in bs})
            kbs.append((M - B) / S)
            say("  %10s %4d %5d %12.4f %12.4f %12.3f %s" %
                (label, B, M, (M - 1) / S, (M - B) / S, p,
                 ",".join(str(g) for g in gone) if gone else "-"))
        say("  схема B колеблется в пределах %.3f-%.3f" % (min(kbs), max(kbs)))
    say("")
    say("  тренд схемы A: оценка растёт монотонно; схема B тренда не показывает")


def section54(seqs, bases, reps):
    rule("[§5.4] Устойчивость: исключение оснований и статус данных")
    M0, S0, B0 = pooled(bases)
    res = []
    for bd in bases:
        bs = [x for x in bases if x.b != bd.b]
        M, S, B = pooled(bs)
        x = KAPPA_LPW * S
        p = 2 * min(stats.gamma.cdf(x, M), 1 - stats.gamma.cdf(x, M))
        res.append((bd.b, (M - 1) / S, (M - B) / S, p))
    ka = [r[1] for r in res]
    kb = [r[2] for r in res]
    ps = [r[3] for r in res]
    say("  исключение по одному основанию: kappa~ (схема A) в пределах [%.3f; %.3f]"
        % (min(ka), max(ka)))
    say("  то же для (M-B)/S (схема B): [%.3f; %.3f]" % (min(kb), max(kb)))
    say("  p (схема A) от %.3f до %.3f" % (min(ps), max(ps)))
    infl = min(res, key=lambda r: r[3])
    say("  наиболее влиятельное основание: b = %d (p = %.3f при удалении)"
        % (infl[0], infl[3]))

    # b = 2 ограничено 50 дважды проверенными показателями
    seqs2 = dict(seqs)
    seqs2[2] = seqs[2][seqs[2] < GIMPS_XVER]
    bs = build(seqs2)
    M, S, B = pooled(bs)
    x = KAPPA_LPW * S
    p = 2 * min(stats.gamma.cdf(x, M), 1 - stats.gamma.cdf(x, M))
    say("  b=2 ограничено %d дважды проверенными показателями: kappa~ = %.3f, "
        "p = %.3f, (M-B)/S = %.3f" % (len(seqs2[2]), (M - 1) / S, p, (M - B) / S))

    # исключение индексов n > 10^6 (статус PRP)
    bs = build(seqs, cap=10 ** 6)
    M, S, B = pooled(bs)
    say("  исключение всех индексов n > 10^6: M = %d, kappa~ = %.3f, (M-B)/S = %.3f"
        % (M, (M - 1) / S, (M - B) / S))

    # наихудший сценарий: последний известный PRP каждого основания составной
    worst = None
    for bd in bases:
        seqs3 = dict(seqs)
        seqs3[bd.b] = seqs[bd.b][:-1]
        bs = build(seqs3)
        M, S, B = pooled(bs)
        val = (M - 1) / S
        if worst is None or val > worst[1]:
            worst = (bd.b, val, (M - B) / S)
    say("  наихудший сценарий (последний член основания оказался составным): "
        "b = %d даёт kappa~ = %.3f, (M-B)/S = %.3f" % worst)


# --------------------------------------------------------------------------
# §6 -- мощность объединённого критерия
# --------------------------------------------------------------------------
def power_gamma(M, r, alpha=0.05):
    """Мощность двустороннего критерия kappa = e^gamma против kappa = r*e^gamma.

    Пивот: kappa_0 * S ~ Gamma(M,1) при H0. Отвергаем, если kappa_0*S вне
    центральной области. При истинном kappa = r*kappa_0 величина kappa_0*S
    распределена как Gamma(M,1)/r.
    """
    lo = stats.gamma.ppf(alpha / 2, M)
    hi = stats.gamma.ppf(1 - alpha / 2, M)
    # P(Gamma(M,1)/r < lo) + P(Gamma(M,1)/r > hi)
    return (stats.gamma.cdf(lo * r, M) + stats.gamma.sf(hi * r, M))


def table7(bases):
    rule("[Табл. 7] Мощность на уровне alpha = 0,05 для отвержения kappa = e^gamma")
    M0, S0, B0 = pooled(bases)
    Mm = [bd for bd in bases if bd.b == 2][0].N
    say("%8s %28s %24s" % ("kappa/e^g", "только Мерсенн (M=%d)" % Mm,
                           "объединение (M=%d)" % M0))
    for r in (1.05, 1.10, 1.15, 1.20, 1.30, 1.50):
        say("%8.2f %28.3f %24.3f" % (r, power_gamma(Mm, r), power_gamma(M0, r)))

    rule("[§6] Разрешающая способность")
    for M, label in ((Mm, "только Мерсенн"), (M0, "объединение")):
        f = optimize.brentq(lambda r: power_gamma(M, r) - 0.8, 1.0001, 4.0)
        say("  %s (M=%d): мощность 80%% против отклонения %+.1f%%"
            % (label, M, 100 * (f - 1)))
    for target in (0.10, 0.05):
        M = optimize.brentq(
            lambda m: power_gamma(m, 1 + target) - 0.8, 20, 200000)
        say("  для мощности 80%% против отклонения %.0f%% нужно M ~ %.0f"
            % (100 * target, M))
    # рост фронтов
    dM = KAPPA_LPW * math.log(2) * sum(1.0 / bd.lnb for bd in bases)
    say("  удвоение всех фронтов даёт %.1f дополнительных событий" % dM)
    M10 = optimize.brentq(lambda m: power_gamma(m, 1.10) - 0.8, 20, 200000)
    need = M10 - M0
    doublings = need / dM
    lg = doublings * math.log10(2)
    say("  недостающие %.0f событий: %.1f удвоений, то есть рост фронтов "
        "в 2^%.1f = %.1f * 10^%d раз"
        % (need, doublings, doublings, 10 ** (lg - int(lg)), int(lg)))
    say("  (округление до 10^21 завышает требуемый рост примерно втрое; "
        "корректно писать 3 * 10^20)")


# --------------------------------------------------------------------------
# §7 -- счётная функция Мерсенна
# --------------------------------------------------------------------------
def newey_west_se(X, resid, lags):
    n, k = X.shape
    XtXinv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtXinv @ S @ XtXinv
    return np.sqrt(np.diag(V))


def section7(seqs, reps):
    rule("[§7] Счётная функция Мерсенна: та же ошибка в ином обличье")
    exps = np.array(sorted(seqs[2]), dtype=float)
    exps = exps[exps < GIMPS_XVER]          # 50 дважды проверенных
    N = len(exps)
    t = np.log(exps)
    t0 = t[0]
    T = math.log(GIMPS_XVER)
    y = np.arange(N, dtype=float)           # нумерация 0..N-1
    X = np.column_stack([np.ones(N), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = N - 2
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * s2))
    say("  N = %d дважды проверенных показателей, T = ln(%d) = %.4f"
        % (N, GIMPS_XVER, T))
    say("  МНК: наклон = %.4f (SE %.4f), свободный член = %.4f (SE %.4f)"
        % (beta[1], se[1], beta[0], se[0]))
    rho1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    say("  автокорреляция остатков rho_1 = %.3f" % rho1)
    nw = newey_west_se(X, resid, lags=8)
    say("  Ньюи -- Уэст (8 лагов): SE свободного члена %.4f (было %.4f), "
        "отношение %.2f" % (nw[0], se[0], nw[0] / se[0]))

    # корректное нулевое распределение: N-1 равномерных точек на (t0, T]
    rng = np.random.default_rng(SEED + 31)
    n_ev = N - 1
    inter = np.zeros(reps)
    slopes = np.zeros(reps)
    maxr = np.zeros(reps)
    midexc = np.zeros(reps)
    for i in range(reps):
        u = np.sort(rng.uniform(t0, T, size=n_ev))
        tt = np.concatenate([[t0], u])
        yy = np.arange(N, dtype=float)
        XX = np.column_stack([np.ones(N), tt])
        bb, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        inter[i] = bb[0]
        slopes[i] = bb[1]
        lam_hat = (N - 1) / (tt[-1] - t0)
        r = yy - lam_hat * (tt - t0)
        maxr[i] = np.max(np.abs(r))
        mid = (tt - t0) / (tt[-1] - t0)
        sel = (mid >= 0.25) & (mid <= 0.75)
        midexc[i] = r[sel].mean() if sel.any() else 0.0
    obs_int = beta[0]
    p_int = 2 * min(np.mean(inter <= obs_int), np.mean(inter >= obs_int))
    say("  корректный нулевой закон свободного члена (%d реализаций): "
        "медиана %.3f, 5-95%% [%.2f; %.2f], p = %.3f"
        % (reps, np.median(inter), np.percentile(inter, 5),
           np.percentile(inter, 95), p_int))

    # (i) бутстрэп
    rng = np.random.default_rng(SEED + 32)
    bs = np.zeros(reps)
    for i in range(reps):
        j = rng.integers(0, N, size=N)
        XX = np.column_stack([np.ones(N), t[j]])
        bb, *_ = np.linalg.lstsq(XX, y[j], rcond=None)
        bs[i] = bb[0]
    say("  (i) бутстрэп по точкам событий: медиана %.3f, SD %.3f, "
        "95%% [%.3f; %.3f], доля отрицательных %.3f"
        % (np.median(bs), bs.std(ddof=1), np.percentile(bs, 2.5),
           np.percentile(bs, 97.5), np.mean(bs < 0)))
    say("      ширина корректного нулевого закона / ширина бутстрэпа = %.1f"
        % ((np.percentile(inter, 97.5) - np.percentile(inter, 2.5)) /
           (np.percentile(bs, 97.5) - np.percentile(bs, 2.5))))

    # (ii) мнимая недодисперсия
    lam_obs = (N - 1) / (t[-1] - t0)
    r_obs = y - lam_obs * (t - t0)
    obs_max = float(np.max(np.abs(r_obs)))
    p_max = 2 * min(np.mean(maxr <= obs_max), np.mean(maxr >= obs_max))
    say("  (ii) max|r| наблюдённое %.3f, медиана симуляций %.3f, p = %.3f"
        % (obs_max, np.median(maxr), p_max))

    # (iii) серединный дефицит
    mid = (t - t0) / (t[-1] - t0)
    sel = (mid >= 0.25) & (mid <= 0.75)
    obs_mid = float(r_obs[sel].mean())
    p_mid = 2 * min(np.mean(midexc <= obs_mid), np.mean(midexc >= obs_mid))
    say("  (iii) средняя экскурсия на средней половине: наблюдённая %.2f, "
        "медиана симуляций %.3f, p = %.3f, доля отрицательных %.3f"
        % (obs_mid, np.median(midexc), p_mid, np.mean(midexc < 0)))

    # §7.2 основание 2 на известном фронте
    lam_B = (N - 1) / (T - t0)
    k2 = lam_B * math.log(2)
    lo = stats.chi2.ppf(0.025, 2 * (N - 1)) / 2 / (T - t0) * math.log(2)
    hi = stats.chi2.ppf(0.975, 2 * (N - 1) + 2) / 2 / (T - t0) * math.log(2)
    say("  [§7.2] схема B для b=2: lambda^ = %.3f против e^gamma/ln2 = %.3f, "
        "kappa^_2 = %.3f" % (lam_B, KAPPA_LPW / math.log(2), k2))
    say("         точный 95%% интервал Гарвуда: [%.3f; %.3f]" % (lo, hi))
    mu0 = KAPPA_LPW / math.log(2) * (T - t0)
    k_obs = N - 1
    # Два общеупотребительных двусторонних пуассоновских p; указываем оба,
    # поскольку они заметно расходятся, а конвенция обычно не оговаривается.
    p_tail = min(1.0, 2 * min(stats.poisson.cdf(k_obs, mu0),
                              stats.poisson.sf(k_obs - 1, mu0)))
    p_mid = min(1.0, 2 * min(stats.poisson.cdf(k_obs - 1, mu0)
                             + 0.5 * stats.poisson.pmf(k_obs, mu0),
                             stats.poisson.sf(k_obs, mu0)
                             + 0.5 * stats.poisson.pmf(k_obs, mu0)))
    say("         пуассоновское p: удвоенный хвост %.2f, mid-p %.2f "
        "(ожидалось %.2f событий, наблюдено %d)" % (p_tail, p_mid, mu0, k_obs))
    # ВАЖНО: условная оценка берётся на ТОМ ЖЕ срезе данных (50 дважды
    # проверенных показателей), иначе сравниваются разные выборки.
    b2 = BaseData(2, exps)
    say("         условная оценка на том же срезе (%d показателей): kappa~_2 = %.3f"
        % (N, b2.kappa_tilde))
    say("         delta_2 = ln(X_ver/n_max) = %.3f при среднем интервале %.3f"
        % (T - t[-1], 1.0 / lam_B))
    # Две конвенции для «среднего интервала»: предсказанный ЛПВ ln b / e^gamma
    # и оценённый 1/lambda^. Приводим обе -- они дают 0.13 и 0.14.
    d2 = T - t[-1]
    lam_lpw = KAPPA_LPW / math.log(2)
    say("         средний интервал: предсказанный ЛПВ %.3f, оценённый %.3f"
        % (1 / lam_lpw, 1 / lam_B))
    say("         Pr(delta*lambda < %.3f) = %.2f  [по ЛПВ]"
        % (d2 * lam_lpw, 1 - math.exp(-d2 * lam_lpw)))
    say("         Pr(delta*lambda < %.3f) = %.2f  [по оценке]"
        % (d2 * lam_B, 1 - math.exp(-d2 * lam_B)))
    say("         для сравнения: kappa~_2 по всем 52 известным показателям = %.3f"
        % BaseData(2, np.array(sorted(seqs[2]), dtype=float)).kappa_tilde)

    # §7.3 знак вторичного члена
    mertens = 0.2614972128476428
    say("  [§7.3] естественная величина вторичного члена из постоянной Мертенса: "
        "%+.3f против наблюдённого %.3f" % (KAPPA_LPW / math.log(2) * mertens, beta[0]))
    return beta, se, nw


# --------------------------------------------------------------------------
# §8 -- структурные поправки
# --------------------------------------------------------------------------
def prime_factors(n):
    fs = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def C_model(bd, model, theta):
    b = bd.b
    if model == "A":
        return 1.0
    if model == "B":
        alpha, beta = theta
        c = 1.0
        for p in prime_factors(b - 1):
            if p != 2:
                c *= (p / (p - 1.0)) ** alpha
        for p in prime_factors(b + 1):
            if p != 2:
                c *= (p / (p - 2.0)) ** beta
        return c
    if model == "C":
        (delta,) = theta
        c = 1.0
        for p in prime_factors(b * b - 1):
            c *= (p / (p - 1.0)) ** delta
        return c
    if model == "D":
        th1, th2, th3 = theta
        return math.exp(th1 * len(prime_factors(b - 1)) +
                        th2 * len(prime_factors(b + 1)) +
                        th3 * math.log(b))
    if model == "Cfix":
        c = 1.0
        for p in prime_factors(b - 1):
            if p != 2:
                c *= p / (p - 1.0)
        for p in prime_factors(b + 1):
            if p != 2:
                c *= p / (p - 2.0)
        return c
    raise ValueError(model)


def loglik_struct(bases, model, theta):
    """Профильное (по kappa) условное лог-правдоподобие (6) с множителем C_b."""
    M = sum(bd.N for bd in bases)
    Sc = sum(C_model(bd, model, theta) * bd.expo for bd in bases)
    if Sc <= 0:
        return -1e18, float("nan")
    kap = M / Sc
    ll = sum(bd.N * math.log(C_model(bd, model, theta) / bd.lnb) for bd in bases)
    ll += M * math.log(kap) - M
    return ll, kap


def table8(bases):
    rule("[Табл. 8] Сравнение структурных моделей")
    M = sum(bd.N for bd in bases)
    specs = [("A", 0, []), ("B", 2, [0.0, 0.0]), ("C", 1, [0.0]),
             ("D", 3, [0.0, 0.0, 0.0])]
    fits = {}
    for name, npar, x0 in specs:
        if npar == 0:
            ll, kap = loglik_struct(bases, name, [])
        else:
            r = optimize.minimize(
                lambda th: -loglik_struct(bases, name, th)[0],
                x0, method="Nelder-Mead",
                options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 200000,
                         "maxfev": 200000})
            ll, kap = loglik_struct(bases, name, r.x)
            fits[name] = r.x
        k = npar + 1                        # плюс kappa
        fits.setdefault(name, np.array([]))
        aic = 2 * k - 2 * ll
        bic = k * math.log(M) - 2 * ll
        fits[name] = (fits[name], ll, k, aic, bic, kap)
    llA = fits["A"][1]
    say("%7s %12s %11s %8s %8s %10s %10s" %
        ("Модель", "параметров", "ln L", "LR", "p", "AIC", "BIC"))
    bics = {}
    for name, npar, _ in specs:
        _, ll, k, aic, bic, kap = fits[name]
        LR = 2 * (ll - llA)
        p = stats.chi2.sf(LR, npar) if npar > 0 else float("nan")
        bics[name] = bic
        say("%7s %12d %11.3f %8.2f %8.3f %10.2f %10.2f" %
            (name, k, ll, LR, p, aic, bic))
    for n_eff, label in ((M, "n = M = %d" % M), (len(bases), "n = B = %d" % len(bases))):
        b2 = {}
        for name, npar, _ in specs:
            _, ll, k, aic, _, kap = fits[name]
            b2[name] = k * math.log(n_eff) - 2 * ll
        mn = min(b2.values())
        w = {n: math.exp(-(v - mn) / 2) for n, v in b2.items()}
        tot = sum(w.values())
        say("  апостериорные вероятности по BIC при %s: %s"
            % (label, " / ".join("%.3f" % (w[n] / tot) for n, _, _ in specs)))
    say("  оценки параметров: " + "; ".join(
        "%s: %s" % (n, np.array2string(fits[n][0], precision=3))
        for n, npar, _ in specs if npar > 0))

    # фиксированная арифметическая поправка
    llf, kapf = loglik_struct(bases, "Cfix", [])
    cs = [C_model(bd, "Cfix", []) for bd in bases]
    say("  фиксированная поправка C^fix: среднее C_b = %.2f, kappa^ = %.3f, "
        "ln L = %.2f" % (np.mean(cs), kapf, llf))
    say("  прямое сравнение с моделью A: ln L %.2f против %.2f, разность %.2f"
        % (llf, llA, llf - llA))


def section8_log(bases):
    rule("[§8] Логарифмические поправки")

    def nll(par):
        kap, c1, c2 = par
        if kap <= 0:
            return 1e18
        tot = 0.0
        for bd in bases:
            tt = bd.t[1:]                   # события строго правее t0
            lam = (kap / bd.lnb) * (1 + c1 / tt + c2 / tt ** 2)
            if np.any(lam <= 0):
                return 1e18
            tot += np.sum(np.log(lam))
            # интеграл (kappa/ln b)(t + c1 ln t - c2/t) от t0 до t_N
            a, bb = bd.t0, float(bd.t[-1])
            F = lambda x: x + c1 * math.log(x) - c2 / x
            tot -= (kap / bd.lnb) * (F(bb) - F(a))
        return -tot

    r0 = nll([1.9648, 0.0, 0.0])
    r = optimize.minimize(nll, [1.9648, 0.0, 0.0], method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-11,
                                   "maxiter": 40000, "maxfev": 40000})
    # нулевая модель: c1=c2=0, kappa свободна
    rn = optimize.minimize_scalar(lambda k: nll([k, 0.0, 0.0]),
                                  bracket=(1.5, 2.0, 2.5))
    LR = 2 * (rn.fun - r.fun)
    say("  kappa^ = %.3f, c1^ = %+.3f, c2^ = %.3f" % (r.x[0], r.x[1], r.x[2]))
    say("  LR = %.2f при 2 степенях свободы, p = %.3f" % (LR, stats.chi2.sf(LR, 2)))


# --------------------------------------------------------------------------
# §2.3 -- независимость по основаниям
# --------------------------------------------------------------------------
def section23(seqs):
    rule("[§2.3, замечание 2] Совпадения индексов между основаниями")
    from collections import Counter
    c = Counter()
    for b, idx in seqs.items():
        for n in idx:
            c[int(n)] += 1
    shared = {n: k for n, k in c.items() if k > 1}
    say("  значений индекса, встречающихся более чем в одном основании: %d"
        % len(shared))
    mx = max(shared.items(), key=lambda kv: kv[1])
    say("  чаще всего: n = %d, встречается в %d основаниях" % mx)
    say("  наибольший совпадающий индекс: n = %d" % max(shared))
    say("  все совпадения при n <= %d" % max(shared))


# --------------------------------------------------------------------------
# §2.2 -- правило отбора оснований и его устойчивость
# --------------------------------------------------------------------------
def check_selection_rule(seqs):
    rule("[§2.2] Правило отбора оснований и устойчивость границы b <= 20")
    extra = load_sequences(SEQ_EXTRA)
    allseq = dict(seqs)
    allseq.update(extra)
    say("  число известных индексов по основаниям (точные степени 4, 8, 9, 16,")
    say("  25, 27 исключены алгебраически):")
    line = []
    for b in sorted(allseq):
        line.append("b=%d:%d" % (b, len(allseq[b])))
    say("    " + "  ".join(line))
    lo = min(len(v) for v in seqs.values())
    say("  минимум по основному набору (2 <= b <= 20): %d индексов" % lo)
    for b in sorted(extra):
        say("  за границей: b=%d -> %d индексов %s" %
            (b, len(extra[b]), "(>= 8!)" if len(extra[b]) >= 8 else "(< 8)"))
    ok = [b for b in sorted(allseq) if len(allseq[b]) >= 8]
    # наибольшая граница, до которой ВСЕ допустимые основания имеют >= 8
    bound = None
    for b in sorted(allseq):
        if len(allseq[b]) >= 8:
            bound = b
        else:
            break
    say("  правило «все допустимые основания до границы имеют >= 8 индексов»")
    say("  даёт границу b <= %d, а не b <= 20: основания %s также проходят"
        % (bound, ", ".join(str(b) for b in sorted(extra) if b <= (bound or 0))))
    say("  вывод: b <= 20 -- условная круглая граница, а не следствие "
        "сформулированного правила")

    # устойчивость вывода при расширении набора
    say("")
    say("  Устойчивость объединённой оценки к расширению набора оснований:")
    say("  %28s %4s %5s %10s %10s %8s" %
        ("набор", "B", "M", "(M-1)/S", "(M-B)/S", "p (A)"))
    variants = [
        ("b <= 20 (основной)", seqs),
        ("b <= 22", {b: v for b, v in allseq.items() if b <= 22}),
        ("b <= 24", {b: v for b, v in allseq.items() if b <= 24}),
        ("b <= 26 (все доступные)", allseq),
    ]
    for name, ss in variants:
        bs = build(ss)
        M, S, B = pooled(bs)
        x = KAPPA_LPW * S
        p = 2 * min(stats.gamma.cdf(x, M), 1 - stats.gamma.cdf(x, M))
        say("  %28s %4d %5d %10.4f %10.4f %8.3f" %
            (name, B, M, (M - 1) / S, (M - B) / S, p))
    # критерий однородности на расширенном наборе
    bs = build(allseq)
    M, S, B = pooled(bs)
    kh = M / S
    Lam = 2 * sum(bd.N * math.log(bd.kappa_hat / kh) for bd in bs)
    say("  критерий однородности на расширенном наборе: Lambda = %.2f, "
        "df = %d, p = %.2f" % (Lam, B - 1, stats.chi2.sf(Lam, B - 1)))


# --------------------------------------------------------------------------
# Рисунки
# --------------------------------------------------------------------------
def figures(seqs, bases, kB, ciB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    M0, S0, B0 = pooled(bases)

    # Рис. 1 -- оценки по основаниям
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(bases))
    for i, bd in enumerate(bases):
        lo, hi = bd.ci()
        ax.plot([i, i], [lo, hi], color="0.4", lw=1.0, zorder=1)
        ax.scatter([i], [bd.kappa_tilde], s=8 + 3.0 * bd.N, color="black", zorder=2)
    ax.axhline(KAPPA_LPW, ls="--", color="k", lw=1.0, label=r"$e^\gamma$")
    # Начертания различны, чтобы рисунок читался и при чёрно-белой печати.
    ax.axhline((M0 - 1) / S0, color="tab:blue", lw=1.2, ls="-.",
               label=r"схема A: $(M-1)/S$")
    ax.axhspan(ciB[0], ciB[1], color="tab:green", alpha=0.15, hatch="///",
               edgecolor="tab:green", lw=0)
    ax.axhline(kB, color="tab:green", lw=1.6, ls="-",
               label=r"схема B: $(M-B)/S$")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(bd.b) for bd in bases])
    ax.set_xlabel("основание $b$")
    ax.set_ylabel(r"$\tilde\kappa_b$")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_kappa.pdf")
    plt.close(fig)

    # Рис. 2 -- усечение
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    trs = [None, 10, 100, 1000, 10000]
    labels = ["нет", "$>10$", "$>10^2$", "$>10^3$", "$>10^4$"]
    ka, kb = [], []
    for tr in trs:
        bs = build(seqs, truncate=tr)
        M, S, B = pooled(bs)
        ka.append((M - 1) / S)
        kb.append((M - B) / S)
    ax.plot(range(len(trs)), ka, "o-.", color="tab:blue",
            label=r"$(M-1)/S$ (схема A)")
    ax.plot(range(len(trs)), kb, "s-", color="tab:green",
            label=r"$(M-B)/S$ (схема B)")
    ax.axhline(KAPPA_LPW, ls="--", color="k", lw=1.0, label=r"$e^\gamma$")
    ax.set_xticks(range(len(trs)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("нижнее усечение по $n$")
    ax.set_ylabel(r"$\hat\kappa$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_truncation.pdf")
    plt.close(fig)

    # Рис. 3 -- мощность
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    rs = np.linspace(1.0, 1.8, 200)
    Mm = [bd for bd in bases if bd.b == 2][0].N
    ax.plot(rs, [power_gamma(Mm, r) for r in rs], ls="--", color="tab:blue",
            label="только Мерсенн ($M=%d$)" % Mm)
    ax.plot(rs, [power_gamma(M0, r) for r in rs], ls="-", color="tab:green",
            label="объединение ($M=%d$)" % M0)
    ax.axhline(0.8, ls=":", color="k", lw=1.0)
    ax.set_xlabel(r"$\kappa/e^\gamma$")
    ax.set_ylabel("мощность")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_power.pdf")
    plt.close(fig)

    # Рис. 4 -- остаточный процесс для b = 2
    exps = np.array(sorted(seqs[2]), dtype=float)
    exps = exps[exps < GIMPS_XVER]
    N = len(exps)
    t = np.log(exps)
    t0, tN = t[0], t[-1]
    lam = (N - 1) / (tN - t0)
    r = np.arange(N) - lam * (t - t0)
    rng = np.random.default_rng(SEED + 41)
    grid = np.linspace(t0, tN, 200)
    sims = np.zeros((4000, len(grid)))
    for i in range(4000):
        u = np.sort(rng.uniform(t0, tN, size=N - 1))
        tt = np.concatenate([[t0], u])
        lam_i = (N - 1) / (tt[-1] - t0)
        rr = np.arange(N) - lam_i * (tt - t0)
        sims[i] = np.interp(grid, tt, rr)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.fill_between(grid, np.percentile(sims, 5, axis=0),
                    np.percentile(sims, 95, axis=0), color="0.85",
                    label="5-95% симуляций")
    ax.plot(grid, np.median(sims, axis=0), color="0.5", lw=1.0, label="медиана")
    ax.step(t, r, where="post", color="black", lw=1.2, label="$b=2$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$t=\ln n$")
    ax.set_ylabel(r"$r(t)$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_resid_bridge.pdf")
    plt.close(fig)
    say("")
    say("  рисунки записаны в %s" % FIGS)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="сокращённое число реализаций Монте-Карло")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="переопределить зерно; результат пишется в "
                         "results_seed<N>.txt, основной results.txt не трогается")
    ap.add_argument("--out", default=None, help="куда писать вывод")
    args = ap.parse_args()

    global SEED
    if args.seed is not None:
        SEED = args.seed
        # рисунки статьи строятся только основным зерном, иначе прогон с другим
        # зерном молча подменил бы figs/ и PDF перестал бы соответствовать тексту
        args.no_figs = True

    R_BIG = 20000 if not args.fast else 2000
    R_MED = 15000 if not args.fast else 1500
    R_SMALL = 5000 if not args.fast else 800

    seqs = load_sequences()
    bases = build(seqs)

    say("analysis.py -- воспроизведение результатов работы")
    say("e^gamma = %.6f, e^-gamma = %.7f, seed = %d"
        % (KAPPA_LPW, G_LPW, SEED))

    table1(bases)
    check_selection_rule(seqs)
    section23(seqs)
    table2(200000 if not args.fast else 20000)
    table5(bases)
    kA, kB = section52(bases)
    table3(bases, R_MED)
    table4(bases, R_BIG)

    rule("[§5.2, ур. 12] Интервал схемы B инверсией симуляции")
    obs, lo, hi, p = scheme_B_interval(bases, R_BIG)
    say("  kappa^_B = %.4f, 95%% ДИ = [%.3f; %.3f], p = %.3f" % (obs, lo, hi, p))

    section43(bases, R_MED)
    section53(bases, R_SMALL)
    table6(seqs)
    section54(seqs, bases, R_MED)
    table7(bases)
    section7(seqs, R_BIG)
    table8(bases)
    section8_log(bases)

    if not args.no_figs:
        figures(seqs, bases, obs, (lo, hi))

    if args.out:
        out = Path(args.out)
    elif args.seed is not None:
        out = HERE / ("results_seed%d.txt" % args.seed)
    else:
        out = HERE / "results.txt"
    out.write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print("\n[записано %s]" % out)


if __name__ == "__main__":
    main()
