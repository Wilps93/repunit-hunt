#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_v2.py -- воспроизведение числовых результатов версий 2 и 3 статьи
«Схема наблюдения в статистике обобщённых репьюнитных простых: точный условный
критерий константы Ленстры -- Померанса -- Вагстаффа» (paper_ru.tex / paper_en.tex).

Отличие от analysis.py (версия 1): основной вывод строится на ТОЧНОМ условном
законе схемы B, K_b = N_b - 1 | D_b ~ Poisson(lambda_b D_b) (предложение 3),
а не на моментной поправке и симуляционной инверсии интервала. Все интервалы и
p-значения объединённого вывода получаются точной инверсией пуассоновского
распределения (Гарвуд) и от генератора случайных чисел не зависят.

Теги вывода:
  [S0]  данные: таблица 1, правило отбора (b <= 26), совпадения индексов
  [S1]  таблица 2: Монте-Карло проверка предложений 1-2
  [S2]  таблица 3 и §3.4: оценки по основаниям, неустойчивость формы оценки
  [S3]  §5.2: объединённые оценки, интервал Гарвуда, покрытие, цена фронта
  [S4]  §3.3, §4.3: экспоненциально малые остатки
  [S5]  §5.3: условный мультиномиальный критерий однородности и его мощность
  [S6]  таблица 5 (усечение): нижнее усечение
  [S7]  §5.4: исключение оснований, статус данных
  [S8]  таблица 7 и §6: мощность объединённого критерия
  [S9]  §7: счётная функция Мерсенна, основание 2 на известном фронте
  [S10] таблица 4: схема B при семи правилах фронта (калибровка гамма-пивота)
  [S11] §4.6: данные не выбирают схему (взвешенное среднее)
  [S12] таблица 8 и §8: структурные модели под условным правдоподобием
Теги [S13]-[S18] выводят stopping_rules.py, lpw_second_order.py, frontiers.py.

Запуск:  python analysis_v2.py           полный прогон -> results_v2.txt, figs_v2/
         python analysis_v2.py --fast    сокращённое число реализаций
Рисунки пишутся в figs_v2/ (русские подписи) и figs_v2/en/ (английские), чтобы
не затирать рисунки версии 1 в figs/.
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

import analysis as v1  # загрузка b-файлов, BaseData, build, pooled, C_model, newey_west_se

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs_v2"
KAPPA = v1.KAPPA_LPW
G_LPW = v1.G_LPW
SEED = 20260914
GIMPS_XVER = v1.GIMPS_XVER

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
# Точный пуассоновский аппарат
# --------------------------------------------------------------------------
def garwood(k, S, level=0.95):
    a = (1 - level) / 2
    lo = stats.chi2.ppf(a, 2 * k) / (2 * S) if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - a, 2 * (k + 1)) / (2 * S)
    return lo, hi


def p_two(k, mu):
    """Двусторонний p по конвенции удвоенного минимального хвоста."""
    return min(1.0, 2 * min(stats.poisson.cdf(k, mu), stats.poisson.sf(k - 1, mu)))


def p_mid(k, mu):
    h = 0.5 * stats.poisson.pmf(k, mu)
    return min(1.0, 2 * min(stats.poisson.cdf(k - 1, mu) + h, stats.poisson.sf(k, mu) + h))


def coverage(S, kappa, level=0.95):
    mu = kappa * S
    ks = np.arange(0, int(mu + 20 * math.sqrt(mu) + 50))
    lo = np.where(ks > 0, stats.chi2.ppf((1 - level) / 2, 2 * np.maximum(ks, 1)) / (2 * S), 0.0)
    hi = stats.chi2.ppf(1 - (1 - level) / 2, 2 * (ks + 1)) / (2 * S)
    return float(stats.poisson.pmf(ks, mu)[(lo <= kappa) & (kappa <= hi)].sum())


def cond(bases):
    """K, S для условного закона схемы B."""
    K = sum(bd.N - 1 for bd in bases)
    S = sum(bd.expo for bd in bases)
    return K, S


def gamma_p(M, S):
    x = KAPPA * S
    return 2 * min(stats.gamma.cdf(x, M), stats.gamma.sf(x, M))


# --------------------------------------------------------------------------
# [S0] данные
# --------------------------------------------------------------------------
def s0(seqs, bases):
    rule("[S0] Таблица 1: использованные последовательности")
    say("%3s %8s %8s %5s %5s %7s %13s %10s %10s" %
        ("b", "OEIS", "простых", "N_b", "K_b", "n_min", "n_max", "D_b", "D_b/ln b"))
    for bd in bases:
        say("%3d %8s %8d %5d %5d %7d %13d %10.4f %10.4f" %
            (bd.b, v1.SEQ[bd.b], bd.n_primes, bd.N, bd.N - 1, bd.nmin, bd.nmax, bd.D, bd.expo))
    M, S, B = v1.pooled(bases)
    say("  M = %d, K = M - B = %d, S = %.4f, B = %d" % (M, M - B, S, B))

    extra = v1.load_sequences(v1.SEQ_EXTRA)
    allseq = dict(seqs)
    allseq.update(extra)
    say("  [§2.2] число индексов за границей: " +
        ", ".join("b=%d:%d" % (b, len(extra[b])) for b in sorted(extra)))
    bs = v1.build(allseq)
    M2, S2, B2 = v1.pooled(bs)
    say("  [§2.2] b <= 26: B = %d, M = %d, (M-1)/S = %.4f, (M-B)/S = %.4f, сдвиг %+.1f%%"
        % (B2, M2, (M2 - 1) / S2, (M2 - B2) / S2, 100 * ((M2 - B2) / S2 / ((M - B) / S) - 1)))

    c = Counter(int(n) for idx in seqs.values() for n in idx)
    shared = {n: k for n, k in c.items() if k > 1}
    mx = max(shared.items(), key=lambda kv: kv[1])
    say("  [замечание 2] совпадающих значений индекса: %d; чаще всего n = %d (%d оснований); "
        "все совпадения при n <= %d" % (len(shared), mx[0], mx[1], max(shared)))
    return allseq


# --------------------------------------------------------------------------
# [S1] таблица 2
# --------------------------------------------------------------------------
def s1(reps):
    rule("[S1] Таблица 2: проверка предложений 1-2 (%d реализаций на строку)" % reps)
    rng = np.random.default_rng(SEED + 1)
    lam = 1.0
    say("%4s | %9s %9s %9s %9s %9s | %9s %9s %9s %9s"
        % ("N=mu", "E[l^]/l", "N/(N-1)", "E[l~]/l", "E[G^]l", "E[b^]l",
           "B:E[G^]l", "теория", "B:E[b1]l", "теория"))
    for N in (7, 10, 16, 22, 51):
        # схема A
        t = np.cumsum(rng.exponential(1 / lam, size=(reps, N)), axis=1)
        D = t[:, -1]
        k = np.arange(N + 1, dtype=float)
        w = (k - k.mean()) / np.sum((k - k.mean()) ** 2)
        beta = np.concatenate([np.zeros((reps, 1)), t], axis=1) @ w
        # схема B, mu = lambda W = N, W = N
        mu = float(N)
        W = mu / lam
        n = rng.poisson(mu, size=reps)
        g_end = np.zeros(reps)
        b1 = []
        for i in range(reps):
            if n[i] >= 1:
                u = np.sort(rng.uniform(0, W, n[i]))
                g_end[i] = u[-1] / n[i]
                if n[i] >= 2:
                    kk = np.arange(1, n[i] + 1, dtype=float)
                    b1.append(np.polyfit(kk, u, 1)[0])
        th_end = 1 - (1 + mu) * math.exp(-mu)
        th_ols = (1 - (1 + mu + mu * mu / 2) * math.exp(-mu)) / th_end
        say("%4d | %9.4f %9.4f %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f %9.4f"
            % (N, np.mean(N / D) / lam, N / (N - 1), np.mean((N - 1) / D) / lam,
               np.mean(D / N) * lam, beta.mean() * lam,
               g_end.mean() * lam, th_end, np.mean(b1) * lam, th_ols))
    say("  схема B: G^ = D/N при N >= 1 и 0 при N = 0; b1 -- наклон МНК без начала")
    say("  отсчёта при условии N >= 2, теория -- формула (BunbiasedOLS)")
    for mu in (7, 10, 16):
        e = math.exp(-mu)
        say("  [S4] mu = %2d: (1+mu)e^-mu = %.3g; дефицит МНК без начала отсчёта = %.3g"
            % (mu, (1 + mu) * e, 1 - (1 - (1 + mu + mu * mu / 2) * e) / (1 - (1 + mu) * e)))


# --------------------------------------------------------------------------
# [S2] оценки по основаниям
# --------------------------------------------------------------------------
def s2(bases):
    rule("[S2] Таблица 3: оценки по основаниям (схема B, точные интервалы Гарвуда)")
    say("%3s %4s %4s %8s %8s %7s %7s %18s %8s" %
        ("b", "N_b", "K_b", "G_end", "G_OLS", "разн.%", "kappa_b", "95% ДИ (Гарвуд)", "mid-PIT"))
    rows = []
    for bd in bases:
        K = bd.N - 1
        s = bd.expo
        lo, hi = garwood(K, s)
        mu = KAPPA * s
        pit = stats.poisson.cdf(K - 1, mu) + 0.5 * stats.poisson.pmf(K, mu)
        d = 100 * (bd.G_ols - bd.G_end) / bd.G_end
        rows.append((bd, K / s, lo, hi, pit, d))
        say("%3d %4d %4d %8.4f %8.4f %+7.1f %7.3f %18s %8.3f" %
            (bd.b, bd.N, K, bd.G_end, bd.G_ols, d, K / s, "[%.3f; %.3f]" % (lo, hi), pit))
    cover = all(r[2] <= KAPPA <= r[3] for r in rows)
    pits = [r[4] for r in rows]
    say("  все интервалы Гарвуда накрывают e^gamma: %s" % cover)
    say("  mid-PIT: min %.3f (b=%d), max %.3f (b=%d); все внутри [0,025; 0,975]: %s"
        % (min(pits), rows[int(np.argmin(pits))][0].b, max(pits), rows[int(np.argmax(pits))][0].b,
           all(0.025 < p < 0.975 for p in pits)))
    say("  b=13: mid-PIT = %.3f" % [r[4] for r in rows if r[0].b == 13][0])

    rule("[S2] §3.4: неустойчивость к выбору формы оценки")
    spr = {bd.b: 100 * abs(bd.G_ols - bd.G_end) / bd.G_end for bd in bases}
    dev = {bd.b: 100 * abs(bd.G_end - G_LPW) / G_LPW for bd in bases}
    top = sorted(spr.items(), key=lambda kv: -kv[1])[:3]
    say("  наибольшие расхождения форм: " + ", ".join("b=%d %.1f%%" % kv for kv in top))
    say("  отклонения от e^-gamma: медиана %.1f%%, от %.1f%% (b=%d) до %.1f%% (b=%d)"
        % (np.median(list(dev.values())), min(dev.values()), min(dev, key=dev.get),
           max(dev.values()), max(dev, key=dev.get)))
    say("  расхождения форм: медиана %.1f%%; макс/медиана отклонений = %.2f"
        % (np.median(list(spr.values())), max(spr.values()) / np.median(list(dev.values()))))
    over = [b for b in spr if spr[b] > dev[b]]
    say("  основания, где выбор формы весит больше отклонения: %s" % over)
    by_end = sorted(bases, key=lambda x: -x.G_end)
    by_ols = sorted(bases, key=lambda x: -x.G_ols)
    b18 = [x for x in bases if x.b == 18][0]
    say("  по G_end (убыв.): " + ", ".join("%d:%.4f" % (x.b, x.G_end) for x in by_end[:4]))
    say("  b=18: G_end = %.4f (место %d), G_OLS = %.4f (место %d), медиана G_OLS = %.4f"
        % (b18.G_end, 1 + by_end.index(b18), b18.G_ols, 1 + by_ols.index(b18),
           np.median([x.G_ols for x in bases])))
    be = sorted(bd.b for bd in bases if bd.G_end < G_LPW)
    bo = sorted(bd.b for bd in bases if bd.G_ols < G_LPW)
    say("  ниже e^-gamma: концевая %d/15 %s; МНК %d/15 %s" % (len(be), be, len(bo), bo))
    return rows


# --------------------------------------------------------------------------
# [S3] объединённые оценки
# --------------------------------------------------------------------------
def s3(bases):
    rule("[S3] §5.2: объединённые оценки")
    M, S, B = v1.pooled(bases)
    K = M - B
    kB = K / S
    lo, hi = garwood(K, S)
    mu = KAPPA * S
    say("  схема B: (M-B)/S = %.4f, 95%% Гарвуд [%.4f; %.4f], p = %.3f (mid-p %.3f), "
        "e^gamma S = %.2f" % (kB, lo, hi, p_two(K, mu), p_mid(K, mu), mu))
    cov = [coverage(S, k) for k in np.linspace(1.5, 2.2, 141)]
    say("  покрытие интервала: %.4f при kappa = e^gamma; от %.4f до %.4f при kappa in [1,5; 2,2]"
        % (coverage(S, KAPPA), min(cov), max(cov)))
    kA = (M - 1) / S
    say("  схема A: (M-1)/S = %.4f, 95%% [%.4f; %.4f], p = %.3f; M/S = %.4f"
        % (kA, stats.gamma.ppf(0.025, M) / S, stats.gamma.ppf(0.975, M) / S, gamma_p(M, S), M / S))
    say("  превышение над e^gamma: A %+.1f%%, B %+.1f%%; разница оценок %.1f%%"
        % (100 * (kA / KAPPA - 1), 100 * (kB / KAPPA - 1), 100 * (kA / kB - 1)))
    sfull = S + B / KAPPA
    say("  SD(kappa_B) = %.4f; при известных фронтах (S_full = S + B/kappa) %.4f; "
        "рост SD %.1f%%, потеря информации %.1f%%"
        % (math.sqrt(KAPPA / S), math.sqrt(KAPPA / sfull),
           100 * (math.sqrt(sfull / S) - 1), 100 * (B / KAPPA) / sfull))
    rem = sum(math.exp(-bd.N) for bd in bases)
    say("  [S4] остаток sum exp(-lambda_b W_b) при lambda_b W_b = N_b: %.2g" % rem)
    return kB, (lo, hi)


# --------------------------------------------------------------------------
# [S5] однородность
# --------------------------------------------------------------------------
def lr_stat(K, s):
    e = K.sum(axis=-1, keepdims=True) * s / s.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * np.sum(np.where(K > 0, K * np.log(K / e), 0.0), axis=-1)


def s5(bases, reps_exact, reps_power):
    rule("[S5] §5.3: условный мультиномиальный критерий однородности")
    K = np.array([bd.N - 1 for bd in bases])
    s = np.array([bd.expo for bd in bases])
    lam = float(lr_stat(K, s))
    rng = np.random.default_rng(SEED + 5)
    hits, done = 0, 0
    while done < reps_exact:
        m = min(100000, reps_exact - done)
        x = rng.multinomial(K.sum(), s / s.sum(), size=m)
        e = K.sum() * s / s.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            t = 2 * np.sum(np.where(x > 0, x * np.log(x / e), 0.0), axis=1)
        hits += int(np.sum(t >= lam - 1e-9))
        done += m
    p = hits / done
    say("  Lambda = %.3f, df = %d, точное p = %.3f (+-%.4f, %d реализаций), chi2 p = %.3f"
        % (lam, len(K) - 1, p, 1.96 * math.sqrt(p * (1 - p) / done), done,
           stats.chi2.sf(lam, len(K) - 1)))
    # критическое значение точного условного критерия
    x = rng.multinomial(K.sum(), s / s.sum(), size=200000)
    e = K.sum() * s / s.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = 2 * np.sum(np.where(x > 0, x * np.log(x / e), 0.0), axis=1)
    crit = float(np.quantile(t0, 0.95))
    say("  критическое значение (95%% квантиль нулевого закона) = %.2f (chi2: %.2f)"
        % (crit, stats.chi2.ppf(0.95, len(K) - 1)))
    rule("[S5] мощность против логнормального разброса kappa_b (%d реализаций на точку)" % reps_power)
    cvs = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    pw = []
    for cv in cvs:
        sig = math.sqrt(math.log(1 + cv * cv))
        kb = rng.lognormal(math.log(KAPPA) - sig * sig / 2, sig, size=(reps_power, len(s)))
        # условно на наблюдённом K: (K_b) | K ~ Multinomial(K; kappa_b s_b / sum)
        pr = kb * s
        pr /= pr.sum(axis=1, keepdims=True)
        Ks = np.array([rng.multinomial(K.sum(), row) for row in pr])
        stat = lr_stat(Ks, s)
        pw.append(float(np.mean(stat > crit)))
    say("  cv:       " + " ".join("%6.2f" % c for c in cvs))
    say("  мощность: " + " ".join("%6.3f" % q for q in pw))
    say("  мощность 80%% при cv ~ %.2f" % float(np.interp(0.8, pw, cvs)))


# --------------------------------------------------------------------------
# [S6] усечение
# --------------------------------------------------------------------------
def s6(seqs):
    rule("[S6] Таблица 5: нижнее усечение (начало отсчёта -- первый уцелевший индекс)")
    say("%14s %3s %4s %9s %9s %18s %7s %7s" %
        ("усечение", "B", "M", "(M-1)/S", "(M-B)/S", "95% ДИ (B)", "p_B", "p_A"))
    out = []
    for label, tr, me in (("нет", None, 2), ("n>10", 10, 2), ("n>10^2", 100, 2),
                          ("n>10^3", 1000, 2), ("n>10^4", 10000, 2), ("n>10^4, N>=1", 10000, 1)):
        bs = v1.build(seqs, truncate=tr, min_events=me)
        M, S, B = v1.pooled(bs)
        lo, hi = garwood(M - B, S)
        pB = p_two(M - B, KAPPA * S)
        out.append((label, (M - 1) / S, (M - B) / S))
        say("%14s %3d %4d %9.4f %9.4f %18s %7.3f %7.3f" %
            (label, B, M, (M - 1) / S, (M - B) / S, "[%.3f; %.3f]" % (lo, hi), pB, gamma_p(M, S)))
    return out


# --------------------------------------------------------------------------
# [S7] устойчивость
# --------------------------------------------------------------------------
def s7(seqs, bases):
    rule("[S7] §5.4: исключение оснований и статус данных")
    res = []
    for bd in bases:
        bs = [x for x in bases if x.b != bd.b]
        K, S = cond(bs)
        res.append((bd.b, K / S, p_two(K, KAPPA * S)))
    lo = min(res, key=lambda r: r[1])
    hi = max(res, key=lambda r: r[1])
    say("  исключение по одному: (M-B)/S от %.3f (без b=%d) до %.3f (без b=%d); p от %.2f до %.2f"
        % (lo[1], lo[0], hi[1], hi[0], min(r[2] for r in res), max(r[2] for r in res)))
    s2 = dict(seqs)
    s2[2] = seqs[2][seqs[2] < GIMPS_XVER]
    bs = v1.build(s2)
    M, S, B = v1.pooled(bs)
    say("  b=2 -- 50 дважды проверенных: (M-1)/S = %.3f, p_A = %.3f; (M-B)/S = %.4f"
        % ((M - 1) / S, gamma_p(M, S), (M - B) / S))
    bs = v1.build(seqs, cap=10 ** 6)
    M, S, B = v1.pooled(bs)
    say("  без индексов n > 10^6: M = %d, (M-1)/S = %.3f, (M-B)/S = %.3f" % (M, (M - 1) / S, (M - B) / S))
    worst = None
    for bd in bases:
        s3_ = dict(seqs)
        s3_[bd.b] = seqs[bd.b][:-1]
        M, S, B = v1.pooled(v1.build(s3_))
        if worst is None or (M - 1) / S > worst[1]:
            worst = (bd.b, (M - 1) / S, (M - B) / S)
    say("  наихудший случай (последний PRP составной): b = %d, (M-1)/S = %.3f, (M-B)/S = %.3f" % worst)


# --------------------------------------------------------------------------
# [S8] мощность
# --------------------------------------------------------------------------
def power(S, r, one_sided=False, alpha=0.05):
    m0 = KAPPA * S
    ks = np.arange(0, int(m0 * r + 12 * math.sqrt(m0 * r) + 60))
    if one_sided:
        rej = stats.poisson.sf(ks - 1, m0) <= alpha
    else:
        rej = 2 * np.minimum(stats.poisson.cdf(ks, m0), stats.poisson.sf(ks - 1, m0)) <= alpha
    return float(stats.poisson.pmf(ks, r * m0)[rej].sum())


def first_crossing(f, grid):
    for x in grid:
        if f(x) >= 0.8:
            return x
    return float("nan")


def s8(bases):
    rule("[S8] Таблица 7: мощность точного пуассоновского критерия, alpha = 0,05")
    K, S = cond(bases)
    b2 = [bd for bd in bases if bd.b == 2][0]
    S2 = b2.expo
    say("  экспозиции: только b=2 S = %.2f (K = %d); объединение S = %.2f (K = %d)"
        % (S2, b2.N - 1, S, K))
    say("%8s %10s %10s %10s %10s" % ("r", "Мерс.2ст", "объед.2ст", "Мерс.1ст", "объед.1ст"))
    for r in (1.05, 1.10, 1.15, 1.20, 1.30, 1.50):
        say("%8.2f %10.3f %10.3f %10.3f %10.3f" %
            (r, power(S2, r), power(S, r), power(S2, r, True), power(S, r, True)))
    g = np.arange(1.0, 2.0, 0.0005)
    say("  80%%: только Мерсенн %+.1f%%, объединение %+.1f%% (односторонний %+.1f%%)"
        % (100 * (first_crossing(lambda r: power(S2, r), g) - 1),
           100 * (first_crossing(lambda r: power(S, r), g) - 1),
           100 * (first_crossing(lambda r: power(S, r, True), g) - 1)))
    dK = KAPPA * math.log(2) * sum(1 / bd.lnb for bd in bases)
    dS = math.log(2) * sum(1 / bd.lnb for bd in bases)   # exposure added by one doubling
    for tgt in (0.10, 0.05):
        # The power of a discrete test is saw-toothed in S. Take the smallest S after
        # which it never drops below 0.8 again: coarse scan, then refine the last dip.
        grid = np.arange(100, 4000, 1.0)
        pw = np.array([power(x, 1 + tgt) for x in grid])
        first = float(grid[int(np.argmax(pw >= 0.8))])
        fine = np.arange(first - 2, first + 80, 0.01)       # continuous scan of the saw-tooth
        pf = np.array([power(x, 1 + tgt) for x in fine])
        Sreq = float(fine[int(np.max(np.nonzero(pf < 0.8)[0])) + 1])
        Kreq = KAPPA * Sreq
        dbl = (Sreq - S) / dS
        say("  отклонение %.0f%%: S ~ %.1f, K ~ %.0f при kappa = e^gamma (%.0f при kappa = %.2f e^gamma), "
            "M ~ %.0f; недостаёт экспозиции %.1f = %.1f удвоений всех фронтов, рост 2^%.1f = %.1e"
            % (100 * tgt, Sreq, Kreq, Kreq * (1 + tgt), 1 + tgt, Kreq + len(bases), Sreq - S, dbl,
               dbl, 2.0 ** dbl))
    say("  удвоение всех фронтов даёт %.2f события (экспозиция %.3f)" % (dK, dS))


# --------------------------------------------------------------------------
# [S9] §7
# --------------------------------------------------------------------------
def s9(seqs, reps_null, reps_boot):
    rule("[S9] §7: счётная функция Мерсенна")
    exps = np.array(sorted(seqs[2]), dtype=float)
    exps = exps[exps < GIMPS_XVER]
    N = len(exps)
    t = np.log(exps)
    t0, T = t[0], math.log(GIMPS_XVER)
    y = np.arange(N, dtype=float)
    X = np.column_stack([np.ones(N), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / (N - 2))))
    nw = v1.newey_west_se(X, resid, lags=8)
    rho1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    say("  N = %d, наклон %.4f, свободный член %.4f (SE %.4f); rho_1 = %.3f; Ньюи -- Уэст %.4f (x%.2f)"
        % (N, beta[1], beta[0], se[0], rho1, nw[0], nw[0] / se[0]))
    rng = np.random.default_rng(SEED + 9)
    inter = np.empty(reps_null)
    maxr = np.empty(reps_null)
    midexc = np.empty(reps_null)
    for i in range(reps_null):
        tt = np.concatenate([[t0], np.sort(rng.uniform(t0, T, N - 1))])
        XX = np.column_stack([np.ones(N), tt])
        inter[i] = np.linalg.lstsq(XX, y, rcond=None)[0][0]
        lam_hat = (N - 1) / (tt[-1] - t0)
        r = y - lam_hat * (tt - t0)
        maxr[i] = np.max(np.abs(r))
        mid = (tt - t0) / (tt[-1] - t0)
        sel = (mid >= 0.25) & (mid <= 0.75)
        midexc[i] = r[sel].mean() if sel.any() else 0.0

    def p2(sim, obs):
        return 2 * min(np.mean(sim <= obs), np.mean(sim >= obs))
    say("  нулевой закон свободного члена (%d): медиана %.3f, 5-95%% [%.2f; %.2f], SD %.3f, p = %.3f"
        % (reps_null, np.median(inter), np.percentile(inter, 5), np.percentile(inter, 95),
           inter.std(ddof=1), p2(inter, beta[0])))
    bs = np.empty(reps_boot)
    for i in range(reps_boot):
        j = rng.integers(0, N, size=N)
        bs[i] = np.linalg.lstsq(np.column_stack([np.ones(N), t[j]]), y[j], rcond=None)[0][0]
    say("  (i) бутстрэп (%d): медиана %.3f, SD %.3f, 95%% [%.3f; %.3f], доля < 0: %.3f"
        % (reps_boot, np.median(bs), bs.std(ddof=1), np.percentile(bs, 2.5),
           np.percentile(bs, 97.5), np.mean(bs < 0)))
    say("      SD нуля / SD бутстрэпа = %.1f; ширина 95%%: %.1f"
        % (inter.std(ddof=1) / bs.std(ddof=1),
           (np.percentile(inter, 97.5) - np.percentile(inter, 2.5)) /
           (np.percentile(bs, 97.5) - np.percentile(bs, 2.5))))
    lam_obs = (N - 1) / (t[-1] - t0)
    r_obs = y - lam_obs * (t - t0)
    say("  (ii) max|r| = %.3f, медиана симуляций %.3f, p = %.3f"
        % (np.max(np.abs(r_obs)), np.median(maxr), p2(maxr, np.max(np.abs(r_obs)))))
    mid = (t - t0) / (t[-1] - t0)
    om = float(r_obs[(mid >= 0.25) & (mid <= 0.75)].mean())
    say("  (iii) средняя экскурсия %.2f, медиана симуляций %.3f, p = %.3f, доля отрицательных %.3f"
        % (om, np.median(midexc), p2(midexc, om), np.mean(midexc < 0)))
    k = N - 1
    lamB = k / (T - t0)
    lo, hi = garwood(k, (T - t0) / math.log(2))
    mu0 = KAPPA / math.log(2) * (T - t0)
    say("  [§7.2] известный фронт: lambda^ = %.4f (ЛПВ %.4f), kappa_2 = %.4f, Гарвуд [%.3f; %.3f], "
        "p = %.3f (mid-p %.3f)" % (lamB, KAPPA / math.log(2), lamB * math.log(2), lo, hi,
                                   p_two(k, mu0), p_mid(k, mu0)))
    b50 = v1.BaseData(2, exps)
    b52 = v1.BaseData(2, np.array(sorted(seqs[2]), dtype=float))
    say("         условная оценка того же среза: %.4f; по всем 52: %.4f"
        % (b50.kappa_tilde, b52.kappa_tilde))
    say("  [§7.3] (e^gamma/ln 2) * Mertens = %+.3f" % (KAPPA / math.log(2) * 0.2614972128476428))
    return t, t0, N


# --------------------------------------------------------------------------
# [S10] калибровка
# --------------------------------------------------------------------------
def simulate_B(bases, reps, rng, W_of):
    Ms = np.zeros(reps, dtype=np.int64)
    Ss = np.zeros(reps)
    for bd in bases:
        lam = KAPPA / bd.lnb
        W = np.broadcast_to(np.asarray(W_of(bd, lam), dtype=float), (reps,)).copy()
        n = rng.poisson(lam * W)
        bad = n < 1
        while np.any(bad):
            n[bad] = rng.poisson(lam * W[bad])
            bad = n < 1
        D = W * rng.random(reps) ** (1.0 / n)
        Ms += n
        Ss += D / bd.lnb
    return Ms, Ss


def s10(bases, reps):
    rule("[S10] Таблица 4: схема B при семи правилах фронта (%d реализаций)" % reps)
    B = len(bases)
    rules = [("E[#] = наблюдённое", lambda bd, l: bd.N / l),
             ("+0,5 среднего интервала", lambda bd, l: (bd.N + 0.5) / l),
             ("+1 средний интервал", lambda bd, l: (bd.N + 1) / l),
             ("+2 средних интервала", lambda bd, l: (bd.N + 2) / l),
             ("+5 средних интервалов", lambda bd, l: (bd.N + 5) / l),
             ("в 1,5 раза дальше по t", lambda bd, l: 1.5 * bd.D),
             ("вдвое дальше по t", lambda bd, l: 2.0 * bd.D)]
    say("%28s %10s %12s %12s" % ("правило", "ошибка I", "E[(M-1)/S]", "E[(M-B)/S]"))
    for name, f in rules:
        rng = np.random.default_rng(SEED + 10)
        Ms, Ss = simulate_B(bases, reps, rng, f)
        x = KAPPA * Ss
        rej = np.mean((x < stats.gamma.ppf(0.025, Ms)) | (x > stats.gamma.ppf(0.975, Ms)))
        say("%28s %10.4f %12.4f %12.4f" % (name, rej, np.mean((Ms - 1) / Ss) / KAPPA,
                                             np.mean((Ms - B) / Ss) / KAPPA))
    rng = np.random.default_rng(SEED + 10)
    M0 = sum(bd.N for bd in bases)
    Ss = sum(rng.gamma(bd.N, bd.lnb / KAPPA, size=reps) / bd.lnb for bd in bases)
    x = KAPPA * Ss
    say("%28s %10.4f" % ("A (контроль)", np.mean((x < stats.gamma.ppf(0.025, M0)) |
                                                 (x > stats.gamma.ppf(0.975, M0)))))


# --------------------------------------------------------------------------
# [S11] данные не выбирают схему
# --------------------------------------------------------------------------
def s11(bases, reps):
    rule("[S11] §4.6: взвешенное среднее несмещённых по-основанию оценок")
    for scheme in ("A", "B"):
        rng = np.random.default_rng(SEED + 11)
        num = np.zeros(reps)
        den = np.zeros(reps)
        for bd in bases:
            lam = KAPPA / bd.lnb
            if scheme == "A":
                n = np.full(reps, bd.N)
                D = rng.gamma(bd.N, 1 / lam, size=reps)
            else:
                W = bd.N / lam
                n = rng.poisson(lam * W, size=reps)
                bad = n < 1
                while np.any(bad):
                    n[bad] = rng.poisson(lam * W, size=int(bad.sum()))
                    bad = n < 1
                D = W * rng.random(reps) ** (1.0 / n)
            num += np.maximum(n - 2, 0) * (n - 1) * bd.lnb / D
            den += np.maximum(n - 2, 0)
        star = num / den
        say("  схема %s: смещение kappa* = %+.2f%%" % (scheme, 100 * (np.mean(star) / KAPPA - 1)))
    say("  [S4] E[lambda~_b]/lambda_b = 1 - e^-mu: при mu = 7 дефицит %.2f%%" % (100 * math.exp(-7)))


# --------------------------------------------------------------------------
# [S12] структурные модели
# --------------------------------------------------------------------------
def ll_cond(bases, model, theta):
    K = np.array([bd.N - 1 for bd in bases], dtype=float)
    cs = np.array([v1.C_model(bd, model, theta) for bd in bases])
    s = np.array([bd.expo for bd in bases])
    if np.any(cs <= 0):
        return -1e18, float("nan")
    kap = K.sum() / np.sum(cs * s)
    mu = kap * cs * s
    return float(np.sum(K * np.log(mu) - mu - special.gammaln(K + 1))), kap


def s12(bases):
    rule("[S12] Таблица 8: структурные модели под условным пуассоновским правдоподобием")
    K = sum(bd.N - 1 for bd in bases)
    specs = [("A", []), ("B", [0.0, 0.0]), ("C", [0.0]), ("D", [0.0, 0.0, 0.0])]
    fits = {}
    for name, x0 in specs:
        if not x0:
            ll, kap = ll_cond(bases, name, [])
        else:
            r = optimize.minimize(lambda th: -ll_cond(bases, name, th)[0], x0, method="Nelder-Mead",
                                  options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 100000,
                                           "maxfev": 100000})
            ll, kap = ll_cond(bases, name, r.x)
        fits[name] = (ll, len(x0) + 1)
    llA = fits["A"][0]
    say("%7s %4s %10s %6s %6s %8s %8s" % ("модель", "k", "ln L", "LR", "p", "AIC", "BIC"))
    for name, x0 in specs:
        ll, k = fits[name]
        lr = 2 * (ll - llA)
        say("%7s %4d %10.3f %6.2f %6s %8.2f %8.2f" %
            (name, k, ll, lr, "%.3f" % stats.chi2.sf(lr, len(x0)) if x0 else "-",
             2 * k - 2 * ll, k * math.log(K) - 2 * ll))
    for n_eff in (K, len(bases)):
        bic = {n: fits[n][1] * math.log(n_eff) - 2 * fits[n][0] for n, _ in specs}
        mn = min(bic.values())
        w = {n: math.exp(-(v - mn) / 2) for n, v in bic.items()}
        tot = sum(w.values())
        say("  апостериорные вероятности по BIC при n = %d: %s"
            % (n_eff, " / ".join("%.3f" % (w[n] / tot) for n, _ in specs)))
    llf, kapf = ll_cond(bases, "Cfix", [])
    say("  C^fix: среднее C_b = %.3f, kappa^ = %.3f, ln L = %.2f против %.2f (разность %.2f)"
        % (np.mean([v1.C_model(bd, "Cfix", []) for bd in bases]), kapf, llf, llA, llf - llA))


# --------------------------------------------------------------------------
# Рисунки
# --------------------------------------------------------------------------
def figures(seqs, bases, rows, kB, ciB, trunc, sameerr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (FIGS / "en").mkdir(parents=True, exist_ok=True)
    M0, S0, B0 = v1.pooled(bases)
    K0 = M0 - B0
    TXT = {
        "ru": dict(base="основание $b$", A=r"схема A: $(M-1)/S$", B=r"схема B: $(M-B)/S$",
                   Bc=r"схема B, $c=\ln 2$: $K/S(c)$ (§9)", trunc="нижнее усечение по $n$",
                   none="нет", mers="только Мерсенн ($K=%d$)", pool="объединение ($K=%d$)",
                   power="мощность", band="5--95% симуляций", med="медиана"),
        "en": dict(base="base $b$", A=r"scheme A: $(M-1)/S$", B=r"scheme B: $(M-B)/S$",
                   Bc=r"scheme B, $c=\ln 2$: $K/S(c)$ (§9)", trunc="lower truncation in $n$",
                   none="none", mers="Mersenne only ($K=%d$)", pool="pooled ($K=%d$)",
                   power="power", band="5--95% of simulations", med="median"),
    }
    # усечение с членом второго порядка c = ln 2
    kc = []
    for tr in (None, 10, 100, 1000, 10000):
        bs = v1.build(seqs, truncate=tr, min_events=2)
        K = sum(bd.N - 1 for bd in bs)
        Sc = sum((bd.D + math.log(2) * math.log(bd.t[-1] / bd.t0)) / bd.lnb for bd in bs)
        kc.append(K / Sc)
    ka = [r[1] for r in trunc[:5]]
    kb = [r[2] for r in trunc[:5]]
    rs = np.linspace(1.0, 1.8, 161)
    b2 = [bd for bd in bases if bd.b == 2][0]
    pw_m = [power(b2.expo, r) for r in rs]
    pw_p = [power(S0, r) for r in rs]
    t, t0, N = sameerr
    lam = (N - 1) / (t[-1] - t0)
    r_obs = np.arange(N) - lam * (t - t0)
    rng = np.random.default_rng(SEED + 41)
    grid = np.linspace(t0, t[-1], 200)
    sims = np.empty((4000, grid.size))
    for i in range(4000):
        tt = np.concatenate([[t0], np.sort(rng.uniform(t0, t[-1], N - 1))])
        sims[i] = np.interp(grid, tt, np.arange(N) - (N - 1) / (tt[-1] - t0) * (tt - t0))

    for lang, out in (("ru", FIGS), ("en", FIGS / "en")):
        T = TXT[lang]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for i, (bd, kb_, lo, hi, _, _) in enumerate(rows):
            ax.plot([i, i], [lo, hi], color="0.4", lw=1.0, zorder=1)
            ax.scatter([i], [kb_], s=8 + 3.0 * bd.N, color="black", zorder=2)
        ax.axhline(KAPPA, ls="--", color="k", lw=1.0, label=r"$e^\gamma$")
        ax.axhline((M0 - 1) / S0, color="tab:blue", lw=1.2, ls="-.", label=T["A"])
        ax.axhspan(ciB[0], ciB[1], color="tab:green", alpha=0.15, hatch="///",
                   edgecolor="tab:green", lw=0)
        ax.axhline(kB, color="tab:green", lw=1.6, label=T["B"])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([str(r[0].b) for r in rows])
        ax.set_xlabel(T["base"])
        ax.set_ylabel(r"$\hat\kappa_b=K_b\ln b/D_b$")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "fig_kappa.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        x = range(5)
        ax.plot(x, ka, "o-.", color="tab:blue", label=T["A"])
        ax.plot(x, kb, "s-", color="tab:green", label=T["B"])
        ax.plot(x, kc, "^:", color="tab:red", label=T["Bc"])
        ax.axhline(KAPPA, ls="--", color="k", lw=1.0, label=r"$e^\gamma$")
        ax.set_xticks(list(x))
        ax.set_xticklabels([T["none"], "$>10$", "$>10^2$", "$>10^3$", "$>10^4$"])
        ax.set_xlabel(T["trunc"])
        ax.set_ylabel(r"$\hat\kappa$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_truncation.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ax.plot(rs, pw_m, ls="--", color="tab:blue", label=T["mers"] % (b2.N - 1))
        ax.plot(rs, pw_p, color="tab:green", label=T["pool"] % K0)
        ax.axhline(0.8, ls=":", color="k", lw=1.0)
        ax.set_xlabel(r"$\kappa/e^\gamma$")
        ax.set_ylabel(T["power"])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_power.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        ax.fill_between(grid, np.percentile(sims, 5, axis=0), np.percentile(sims, 95, axis=0),
                        color="0.85", label=T["band"])
        ax.plot(grid, np.median(sims, axis=0), color="0.5", lw=1.0, label=T["med"])
        ax.step(t, r_obs, where="post", color="black", lw=1.2, label="$b=2$")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(r"$t=\ln n$")
        ax.set_ylabel(r"$r(t)$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_resid_bridge.pdf")
        plt.close(fig)
    say("")
    say("  [рисунки] записаны в %s и %s; kappa(c=ln2) по усечению: %s"
        % (FIGS.name, (FIGS / "en").relative_to(HERE), ", ".join("%.4f" % v for v in kc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="другое зерно: вывод в results_v2_seed<N>.txt, рисунки не строятся")
    args = ap.parse_args()
    f = 10 if args.fast else 1
    global SEED
    if args.seed is not None:
        SEED = args.seed
        args.no_figs = True

    seqs = v1.load_sequences()
    bases = v1.build(seqs)
    say("analysis_v2.py; e^gamma = %.6f, e^-gamma = %.7f, seed = %d" % (KAPPA, G_LPW, SEED))
    s0(seqs, bases)
    s1(60000 // f)
    rows = s2(bases)
    kB, ciB = s3(bases)
    s5(bases, 2_000_000 // f, 12000 // f)
    trunc = s6(seqs)
    s7(seqs, bases)
    s8(bases)
    sameerr = s9(seqs, 60000 // f, 20000 // f)
    s10(bases, 50000 // f)
    s11(bases, 50000 // f)
    s12(bases)
    if not args.no_figs:
        figures(seqs, bases, rows, kB, ciB, trunc, sameerr)
    if not args.fast:
        name = "results_v2.txt" if args.seed is None else "results_v2_seed%d.txt" % args.seed
        (HERE / name).write_text("\n".join(OUT) + "\n", encoding="utf-8")
        print("\n[записано %s]" % name)


if __name__ == "__main__":
    main()
