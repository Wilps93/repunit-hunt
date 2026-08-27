//! CPU-сито по малым делителям q.
//!
//! ИДЕЯ: ПЕРЕБИРАЕМ q, А НЕ k
//! --------------------------
//! Наивный ход — для каждого k перебрать q = 2mk+1 до предела Q — стоит
//! Θ(Q/2 · Σ 1/k) модульных возведений. Но структура задачи позволяет
//! развернуть цикл наизнанку и получить почти линейную стоимость:
//!
//! для простого k делитель q даёт `ord_q(b) ∈ {1, k}`, а `ord_q(b) = k`
//! влечёт `k | q-1`. Значит ОДНО простое q может «убить» лишь те k,
//! которые входят в разложение q-1 — их не больше десятка. Поэтому мы
//! перебираем простые q ≤ Q, факторизуем q-1 (пробным делением до √q) и
//! проверяем только те его простые множители k, что попали в рабочий
//! диапазон. Итог: ~π(Q) факторизаций вместо миллиардов powmod.
//!
//! КРАЕВЫЕ СЛУЧАИ (те же, что и в GPU-ядре, см. native/cuda/tf_kernel.cu):
//!
//! * `q | b` => R_k ≡ 1 (mod q) — q не делитель никогда;
//! * `q | (b-1)` => R_k ≡ k (mod q) — q делитель ⟺ q | k, а для простого k
//!   это значит ровно k = q;
//! * иначе => q делитель ⟺ b^k ≡ 1 (mod q).
//!
//! Проверка `b^k ≡ 1` в общей ветке безопасна именно потому, что случай
//! `b ≡ 1 (mod q)` уже отсечён: без него powmod давал бы ложное срабатывание.

use rayon::prelude::*;
use std::collections::HashMap;

use super::kbase::simple_sieve;

pub struct SmallSieve {
    /// k -> наименьший найденный делитель q.
    factors: HashMap<u64, u64>,
    q_limit: u64,
}

impl SmallSieve {
    /// Построить сито: простые q ≤ `q_limit`, показатели k ∈ [k_min, k_max).
    pub fn build(base: u64, q_limit: u64, k_min: u64, k_max: u64) -> Self {
        assert!(base >= 2, "base >= 2");
        if q_limit < 2 || k_max <= k_min {
            return Self { factors: HashMap::new(), q_limit };
        }

        // Простые q ≤ q_limit. Они же служат базисом пробного деления q-1.
        let primes = simple_sieve(q_limit);

        let hits: Vec<(u64, u64)> = primes
            .par_iter()
            .flat_map_iter(|&q| {
                let mut out: Vec<(u64, u64)> = Vec::new();
                let b_mod = base % q;

                if b_mod == 0 {
                    // q | b  =>  R_k ≡ 1 (mod q): не делитель ни при каком k.
                    return out.into_iter();
                }
                if b_mod == 1 {
                    // q | (b-1)  =>  q | R_k  ⟺  q | k  ⟺  k = q (k простое).
                    if q >= k_min && q < k_max && is_proper_divisor(base, q, q) {
                        out.push((q, q));
                    }
                    return out.into_iter();
                }

                // Общий случай: кандидаты k — простые множители q-1.
                for k in prime_factors_in_range(q - 1, k_min, k_max) {
                    if pow_mod(b_mod, k, q) == 1 && is_proper_divisor(base, k, q) {
                        out.push((k, q));
                    }
                }
                out.into_iter()
            })
            .collect();

        let mut factors: HashMap<u64, u64> = HashMap::with_capacity(hits.len());
        for (k, q) in hits {
            factors.entry(k).and_modify(|old| *old = (*old).min(q)).or_insert(q);
        }
        Self { factors, q_limit }
    }

    /// Найденный малый делитель R_k(b), если он есть.
    pub fn find_factor(&self, k: u64) -> Option<u64> {
        self.factors.get(&k).copied()
    }

    /// Сколько показателей уже отсеяно.
    pub fn eliminated(&self) -> usize {
        self.factors.len()
    }

    pub fn q_limit(&self) -> u64 {
        self.q_limit
    }
}

/// Различные простые множители `n`, попадающие в [lo, hi).
/// Пробное деление до √n: для q ≤ 2^24 это ≤ 2048 итераций на кандидата.
fn prime_factors_in_range(mut n: u64, lo: u64, hi: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let in_range = |p: u64| p >= lo && p < hi;

    if n % 2 == 0 {
        if in_range(2) {
            out.push(2);
        }
        while n % 2 == 0 {
            n /= 2;
        }
    }

    let mut p = 3u64;
    while p * p <= n {
        if n % p == 0 {
            if in_range(p) {
                out.push(p);
            }
            while n % p == 0 {
                n /= p;
            }
        }
        p += 2;
    }
    // Остаток > 1 — сам простой.
    if n > 1 && in_range(n) {
        out.push(n);
    }
    out
}

/// Является ли q СОБСТВЕННЫМ делителем, то есть `q < R_k(b)`?
///
/// Иначе кандидат отсеивался бы собственным значением: например
/// R_2(10) = 11, и «делитель» 11 честно делит его нацело — но это само число,
/// а R_2(10) простое. Проверка нужна только при крошечных k: уже при
/// `(k-1)·log2(b) > 64` репьюнит заведомо больше любого q типа u64.
fn is_proper_divisor(b: u64, k: u64, q: u64) -> bool {
    let qq = q as u128;
    let mut r: u128 = 1; // R_1 = 1
    for _ in 1..k {
        // r <= q < 2^64 на входе в итерацию => r*b + 1 < 2^128, переполнения нет
        r = r * b as u128 + 1;
        if r > qq {
            return true;
        }
    }
    false
}

/// b^e mod m через двоичное возведение (m < 2^63, промежуточные — в u128).
fn pow_mod(b: u64, e: u64, m: u64) -> u64 {
    let mut result: u128 = 1;
    let mut base = (b % m) as u128;
    let m128 = m as u128;
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            result = result * base % m128;
        }
        base = base * base % m128;
        e >>= 1;
    }
    result as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_repunit_factors_base10() {
        // R_3 = 111 = 3·37, R_5 = 11111 = 41·271, R_7 = 239·4649.
        let s = SmallSieve::build(10, 100_000, 2, 25);
        assert_eq!(s.find_factor(3), Some(3));
        assert_eq!(s.find_factor(5), Some(41));
        assert_eq!(s.find_factor(7), Some(239));
        assert_eq!(s.find_factor(11), Some(21649));
        assert_eq!(s.find_factor(13), Some(53));

        // R_2 = 11, R_19 и R_23 — простые: делителей быть не должно.
        assert_eq!(s.find_factor(2), None);
        assert_eq!(s.find_factor(19), None);
        assert_eq!(s.find_factor(23), None);
        // Наименьший делитель R_17 равен 2071723 > предела сита.
        assert_eq!(s.find_factor(17), None);
    }

    #[test]
    fn prime_repunit_is_not_sieved_by_itself() {
        // Регрессия: R_2(10) = 11 — простое, но 11 = 2·1·5+1 «делит» его нацело.
        // Делитель обязан быть собственным, иначе кандидат теряется.
        let s = SmallSieve::build(10, 1000, 2, 40);
        assert_eq!(s.find_factor(2), None, "R_2(10)=11 простое");
        // R_3(4) = 21, R_2(4) = 5 — простое, 5 = 2·1·2+1
        let s4 = SmallSieve::build(4, 1000, 2, 40);
        assert_eq!(s4.find_factor(2), None, "R_2(4)=5 простое");
        // Настоящие собственные делители при этом обязаны находиться.
        assert_eq!(s4.find_factor(3), Some(3));
    }

    #[test]
    fn edge_case_q_divides_b_minus_one() {
        // b = 4: q = 3 делит b-1, значит 3 | R_k ⟺ 3 | k ⟺ k = 3.
        // R_3(4) = 21 = 3·7 — делится; R_5(4) = 341 = 11·31 — на 3 не делится.
        let s = SmallSieve::build(4, 1000, 2, 20);
        assert_eq!(s.find_factor(3), Some(3));
        assert_ne!(s.find_factor(5), Some(3));
    }

    #[test]
    fn edge_case_q_divides_b() {
        // b = 6, q = 3 делит b: R_k(6) ≡ 1 (mod 3) — никогда не делитель.
        let s = SmallSieve::build(6, 50, 2, 30);
        for k in [2u64, 3, 5, 7, 11, 13] {
            assert_ne!(s.find_factor(k), Some(3), "k={k}");
        }
    }

    #[test]
    fn factors_match_brute_force() {
        // Полная сверка с прямым делением для небольших параметров.
        let (base, q_limit) = (7u64, 5000u64);
        let s = SmallSieve::build(base, q_limit, 2, 60);
        for k in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59] {
            let brute = simple_sieve(q_limit).into_iter().find(|&q| {
                // q | R_k(b) проверяем честно: (b^k - 1)/(b-1) mod q,
                // и требуем СОБСТВЕННОГО делителя (q < R_k), иначе простое
                // число «отсеивалось» бы самим собой.
                let divides = if base % q == 1 {
                    k % q == 0
                } else {
                    base % q != 0 && pow_mod(base % q, k, q) == 1
                };
                divides && is_proper_divisor(base, k, q)
            });
            assert_eq!(s.find_factor(k), brute, "k={k}");
        }
    }
}
