//! Генератор простых показателей k — сегментное сито Эратосфена.
//!
//! ПОЧЕМУ ТОЛЬКО ПРОСТЫЕ k. Для d | k верно R_d(b) | R_k(b), поэтому при
//! составном k репьюнит заведомо составной. Перебирать имеет смысл только
//! простые k (это же обстоятельство даёт форму делителей q = 2mk+1).
//!
//! Сегментация: базовые простые до sqrt(k_max) хранятся один раз, дальше
//! идём окнами по `SEG_BITS` чисел — рабочее множество влезает в L2 и
//! генератор не зависит от величины k_max по памяти.

/// Размер сегмента в числах (256 KiB булевых флагов).
const SEG_SIZE: usize = 1 << 18;

pub struct PrimeIter {
    /// Простые до sqrt(hi) — «базис» для вычёркивания.
    base: Vec<u64>,
    /// Текущее окно [seg_lo, seg_lo + SEG_SIZE).
    seg_lo: u64,
    hi: u64,
    flags: Vec<bool>,
    /// Позиция внутри окна.
    pos: usize,
    done: bool,
}

impl PrimeIter {
    /// Простые k из полуинтервала [lo, hi).
    pub fn new(lo: u64, hi: u64) -> Self {
        let lo = lo.max(2);
        let hi = hi.max(lo);
        let limit = (hi as f64).sqrt() as u64 + 2;
        Self {
            base: simple_sieve(limit),
            seg_lo: lo,
            hi,
            flags: vec![true; SEG_SIZE],
            pos: SEG_SIZE, // заставляет заполнить первый сегмент при первом next()
            done: lo >= hi,
        }
    }

    /// Заполнить текущее окно, вернуть false, если окон больше нет.
    fn fill_segment(&mut self) -> bool {
        if self.seg_lo >= self.hi {
            return false;
        }
        let hi = (self.seg_lo + SEG_SIZE as u64).min(self.hi);
        let len = (hi - self.seg_lo) as usize;
        self.flags[..len].fill(true);
        if len < SEG_SIZE {
            self.flags[len..].fill(false);
        }

        for &p in &self.base {
            if p * p >= hi {
                break;
            }
            // Первое кратное p внутри окна, но не меньше p^2.
            let mut start = self.seg_lo.div_ceil(p) * p;
            if start < p * p {
                start = p * p;
            }
            let mut i = start;
            while i < hi {
                self.flags[(i - self.seg_lo) as usize] = false;
                i += p;
            }
        }
        // 0 и 1 простыми не являются.
        for n in [0u64, 1u64] {
            if (self.seg_lo..hi).contains(&n) {
                self.flags[(n - self.seg_lo) as usize] = false;
            }
        }
        self.pos = 0;
        true
    }
}

impl Iterator for PrimeIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        loop {
            if self.done {
                return None;
            }
            if self.pos >= SEG_SIZE && !self.fill_segment() {
                self.done = true;
                return None;
            }
            while self.pos < SEG_SIZE {
                let n = self.seg_lo + self.pos as u64;
                if n >= self.hi {
                    self.done = true;
                    return None;
                }
                let is_prime = self.flags[self.pos];
                self.pos += 1;
                if is_prime {
                    return Some(n);
                }
            }
            // Окно исчерпано — переходим к следующему.
            self.seg_lo += SEG_SIZE as u64;
            self.pos = SEG_SIZE;
        }
    }
}

/// Обычное сито Эратосфена до `limit` включительно.
pub fn simple_sieve(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let n = limit as usize + 1;
    let mut flags = vec![true; n];
    flags[0] = false;
    flags[1] = false;
    let mut p = 2usize;
    while p * p < n {
        if flags[p] {
            let mut i = p * p;
            while i < n {
                flags[i] = false;
                i += p;
            }
        }
        p += 1;
    }
    flags
        .iter()
        .enumerate()
        .filter(|(_, &f)| f)
        .map(|(i, _)| i as u64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_range() {
        let v: Vec<u64> = PrimeIter::new(0, 30).collect();
        assert_eq!(v, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn offset_range_matches_simple_sieve() {
        let all = simple_sieve(300_000);
        let want: Vec<u64> = all.iter().copied().filter(|&p| p >= 100_000).collect();
        let got: Vec<u64> = PrimeIter::new(100_000, 300_001).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn crosses_segment_boundary() {
        let n = SEG_SIZE as u64;
        let got: Vec<u64> = PrimeIter::new(n - 50, n + 50).collect();
        let want: Vec<u64> = simple_sieve(n + 50)
            .into_iter()
            .filter(|&p| p >= n - 50 && p < n + 50)
            .collect();
        assert_eq!(got, want);
    }
}
