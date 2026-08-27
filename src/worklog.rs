//! Crash-safe журнал: append-only JSONL + периодический атомарный снапшот
//! множества завершённых k (для resume после перезапуска).

use crate::pipeline::Outcome;
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

pub struct WorkLog {
    writer: Mutex<BufWriter<File>>,
    done: parking_lot::RwLock<HashSet<u64>>,
}

impl WorkLog {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path: PathBuf = path.into();
        let mut done = HashSet::new();
        if path.exists() {
            let f = File::open(&path)?;
            // map_while, а не flatten(): на повторяющейся ошибке чтения
            // flatten() крутился бы вечно, а нам нужно просто остановиться.
            for line in BufReader::new(f).lines().map_while(Result::ok) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(k) = v.get("k").and_then(|x| x.as_u64()) {
                        done.insert(k);
                    }
                }
            }
        }
        let f = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::with_capacity(1 << 16, f)),
            done: parking_lot::RwLock::new(done),
        })
    }

    pub fn is_done(&self, k: u64) -> bool {
        self.done.read().contains(&k)
    }

    pub fn record(&self, o: &Outcome) -> Result<()> {
        let (k, json) = match o {
            Outcome::SmallFactor { k, q } =>
                (*k, serde_json::json!({"k":k,"status":"factored","q":q,"stage":"small"})),
            // q может не влезать в u64 (до 2^127), поэтому пишем десятичной строкой.
            Outcome::GpuFactor { k, q } =>
                (*k, serde_json::json!({"k":k,"status":"factored","q":q.to_string(),"stage":"gpu"})),
            Outcome::Pm1Factor { k, q } =>
                (*k, serde_json::json!({"k":k,"status":"factored","q":q,"stage":"pm1"})),
            // backend пишем, чтобы `--verify` знал, каким путём получен
            // вердикт, и мог пересчитать ДРУГИМ (см. src/verify.rs).
            Outcome::Composite { k, bits, secs, backend } =>
                (*k, serde_json::json!({"k":k,"status":"composite","bits":bits,
                                        "secs":secs,"backend":backend})),
            Outcome::Prp { k, bits, secs, backend } =>
                (*k, serde_json::json!({"k":k,"status":"PRP","bits":bits,
                                        "secs":secs,"backend":backend})),
            // Нерешённый показатель. Пишем как отдельный статус, а не молчим:
            // иначе он неотличим от ещё не дошедшего до проверки.
            Outcome::Failed { k, bits, reason } =>
                (*k, serde_json::json!({"k":k,"status":"failed","bits":bits,
                                        "reason":reason})),
        };
        self.done.write().insert(k);
        let mut w = self.writer.lock();
        writeln!(w, "{json}")?;
        // PRP-находки флашим немедленно — их терять нельзя.
        if matches!(o, Outcome::Prp { .. } | Outcome::Pm1Factor { .. }
                       | Outcome::Failed { .. }) {
            w.flush()?;
        }
        Ok(())
    }

    /// Прочитать журнал целиком — для режима перепроверки (`--verify`).
    pub fn read_records(path: &std::path::Path) -> Result<Vec<serde_json::Value>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let f = File::open(path)?;
        Ok(BufReader::new(f)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(&l).ok())
            .collect())
    }

    pub fn flush(&self) -> Result<()> {
        self.writer.lock().flush()?;
        Ok(())
    }

}