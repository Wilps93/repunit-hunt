use anyhow::Result;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub struct Report<'a> {
    pub base: u64,
    pub k_min: u64,
    pub k_max: u64,
    pub prp_exponents: &'a [u64],
    pub note: &'a str,
}

pub fn write_json(path: &Path, rep: &Report) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(rep)?)?;
    std::fs::rename(&tmp, path)?;   // атомарная замена
    Ok(())
}