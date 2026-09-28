//! A generic on-disk, size-bounded, LRU-evicted byte-blob cache keyed by an
//! arbitrary string.
//!
//! "KV cache" here is a deliberately generic name: this crate has no
//! opinion on what you store under a key - a transformer's actual
//! attention key/value tensors, a model's resumable generation context, a
//! rendered report, anything. It exists because that decision (what's
//! worth memoizing to disk, and under what key) is always the caller's,
//! while "evict the least-recently-used entry once we're over budget" is
//! the same problem every time.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};

const ENTRIES: TableDefinition<&str, &str> = TableDefinition::new("entries");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");

const DEFAULT_CAP_BYTES: u64 = 50 * 1024 * 1024 * 1024; // 50 GB

#[derive(Serialize, Deserialize)]
struct StoredEntry {
    value: Vec<u8>,
    last_accessed: u64,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub entry_count: usize,
    pub total_bytes: u64,
    pub capacity_bytes: u64,
}

pub struct Cache {
    db: Database,
    capacity_bytes: u64,
}

impl Cache {
    /// Opens (creating if needed) a cache at `path` with the given byte
    /// capacity. Use [`default_capacity_bytes`] to compute a sensible
    /// capacity from available disk space if the caller has no opinion.
    pub fn open(path: &Path, capacity_bytes: u64) -> Result<Self> {
        let db = Database::create(path)
            .with_context(|| format!("failed to open cache at {}", path.display()))?;
        let txn = db.begin_write()?;
        txn.open_table(ENTRIES)?;
        txn.open_table(META)?;
        txn.commit()?;
        Ok(Self { db, capacity_bytes })
    }

    /// Reads a value, bumping its recency on a hit. `None` on a miss.
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let value = {
            let txn = self.db.begin_read()?;
            let table = txn.open_table(ENTRIES)?;
            match table.get(key)? {
                Some(v) => Some(serde_json::from_str::<StoredEntry>(v.value())?.value),
                None => None,
            }
        };
        if value.is_some() {
            self.touch(key)?;
        }
        Ok(value)
    }

    fn touch(&self, key: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(ENTRIES)?;
            let existing = table.get(key)?.map(|v| v.value().to_string());
            if let Some(v) = existing {
                let mut entry: StoredEntry = serde_json::from_str(&v)?;
                entry.last_accessed = now();
                table.insert(key, serde_json::to_string(&entry)?.as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Writes a value under `key`, evicting the least-recently-used
    /// entries (as needed, oldest first) until the cache is back within
    /// its byte capacity. A single value larger than the whole capacity is
    /// still stored - eviction empties everything else first, but this
    /// crate does not refuse a write, since "the one thing you asked to
    /// cache doesn't fit" is a policy question for the caller, not this
    /// crate, to decide what to do about.
    pub fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let size = value.len() as u64;
        let txn = self.db.begin_write()?;
        {
            let mut entries = txn.open_table(ENTRIES)?;
            let mut meta = txn.open_table(META)?;

            let old_size = match entries.get(key)? {
                Some(v) => serde_json::from_str::<StoredEntry>(v.value())?.size,
                None => 0,
            };
            let mut total: u64 = meta
                .get("total_bytes")?
                .and_then(|v| v.value().parse().ok())
                .unwrap_or(0);
            total = total.saturating_sub(old_size);

            // Evict oldest-first until there is room for the new entry.
            while total + size > self.capacity_bytes {
                let oldest_key = {
                    let mut oldest: Option<(String, u64)> = None;
                    for row in entries.iter()? {
                        let (k, v) = row?;
                        if k.value() == key {
                            continue; // being replaced anyway, not a real eviction candidate
                        }
                        let entry: StoredEntry = serde_json::from_str(v.value())?;
                        if oldest.as_ref().map(|(_, t)| entry.last_accessed < *t).unwrap_or(true) {
                            oldest = Some((k.value().to_string(), entry.last_accessed));
                        }
                    }
                    oldest.map(|(k, _)| k)
                };
                match oldest_key {
                    Some(k) => {
                        if let Some(v) = entries.remove(k.as_str())? {
                            let evicted: StoredEntry = serde_json::from_str(v.value())?;
                            total = total.saturating_sub(evicted.size);
                        }
                    }
                    None => break, // nothing left to evict
                }
            }

            let entry = StoredEntry { value: value.to_vec(), last_accessed: now(), size };
            entries.insert(key, serde_json::to_string(&entry)?.as_str())?;
            meta.insert("total_bytes", (total + size).to_string().as_str())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Deletes `key` if present, adjusting the tracked total accordingly.
    /// Returns whether it existed. For a caller that needs a stale entry
    /// gone *now* rather than waiting for LRU pressure to age it out - e.g.
    /// when whatever the entry was keyed against (a document, a session)
    /// no longer applies and should not linger even briefly.
    pub fn remove(&self, key: &str) -> Result<bool> {
        let txn = self.db.begin_write()?;
        let existed = {
            let mut entries = txn.open_table(ENTRIES)?;
            let mut meta = txn.open_table(META)?;
            let removed_size: Option<u64> = entries.remove(key)?.map(|v| serde_json::from_str::<StoredEntry>(v.value())).transpose()?.map(|e| e.size);
            match removed_size {
                Some(size) => {
                    let total: u64 = meta
                        .get("total_bytes")?
                        .and_then(|v| v.value().parse().ok())
                        .unwrap_or(0);
                    meta.insert("total_bytes", total.saturating_sub(size).to_string().as_str())?;
                    true
                }
                None => false,
            }
        };
        txn.commit()?;
        Ok(existed)
    }

    pub fn stats(&self) -> Result<CacheStats> {
        let txn = self.db.begin_read()?;
        let entries = txn.open_table(ENTRIES)?;
        let meta = txn.open_table(META)?;
        let total_bytes: u64 = meta
            .get("total_bytes")?
            .and_then(|v| v.value().parse().ok())
            .unwrap_or(0);
        Ok(CacheStats {
            entry_count: entries.len()? as usize,
            total_bytes,
            capacity_bytes: self.capacity_bytes,
        })
    }
}

/// Nanosecond resolution, not milliseconds - `last_accessed` only needs to
/// establish a strict recency *order*, and millisecond resolution ties
/// easily within a single fast `put`/`get` sequence (confirmed: caused
/// `eviction_removes_the_least_recently_used_entry_first` to flake, since a
/// tie falls back to redb's iteration order, not true recency).
fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
}

/// `min(available space on the volume containing `path` / 2, 50 GB)` - a
/// sensible default when the caller has no stronger opinion about how much
/// disk this cache should be allowed to use.
pub fn default_capacity_bytes(path: &Path) -> Result<u64> {
    let base = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(base).ok();
    let available = fs4::available_space(base)
        .with_context(|| format!("failed to read available disk space at {}", base.display()))?;
    Ok((available / 2).min(DEFAULT_CAP_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_cache_path(label: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("kvcache-test-{label}-{}-{n}.redb", std::process::id()))
    }

    struct TempPath(std::path::PathBuf);
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn put_then_get_round_trips() {
        let path = TempPath(temp_cache_path("roundtrip"));
        let cache = Cache::open(&path.0, 1_000_000).unwrap();
        cache.put("k1", b"hello").unwrap();
        assert_eq!(cache.get("k1").unwrap(), Some(b"hello".to_vec()));
        assert_eq!(cache.get("missing").unwrap(), None);
    }

    #[test]
    fn stats_report_size_and_count() {
        let path = TempPath(temp_cache_path("stats"));
        let cache = Cache::open(&path.0, 1_000_000).unwrap();
        cache.put("a", b"12345").unwrap();
        cache.put("b", b"1234567890").unwrap();
        let stats = cache.stats().unwrap();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.total_bytes, 15);
        assert_eq!(stats.capacity_bytes, 1_000_000);
    }

    #[test]
    fn replacing_a_key_updates_size_correctly() {
        let path = TempPath(temp_cache_path("replace"));
        let cache = Cache::open(&path.0, 1_000_000).unwrap();
        cache.put("a", b"12345").unwrap();
        cache.put("a", b"1").unwrap();
        let stats = cache.stats().unwrap();
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.total_bytes, 1);
    }

    #[test]
    fn eviction_removes_the_least_recently_used_entry_first() {
        let path = TempPath(temp_cache_path("evict"));
        // Capacity for two 5-byte entries, not three.
        let cache = Cache::open(&path.0, 10).unwrap();
        cache.put("a", b"aaaaa").unwrap();
        cache.put("b", b"bbbbb").unwrap();
        // Touch "a" so "b" becomes the least-recently-used.
        cache.get("a").unwrap();
        cache.put("c", b"ccccc").unwrap();

        assert_eq!(cache.get("a").unwrap(), Some(b"aaaaa".to_vec()), "a was touched, should survive");
        assert_eq!(cache.get("b").unwrap(), None, "b was least-recently-used, should be evicted");
        assert_eq!(cache.get("c").unwrap(), Some(b"ccccc".to_vec()));
        assert!(cache.stats().unwrap().total_bytes <= 10);
    }

    #[test]
    fn remove_deletes_an_entry_and_frees_its_bytes() {
        let path = TempPath(temp_cache_path("remove"));
        let cache = Cache::open(&path.0, 1_000_000).unwrap();
        cache.put("a", b"12345").unwrap();
        cache.put("b", b"1234567890").unwrap();

        assert!(cache.remove("a").unwrap(), "should report the key existed");
        assert_eq!(cache.get("a").unwrap(), None, "removed entry should be gone");
        assert_eq!(cache.get("b").unwrap(), Some(b"1234567890".to_vec()), "unrelated entry untouched");
        assert_eq!(cache.stats().unwrap().total_bytes, 10, "freed bytes reflected in total");
    }

    #[test]
    fn remove_on_a_missing_key_is_a_harmless_no_op() {
        let path = TempPath(temp_cache_path("remove-missing"));
        let cache = Cache::open(&path.0, 1_000_000).unwrap();
        assert!(!cache.remove("nope").unwrap(), "should report the key did not exist");
    }

    #[test]
    fn default_capacity_is_half_available_space_capped_at_50gb() {
        let path = TempPath(temp_cache_path("defaultcap"));
        let cap = default_capacity_bytes(&path.0).unwrap();
        assert!(cap > 0);
        assert!(cap <= DEFAULT_CAP_BYTES);
    }
}
