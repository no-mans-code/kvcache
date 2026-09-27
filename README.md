# kvcache

A generic on-disk, size-bounded, LRU-evicted byte-blob cache keyed by an arbitrary string.

"KV cache" is a deliberately generic name here — this crate has no opinion on what you store under a key: a transformer's actual attention key/value tensors, a model's resumable generation context, a rendered report, anything. What's worth memoizing to disk, and under what key, is always the caller's decision; "evict the least-recently-used entry once we're over budget" is the same problem every time, so that part lives here.

## Usage

```toml
[dependencies]
kvcache = { git = "https://github.com/no-mans-code/kvcache" }
```

```rust
use kvcache::{Cache, default_capacity_bytes};
use std::path::Path;

let path = Path::new("./cache.redb");
let capacity = default_capacity_bytes(path)?; // min(available disk space / 2, 50 GB)
let cache = Cache::open(path, capacity)?;

cache.put("my-key", b"whatever bytes you want cached")?;
assert_eq!(cache.get("my-key")?, Some(b"whatever bytes you want cached".to_vec()));

let stats = cache.stats()?;
println!("{} entries, {}/{} bytes", stats.entry_count, stats.total_bytes, stats.capacity_bytes);
```

Backed by `redb` (pure Rust, no C toolchain needed). A `put` on a full cache evicts the least-recently-used entries (oldest `get`/`put` first) until there's room, then writes the new entry regardless of how much was evicted — a single value larger than the whole capacity is a policy question for the caller, not this crate.

## Origin

Built for [docuzent](https://github.com/no-mans-code/docuzent)'s single-document Q&A mode, to persist Ollama's resumable generation context per `(model, context_size, document_hash)` so asking a second question about the same document can skip reprocessing it. That use is documented in docuzent's own README/Blueprint, not here — this crate stays domain-agnostic on purpose.

## License

MIT
