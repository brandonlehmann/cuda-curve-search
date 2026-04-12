use clap::Parser;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Serialize, Deserialize};

#[allow(dead_code)]
mod math;
#[allow(dead_code)]
mod search;
#[allow(dead_code)]
mod verify;
#[allow(dead_code)]
mod gpu;

use gpu::pipeline::CompactRecord;
use math::types::*;
use math::ranking::{finalize_results_with_mode, tag_playbook_results, passes_playbook_filters};
use search::{search_discriminant, search_discriminant_profiled};

#[derive(Parser)]
#[command(about = "Elliptic Curve Cycle Search")]
struct Args {
    #[arg(long, default_value_t = 3)]
    start_d: u64,

    #[arg(long)]
    end_d: Option<u64>,

    #[arg(long)]
    count: Option<u64>,

    #[arg(long)]
    random: bool,

    #[arg(long, default_value_t = 100)]
    min_twist: u32,

    #[arg(long, default_value_t = 30)]
    top: usize,

    #[arg(long, default_value = "candidates.jsonl")]
    output: String,

    #[arg(long, default_value_t = 0)]
    chunk_size: usize,

    #[arg(long, default_value_t = 2)]
    progress_secs: u64,

    #[arg(long, default_value = "security")]
    ranking_mode: String,

    /// Comma-separated list of max gamma bit thresholds
    #[arg(long, default_value = "127,128", value_delimiter = ',')]
    max_gamma_bits: Vec<u32>,

    /// Comma-separated list of allowed q mod 8 values
    #[arg(long, default_value = "3,5", value_delimiter = ',')]
    q_mod_8: Vec<u8>,

    /// Pollard-rho budget per twist order in seconds
    #[arg(long, default_value_t = 2.0)]
    rho_seconds: f64,

    /// Force CPU-only path
    #[arg(long)]
    cpu_only: bool,

    /// Optional path for verified results JSON
    #[arg(long)]
    verified_output: Option<String>,

    /// Enable stage profiling
    #[arg(long)]
    profile_stages: bool,

    /// Number of CPU threads (0 = auto-detect)
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

// ── Output path derivation ──────────────────────────────────────────────────

struct OutputPaths {
    gpu: String,
    scanned: String,
    filtered: String,
    top: String,
    winner: String,
    checkpoint: String,
}

fn derive_output_paths(base: &str) -> OutputPaths {
    let stem = base
        .trim_end_matches(".jsonl")
        .trim_end_matches(".json");
    OutputPaths {
        gpu: format!("{}_gpu.jsonl", stem),
        scanned: format!("{}_scanned.jsonl", stem),
        filtered: format!("{}_filtered.jsonl", stem),
        top: format!("{}_top.json", stem),
        winner: format!("{}_winner.json", stem),
        checkpoint: format!("{}_checkpoint.json", stem),
    }
}

// ── Checkpoint for GPU resume ───────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
struct Checkpoint {
    scanned_ranges: Vec<(u64, u64)>,
    version: u32,
}

impl Checkpoint {
    fn new() -> Self {
        Checkpoint { scanned_ranges: Vec::new(), version: 1 }
    }
}

fn load_checkpoint(path: &str) -> Checkpoint {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(Checkpoint::new)
}

fn save_checkpoint(path: &str, checkpoint: &Checkpoint) {
    if let Ok(json) = serde_json::to_string_pretty(checkpoint) {
        let tmp = format!("{}.tmp", path);
        if fs::write(&tmp, &json).is_ok() {
            let _ = fs::rename(&tmp, path);
        }
    }
}

fn merge_ranges(ranges: &mut Vec<(u64, u64)>) {
    if ranges.len() <= 1 { return; }
    ranges.sort_by_key(|&(s, _)| s);
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for &(s, e) in ranges.iter() {
        if let Some(last) = merged.last_mut() {
            if s <= last.1.saturating_add(1) {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }
    *ranges = merged;
}

fn compute_gaps(requested: (u64, u64), scanned: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let (req_start, req_end) = requested;
    let mut gaps = Vec::new();
    let mut cursor = req_start;
    for &(s, e) in scanned {
        if s > req_end || e < req_start { continue; }
        let s = s.max(req_start);
        let e = e.min(req_end);
        if cursor < s {
            gaps.push((cursor, s - 1));
        }
        cursor = e.saturating_add(1);
    }
    if cursor <= req_end {
        gaps.push((cursor, req_end));
    }
    gaps
}

// ── File I/O helpers ────────────────────────────────────────────────────────

fn open_append(path: &str) -> std::fs::File {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {}", path, e))
}

fn append_jsonl<T: Serialize>(file: &Mutex<std::fs::File>, item: &T) {
    if let Ok(line) = serde_json::to_string(item) {
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(f, "{}", line);
            let _ = f.flush();
        }
    }
}

fn write_json_atomic<T: Serialize>(path: &str, item: &T) {
    if let Ok(json) = serde_json::to_string_pretty(item) {
        let tmp = format!("{}.tmp", path);
        if fs::write(&tmp, &json).is_ok() {
            let _ = fs::rename(&tmp, path);
        }
    }
}

fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &str) -> Vec<T> {
    let Ok(file) = fs::File::open(path) else { return Vec::new(); };
    std::io::BufReader::new(file)
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect()
}

// ── Formatting helpers ──────────────────────────────────────────────────────

fn average_ns(duration: Duration, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        duration.as_nanos() as f64 / count as f64
    }
}

fn print_profile_stage(
    label: &str,
    duration: Duration,
    operations: u64,
    passes: Option<(u64, u64)>,
    total_seconds: f64,
) {
    let share = if total_seconds == 0.0 {
        0.0
    } else {
        duration.as_secs_f64() / total_seconds * 100.0
    };
    match passes {
        Some((passed, total)) => eprintln!(
            "  {:<12} {:>7.3}s  {:>5.1}%  avg {:>10.0} ns/op  passes {}/{}",
            label,
            duration.as_secs_f64(),
            share,
            average_ns(duration, operations),
            passed,
            total
        ),
        None => eprintln!(
            "  {:<12} {:>7.3}s  {:>5.1}%  avg {:>10.0} ns/op  ops {}",
            label,
            duration.as_secs_f64(),
            share,
            average_ns(duration, operations),
            operations
        ),
    }
}

fn print_stage_profile(profile: &SearchProfile) {
    let total_measured = profile.total_measured_time();
    let total_seconds = total_measured.as_secs_f64();

    eprintln!();
    eprintln!("Stage profile:");
    eprintln!(
        "  Measured search time: {:.3}s across {} D values",
        total_seconds, profile.discriminants_checked
    );
    if let Some(dps) = profile.measured_d_per_second() {
        eprintln!("  Measured throughput: {:.0} D/s", dps);
    }

    print_profile_stage(
        "squarefree",
        profile.squarefree_time,
        profile.discriminants_checked,
        Some((profile.squarefree_passes, profile.discriminants_checked)),
        total_seconds,
    );
    print_profile_stage(
        "jacobi",
        profile.jacobi_time,
        profile.squarefree_passes,
        Some((profile.jacobi_passes, profile.squarefree_passes)),
        total_seconds,
    );
    print_profile_stage(
        "solve_4p",
        profile.solve_4p_time,
        profile.jacobi_passes,
        Some((profile.solve_4p_passes, profile.jacobi_passes)),
        total_seconds,
    );
    print_profile_stage(
        "q_prime",
        profile.q_prime_time,
        profile.q_candidates,
        Some((profile.prime_q_passes, profile.q_candidates)),
        total_seconds,
    );
    print_profile_stage(
        "twists",
        profile.twist_time,
        profile.twist_candidates,
        Some((profile.twist_candidates, profile.prime_q_passes)),
        total_seconds,
    );
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn eprint_flush(msg: &str) {
    use std::io::Write;
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "{}", msg);
    let _ = stderr.flush();
}

fn format_si(val: f64) -> String {
    if val >= 1e15 { format!("{:.1}P", val / 1e15) }
    else if val >= 1e12 { format!("{:.1}T", val / 1e12) }
    else if val >= 1e9 { format!("{:.1}G", val / 1e9) }
    else if val >= 1e6 { format!("{:.1}M", val / 1e6) }
    else if val >= 1e3 { format!("{:.1}K", val / 1e3) }
    else { format!("{:.0}", val) }
}

fn format_duration(secs: f64) -> String {
    if secs < 0.0 { return "??d??h??m??s".to_string(); }
    let s = secs as u64;
    format!("{:02}d{:02}h{:02}m{:02}s", s / 86400, (s % 86400) / 3600, (s % 3600) / 60, s % 60)
}

// ── Top-N management ────────────────────────────────────────────────────────

fn insert_into_top_n(
    top: &mut Vec<CycleResult>,
    entry: CycleResult,
    max_n: usize,
) -> bool {
    // Insert maintaining sort by min_twist_bits descending, then gamma_bits ascending
    let pos = top.iter().position(|r| {
        entry.min_twist_bits > r.min_twist_bits
            || (entry.min_twist_bits == r.min_twist_bits && entry.gamma_bits < r.gamma_bits)
    }).unwrap_or(top.len());
    if pos < max_n {
        top.insert(pos, entry);
        top.truncate(max_n);
        true
    } else {
        false
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // Configure thread pool
    {
        let threads = if args.threads > 0 {
            args.threads
        } else {
            num_cpus()
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    // Validate mutually exclusive args
    if args.random && args.end_d.is_some() {
        eprintln!("error: --random and --end-d are mutually exclusive");
        std::process::exit(2);
    }
    if args.end_d.is_some() && args.count.is_some() {
        eprintln!("error: --end-d and --count are mutually exclusive");
        std::process::exit(2);
    }

    let start_d = if args.random {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(3..=u64::MAX)
    } else {
        args.start_d.max(3)
    };

    let end_d = match (args.end_d, args.count) {
        (Some(e), None) => e,
        (None, Some(c)) => start_d.saturating_add(c - 1),
        (None, None) => start_d.saturating_add(100_000_000 - 1),
        _ => unreachable!(),
    };
    if start_d > end_d {
        eprintln!("error: start_d ({}) must be <= end_d ({})", start_d, end_d);
        std::process::exit(2);
    }
    let total_discriminants = end_d - start_d + 1;

    let ranking_mode = match args.ranking_mode.as_str() {
        "balanced" => RankingMode::Balanced,
        _ => RankingMode::Security,
    };

    let paths = derive_output_paths(&args.output);

    // Check for PARI/GP
    let gp_available = std::process::Command::new("gp")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    eprintln!("==========================================================================");
    eprintln!("Elliptic Curve Cycle Search (cuda-curve-search)");
    eprintln!("  Base field: p = 2^255 - 19 (255 bits)");
    eprintln!("  Search range: D = {:>12} .. {:>12}", start_d, end_d);
    eprintln!("  Min twist security: {} bits", args.min_twist);
    eprintln!("  Max gamma bits: {:?}", args.max_gamma_bits);
    eprintln!("  Allowed q mod 8: {:?}", args.q_mod_8);
    eprintln!("  Threads: {}", rayon::current_num_threads());
    eprintln!("  PARI/GP: {}", if gp_available { "available" } else { "NOT FOUND (some composites may be dropped)" });
    eprintln!("  Output base: {}", args.output);
    eprintln!("==========================================================================");
    eprintln!();

    // Configure Pollard-rho budget from CLI
    math::twist::set_rho_budget(args.rho_seconds);

    // ── Resume: load checkpoint and existing results ────────────────────────

    let checkpoint = load_checkpoint(&paths.checkpoint);

    // Load ALL pre-filter CycleResults from scanned file (source of truth for CPU work)
    let existing_scanned: Vec<CycleResult> = load_jsonl(&paths.scanned);

    // Build set of processed discriminants so we can skip them in CPU phase
    let mut processed_discriminants: HashSet<u64> = HashSet::new();
    for r in &existing_scanned {
        processed_discriminants.insert(r.discriminant_u64);
    }

    // Re-filter existing scanned results with CURRENT params (handles filter changes)
    let mut seen_q: HashSet<String> = HashSet::new();
    let mut top_n: Vec<CycleResult> = Vec::new();
    let mut initial_filtered: Vec<CycleResult> = Vec::new();
    for r in &existing_scanned {
        if r.min_twist_bits >= args.min_twist
            && passes_playbook_filters(r, &args.max_gamma_bits, &args.q_mod_8)
            && seen_q.insert(r.q.clone())
        {
            insert_into_top_n(&mut top_n, r.clone(), args.top);
            initial_filtered.push(r.clone());
        }
    }

    // Rewrite filtered.jsonl with re-filtered results (it's a derived output)
    if !initial_filtered.is_empty() {
        let mut f = std::fs::File::create(&paths.filtered)
            .expect("failed to create filtered file");
        for r in &initial_filtered {
            if let Ok(line) = serde_json::to_string(r) {
                let _ = writeln!(f, "{}", line);
            }
        }
        let _ = f.flush();
    }

    let resumed_scanned = existing_scanned.len();
    let resumed_filtered = initial_filtered.len();
    let resumed_processed = processed_discriminants.len();
    if resumed_scanned > 0 {
        eprintln!("  Resume: {} scanned results, {} passing filters, {} unique discriminants processed",
            resumed_scanned, resumed_filtered, resumed_processed);
    }

    // Compact GPU file on disk: deduplicate by (discriminant, trace, flags)
    {
        let raw: Vec<CompactRecord> = load_jsonl(&paths.gpu);
        if !raw.is_empty() {
            let before = raw.len();
            let mut seen: HashSet<(u64, [u64; 4], u8)> = HashSet::with_capacity(before);
            let deduped: Vec<CompactRecord> = raw
                .into_iter()
                .filter(|rec| {
                    let flags = (rec.positive_q_passes as u8) | ((rec.negative_q_passes as u8) << 1);
                    seen.insert((rec.discriminant, rec.trace, flags))
                })
                .collect();
            let after = deduped.len();
            if after < before {
                // Atomic rewrite
                let tmp = format!("{}.tmp", paths.gpu);
                let mut f = std::fs::File::create(&tmp)
                    .expect("failed to create gpu tmp file");
                for rec in &deduped {
                    if let Ok(line) = serde_json::to_string(rec) {
                        let _ = writeln!(f, "{}", line);
                    }
                }
                let _ = f.flush();
                let _ = fs::rename(&tmp, &paths.gpu);
                eprintln!("  Compacted GPU file: {} -> {} records ({} duplicates removed)",
                    before, after, before - after);
            }
        }
    }

    // Open output files
    let gpu_file = Mutex::new(open_append(&paths.gpu));
    let scanned_file = Mutex::new(open_append(&paths.scanned));
    let filtered_file = Mutex::new(open_append(&paths.filtered));

    let start = Instant::now();
    let passing_found = AtomicU64::new(resumed_filtered as u64);
    let all_results: Mutex<Vec<CycleResult>> = Mutex::new(existing_scanned);
    let seen_q = Mutex::new(seen_q);
    let top_n = Mutex::new(top_n);
    let min_twist = args.min_twist;
    let profile_stages = args.profile_stages;
    let profile_totals = Mutex::new(SearchProfile::default());
    let _progress_interval = args.progress_secs;
    let max_gamma_bits = args.max_gamma_bits.clone();
    let q_mod_8 = args.q_mod_8.clone();
    let top_max = args.top;
    let top_path = paths.top.clone();
    let winner_path = paths.winner.clone();

    // Helper closure: process a CycleResult through filter/top/winner pipeline
    let handle_result = |r: &CycleResult,
                         seen_q: &Mutex<HashSet<String>>,
                         passing_found: &AtomicU64,
                         filtered_file: &Mutex<std::fs::File>,
                         all_results: &Mutex<Vec<CycleResult>>,
                         top_n: &Mutex<Vec<CycleResult>>,
                         top_path: &str,
                         winner_path: &str,
                         min_twist: u32,
                         max_gamma_bits: &[u32],
                         q_mod_8: &[u8],
                         top_max: usize| {
        all_results.lock().unwrap().push(r.clone());

        if r.min_twist_bits >= min_twist && passes_playbook_filters(r, max_gamma_bits, q_mod_8) {
            {
                let mut sq = seen_q.lock().unwrap();
                if !sq.insert(r.q.clone()) {
                    return;
                }
            }

            passing_found.fetch_add(1, Ordering::Relaxed);
            append_jsonl(filtered_file, r);

            eprint_flush(&format!(
                "  >>> D={:>10}  twist=({:>3},{:>3})  q=2^{}-c  gamma={} bits  qmod8={}",
                r.discriminant,
                r.twist_bits_curve_a,
                r.twist_bits_curve_b,
                r.crandall_k,
                r.gamma_bits,
                r.q_mod_8,
            ));

            // Update top-N and winner
            let mut tn = top_n.lock().unwrap();
            if insert_into_top_n(&mut tn, r.clone(), top_max) {
                write_json_atomic(top_path, &*tn);
                if tn.first().map(|f| f.q == r.q).unwrap_or(false) {
                    write_json_atomic(winner_path, r);
                }
            }
        }
    };

    // ── GPU search path ─────────────────────────────────────────────────────

    let mut gpu_handled = false;
    if !args.cpu_only {
        match gpu::context::GpuContext::try_new() {
            Some(gpu_ctx) => {
                eprintln!("  GPU: {} ({} SMs)", gpu_ctx.device_name, gpu_ctx.sm_count);
                let chunk = if args.chunk_size > 0 {
                    args.chunk_size
                } else {
                    ((gpu_ctx.sm_count as usize) * 256 * 256).max(1_000_000)
                };
                eprintln!("  GPU chunk size: {}", chunk);

                // Compute gaps (GPU ranges not yet scanned)
                let mut sorted_scanned = checkpoint.scanned_ranges.clone();
                merge_ranges(&mut sorted_scanned);
                let gaps = compute_gaps((start_d, end_d), &sorted_scanned);
                let gap_total: u64 = gaps.iter().map(|(s, e)| e - s + 1).sum();

                // Collect records from the current GPU scan (empty if nothing to scan)
                let mut current_run_records: Vec<CompactRecord> = Vec::new();

                if gap_total == 0 {
                    eprintln!("  GPU: all {} D already scanned (resume), skipping GPU phase", total_discriminants);
                } else {
                    if gap_total < total_discriminants {
                        eprintln!("  GPU: resuming — {} of {} D remaining ({} gaps)",
                            gap_total, total_discriminants, gaps.len());
                    }

                    let last_progress = std::cell::Cell::new(Instant::now());

                    let checkpoint_mutex = Mutex::new(checkpoint.clone());
                    let checkpoint_path = paths.checkpoint.clone();

                    let scan_result = gpu::pipeline::gpu_scan_ranges(
                        &gpu_ctx,
                        &gaps,
                        chunk,
                        // on_gpu_progress
                        |gpu_d, _gpu_secs, chunk_rate, candidates, total_d| {
                            let now = Instant::now();
                            if now.duration_since(last_progress.get()).as_secs() >= _progress_interval {
                                let rate_str = if chunk_rate >= 1_000_000.0 {
                                    format!("{:.0}M D/s", chunk_rate / 1_000_000.0)
                                } else {
                                    format!("{:.0} D/s", chunk_rate)
                                };
                                let d_str = format_si(gpu_d as f64);
                                let total_str = format_si(total_d as f64);
                                let eta_secs = if chunk_rate > 0.0 {
                                    (total_d - gpu_d) as f64 / chunk_rate
                                } else {
                                    0.0
                                };
                                eprint_flush(&format!(
                                    "  gpu: {} / {}  {}  candidates: {}  ETA: {}  wall: {:.1}s",
                                    d_str, total_str, rate_str, candidates,
                                    format_duration(eta_secs),
                                    start.elapsed().as_secs_f64(),
                                ));
                                last_progress.set(now);
                            }
                        },
                        // on_gpu_candidates: write each CompactRecord to GPU file, update checkpoint
                        |records, chunk_start, chunk_end| {
                            for rec in records {
                                append_jsonl(&gpu_file, rec);
                            }
                            // Update checkpoint incrementally so crash-resume doesn't
                            // re-scan (and duplicate) already-written records
                            let mut cp = checkpoint_mutex.lock().unwrap();
                            cp.scanned_ranges.push((chunk_start, chunk_end));
                            merge_ranges(&mut cp.scanned_ranges);
                            save_checkpoint(&checkpoint_path, &cp);
                        },
                    );

                    match scan_result {
                        Ok((new_records, gpu_time, gpu_d)) => {
                            current_run_records = new_records;
                            let gpu_rate = if gpu_time > 0.0 { gpu_d as f64 / gpu_time } else { 0.0 };
                            let rate_str = if gpu_rate >= 1_000_000.0 {
                                format!("{:.0}M D/s", gpu_rate / 1_000_000.0)
                            } else {
                                format!("{:.0} D/s", gpu_rate)
                            };
                            eprint_flush(&format!(
                                "  GPU done: {} D in {:.3}s ({})  candidates: {}",
                                gpu_d, gpu_time, rate_str,
                                current_run_records.len(),
                            ));
                        }
                        Err(e) => {
                            eprintln!("GPU scan error: {}, falling back to CPU", e);
                        }
                    }
                }

                // ── CPU phase: process all unprocessed GPU records ──────────

                // Build the unprocessed set: current run's records + any old
                // records from prior runs that never completed CPU processing.
                // The file on disk contains both old and current-run records,
                // so we only load old ones that aren't duplicated by current run.
                let current_run_discriminants: HashSet<u64> = current_run_records
                    .iter()
                    .map(|rec| rec.discriminant)
                    .collect();
                let old_unprocessed: Vec<CompactRecord> = load_jsonl::<CompactRecord>(&paths.gpu)
                    .into_iter()
                    .filter(|rec| {
                        !processed_discriminants.contains(&rec.discriminant)
                            && !current_run_discriminants.contains(&rec.discriminant)
                    })
                    .collect();
                let n_old = old_unprocessed.len();
                let n_current = current_run_records.len();
                let mut unprocessed = current_run_records;
                unprocessed.extend(old_unprocessed);
                if n_old > 0 {
                    eprintln!("  CPU: {} from current scan + {} unprocessed from prior runs",
                        n_current, n_old);
                }

                let n_unprocessed = unprocessed.len();
                if n_unprocessed == 0 {
                    eprintln!("  CPU: all records already processed (resume), skipping twist phase");
                } else {
                    if resumed_processed > 0 {
                        eprintln!("  CPU: resuming — {} unprocessed records (skipping {} already done)",
                            n_unprocessed, resumed_processed);
                    }
                    eprint_flush(&format!("  Factoring {} candidates...", n_unprocessed));

                    let last_progress = std::cell::Cell::new(Instant::now());
                    let last_done_count = std::cell::Cell::new(0u64);
                    let last_done_time = std::cell::Cell::new(Instant::now());
                    // Rate is frozen when no progress is being made (stalled on hard candidate)
                    let frozen_rate = std::cell::Cell::new(0.0f64);
                    let wait_start = std::cell::Cell::new(None::<Instant>);

                    let (cpu_time, cpu_processed) = gpu::pipeline::process_records_cpu(
                        unprocessed,
                        // on_result: write to scanned (pre-filter), then run through filter pipeline
                        |r| {
                            append_jsonl(&scanned_file, r);
                            handle_result(
                                r, &seen_q, &passing_found, &filtered_file,
                                &all_results, &top_n, &top_path, &winner_path,
                                min_twist, &max_gamma_bits, &q_mod_8, top_max,
                            );
                        },
                        // on_record_done (no-op, tracking via scanned CycleResults)
                        |_rec| {},
                        // on_cpu_progress
                        |cpu_done, total| {
                            let now = Instant::now();
                            if now.duration_since(last_progress.get()).as_secs() >= _progress_interval {
                                let found = passing_found.load(Ordering::Relaxed);
                                let prev_done = last_done_count.get();

                                if cpu_done > prev_done {
                                    // Progress was made — compute real rate from this window
                                    let dt = now.duration_since(last_done_time.get()).as_secs_f64();
                                    let delta = (cpu_done - prev_done) as f64;
                                    let rate = if dt > 0.0 { delta / dt } else { 0.0 };
                                    frozen_rate.set(rate);
                                    last_done_count.set(cpu_done);
                                    last_done_time.set(now);
                                    wait_start.set(None);

                                    let remaining = total - cpu_done;
                                    let eta_secs = if rate > 0.0 { remaining as f64 / rate } else { -1.0 };
                                    eprint_flush(&format!(
                                        "  twist: {}/{} done ({:.1} cand/s)  found: {}  ETA: {}  wall: {:.1}s",
                                        cpu_done, total, rate, found,
                                        format_duration(eta_secs),
                                        start.elapsed().as_secs_f64(),
                                    ));
                                } else {
                                    // Stalled — show frozen rate and stall duration
                                    let wait_t = wait_start.get().unwrap_or(now);
                                    if wait_start.get().is_none() {
                                        wait_start.set(Some(now));
                                    }
                                    let wait_secs = now.duration_since(wait_t).as_secs_f64();
                                    let rate = frozen_rate.get();
                                    let remaining = total - cpu_done;
                                    let eta_secs = if rate > 0.0 { remaining as f64 / rate } else { -1.0 };
                                    eprint_flush(&format!(
                                        "  twist: {}/{} done ({:.1} cand/s)  found: {}  ETA: {}  wall: {:.1}s  [waiting {:.0}s]",
                                        cpu_done, total, rate, found,
                                        format_duration(eta_secs),
                                        start.elapsed().as_secs_f64(),
                                        wait_secs,
                                    ));
                                }

                                last_progress.set(now);
                            }
                        },
                    );

                    eprintln!();
                    eprintln!("  Twist: {:>10} cand in {:>6.2}s  ({:>8.1} cand/s)",
                        cpu_processed, cpu_time,
                        if cpu_time > 0.0 { cpu_processed as f64 / cpu_time } else { 0.0 });
                }

                gpu_handled = true;
            }
            None => {
                eprintln!("  GPU: not available, using CPU path");
            }
        }
    }

    // ── CPU-only search path (fallback) ─────────────────────────────────────

    if !gpu_handled {
        let counter = AtomicU64::new(0);
        let file = Mutex::new(open_append(&paths.filtered));

        (start_d..=end_d).into_par_iter().for_each(|d| {
            let (results, profile) = if profile_stages {
                let (results, profile) = search_discriminant_profiled(d);
                (results, Some(profile))
            } else {
                (search_discriminant(d), None)
            };

            if let Some(profile) = profile {
                profile_totals.lock().unwrap().merge(&profile);
            }

            let cnt = counter.fetch_add(1, Ordering::Relaxed);
            if !results.is_empty() {
                for r in &results {
                    if r.min_twist_bits >= min_twist && passes_playbook_filters(r, &max_gamma_bits, &q_mod_8) {
                        {
                            let mut sq = seen_q.lock().unwrap();
                            if !sq.insert(r.q.clone()) {
                                continue;
                            }
                        }

                        passing_found.fetch_add(1, Ordering::Relaxed);

                        if let Ok(line) = serde_json::to_string(r) {
                            if let Ok(mut f) = file.lock() {
                                let _ = writeln!(f, "{}", line);
                                let _ = f.flush();
                            }
                        }

                        eprintln!(
                            "  >>> D={:>10}  twist=({:>3},{:>3})  q=2^{}-c  gamma={} bits  qmod8={}",
                            r.discriminant,
                            r.twist_bits_curve_a,
                            r.twist_bits_curve_b,
                            r.crandall_k,
                            r.gamma_bits,
                            r.q_mod_8,
                        );
                    }
                }
                all_results.lock().unwrap().extend(results);
            }
            if cnt % 500_000 == 0 && cnt > 0 {
                let processed = cnt + 1;
                let elapsed = start.elapsed().as_secs_f64();
                let rate = processed as f64 / elapsed;
                let pct = processed as f64 / total_discriminants as f64 * 100.0;
                let eta = (total_discriminants - processed) as f64 / rate;
                let found = passing_found.load(Ordering::Relaxed);
                eprint_flush(&format!(
                    "  D={:>12} ({:5.1}%)  passing: {}  rate: {:.0} D/s  ETA: {:.0}s",
                    start_d + cnt,
                    pct,
                    found,
                    rate,
                    eta
                ));
            }
        });
    }

    // ── Final summary ───────────────────────────────────────────────────────

    let elapsed = start.elapsed();
    let mut results = all_results.into_inner().unwrap();
    let total = results.len();

    tag_playbook_results(&mut results, &args.max_gamma_bits, &args.q_mod_8);
    let results = finalize_results_with_mode(results, min_twist, ranking_mode, &args.max_gamma_bits, &args.q_mod_8);

    eprintln!();
    eprintln!("==========================================================================");
    eprintln!("Search complete in {:.1}s", elapsed.as_secs_f64());
    eprintln!("  Discriminants searched: {}", total_discriminants);
    eprintln!("  Total cycles found: {}", total);
    eprintln!(
        "  Passing (>={}-bit twist, deduplicated): {}",
        min_twist,
        results.len()
    );

    if !results.is_empty() {
        eprintln!();
        eprintln!("────────────────────────────────────────────────────────────────────────────────────────────");
        eprintln!(
            "{:>4}  {:>16}  {:>9}  {:>9}  {:>5}  {:>6}  {:>6}  {}",
            "Rank", "D", "Twist(H)", "Twist(S)", "Min", "Gamma", "q%8", "Tags"
        );
        eprintln!("────────────────────────────────────────────────────────────────────────────────────────────");
        for (i, r) in results.iter().take(args.top).enumerate() {
            let tags = if r.playbook_tags.is_empty() {
                String::new()
            } else {
                r.playbook_tags.join(", ")
            };
            eprintln!(
                "{:>4}  {:>16}  {:>9}  {:>9}  {:>5}  {:>6}  {:>6}  {}",
                i + 1,
                r.discriminant,
                r.twist_bits_curve_a,
                r.twist_bits_curve_b,
                r.min_twist_bits,
                r.gamma_bits,
                r.q_mod_8,
                tags,
            );
        }

        eprintln!();
        eprintln!("Best candidate:");
        let best = &results[0];
        eprintln!("  Discriminant: {}", best.discriminant);
        eprintln!("  Trace: {}", best.t);
        eprintln!("  p: {}", best.p);
        eprintln!("  q: {}", best.q);
        eprintln!("  q = 2^{} - {}", best.crandall_k, best.crandall_c);
        eprintln!("  gamma: {} bits", best.gamma_bits);
        eprintln!("  q mod 8: {}", best.q_mod_8);
        eprintln!("  Twist security (curve A): {} bits", best.twist_bits_curve_a);
        eprintln!("  Twist security (curve B): {} bits", best.twist_bits_curve_b);
    }

    // Write final top/winner files
    write_json_atomic(&paths.top, &results.iter().take(args.top).collect::<Vec<_>>());
    if let Some(best) = results.first() {
        write_json_atomic(&paths.winner, best);
    }

    eprintln!();
    eprintln!("Output files:");
    eprintln!("  GPU records:  {}", paths.gpu);
    eprintln!("  Scanned:      {}", paths.scanned);
    eprintln!("  Filtered:     {}", paths.filtered);
    eprintln!("  Top {}:       {}", args.top, paths.top);
    eprintln!("  Winner:       {}", paths.winner);
    eprintln!("  Checkpoint:   {}", paths.checkpoint);

    if profile_stages {
        let profile = profile_totals.into_inner().unwrap();
        print_stage_profile(&profile);
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use num_integer::Integer;
    use num_traits::{One, Zero};

    use crate::math::sieve::{is_squarefree, P};
    use crate::math::jacobi::{jacobi_symbol_discriminant, passes_local_solve_sieve_discriminant};
    use crate::math::cornacchia::{sqrt_mod, solve_4p};

    use super::*;

    #[test]
    fn test_merge_ranges() {
        let mut r = vec![(1, 5), (3, 8), (10, 15)];
        merge_ranges(&mut r);
        assert_eq!(r, vec![(1, 8), (10, 15)]);

        let mut r = vec![(1, 5), (6, 10)];
        merge_ranges(&mut r);
        assert_eq!(r, vec![(1, 10)]);

        let mut r = vec![(5, 10), (1, 3)];
        merge_ranges(&mut r);
        assert_eq!(r, vec![(1, 3), (5, 10)]);
    }

    #[test]
    fn test_compute_gaps() {
        let gaps = compute_gaps((1, 20), &[(5, 10), (15, 18)]);
        assert_eq!(gaps, vec![(1, 4), (11, 14), (19, 20)]);

        let gaps = compute_gaps((1, 10), &[(1, 10)]);
        assert_eq!(gaps, vec![]);

        let gaps = compute_gaps((5, 15), &[(1, 7), (12, 20)]);
        assert_eq!(gaps, vec![(8, 11)]);

        let gaps = compute_gaps((1, 10), &[]);
        assert_eq!(gaps, vec![(1, 10)]);
    }

    #[test]
    fn test_derive_output_paths() {
        let p = derive_output_paths("candidates.jsonl");
        assert_eq!(p.gpu, "candidates_gpu.jsonl");
        assert_eq!(p.filtered, "candidates_filtered.jsonl");
        assert_eq!(p.top, "candidates_top.json");
        assert_eq!(p.checkpoint, "candidates_checkpoint.json");

        let p = derive_output_paths("output/results.json");
        assert_eq!(p.gpu, "output/results_gpu.jsonl");
        assert_eq!(p.winner, "output/results_winner.json");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.0), "00d00h00m00s");
        assert_eq!(format_duration(61.0), "00d00h01m01s");
        assert_eq!(format_duration(3661.0), "00d01h01m01s");
        assert_eq!(format_duration(86400.0 + 3600.0 + 60.0 + 1.0), "01d01h01m01s");
        assert_eq!(format_duration(-1.0), "??d??h??m??s");
    }

    #[test]
    fn debug_twist_factorization() {
        use crate::math::twist::{twist_security_bits, pollard_rho_factor, pari_factor, bigint_to_rug};

        let p = &*P;
        let d: u64 = 15203;
        let d_big = BigInt::from(d);
        let neg_d = (p - &d_big).mod_floor(p);
        let root = sqrt_mod(&neg_d, p).unwrap();
        let solutions = solve_4p(d, p);
        assert!(!solutions.is_empty());

        let (t, _s) = &solutions[0];
        let q = p + BigInt::one() - t;
        println!("D={d}, t={t}");
        println!("q = {q}");
        println!("q bits = {}", q.bits());

        // Twist orders
        let t_b = BigInt::from(2u64) - t;
        let twist_a = p + BigInt::one() + t;
        let twist_b = &q + BigInt::one() + &t_b;
        println!("twist_a = {twist_a}");
        println!("twist_b = {twist_b}");

        // Try factoring twist_a
        println!("\n=== Factoring twist_a ===");
        let (bits_a, exact_a) = twist_security_bits(&twist_a);
        println!("twist_a: bits={bits_a}, exact={exact_a}");

        // Try factoring twist_b
        println!("\n=== Factoring twist_b ===");
        let (bits_b, exact_b) = twist_security_bits(&twist_b);
        println!("twist_b: bits={bits_b}, exact={exact_b}");

        // Manually test PARI on the cofactor
        println!("\n=== Manual PARI test ===");
        let rug_twist_a = bigint_to_rug(&twist_a);
        let pari_result = pari_factor(&rug_twist_a);
        println!("pari_factor(twist_a) = {:?}", pari_result);
    }

    #[test]
    fn debug_discriminant_91515() {
        let d: u64 = 91515;
        let p = &*P;

        println!("D = {d}");
        println!("  d % 4 = {}", d % 4);
        println!("  squarefree: {}", is_squarefree(d));
        println!("  jacobi(-D, p) = {}", jacobi_symbol_discriminant(d));
        println!("  passes local solve sieve: {}", passes_local_solve_sieve_discriminant(d));

        let d_big = BigInt::from(d);
        let neg_d = (p - &d_big).mod_floor(p);
        let sqrt_neg_d = sqrt_mod(&neg_d, p);
        match &sqrt_neg_d {
            Some(root) => {
                println!("  sqrt(-D) mod p exists");
                let root_sq = root.modpow(&BigInt::from(2u64), p);
                assert_eq!(root_sq, neg_d);
                println!("  root verified");
                println!("  root is even: {}", (root % BigInt::from(2u64)).is_zero());
            }
            None => panic!("No sqrt exists!"),
        }

        let solutions = solve_4p(d, p);
        println!("  solve_4p solutions: {}", solutions.len());
        for (i, (t, s)) in solutions.iter().enumerate() {
            let lhs = t * t + &d_big * s * s;
            let four_p = p * 4u64;
            assert_eq!(lhs, four_p);
            println!("    solution {i}: t = {t}");
            println!("    t is even: {}", (t % BigInt::from(2u64)).is_zero());
            println!("    t bits: {}", t.bits());
        }

        // Trace through the odd path manually
        let root = sqrt_neg_d.unwrap();
        println!("\n  === Odd path trace (d % 4 == 3) ===");
        let r0 = if (root.clone() % BigInt::from(2u64)).is_zero() {
            println!("  root is even, using p - root");
            p - &root
        } else {
            println!("  root is odd, using root directly");
            root.clone()
        };
        println!("  r0 = {r0}");
        println!("  r0 is odd: {}", (r0.clone() % BigInt::from(2u64)) == BigInt::one());

        let four_p = p * 4u64;
        let sqrt_4p = {
            let mut x = BigInt::one() << ((four_p.bits() + 1) / 2);
            loop {
                let y = (&x + &four_p / &x) >> 1;
                if y >= x { break x; }
                x = y;
            }
        };
        let two_p = p * 2u64;
        println!("  2p bits: {}", two_p.bits());
        println!("  sqrt(4p) bits: {}", sqrt_4p.bits());

        let mut a = two_p;
        let mut b = r0;
        let mut steps = 0;
        while b > sqrt_4p {
            let temp = a.mod_floor(&b);
            a = b;
            b = temp;
            steps += 1;
        }
        println!("  Euclidean steps: {steps}");
        println!("  t = b = {b}");
        println!("  t bits: {}", b.bits());
        println!("  t is odd: {}", (b.clone() % BigInt::from(2u64)) == BigInt::one());

        let rem = &four_p - &b * &b;
        println!("  4p - t^2 = {rem}");
        println!("  (4p - t^2) % d = {}", &rem % &d_big);

        // Now trace through the even path to confirm it fails
        println!("\n  === Even path trace ===");
        let root2 = sqrt_mod(&neg_d, p).unwrap();
        let r0_even = if &root2 * 2 < *p {
            p - &root2
        } else {
            root2.clone()
        };
        let sqrt_p = {
            let mut x = BigInt::one() << ((p.bits() + 1) / 2);
            loop {
                let y = (&x + p / &x) >> 1;
                if y >= x { break x; }
                x = y;
            }
        };
        let mut a2 = p.clone();
        let mut b2 = r0_even;
        let mut steps2 = 0;
        while b2 > sqrt_p {
            let temp = a2.mod_floor(&b2);
            a2 = b2;
            b2 = temp;
            steps2 += 1;
        }
        println!("  Euclidean steps: {steps2}");
        println!("  b = {b2}");
        let rem2 = p - &b2 * &b2;
        println!("  p - b^2 = {rem2}");
        println!("  (p - b^2) % d = {}", &rem2 % &d_big);
        let is_div = (&rem2 % &d_big).is_zero() && rem2 > BigInt::zero();
        println!("  Even path produces solution: {is_div}");
    }
}
