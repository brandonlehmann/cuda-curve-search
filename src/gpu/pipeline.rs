use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use serde::{Serialize, Deserialize};

use crate::math::types::*;
use crate::math::u256::u256_to_bigint;
use crate::math::ranking::build_cycle_result_from_trace;
use super::context::GpuContext;
use super::workspace::*;
use super::kernels;

#[derive(Clone, Serialize, Deserialize)]
pub struct CompactRecord {
    pub discriminant: u64,
    pub trace: [u64; 4],
    pub positive_q_passes: bool,
    pub negative_q_passes: bool,
}

fn decode_compact_records(stats: &[u32], records: &[u64]) -> Vec<CompactRecord> {
    let record_count = stats[COMPACT_STAT_RECORD_COUNT] as usize;
    (0..record_count)
        .map(|i| {
            let offset = i * COMPACT_RECORD_LIMBS;
            CompactRecord {
                discriminant: records[offset],
                trace: [
                    records[offset + 1],
                    records[offset + 2],
                    records[offset + 3],
                    records[offset + 4],
                ],
                positive_q_passes: (records[offset + 5] & COMPACT_FLAG_POSITIVE_Q) != 0,
                negative_q_passes: (records[offset + 5] & COMPACT_FLAG_NEGATIVE_Q) != 0,
            }
        })
        .collect()
}

pub fn finalize_one_record(rec: &CompactRecord) -> Vec<CycleResult> {
    let t_bigint = u256_to_bigint(&rec.trace);
    let mut results = Vec::new();
    if rec.positive_q_passes {
        if let Some(r) = build_cycle_result_from_trace(rec.discriminant, t_bigint.clone(), None) {
            results.push(r);
        }
    }
    if rec.negative_q_passes {
        if let Some(r) = build_cycle_result_from_trace(rec.discriminant, -&t_bigint, None) {
            results.push(r);
        }
    }
    results
}

/// Live stats from the decoupled pipeline.
pub struct PipelineStats {
    pub gpu_d_scanned: u64,
    pub gpu_time_secs: f64,
    pub gpu_candidates_produced: u64,
    pub cpu_candidates_factored: u64,
    pub cpu_time_secs: f64,
    pub wall_time_secs: f64,
}

impl PipelineStats {
    pub fn gpu_d_per_sec(&self) -> f64 {
        if self.gpu_time_secs > 0.0 { self.gpu_d_scanned as f64 / self.gpu_time_secs } else { 0.0 }
    }
    pub fn cpu_cand_per_sec(&self) -> f64 {
        if self.cpu_time_secs > 0.0 { self.cpu_candidates_factored as f64 / self.cpu_time_secs } else { 0.0 }
    }
}

/// Scan GPU over multiple disjoint ranges, returning all CompactRecords found.
///
/// Callbacks:
/// - `on_gpu_progress(d_scanned, gpu_secs, chunk_d_per_sec, candidates_so_far, total_d)`
/// - `on_gpu_candidates(&[CompactRecord], chunk_start_d, chunk_end_d)`: called after each chunk
pub fn gpu_scan_ranges(
    gpu: &GpuContext,
    ranges: &[(u64, u64)],
    chunk_size: usize,
    mut on_gpu_progress: impl FnMut(u64, f64, f64, u64, u64),
    mut on_gpu_candidates: impl FnMut(&[CompactRecord], u64, u64),
) -> Result<(Vec<CompactRecord>, f64, u64), Box<dyn std::error::Error + Send + Sync>> {
    let total_d: u64 = ranges.iter().map(|(s, e)| e - s + 1).sum();
    let mut ws = GpuWorkspace::new(gpu, chunk_size)?;
    let mut all_records: Vec<CompactRecord> = Vec::new();
    let mut gpu_time: f64 = 0.0;
    let mut gpu_d: u64 = 0;

    for &(range_start, range_end) in ranges {
        let mut d = range_start;
        while d <= range_end {
            let count = ((range_end - d + 1) as usize).min(chunk_size);
            let chunk_start = d;
            let chunk_end = d + count as u64 - 1;
            let t0 = Instant::now();
            kernels::launch_prefilter_and_solve(gpu, &mut ws, d, count)?;
            let (stats, records) = kernels::read_results(&ws)?;
            let chunk_secs = t0.elapsed().as_secs_f64();
            let chunk_rate = if chunk_secs > 0.0 { count as f64 / chunk_secs } else { 0.0 };
            gpu_time += chunk_secs;
            gpu_d += count as u64;
            let new_records = decode_compact_records(&stats, &records);
            on_gpu_candidates(&new_records, chunk_start, chunk_end);
            all_records.extend(new_records);
            d += count as u64;
            on_gpu_progress(gpu_d, gpu_time, chunk_rate, all_records.len() as u64, total_d);
        }
    }

    Ok((all_records, gpu_time, gpu_d))
}

/// CPU twist factoring over CompactRecords, streaming CycleResults as they complete.
///
/// Returns (cpu_time_secs, records_processed).
pub fn process_records_cpu(
    records: Vec<CompactRecord>,
    mut on_result: impl FnMut(&CycleResult) + Send,
    mut on_record_done: impl FnMut(&CompactRecord) + Send,
    mut on_cpu_progress: impl FnMut(u64, u64),
) -> (f64, u64) {
    let total = records.len() as u64;
    if total == 0 {
        return (0.0, 0);
    }

    enum CpuMessage {
        Result(CycleResult),
        RecordDone(CompactRecord),
    }

    let (tx, rx) = mpsc::channel::<CpuMessage>();
    let cpu_done = Arc::new(AtomicU64::new(0));
    let cpu_done_clone = cpu_done.clone();

    let worker = std::thread::spawn(move || {
        let t0 = Instant::now();
        records.par_iter().for_each(|rec| {
            let results = finalize_one_record(rec);
            for r in results {
                let _ = tx.send(CpuMessage::Result(r));
            }
            let _ = tx.send(CpuMessage::RecordDone(rec.clone()));
            cpu_done_clone.fetch_add(1, Ordering::Relaxed);
        });
        drop(tx);
        t0.elapsed().as_secs_f64()
    });

    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(CpuMessage::Result(r)) => {
                on_result(&r);
                on_cpu_progress(cpu_done.load(Ordering::Relaxed), total);
            }
            Ok(CpuMessage::RecordDone(rec)) => {
                on_record_done(&rec);
                on_cpu_progress(cpu_done.load(Ordering::Relaxed), total);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                on_cpu_progress(cpu_done.load(Ordering::Relaxed), total);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let cpu_time = worker.join().unwrap_or(0.0);
    let processed = cpu_done.load(Ordering::Relaxed);
    (cpu_time, processed)
}

/// Run the GPU pipeline: GPU scan first, then stream CPU results as they complete.
///
/// Callbacks:
/// - `on_result`: called for each finalized CycleResult as soon as CPU finishes it
/// - `on_gpu_progress(d_scanned, gpu_secs, chunk_d_per_sec, candidates_so_far, total_d)`
/// - `on_gpu_done(d_scanned, gpu_secs, n_candidates)`
/// - `on_gpu_candidates(&[CompactRecord], chunk_start, chunk_end)`: called after each GPU chunk
/// - `on_cpu_progress(done, total)`: called periodically during CPU phase
pub fn search_range_gpu_pipeline(
    gpu: &GpuContext,
    start_d: u64,
    end_d: u64,
    chunk_size: usize,
    on_result: impl FnMut(&CycleResult) + Send,
    on_gpu_progress: impl FnMut(u64, f64, f64, u64, u64),
    mut on_gpu_done: impl FnMut(u64, f64, u64),
    on_gpu_candidates: impl FnMut(&[CompactRecord], u64, u64),
    on_record_done: impl FnMut(&CompactRecord) + Send,
    on_cpu_progress: impl FnMut(u64, u64),
) -> Result<PipelineStats, Box<dyn std::error::Error + Send + Sync>> {
    let wall_start = Instant::now();

    let (all_records, gpu_time, gpu_d) = gpu_scan_ranges(
        gpu, &[(start_d, end_d)], chunk_size, on_gpu_progress, on_gpu_candidates,
    )?;
    let total_candidates = all_records.len() as u64;

    on_gpu_done(gpu_d, gpu_time, total_candidates);

    let (cpu_time, cpu_processed) = process_records_cpu(
        all_records, on_result, on_record_done, on_cpu_progress,
    );

    Ok(PipelineStats {
        gpu_d_scanned: gpu_d,
        gpu_time_secs: gpu_time,
        gpu_candidates_produced: total_candidates,
        cpu_candidates_factored: cpu_processed,
        cpu_time_secs: cpu_time,
        wall_time_secs: wall_start.elapsed().as_secs_f64(),
    })
}
