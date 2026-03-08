use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::context::GpuContext;
use super::workspace::*;

const DEFAULT_BLOCK_SIZE: u32 = 256;
const HIGH_D_RANGE_SQUAREFREE_THRESHOLD: u64 = 100_000_000;

pub fn launch_prefilter_and_solve(
    gpu: &GpuContext,
    ws: &mut GpuWorkspace,
    start_d: u64,
    count: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if count == 0 {
        return Ok(());
    }

    let count_u32 = count as u32;
    let prefilter_grid = count_u32.div_ceil(DEFAULT_BLOCK_SIZE);
    let queued_solve_grid = (gpu.sm_count * 4).max(1);
    let stream = ws.stream.clone();

    // Zero out stats
    stream.memset_zeros(&mut ws.compact_stats_device)?;

    let high_d = start_d.saturating_add(count as u64) >= HIGH_D_RANGE_SQUAREFREE_THRESHOLD;

    if high_d {
        // High-D path: mark squareful bits then compact
        stream.memset_zeros(&mut ws.squareful_bits_device)?;

        let squarefree_grid = gpu.prime_square_count.div_ceil(DEFAULT_BLOCK_SIZE);

        let mark_fn = gpu.module.load_function("mark_squareful_range_bits")?;
        let config = LaunchConfig {
            grid_dim: (squarefree_grid, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&mark_fn)
                .arg(&start_d)
                .arg(&count_u32)
                .arg(&gpu.prime_squares)
                .arg(&gpu.prime_square_count)
                .arg(&mut ws.squareful_bits_device)
                .launch(config)?;
        }

        let compact_fn = gpu.module.load_function("solve_stage_compact_from_squarefree_bits")?;
        let config = LaunchConfig {
            grid_dim: (prefilter_grid, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&compact_fn)
                .arg(&start_d)
                .arg(&count_u32)
                .arg(&ws.squareful_bits_device)
                .arg(&mut ws.compact_stats_device)
                .arg(&mut ws.discriminants_device)
                .launch(config)?;
        }
    } else {
        // Low-D path: combined prefilter
        let prefilter_fn = gpu.module.load_function("solve_stage_prefilter_range_compact")?;
        let config = LaunchConfig {
            grid_dim: (prefilter_grid, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&prefilter_fn)
                .arg(&start_d)
                .arg(&count_u32)
                .arg(&gpu.prime_squares)
                .arg(&gpu.prime_square_count)
                .arg(&mut ws.compact_stats_device)
                .arg(&mut ws.discriminants_device)
                .launch(config)?;
        }
    }

    // Launch solver kernel
    let solve_fn = gpu.module.load_function("solve_verified_qsieve_compact_queued")?;
    let config = LaunchConfig {
        grid_dim: (queued_solve_grid, 1, 1),
        block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&solve_fn)
            .arg(&ws.discriminants_device)
            .arg(&count_u32)
            .arg(&gpu.primality_sieve_primes)
            .arg(&gpu.primality_sieve_prime_count)
            .arg(&gpu.sqrt_m1)
            .arg(&gpu.p_u256)
            .arg(&gpu.p_plus_one_u256)
            .arg(&gpu.p_u320)
            .arg(&gpu.four_p_u320)
            .arg(&gpu.two_p_u256)
            .arg(&gpu.half_p_u256)
            .arg(&gpu.sqrt_p_u256)
            .arg(&gpu.sqrt_four_p_u256)
            .arg(&mut ws.compact_stats_device)
            .arg(&mut ws.compact_records_device)
            .launch(config)?;
    }

    stream.synchronize()?;
    Ok(())
}

pub fn read_results(ws: &GpuWorkspace) -> Result<(Vec<u32>, Vec<u64>), Box<dyn std::error::Error + Send + Sync>> {
    let stream = &ws.stream;
    let stats = stream.clone_dtoh(&ws.compact_stats_device)?;
    let record_count = stats[COMPACT_STAT_RECORD_COUNT] as usize;

    let records = if record_count > 0 {
        // Read only the used portion
        let total_limbs = record_count * COMPACT_RECORD_LIMBS;
        let view = ws.compact_records_device.slice(0..total_limbs);
        stream.clone_dtoh(&view)?
    } else {
        Vec::new()
    };

    Ok((stats, records))
}
