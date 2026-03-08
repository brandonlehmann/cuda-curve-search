use rayon::prelude::*;

use crate::math::types::*;
use crate::math::sieve::is_squarefree;
use crate::math::jacobi::{jacobi_symbol_discriminant, passes_local_solve_sieve_discriminant};
use crate::math::cornacchia::solve_4p_single_with_backend;
use crate::math::ranking::{build_cycle_results, finalize_results};

fn search_discriminant_after_squarefree_with_backend<B: Solve4pKernelBackend>(
    d: u64,
    backend: &B,
) -> Vec<CycleResult> {
    if jacobi_symbol_discriminant(d) != 1 || !passes_local_solve_sieve_discriminant(d) {
        return Vec::new();
    }
    search_discriminant_after_prefilter_with_backend(d, backend)
}

fn search_discriminant_after_prefilter_with_backend<B: Solve4pKernelBackend>(
    d: u64,
    backend: &B,
) -> Vec<CycleResult> {
    let solutions = solve_4p_single_with_backend(d, backend);
    if solutions.is_empty() {
        return Vec::new();
    }
    build_cycle_results(d, &solutions, None)
}

pub fn search_discriminant_with_backend<B: Solve4pKernelBackend>(
    d: u64,
    backend: &B,
) -> Vec<CycleResult> {
    if !is_squarefree(d) {
        return Vec::new();
    }
    search_discriminant_after_squarefree_with_backend(d, backend)
}

pub fn search_discriminant(d: u64) -> Vec<CycleResult> {
    search_discriminant_with_backend(d, &CPU_BIGINT_SOLVE4P_BACKEND)
}

pub fn search_discriminant_profiled_with_backend<B: Solve4pKernelBackend>(
    d: u64,
    backend: &B,
) -> (Vec<CycleResult>, SearchProfile) {
    let mut profile = SearchProfile {
        discriminants_checked: 1,
        ..SearchProfile::default()
    };

    let squarefree_start = std::time::Instant::now();
    let squarefree = is_squarefree(d);
    profile.squarefree_time += squarefree_start.elapsed();
    if !squarefree {
        return (Vec::new(), profile);
    }
    profile.squarefree_passes += 1;

    let jacobi_start = std::time::Instant::now();
    let jacobi_ok = jacobi_symbol_discriminant(d) == 1;
    profile.jacobi_time += jacobi_start.elapsed();
    if !jacobi_ok {
        return (Vec::new(), profile);
    }
    profile.jacobi_passes += 1;

    if !passes_local_solve_sieve_discriminant(d) {
        return (Vec::new(), profile);
    }

    let solve_start = std::time::Instant::now();
    let solutions = solve_4p_single_with_backend(d, backend);
    profile.solve_4p_time += solve_start.elapsed();
    if solutions.is_empty() {
        return (Vec::new(), profile);
    }
    profile.solve_4p_passes += 1;

    let results = build_cycle_results(d, &solutions, Some(&mut profile));
    (results, profile)
}

pub fn search_discriminant_profiled(d: u64) -> (Vec<CycleResult>, SearchProfile) {
    search_discriminant_profiled_with_backend(d, &CPU_BIGINT_SOLVE4P_BACKEND)
}

pub fn search_range_cpu(
    start_d: u64,
    end_d: u64,
    min_twist: u32,
) -> SearchSummary {
    let results: Vec<CycleResult> = (start_d..=end_d)
        .into_par_iter()
        .flat_map(|d| search_discriminant(d))
        .collect();

    let total_cycles_found = results.len();
    let results = finalize_results(results, min_twist, &[], &[]);

    SearchSummary {
        total_discriminants: end_d - start_d + 1,
        total_cycles_found,
        results,
    }
}
