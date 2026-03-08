use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Instant;

use crate::math::sieve::P;
use crate::math::types::*;
use crate::math::primality::is_probable_prime;
use crate::math::twist::twist_security;

pub fn bigint_bits(n: &BigInt) -> u32 {
    if n.is_zero() {
        return 0;
    }
    n.abs().bits() as u32
}

fn parse_decimal_bigint(value: &str) -> Option<BigInt> {
    BigInt::parse_bytes(value.as_bytes(), 10)
}

pub fn cycle_result_c_bits(result: &CycleResult) -> u32 {
    result.gamma_bits
}

pub fn passes_playbook_filters(result: &CycleResult, max_gamma_bits: &[u32], allowed_q_mod_8: &[u8]) -> bool {
    let gb = result.gamma_bits;
    // Use the loosest threshold
    let max_threshold = max_gamma_bits.iter().copied().max().unwrap_or(u32::MAX);
    if gb > max_threshold {
        return false;
    }
    if !allowed_q_mod_8.is_empty() && !allowed_q_mod_8.contains(&result.q_mod_8) {
        return false;
    }
    true
}

pub fn tag_playbook_results(results: &mut [CycleResult], max_gamma_bits: &[u32], allowed_q_mod_8: &[u8]) {
    for result in results.iter_mut() {
        let gb = result.gamma_bits;
        let q_ok = allowed_q_mod_8.is_empty() || allowed_q_mod_8.contains(&result.q_mod_8);
        if !q_ok {
            continue;
        }
        for &threshold in max_gamma_bits {
            if gb <= threshold {
                let q_mod_str: Vec<String> = allowed_q_mod_8.iter().map(|v| v.to_string()).collect();
                result.playbook_tags.push(format!(
                    "gamma<={},qmod8=[{}]",
                    threshold,
                    q_mod_str.join(",")
                ));
            }
        }
    }
}

fn balanced_ranking_score(result: &CycleResult) -> i64 {
    result.min_twist_bits as i64 * 100 - result.gamma_bits as i64
}

fn is_fully_verified(r: &CycleResult) -> bool {
    r.twist_fully_factored_curve_a && r.twist_fully_factored_curve_b
}

fn compare_cycle_results(a: &CycleResult, b: &CycleResult, mode: RankingMode) -> Ordering {
    let a_verified = is_fully_verified(a);
    let b_verified = is_fully_verified(b);
    if a_verified != b_verified {
        return if b_verified { Ordering::Greater } else { Ordering::Less };
    }
    match mode {
        RankingMode::Security => b
            .min_twist_bits
            .cmp(&a.min_twist_bits)
            .then_with(|| a.gamma_bits.cmp(&b.gamma_bits))
            .then_with(|| a.discriminant_u64.cmp(&b.discriminant_u64)),
        RankingMode::Balanced => balanced_ranking_score(b)
            .cmp(&balanced_ranking_score(a))
            .then_with(|| b.min_twist_bits.cmp(&a.min_twist_bits))
            .then_with(|| a.gamma_bits.cmp(&b.gamma_bits))
            .then_with(|| a.discriminant_u64.cmp(&b.discriminant_u64)),
    }
}

pub fn cycle_result_is_better(
    candidate: &CycleResult,
    current_best: &CycleResult,
    mode: RankingMode,
) -> bool {
    compare_cycle_results(candidate, current_best, mode) == Ordering::Less
}

fn crandall_info(q: &BigInt) -> (u32, BigInt) {
    let k = bigint_bits(q);
    let two_k = BigInt::one() << k;
    let c = &two_k - q;
    (k, c)
}

pub fn build_cycle_result_from_trace(
    d: u64,
    t_a: BigInt,
    mut profile: Option<&mut SearchProfile>,
) -> Option<CycleResult> {
    if let Some(profile) = profile.as_deref_mut() {
        profile.q_candidates += 1;
    }
    let q_start = profile.as_ref().map(|_| Instant::now());
    let q = &*P + 1u64 - &t_a;
    let q_is_prime = q > BigInt::from(2u64) && is_probable_prime(&q);
    if let (Some(profile), Some(q_start)) = (profile.as_deref_mut(), q_start) {
        profile.q_prime_time += q_start.elapsed();
    }
    if !q_is_prime {
        return None;
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.prime_q_passes += 1;
    }
    build_cycle_result_from_trace_with_known_prime_q(
        d, t_a, q, profile,
    )
}

pub fn build_cycle_result_from_trace_with_known_prime_q(
    d: u64,
    t_a: BigInt,
    q: BigInt,
    mut profile: Option<&mut SearchProfile>,
) -> Option<CycleResult> {
    let twist_start = profile.as_ref().map(|_| Instant::now());
    let twist_a_order = &*P + 1u64 + &t_a;
    let t_b = BigInt::from(2u64) - &t_a;
    let twist_b_order = &q + 1u64 + &t_b;
    let tw_a = twist_security(&twist_a_order);
    let tw_b = twist_security(&twist_b_order);
    if let (Some(profile), Some(twist_start)) = (profile.as_deref_mut(), twist_start) {
        profile.twist_time += twist_start.elapsed();
        profile.twist_candidates += 1;
    }
    // Mandatory full factorization: reject if either twist couldn't be fully factored
    if !tw_a.exact || !tw_b.exact {
        return None;
    }
    let (k, c) = crandall_info(&q);
    let q_mod_8 = (&q % BigInt::from(8u64))
        .to_u64()
        .map(|v| v as u8)
        .unwrap_or(0);
    let g_bits = bigint_bits(&c);
    Some(CycleResult {
        discriminant: d.to_string(),
        discriminant_u64: d,
        t: t_a.abs().to_string(),
        q: q.to_string(),
        gamma: c.abs().to_string(),
        gamma_bits: g_bits,
        q_mod_8,
        twist_bits_curve_a: tw_a.bits,
        twist_bits_curve_b: tw_b.bits,
        twist_factor_curve_a: tw_a.factor_string,
        twist_factor_curve_b: tw_b.factor_string,
        twist_fully_factored_curve_a: tw_a.exact,
        twist_fully_factored_curve_b: tw_b.exact,
        min_twist_bits: tw_a.bits.min(tw_b.bits),
        p: P.to_string(),
        crandall_k: k,
        crandall_c: c.to_string(),
        rank: None,
        playbook_tags: Vec::new(),
    })
}

pub fn build_cycle_result_from_trace_with_known_prime_q_and_twists(
    d: u64,
    t_a: BigInt,
    q: BigInt,
    tw_a_bits: u32,
    tw_a_exact: bool,
    tw_b_bits: u32,
    tw_b_exact: bool,
) -> CycleResult {
    let (k, c) = crandall_info(&q);
    let q_mod_8 = (&q % BigInt::from(8u64))
        .to_u64()
        .map(|v| v as u8)
        .unwrap_or(0);
    let g_bits = bigint_bits(&c);
    CycleResult {
        discriminant: d.to_string(),
        discriminant_u64: d,
        t: t_a.abs().to_string(),
        q: q.to_string(),
        gamma: c.abs().to_string(),
        gamma_bits: g_bits,
        q_mod_8,
        twist_bits_curve_a: tw_a_bits,
        twist_bits_curve_b: tw_b_bits,
        twist_factor_curve_a: String::new(),
        twist_factor_curve_b: String::new(),
        twist_fully_factored_curve_a: tw_a_exact,
        twist_fully_factored_curve_b: tw_b_exact,
        min_twist_bits: tw_a_bits.min(tw_b_bits),
        p: P.to_string(),
        crandall_k: k,
        crandall_c: c.to_string(),
        rank: None,
        playbook_tags: Vec::new(),
    }
}

pub fn build_cycle_result_for_sign(
    d: u64,
    solution: &Solve4pSolution,
    sign: i64,
    profile: Option<&mut SearchProfile>,
) -> Option<CycleResult> {
    build_cycle_result_from_trace(d, &solution.t * sign, profile)
}

pub fn build_cycle_results(
    d: u64,
    solutions: &[Solve4pSolution],
    mut profile: Option<&mut SearchProfile>,
) -> Vec<CycleResult> {
    let mut results = Vec::new();
    for solution in solutions {
        for sign in &[1i64, -1i64] {
            if let Some(result) =
                build_cycle_result_for_sign(d, solution, *sign, profile.as_deref_mut())
            {
                results.push(result);
            }
        }
    }
    results
}

pub fn finalize_results_with_mode(
    results: Vec<CycleResult>,
    min_twist: u32,
    mode: RankingMode,
    max_gamma_bits: &[u32],
    allowed_q_mod_8: &[u8],
) -> Vec<CycleResult> {
    let mut best_by_q: HashMap<String, CycleResult> = HashMap::new();
    for result in results
        .into_iter()
        .filter(|result| result.min_twist_bits >= min_twist && passes_playbook_filters(result, max_gamma_bits, allowed_q_mod_8))
    {
        match best_by_q.get_mut(&result.q) {
            Some(current_best) => {
                if cycle_result_is_better(&result, current_best, mode) {
                    *current_best = result;
                }
            }
            None => {
                best_by_q.insert(result.q.clone(), result);
            }
        }
    }
    let mut ranked: Vec<CycleResult> = best_by_q.into_values().collect();
    ranked.sort_by(|a, b| compare_cycle_results(a, b, mode));
    for (i, result) in ranked.iter_mut().enumerate() {
        result.rank = Some(i as u32 + 1);
    }
    ranked
}

pub fn finalize_results(results: Vec<CycleResult>, min_twist: u32, max_gamma_bits: &[u32], allowed_q_mod_8: &[u8]) -> Vec<CycleResult> {
    finalize_results_with_mode(results, min_twist, RankingMode::Security, max_gamma_bits, allowed_q_mod_8)
}
