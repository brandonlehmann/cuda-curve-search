use rug::integer::IsPrime;
use rug::Integer;
use std::time::Duration;

use crate::math::types::CycleResult;
use crate::math::sieve::sieve_primes;
use crate::math::twist::{pollard_rho_factor, pari_factor};

#[derive(Clone, Debug)]
pub struct VerificationResult {
    pub discriminant: u64,
    pub q_prime: bool,
    pub class_number: u64,
    pub class_verdict: &'static str,
    pub twist_a_bits: u32,
    pub twist_a_exact: bool,
    pub twist_b_bits: u32,
    pub twist_b_exact: bool,
    pub min_twist_verified: u32,
    pub both_exact: bool,
    pub mov_ok: bool,
    pub c_bits: u32,
    pub notes: Vec<String>,
}

impl VerificationResult {
    pub fn verdict(&self) -> &'static str {
        if self.notes.iter().any(|n| n.contains("NOT PRIME") || n.contains("MOV")) {
            "FAIL"
        } else if self.notes.is_empty() {
            "PASS"
        } else {
            "WARN"
        }
    }
}

lazy_static::lazy_static! {
    static ref P: Integer = (Integer::from(1u64) << 255) - Integer::from(19u64);
    static ref VERIFY_PRIMES: Vec<u64> = sieve_primes(2_000_000);
    static ref TRIAL_PRIMES: Vec<u64> = sieve_primes(1_000_000);
}

fn kronecker_euler(a: i64, p: u64) -> i64 {
    let p_int = Integer::from(p);
    let a_mod = {
        let a_int = Integer::from(a);
        let r = a_int % &p_int;
        if r < 0 { r + &p_int } else { r }
    };
    if a_mod == 0 {
        return 0;
    }
    let exp = (Integer::from(&p_int) - 1u32) / 2u32;
    let result = Integer::from(&a_mod).pow_mod(&exp, &p_int).unwrap();
    if result == 1 {
        1
    } else if result == Integer::from(&p_int) - 1u32 {
        -1
    } else {
        0
    }
}

pub fn estimate_class_number(d: u64) -> u64 {
    let neg_d = -(d as i64);
    let mut log_l = 0.0f64;
    for &p in VERIFY_PRIMES.iter() {
        let chi = kronecker_euler(neg_d, p);
        if chi != 0 {
            log_l -= (1.0 - (chi as f64) / (p as f64)).ln();
        }
    }
    let l_val = log_l.exp();
    let h = (d as f64).sqrt() / std::f64::consts::PI * l_val;
    h.round().max(1.0) as u64
}

pub fn class_number_verdict(h: u64) -> &'static str {
    if h < 10_000 {
        "TRIVIAL"
    } else if h < 100_000 {
        "FEASIBLE"
    } else if h < 1_000_000 {
        "EXPENSIVE"
    } else {
        "VERY EXPENSIVE"
    }
}

fn check_embedding_degree(base: &Integer, modulus: &Integer, max_k: u64) -> Option<u64> {
    let base_mod = Integer::from(base) % modulus;
    let mut acc = base_mod.clone();
    for k in 1..=max_k {
        if acc == 1 {
            return Some(k);
        }
        acc = (acc * &base_mod) % modulus;
    }
    None
}

fn gmp_is_prime(n: &Integer) -> bool {
    n.is_probably_prime(25) != IsPrime::No
}

pub fn enhanced_twist_security(twist_order: &Integer, rho_seconds: f64) -> (u32, bool) {
    if *twist_order <= 1 {
        return (0, true);
    }
    let mut cofactor = twist_order.clone();
    for &f in TRIAL_PRIMES.iter() {
        let f_int = Integer::from(f);
        while Integer::from(&cofactor % &f_int) == 0 {
            cofactor /= &f_int;
        }
        if Integer::from(&f_int * &f_int) > cofactor {
            break;
        }
    }
    if cofactor == 1 {
        return (0, true);
    }
    if gmp_is_prime(&cofactor) {
        return (cofactor.significant_bits(), true);
    }
    let budget = Duration::from_secs_f64(rho_seconds);
    if let Some(bits) = pollard_rho_factor(&cofactor, budget) {
        return (bits, true);
    }
    if let Some(bits) = pari_factor(&cofactor) {
        return (bits, true);
    }
    let bits = cofactor.significant_bits();
    let exact = cofactor == 1 || gmp_is_prime(&cofactor);
    (bits, exact)
}

pub fn verify_candidate(rec: &CycleResult, rho_seconds: f64) -> VerificationResult {
    let q: Integer = rec.q.parse().expect("invalid q");
    let trace_a: Integer = rec.t.parse().expect("invalid t");
    let trace_b: Integer = Integer::from(2) - &trace_a;

    let q_prime = gmp_is_prime(&q);
    let h = estimate_class_number(rec.discriminant_u64);
    let h_verdict = class_number_verdict(h);

    let twist_a_order = Integer::from(&*P) + 1u32 + &trace_a;
    let twist_b_order = Integer::from(&q) + 1u32 + &trace_b;

    let (tw_a_bits, tw_a_exact) = enhanced_twist_security(&twist_a_order, rho_seconds);
    let (tw_b_bits, tw_b_exact) = enhanced_twist_security(&twist_b_order, rho_seconds);

    let mov_a = check_embedding_degree(&*P, &q, 10_000);
    let mov_b = check_embedding_degree(&q, &*P, 10_000);
    let mov_ok = mov_a.is_none() && mov_b.is_none();

    let min_tw = tw_a_bits.min(tw_b_bits);
    let both_exact = tw_a_exact && tw_b_exact;

    let c_int: Integer = rec.crandall_c.parse().unwrap_or_default();
    let c_bits = if c_int == 0 { 0 } else { c_int.significant_bits() };

    let mut notes = Vec::new();
    if !q_prime {
        notes.push("q NOT PRIME".to_string());
    }
    if h >= 1_000_000 {
        notes.push(format!("class number ~{h} (construction very expensive)"));
    }
    if !mov_ok {
        notes.push("MOV embedding degree too small".to_string());
    }
    if !both_exact {
        notes.push("twist security approximate".to_string());
    }

    VerificationResult {
        discriminant: rec.discriminant_u64,
        q_prime,
        class_number: h,
        class_verdict: h_verdict,
        twist_a_bits: tw_a_bits,
        twist_a_exact: tw_a_exact,
        twist_b_bits: tw_b_bits,
        twist_b_exact: tw_b_exact,
        min_twist_verified: min_tw,
        both_exact,
        mov_ok,
        c_bits,
        notes,
    }
}
