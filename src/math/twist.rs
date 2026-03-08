use num_bigint::BigInt;
use num_traits::{One, Zero};
use rug::integer::IsPrime;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::math::sieve::SMALL_PRIMES;
use crate::math::primality::is_probable_prime;
use crate::math::ranking::bigint_bits;

fn rug_is_prime(n: &rug::Integer) -> bool {
    n.is_probably_prime(25) != IsPrime::No
}

pub fn bigint_to_rug(n: &BigInt) -> rug::Integer {
    rug::Integer::parse(n.to_string())
        .expect("valid decimal")
        .into()
}

/// Global configurable Pollard-rho budget (seconds per twist order).
/// Set from CLI via `set_rho_budget()`.
static RHO_BUDGET_SECS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(
    // Default: 2 seconds, stored as f64 bits
    2.0f64.to_bits()
);

pub fn set_rho_budget(seconds: f64) {
    RHO_BUDGET_SECS.store(seconds.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

fn rho_budget() -> Duration {
    let bits = RHO_BUDGET_SECS.load(std::sync::atomic::Ordering::Relaxed);
    Duration::from_secs_f64(f64::from_bits(bits))
}

/// Result of twist security analysis, including the factorization.
pub struct TwistSecurityResult {
    /// Bits of the largest prime factor of the twist order
    pub bits: u32,
    /// Whether the twist order was fully factored
    pub exact: bool,
    /// Factorization string, e.g. "3 * 7 * 123456789..."
    pub factor_string: String,
}

pub fn twist_security_bits(n: &BigInt) -> (u32, bool) {
    let r = twist_security(n);
    (r.bits, r.exact)
}

pub fn twist_security(n: &BigInt) -> TwistSecurityResult {
    if *n <= BigInt::one() {
        return TwistSecurityResult { bits: 0, exact: true, factor_string: n.to_string() };
    }
    if is_probable_prime(n) {
        return TwistSecurityResult { bits: bigint_bits(n), exact: true, factor_string: n.to_string() };
    }

    // Collect small factors
    let mut factors: Vec<BigInt> = Vec::new();
    let mut cofactor = n.clone();
    for &f in SMALL_PRIMES.iter() {
        let f_big = BigInt::from(f);
        while (&cofactor % &f_big).is_zero() {
            cofactor /= &f_big;
            factors.push(f_big.clone());
        }
        if &f_big * &f_big > cofactor {
            break;
        }
    }

    if cofactor == BigInt::one() {
        let fs = format_factors(&factors);
        return TwistSecurityResult { bits: 0, exact: true, factor_string: fs };
    }

    if is_probable_prime(&cofactor) {
        let bits = bigint_bits(&cofactor);
        factors.push(cofactor);
        let fs = format_factors(&factors);
        return TwistSecurityResult { bits, exact: true, factor_string: fs };
    }

    // Try advanced factoring on the composite cofactor
    let rug_cofactor = bigint_to_rug(&cofactor);

    if let Some(extra_factors) = pollard_rho_full_factor(&rug_cofactor, rho_budget()) {
        for f in &extra_factors {
            factors.push(rug_to_bigint(f));
        }
        let largest = extra_factors.iter().max().unwrap();
        let bits = largest.significant_bits();
        let fs = format_factors(&factors);
        return TwistSecurityResult { bits, exact: true, factor_string: fs };
    }

    if let Some(pari_factors) = pari_factor_full(&rug_cofactor) {
        for f in &pari_factors {
            factors.push(rug_to_bigint(f));
        }
        let largest = pari_factors.iter().max().unwrap();
        let bits = largest.significant_bits();
        let fs = format_factors(&factors);
        return TwistSecurityResult { bits, exact: true, factor_string: fs };
    }

    // Failed to fully factor
    factors.push(cofactor.clone());
    let fs = format_factors(&factors);
    TwistSecurityResult { bits: bigint_bits(&cofactor), exact: false, factor_string: fs }
}

fn rug_to_bigint(n: &rug::Integer) -> BigInt {
    BigInt::parse_bytes(n.to_string().as_bytes(), 10).unwrap()
}

fn format_factors(factors: &[BigInt]) -> String {
    if factors.is_empty() {
        return "1".to_string();
    }
    // Group and sort
    let mut sorted = factors.to_vec();
    sorted.sort();
    // Deduplicate with exponents
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let f = &sorted[i];
        let mut count = 1;
        while i + count < sorted.len() && &sorted[i + count] == f {
            count += 1;
        }
        if count == 1 {
            parts.push(f.to_string());
        } else {
            parts.push(format!("{}^{}", f, count));
        }
        i += count;
    }
    parts.join(" * ")
}

pub fn pollard_rho_factor(composite: &rug::Integer, budget: Duration) -> Option<u32> {
    pollard_rho_full_factor(composite, budget)
        .map(|factors| {
            factors.iter()
                .max()
                .map(|f| f.significant_bits())
                .unwrap_or(0)
        })
}

/// Fully factor via Pollard-rho. Returns all prime factors (with repetition) or None.
pub fn pollard_rho_full_factor(composite: &rug::Integer, budget: Duration) -> Option<Vec<rug::Integer>> {
    let deadline = Instant::now() + budget;
    let mut remaining = composite.clone();
    let mut factors = Vec::new();

    for &f in SMALL_PRIMES.iter() {
        let f_rug = rug::Integer::from(f);
        while rug::Integer::from(&remaining % &f_rug) == 0 {
            remaining /= &f_rug;
            factors.push(f_rug.clone());
        }
        if rug::Integer::from(&f_rug * &f_rug) > remaining {
            break;
        }
    }
    if remaining == 1 {
        return Some(factors);
    }
    if rug_is_prime(&remaining) {
        factors.push(remaining);
        return Some(factors);
    }
    for _ in 0..30 {
        if Instant::now() >= deadline {
            break;
        }
        match pollard_rho_find_factor(&remaining, deadline) {
            Some(factor) if factor != remaining && factor > 1 => {
                while rug::Integer::from(&remaining % &factor) == 0 {
                    remaining /= &factor;
                    factors.push(factor.clone());
                }
                if remaining == 1 {
                    return Some(factors);
                }
                if rug_is_prime(&remaining) {
                    factors.push(remaining);
                    return Some(factors);
                }
            }
            _ => break,
        }
    }
    if remaining == 1 {
        Some(factors)
    } else if rug_is_prime(&remaining) {
        factors.push(remaining);
        Some(factors)
    } else {
        None
    }
}

fn pollard_rho_find_factor(n: &rug::Integer, deadline: Instant) -> Option<rug::Integer> {
    if rug::Integer::from(n % 2u32) == 0 {
        return Some(rug::Integer::from(2u32));
    }
    if rug_is_prime(n) {
        return Some(n.clone());
    }
    for c_val in 1u64..512 {
        let c = rug::Integer::from(c_val);
        let mut x = rug::Integer::from(2u32);
        let mut y = rug::Integer::from(2u32);
        let mut d = rug::Integer::from(1u32);
        let mut step = 0u32;
        let mut product = rug::Integer::from(1u32);
        let batch = 128u32;
        while d == 1 {
            for _ in 0..batch {
                x = (rug::Integer::from(&x * &x) + &c) % n;
                y = (rug::Integer::from(&y * &y) + &c) % n;
                y = (rug::Integer::from(&y * &y) + &c) % n;
                let diff = rug::Integer::from(&x - &y).abs();
                product = (product * &diff) % n;
            }
            d = rug::Integer::from(product.gcd_ref(n));
            product = rug::Integer::from(1u32);
            step += batch;
            if step % (batch * 4) == 0 && Instant::now() >= deadline {
                return None;
            }
        }
        if d != 1 && &d != n {
            return Some(d);
        }
        if Instant::now() >= deadline {
            return None;
        }
    }
    None
}

pub fn pari_factor(composite: &rug::Integer) -> Option<u32> {
    pari_factor_full(composite)
        .map(|factors| {
            factors.iter()
                .max()
                .map(|f| f.significant_bits())
                .unwrap_or(0)
        })
}

/// Fully factor via PARI/GP. Returns all prime factors (with repetition) or None.
pub fn pari_factor_full(composite: &rug::Integer) -> Option<Vec<rug::Integer>> {
    let n_str = composite.to_string();
    // PARI script that outputs each prime factor and its exponent, one per line
    let script = format!(
        r#"default(parisizemax, "2G");
{{
  my(n, F, nrows);
  n = {n_str};
  F = factor(n);
  nrows = #F~;
  for(i = 1, nrows,
    print("PFACTOR\t", F[i,1], "\t", F[i,2])
  );
  print("FACTOR_DONE");
}}
\q
"#
    );
    let output = Command::new("gp")
        .args(["-q", "-s", "512M"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(script.as_bytes());
            }
            child.wait_with_output().ok()
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut factors = Vec::new();
    let mut found_done = false;
    for line in stdout.lines() {
        let line = line.trim();
        if line == "FACTOR_DONE" {
            found_done = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("PFACTOR\t") {
            let parts: Vec<&str> = rest.split('\t').collect();
            if parts.len() == 2 {
                if let (Ok(prime), Ok(exp)) = (
                    parts[0].trim().parse::<rug::Integer>(),
                    parts[1].trim().parse::<u32>(),
                ) {
                    for _ in 0..exp {
                        factors.push(prime.clone());
                    }
                }
            }
        }
    }
    if found_done && !factors.is_empty() {
        Some(factors)
    } else {
        None
    }
}
