use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use crate::math::sieve::P;
use crate::math::types::*;
use crate::math::ranking::build_cycle_results;

fn mod_pow(base: &BigInt, exp: &BigInt, modulus: &BigInt) -> BigInt {
    base.modpow(exp, modulus)
}

fn isqrt(n: &BigInt) -> BigInt {
    if n.is_zero() {
        return BigInt::zero();
    }
    let mut x = BigInt::one() << ((n.bits() + 1) / 2);
    loop {
        let y = (&x + n / &x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

fn is_perfect_square(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    let s = isqrt(n);
    if &s * &s == *n { Some(s) } else { None }
}

pub fn sqrt_mod(a: &BigInt, p: &BigInt) -> Option<BigInt> {
    let a = a.mod_floor(p);
    if a.is_zero() {
        return Some(BigInt::zero());
    }
    let p_minus_1 = p - 1u64;
    let two = BigInt::from(2u64);
    let euler = &p_minus_1 / &two;
    if mod_pow(&a, &euler, p) != BigInt::one() {
        return None;
    }
    if p % &BigInt::from(4u64) == BigInt::from(3u64) {
        let exp = (p + 1u64) / &BigInt::from(4u64);
        return Some(mod_pow(&a, &exp, p));
    }
    let mut q = p_minus_1.clone();
    let mut s = 0u32;
    while (&q % &two).is_zero() {
        q /= &two;
        s += 1;
    }
    let mut z = BigInt::from(2u64);
    while mod_pow(&z, &euler, p) != p_minus_1 {
        z += 1u64;
    }
    let mut m = s;
    let mut c = mod_pow(&z, &q, p);
    let mut t = mod_pow(&a, &q, p);
    let exp_r = (&q + 1u64) / &two;
    let mut r = mod_pow(&a, &exp_r, p);

    loop {
        if t == BigInt::one() {
            return Some(r);
        }
        let mut i = 1u32;
        let mut temp = mod_pow(&t, &two, p);
        while temp != BigInt::one() {
            temp = mod_pow(&temp, &two, p);
            i += 1;
        }
        let b = mod_pow(&c, &(BigInt::one() << (m - i - 1)), p);
        m = i;
        c = mod_pow(&b, &two, p);
        t = (&t * &c).mod_floor(p);
        r = (&r * &b).mod_floor(p);
    }
}

fn cornacchia_with_root(d: u64, m: &BigInt, r0: &BigInt) -> Option<(BigInt, BigInt)> {
    let d_big = BigInt::from(d);
    if d_big >= *m {
        return None;
    }
    let r0 = if r0 * 2 < *m { m - r0 } else { r0.clone() };
    let bound = isqrt(m);
    let mut a = m.clone();
    let mut b = r0;
    while b > bound {
        let temp = a.mod_floor(&b);
        a = b;
        b = temp;
    }
    let rem = m - &b * &b;
    if rem.is_zero() || !(&rem % &d_big).is_zero() {
        return None;
    }
    let c = &rem / &d_big;
    is_perfect_square(&c).map(|s| (b, s))
}

pub fn solve_4p_with_root(d: u64, p_val: &BigInt, sqrt_neg_d: &BigInt) -> Vec<(BigInt, BigInt)> {
    let mut results = Vec::new();
    if let Some((u, v)) = cornacchia_with_root(d, p_val, sqrt_neg_d) {
        results.push((&u * 2, &v * 2));
    }
    if d % 4 == 3 {
        let d_big = BigInt::from(d);
        let mut r0 = sqrt_neg_d.clone();
        if (&r0 % 2u64).is_zero() {
            r0 = p_val - &r0;
        }
        let four_p = p_val * 4u64;
        let bound = isqrt(&four_p);
        let two_p = p_val * 2u64;
        let mut a = two_p;
        let mut b = r0;
        while b > bound {
            let temp = a.mod_floor(&b);
            a = b;
            b = temp;
        }
        let t = b;
        let rem = &four_p - &t * &t;
        if rem > BigInt::zero() && (&rem % &d_big).is_zero() {
            let s_sq = &rem / &d_big;
            if let Some(s) = is_perfect_square(&s_sq) {
                if !s.is_zero() && (&t % 2u64) == BigInt::one() && (&s % 2u64) == BigInt::one() {
                    let dominated = results.iter().any(|(rt, _)| *rt == t);
                    if !dominated {
                        results.push((t, s));
                    }
                }
            }
        }
    }
    results
}

pub fn solve_4p(d: u64, p_val: &BigInt) -> Vec<(BigInt, BigInt)> {
    let d_big = BigInt::from(d);
    let neg_d = (p_val - &d_big).mod_floor(p_val);
    match sqrt_mod(&neg_d, p_val) {
        Some(root) => solve_4p_with_root(d, p_val, &root),
        None => Vec::new(),
    }
}

pub fn solve_4p_from_precomputed_root(d: u64, sqrt_neg_d: &BigInt) -> Vec<Solve4pSolution> {
    solve_4p_with_root(d, &P, sqrt_neg_d)
        .into_iter()
        .map(|(t, s)| Solve4pSolution { t, s })
        .collect()
}

pub fn search_discriminant_after_prefilter_with_root(
    d: u64,
    sqrt_neg_d: &BigInt,
) -> Vec<CycleResult> {
    let solutions = solve_4p_from_precomputed_root(d, sqrt_neg_d);
    if solutions.is_empty() {
        Vec::new()
    } else {
        build_cycle_results(d, &solutions, None)
    }
}

impl Solve4pKernelBackend for CpuBigIntSolve4pBackend {
    fn label(&self) -> &'static str {
        "cpu-bigint"
    }

    fn solve_batch(&self, work: &[Solve4pWorkItem]) -> Vec<Solve4pWorkResult> {
        work.iter()
            .map(|item| Solve4pWorkResult {
                discriminant: item.discriminant,
                solutions: solve_4p(item.discriminant, &P)
                    .into_iter()
                    .map(|(t, s)| Solve4pSolution { t, s })
                    .collect(),
            })
            .collect()
    }
}

pub fn solve_4p_batch_with_backend<B: Solve4pKernelBackend>(
    discriminants: &[u64],
    backend: &B,
) -> Vec<Solve4pWorkResult> {
    let work: Vec<Solve4pWorkItem> = discriminants
        .iter()
        .copied()
        .map(|discriminant| Solve4pWorkItem { discriminant })
        .collect();
    backend.solve_batch(&work)
}

pub fn solve_4p_single_with_backend<B: Solve4pKernelBackend>(
    d: u64,
    backend: &B,
) -> Vec<Solve4pSolution> {
    let mut results = solve_4p_batch_with_backend(&[d], backend);
    match results.pop() {
        Some(result) => {
            debug_assert_eq!(result.discriminant, d);
            result.solutions
        }
        None => Vec::new(),
    }
}
