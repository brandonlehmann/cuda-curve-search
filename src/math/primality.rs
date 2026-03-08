use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::math::sieve::PRIMALITY_SIEVE_PRIMES;

fn mod_pow(base: &BigInt, exp: &BigInt, modulus: &BigInt) -> BigInt {
    base.modpow(exp, modulus)
}

pub fn is_probable_prime(n: &BigInt) -> bool {
    let two = BigInt::from(2u64);
    if *n < two {
        return false;
    }
    if *n < BigInt::from(4u64) {
        return true;
    }
    if (n % &two).is_zero() {
        return false;
    }
    if !passes_small_prime_sieve_bigint(n) {
        return false;
    }
    let n_minus_1 = n - 1u64;
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    while (&d % &two).is_zero() {
        d /= &two;
        r += 1;
    }
    const WITNESSES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    'witness: for &a_val in &WITNESSES {
        let a = BigInt::from(a_val);
        if a >= *n {
            continue;
        }
        let mut x = mod_pow(&a, &d, n);
        if x == BigInt::one() || x == n_minus_1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_pow(&x, &two, n);
            if x == n_minus_1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

pub fn passes_small_prime_sieve_bigint(n: &BigInt) -> bool {
    for &p in PRIMALITY_SIEVE_PRIMES.iter() {
        let p_big = BigInt::from(p);
        if n == &p_big {
            return true;
        }
        if (n % &p_big).is_zero() {
            return false;
        }
    }
    true
}
