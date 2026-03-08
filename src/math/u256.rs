use std::cmp::Ordering;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use num_integer::Integer;

use crate::math::sieve::{P, SMALL_PRIMES, PRIMALITY_SIEVE_PRIMES};
use crate::math::twist::{pollard_rho_factor, pari_factor, bigint_to_rug};

use std::time::Duration;

pub fn bigint_to_field51(n: &BigInt) -> [u64; 5] {
    let mask = (1u64 << 51) - 1;
    let mut value = n.mod_floor(&P);
    let mut limbs = [0u64; 5];
    for limb in &mut limbs {
        let chunk = (&value & BigInt::from(mask))
            .to_u64()
            .expect("field limb should fit in 51 bits");
        *limb = chunk;
        value >>= 51;
    }
    limbs
}

pub fn field51_to_bigint(limbs: &[u64; 5]) -> BigInt {
    limbs
        .iter()
        .enumerate()
        .fold(BigInt::zero(), |acc, (i, limb)| {
            acc + (BigInt::from(*limb) << (51 * i))
        })
}

pub fn sqrt_m1_field51_host() -> [u64; 5] {
    let exponent = (&*P - 1u64) >> 2;
    let sqrt_m1 = BigInt::from(2u64).modpow(&exponent, &P);
    bigint_to_field51(&sqrt_m1)
}

pub fn bigint_to_u256(n: &BigInt) -> [u64; 4] {
    let mut value = n.clone();
    let mask = BigInt::from(u64::MAX);
    let mut limbs = [0u64; 4];
    for limb in &mut limbs {
        *limb = (&value & &mask)
            .to_u64()
            .expect("u256 limb should fit in 64 bits");
        value >>= 64;
    }
    debug_assert!(value.is_zero(), "value does not fit into 256 bits");
    limbs
}

pub fn bigint_to_u320(n: &BigInt) -> [u64; 5] {
    let mut value = n.clone();
    let mask = BigInt::from(u64::MAX);
    let mut limbs = [0u64; 5];
    for limb in &mut limbs {
        *limb = (&value & &mask)
            .to_u64()
            .expect("u320 limb should fit in 64 bits");
        value >>= 64;
    }
    debug_assert!(value.is_zero(), "value does not fit into 320 bits");
    limbs
}

pub fn u256_to_bigint(limbs: &[u64; 4]) -> BigInt {
    limbs
        .iter()
        .enumerate()
        .fold(BigInt::zero(), |acc, (i, limb)| {
            acc + (BigInt::from(*limb) << (64 * i))
        })
}

pub fn u256_is_zero(limbs: &[u64; 4]) -> bool {
    limbs.iter().all(|&limb| limb == 0)
}

pub fn u256_add(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut out = [0u64; 4];
    let mut carry = false;
    for i in 0..out.len() {
        let (sum, carry_a) = a[i].overflowing_add(b[i]);
        let (sum, carry_b) = sum.overflowing_add(u64::from(carry));
        out[i] = sum;
        carry = carry_a || carry_b;
    }
    (out, carry)
}

pub fn u256_sub(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut out = [0u64; 4];
    let mut borrow = false;
    for i in 0..out.len() {
        let (diff, borrow_a) = a[i].overflowing_sub(b[i]);
        let (diff, borrow_b) = diff.overflowing_sub(u64::from(borrow));
        out[i] = diff;
        borrow = borrow_a || borrow_b;
    }
    (out, borrow)
}

pub fn u256_add_u64(a: &[u64; 4], b: u64) -> ([u64; 4], bool) {
    let mut out = *a;
    let (sum, carry0) = out[0].overflowing_add(b);
    out[0] = sum;
    let mut carry = carry0;
    for limb in out.iter_mut().skip(1) {
        if !carry {
            break;
        }
        let (sum, next_carry) = limb.overflowing_add(1);
        *limb = sum;
        carry = next_carry;
    }
    (out, carry)
}

pub fn u256_bits(limbs: &[u64; 4]) -> u32 {
    for (index, limb) in limbs.iter().enumerate().rev() {
        if *limb != 0 {
            return ((index as u32) * 64) + (64 - limb.leading_zeros());
        }
    }
    0
}

pub fn u256_is_one(limbs: &[u64; 4]) -> bool {
    limbs[0] == 1 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}

fn u256_from_u64(value: u64) -> [u64; 4] {
    [value, 0, 0, 0]
}

fn u256_cmp(a: &[u64; 4], b: &[u64; 4]) -> Ordering {
    for index in (0..a.len()).rev() {
        match a[index].cmp(&b[index]) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    Ordering::Equal
}

fn u256_cmp_u64(a: &[u64; 4], b: u64) -> Ordering {
    if a[1] != 0 || a[2] != 0 || a[3] != 0 {
        Ordering::Greater
    } else {
        a[0].cmp(&b)
    }
}

fn u256_is_even(limbs: &[u64; 4]) -> bool {
    (limbs[0] & 1) == 0
}

fn u256_bit(limbs: &[u64; 4], bit: u32) -> bool {
    let limb_index = (bit / 64) as usize;
    let shift = bit % 64;
    limb_index < limbs.len() && ((limbs[limb_index] >> shift) & 1) == 1
}

fn u256_trailing_zeros(limbs: &[u64; 4]) -> u32 {
    for (index, limb) in limbs.iter().enumerate() {
        if *limb != 0 {
            return (index as u32) * 64 + limb.trailing_zeros();
        }
    }
    256
}

pub fn u256_shr(limbs: &[u64; 4], shift: u32) -> [u64; 4] {
    if shift >= 256 {
        return [0u64; 4];
    }
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;
    let mut out = [0u64; 4];
    for (index, limb) in out.iter_mut().enumerate() {
        let source_index = index + word_shift;
        if source_index >= limbs.len() {
            break;
        }
        *limb = limbs[source_index] >> bit_shift;
        if bit_shift != 0 && source_index + 1 < limbs.len() {
            *limb |= limbs[source_index + 1] << (64 - bit_shift);
        }
    }
    out
}

pub fn u256_div_mod_u64(limbs: &[u64; 4], divisor: u64) -> ([u64; 4], u64) {
    debug_assert!(divisor != 0, "divisor must be non-zero");
    let mut out = [0u64; 4];
    let mut remainder = 0u128;
    for index in (0..limbs.len()).rev() {
        let wide = (remainder << 64) | u128::from(limbs[index]);
        out[index] = (wide / u128::from(divisor)) as u64;
        remainder = wide % u128::from(divisor);
    }
    (out, remainder as u64)
}

pub fn u256_cofactor_bits_at_least(limbs: &[u64; 4], threshold: u32) -> bool {
    if threshold == 0 {
        return true;
    }
    let mut cofactor = *limbs;
    if u256_bits(&cofactor) < threshold {
        return false;
    }
    for &prime in SMALL_PRIMES.iter() {
        loop {
            let (quotient, remainder) = u256_div_mod_u64(&cofactor, prime);
            if remainder != 0 {
                break;
            }
            cofactor = quotient;
            if u256_is_one(&cofactor) || u256_bits(&cofactor) < threshold {
                return false;
            }
        }
        if cofactor[1] == 0 && cofactor[2] == 0 && cofactor[3] == 0 && prime * prime > cofactor[0] {
            break;
        }
    }
    u256_bits(&cofactor) >= threshold
}

pub fn u256_mod_u64(limbs: &[u64; 4], modulus: u64) -> u64 {
    debug_assert!(modulus != 0, "modulus must be non-zero");
    let mut remainder = 0u128;
    for limb in limbs.iter().rev() {
        remainder = ((remainder << 64) | u128::from(*limb)) % u128::from(modulus);
    }
    remainder as u64
}

pub fn passes_small_prime_sieve_u256(limbs: &[u64; 4]) -> bool {
    for &p in PRIMALITY_SIEVE_PRIMES.iter() {
        if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 && limbs[0] == p {
            return true;
        }
        if u256_mod_u64(limbs, p) == 0 {
            return false;
        }
    }
    true
}

fn u256_double_mod(limbs: &[u64; 4], modulus: &[u64; 4]) -> [u64; 4] {
    let mut wide = [0u64; 5];
    let mut carry = 0u64;
    for (index, limb) in limbs.iter().enumerate() {
        wide[index] = (*limb << 1) | carry;
        carry = *limb >> 63;
    }
    wide[4] = carry;
    if wide[4] != 0 || u256_cmp(&[wide[0], wide[1], wide[2], wide[3]], modulus) != Ordering::Less {
        let mut borrow = false;
        for index in 0..4 {
            let (diff, borrow_a) = wide[index].overflowing_sub(modulus[index]);
            let (diff, borrow_b) = diff.overflowing_sub(u64::from(borrow));
            wide[index] = diff;
            borrow = borrow_a || borrow_b;
        }
        let (diff_high, borrow_high) = wide[4].overflowing_sub(u64::from(borrow));
        debug_assert!(!borrow_high, "u256 double-mod subtraction underflowed");
        wide[4] = diff_high;
    }
    [wide[0], wide[1], wide[2], wide[3]]
}

fn u256_pow_two_mod(modulus: &[u64; 4], exponent_bits: usize) -> [u64; 4] {
    let mut value = u256_from_u64(1);
    for _ in 0..exponent_bits {
        value = u256_double_mod(&value, modulus);
    }
    value
}

fn add_u64_with_carry(words: &mut [u64], start_index: usize, mut carry: u64) {
    let mut index = start_index;
    while carry != 0 && index < words.len() {
        let (sum, overflow) = words[index].overflowing_add(carry);
        words[index] = sum;
        carry = u64::from(overflow);
        index += 1;
    }
    debug_assert_eq!(carry, 0, "carry propagated past end of wide integer");
}

fn montgomery_n0_inv(n0: u64) -> u64 {
    debug_assert_eq!(n0 & 1, 1, "Montgomery modulus must be odd");
    let mut inv = 1u64;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

struct U256MontgomeryContext {
    modulus: [u64; 4],
    n0_inv: u64,
    r2: [u64; 4],
    one_mont: [u64; 4],
    modulus_minus_one_mont: [u64; 4],
}

impl U256MontgomeryContext {
    fn new(modulus: &[u64; 4]) -> Self {
        let n0_inv = montgomery_n0_inv(modulus[0]);
        let r2 = u256_pow_two_mod(modulus, 512);
        let one = u256_from_u64(1);
        let one_mont = u256_montgomery_mul(&one, &r2, modulus, n0_inv);
        let (modulus_minus_one, borrow) = u256_sub(modulus, &one);
        debug_assert!(!borrow, "Montgomery modulus must be > 0");
        let modulus_minus_one_mont = u256_montgomery_mul(&modulus_minus_one, &r2, modulus, n0_inv);
        Self {
            modulus: *modulus,
            n0_inv,
            r2,
            one_mont,
            modulus_minus_one_mont,
        }
    }

    fn to_montgomery(&self, value: &[u64; 4]) -> [u64; 4] {
        u256_montgomery_mul(value, &self.r2, &self.modulus, self.n0_inv)
    }

    fn multiply(&self, a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        u256_montgomery_mul(a, b, &self.modulus, self.n0_inv)
    }

    fn pow_u64(&self, base: u64, exponent: &[u64; 4]) -> [u64; 4] {
        let mut accumulator = self.one_mont;
        let base_mont = self.to_montgomery(&u256_from_u64(base));
        for bit in (0..u256_bits(exponent)).rev() {
            accumulator = self.multiply(&accumulator, &accumulator);
            if u256_bit(exponent, bit) {
                accumulator = self.multiply(&accumulator, &base_mont);
            }
        }
        accumulator
    }
}

fn u256_montgomery_mul(a: &[u64; 4], b: &[u64; 4], modulus: &[u64; 4], n0_inv: u64) -> [u64; 4] {
    let mut wide = [0u64; 9];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let index = i + j;
            let accum = u128::from(wide[index]) + u128::from(a[i]) * u128::from(b[j]) + carry;
            wide[index] = accum as u64;
            carry = accum >> 64;
        }
        add_u64_with_carry(&mut wide, i + 4, carry as u64);
    }
    for i in 0..4 {
        let m = wide[i].wrapping_mul(n0_inv);
        let mut carry = 0u128;
        for j in 0..4 {
            let index = i + j;
            let accum = u128::from(wide[index]) + u128::from(m) * u128::from(modulus[j]) + carry;
            wide[index] = accum as u64;
            carry = accum >> 64;
        }
        add_u64_with_carry(&mut wide, i + 4, carry as u64);
    }
    let mut out = [wide[4], wide[5], wide[6], wide[7]];
    let mut high = wide[8];
    if high != 0 || u256_cmp(&out, modulus) != Ordering::Less {
        let (next, borrow) = u256_sub(&out, modulus);
        out = next;
        let (next_high, borrow_high) = high.overflowing_sub(u64::from(borrow));
        debug_assert!(!borrow_high, "Montgomery final reduction underflowed");
        high = next_high;
    }
    debug_assert_eq!(high, 0, "Montgomery reduction should fit in 256 bits");
    out
}

pub fn is_probable_prime_u256(limbs: &[u64; 4]) -> bool {
    if u256_is_zero(limbs) || u256_cmp_u64(limbs, 1) == Ordering::Equal {
        return false;
    }
    if u256_cmp_u64(limbs, 2) == Ordering::Equal || u256_cmp_u64(limbs, 3) == Ordering::Equal {
        return true;
    }
    if u256_is_even(limbs) {
        return false;
    }
    if !passes_small_prime_sieve_u256(limbs) {
        return false;
    }
    let (n_minus_one, borrow) = u256_sub(limbs, &u256_from_u64(1));
    debug_assert!(!borrow, "n - 1 underflowed in u256 primality test");
    let shift = u256_trailing_zeros(&n_minus_one);
    let d = u256_shr(&n_minus_one, shift);
    let context = U256MontgomeryContext::new(limbs);
    const WITNESSES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    'witness: for &witness in &WITNESSES {
        if u256_cmp_u64(limbs, witness) != Ordering::Greater {
            continue;
        }
        let mut x = context.pow_u64(witness, &d);
        if x == context.one_mont || x == context.modulus_minus_one_mont {
            continue;
        }
        for _ in 0..shift.saturating_sub(1) {
            x = context.multiply(&x, &x);
            if x == context.modulus_minus_one_mont {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

pub fn twist_security_bits_u256(limbs: &[u64; 4]) -> (u32, bool) {
    if u256_is_zero(limbs) || u256_is_one(limbs) {
        return (0, true);
    }
    if is_probable_prime_u256(limbs) {
        return (u256_bits(limbs), true);
    }
    let mut cofactor = *limbs;
    for &prime in SMALL_PRIMES.iter() {
        loop {
            let (quotient, remainder) = u256_div_mod_u64(&cofactor, prime);
            if remainder != 0 {
                break;
            }
            cofactor = quotient;
        }
        if cofactor[1] == 0
            && cofactor[2] == 0
            && cofactor[3] == 0
            && prime.saturating_mul(prime) > cofactor[0]
        {
            break;
        }
    }
    if u256_is_one(&cofactor) {
        return (0, true);
    }
    if is_probable_prime_u256(&cofactor) {
        return (u256_bits(&cofactor), true);
    }
    let cofactor_bigint = u256_to_bigint(&cofactor);
    let rug_cofactor = bigint_to_rug(&cofactor_bigint);
    if let Some(bits) = pollard_rho_factor(&rug_cofactor, Duration::from_secs(2)) {
        return (bits, true);
    }
    if let Some(bits) = pari_factor(&rug_cofactor) {
        return (bits, true);
    }
    (u256_bits(&cofactor), false)
}

// Curve constant helpers
pub fn curve_prime_u320() -> [u64; 5] {
    bigint_to_u320(&P)
}

pub fn curve_four_p_u320() -> [u64; 5] {
    bigint_to_u320(&(&*P * 4u64))
}

pub fn curve_prime_u256() -> [u64; 4] {
    bigint_to_u256(&P)
}

pub fn curve_prime_plus_one_u256() -> [u64; 4] {
    bigint_to_u256(&(&*P + 1u64))
}

pub fn curve_two_p_u256() -> [u64; 4] {
    bigint_to_u256(&(&*P * 2u64))
}

pub fn curve_half_floor_u256() -> [u64; 4] {
    bigint_to_u256(&((&*P - 1u64) >> 1))
}

pub fn curve_sqrt_p_u256() -> [u64; 4] {
    let isqrt_val = {
        let n = &*P;
        let mut x = num_bigint::BigInt::from(1u64) << ((n.bits() + 1) / 2);
        loop {
            let y = (&x + n / &x) >> 1;
            if y >= x {
                break x;
            }
            x = y;
        }
    };
    bigint_to_u256(&isqrt_val)
}

pub fn curve_sqrt_four_p_u256() -> [u64; 4] {
    let four_p = &*P * 4u64;
    let isqrt_val = {
        let n = &four_p;
        let mut x = num_bigint::BigInt::from(1u64) << ((n.bits() + 1) / 2);
        loop {
            let y = (&x + n / &x) >> 1;
            if y >= x {
                break x;
            }
            x = y;
        }
    };
    bigint_to_u256(&isqrt_val)
}
