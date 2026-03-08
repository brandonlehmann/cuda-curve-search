const P_MOD_4: u64 = 1;
const P_MOD_8: u64 = 5;

fn mul_mod_u64(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

fn pow_mod_u64(mut base: u64, mut exp: u32, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut acc = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod_u64(acc, base, modulus);
        }
        base = mul_mod_u64(base, base, modulus);
        exp >>= 1;
    }
    acc
}

pub fn p_mod_u64(modulus: u64) -> u64 {
    if modulus == 0 {
        return 0;
    }
    let two_pow = pow_mod_u64(2, 255, modulus);
    (two_pow + modulus - (19 % modulus)) % modulus
}

pub fn jacobi_symbol_u64(mut a: u64, mut n: u64) -> i32 {
    debug_assert!(n > 0 && n % 2 == 1);
    a %= n;
    let mut result = 1i32;
    while a != 0 {
        while a & 1 == 0 {
            a >>= 1;
            let n_mod_8 = n & 7;
            if n_mod_8 == 3 || n_mod_8 == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if (a & 3) == 3 && (n & 3) == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 { result } else { 0 }
}

pub fn jacobi_symbol_discriminant(d: u64) -> i32 {
    if d == 0 {
        return 0;
    }
    let mut a = d;
    let mut result = 1i32;
    while a & 1 == 0 {
        a >>= 1;
        if P_MOD_8 == 3 || P_MOD_8 == 5 {
            result = -result;
        }
    }
    if a == 1 {
        return result;
    }
    debug_assert_eq!(P_MOD_4, 1);
    result * jacobi_symbol_u64(p_mod_u64(a), a)
}

pub fn passes_local_solve_sieve_discriminant(d: u64) -> bool {
    d & 1 != 0
        && d % 7 != 0
        && d % 11 != 0
        && d % 13 != 0
        && d % 17 != 0
        && d % 19 != 0
        && d % 29 != 0
        && d % 31 != 0
        && d % 43 != 0
        && d % 53 != 0
        && d % 61 != 0
        && d % 67 != 0
        && d % 71 != 0
        && d % 73 != 0
        && d % 79 != 0
        && d % 89 != 0
        && d % 97 != 0
        && d % 101 != 0
        && d % 103 != 0
        && d % 107 != 0
        && d % 109 != 0
        && d % 127 != 0
        && d % 131 != 0
        && d % 139 != 0
        && d % 149 != 0
        && d % 151 != 0
        && d % 163 != 0
        && d % 167 != 0
        && d % 179 != 0
        && d % 191 != 0
        && d % 199 != 0
        && d % 229 != 0
        && d % 257 != 0
        && d % 263 != 0
        && d % 277 != 0
        && d % 331 != 0
        && d % 337 != 0
        && d % 349 != 0
        && d % 367 != 0
        && d % 379 != 0
        && d % 389 != 0
}
