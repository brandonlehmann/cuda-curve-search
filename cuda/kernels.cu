// Elliptic Curve Cycle Search - CUDA Kernels
// GPU implementation of discriminant prefilter, Cornacchia solve, and q-sieve stages.
//
// p = 2^255 - 19  (the Ed25519 / Curve25519 prime)

#include "local_solve_sieve_wheel.inc"
#include "squarefree_sieve_wheel.inc"

// ============================================================================
// Constants
// ============================================================================

// Extended local-solve sieve primes (beyond the wheel primes 7,11,13,17,19).
// D must not be divisible by any of these for the local-solve check.
__device__ __constant__ unsigned long long LOCAL_SOLVE_EXTENDED_PRIMES[] = {
    29, 31, 43, 53, 61, 67, 71, 73, 79, 89, 97,
    101, 103, 107, 109, 127, 131, 139, 149, 151,
    163, 167, 179, 191, 199, 229, 257, 263, 277,
    331, 337, 349, 367, 379, 389
};
constexpr unsigned int LOCAL_SOLVE_EXTENDED_PRIME_COUNT = 35U;

// p mod 8 for the Jacobi computation
constexpr unsigned long long P_MOD_8 = 5ULL;

// Mask for 51-bit field limbs
constexpr unsigned long long F51_MASK = (1ULL << 51) - 1ULL;

// ============================================================================
// u256 arithmetic  (4 x u64 limbs, little-endian)
// ============================================================================

__device__ __forceinline__
unsigned int u256_add(const unsigned long long* a, const unsigned long long* b,
                      unsigned long long* out) {
    unsigned long long carry = 0ULL;
    for (int i = 0; i < 4; i++) {
        unsigned long long sum = a[i] + b[i];
        unsigned long long c1 = (sum < a[i]) ? 1ULL : 0ULL;
        unsigned long long sum2 = sum + carry;
        unsigned long long c2 = (sum2 < sum) ? 1ULL : 0ULL;
        out[i] = sum2;
        carry = c1 + c2;
    }
    return (unsigned int)carry;
}

__device__ __forceinline__
unsigned int u256_sub(const unsigned long long* a, const unsigned long long* b,
                      unsigned long long* out) {
    unsigned long long borrow = 0ULL;
    for (int i = 0; i < 4; i++) {
        unsigned long long diff = a[i] - b[i];
        unsigned long long b1 = (diff > a[i]) ? 1ULL : 0ULL;
        unsigned long long diff2 = diff - borrow;
        unsigned long long b2 = (diff2 > diff) ? 1ULL : 0ULL;
        out[i] = diff2;
        borrow = b1 + b2;
    }
    return (unsigned int)borrow;
}

// Returns -1 if a < b, 0 if a == b, 1 if a > b
__device__ __forceinline__
int u256_cmp(const unsigned long long* a, const unsigned long long* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ __forceinline__
void u256_shr(const unsigned long long* limbs, unsigned int shift, unsigned long long* out) {
    if (shift >= 256) {
        out[0] = out[1] = out[2] = out[3] = 0ULL;
        return;
    }
    unsigned int word_shift = shift / 64;
    unsigned int bit_shift = shift % 64;
    for (int i = 0; i < 4; i++) {
        unsigned int src = i + word_shift;
        if (src >= 4) {
            out[i] = 0ULL;
        } else {
            out[i] = limbs[src] >> bit_shift;
            if (bit_shift != 0 && src + 1 < 4) {
                out[i] |= limbs[src + 1] << (64 - bit_shift);
            }
        }
    }
}

// Schoolbook 4x4 -> 8 limb multiplication using __int128
__device__ __forceinline__
void u256_mul_schoolbook(const unsigned long long* a, const unsigned long long* b,
                         unsigned long long* out8) {
    for (int i = 0; i < 8; i++) out8[i] = 0ULL;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a[i] * b[j]
                                   + (unsigned __int128)out8[i + j]
                                   + carry;
            out8[i + j] = (unsigned long long)prod;
            carry = prod >> 64;
        }
        out8[i + 4] = (unsigned long long)carry;
    }
}

// Division of u256 by u64, returns remainder; quotient written to out
__device__ __forceinline__
unsigned long long u256_div_mod_u64(const unsigned long long* limbs, unsigned long long divisor,
                                     unsigned long long* quotient) {
    unsigned __int128 rem = 0;
    for (int i = 3; i >= 0; i--) {
        unsigned __int128 wide = (rem << 64) | (unsigned __int128)limbs[i];
        quotient[i] = (unsigned long long)(wide / (unsigned __int128)divisor);
        rem = wide % (unsigned __int128)divisor;
    }
    return (unsigned long long)rem;
}

// Mod of u256 by u64 (no quotient needed)
__device__ __forceinline__
unsigned long long u256_mod_u64(const unsigned long long* limbs, unsigned long long modulus) {
    unsigned __int128 rem = 0;
    for (int i = 3; i >= 0; i--) {
        rem = ((rem << 64) | (unsigned __int128)limbs[i]) % (unsigned __int128)modulus;
    }
    return (unsigned long long)rem;
}

__device__ __forceinline__
bool u256_is_zero(const unsigned long long* limbs) {
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
}

__device__ __forceinline__
unsigned int u256_bits(const unsigned long long* limbs) {
    for (int i = 3; i >= 0; i--) {
        if (limbs[i] != 0) {
            return (unsigned int)(i * 64) + (64 - __clzll(limbs[i]));
        }
    }
    return 0;
}

__device__ __forceinline__
void u256_from_u64(unsigned long long val, unsigned long long* out) {
    out[0] = val;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
}

__device__ __forceinline__
void u256_copy(const unsigned long long* src, unsigned long long* dst) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

// ============================================================================
// u320 arithmetic  (5 x u64 limbs, little-endian)
// ============================================================================

__device__ __forceinline__
unsigned int u320_sub(const unsigned long long* a, const unsigned long long* b,
                      unsigned long long* out) {
    unsigned long long borrow = 0ULL;
    for (int i = 0; i < 5; i++) {
        unsigned long long diff = a[i] - b[i];
        unsigned long long b1 = (diff > a[i]) ? 1ULL : 0ULL;
        unsigned long long diff2 = diff - borrow;
        unsigned long long b2 = (diff2 > diff) ? 1ULL : 0ULL;
        out[i] = diff2;
        borrow = b1 + b2;
    }
    return (unsigned int)borrow;
}

__device__ __forceinline__
unsigned int u320_add(const unsigned long long* a, const unsigned long long* b,
                      unsigned long long* out) {
    unsigned long long carry = 0ULL;
    for (int i = 0; i < 5; i++) {
        unsigned long long sum = a[i] + b[i];
        unsigned long long c1 = (sum < a[i]) ? 1ULL : 0ULL;
        unsigned long long sum2 = sum + carry;
        unsigned long long c2 = (sum2 < sum) ? 1ULL : 0ULL;
        out[i] = sum2;
        carry = c1 + c2;
    }
    return (unsigned int)carry;
}

// ============================================================================
// field51 arithmetic  (5 x 51-bit limbs, mod p = 2^255 - 19)
// ============================================================================

__device__ __forceinline__
void f51_reduce(unsigned long long* limbs) {
    // Carry propagation
    for (int i = 0; i < 4; i++) {
        limbs[i + 1] += limbs[i] >> 51;
        limbs[i] &= F51_MASK;
    }
    // Top limb overflow: limbs[4] might be >= 2^51
    // 2^255 = 19 mod p, so overflow wraps with factor 19
    unsigned long long top_carry = limbs[4] >> 51;
    limbs[4] &= F51_MASK;
    limbs[0] += top_carry * 19ULL;
    // Propagate again
    for (int i = 0; i < 4; i++) {
        limbs[i + 1] += limbs[i] >> 51;
        limbs[i] &= F51_MASK;
    }
    top_carry = limbs[4] >> 51;
    limbs[4] &= F51_MASK;
    limbs[0] += top_carry * 19ULL;
    limbs[1] += limbs[0] >> 51;
    limbs[0] &= F51_MASK;
}

__device__ __forceinline__
void f51_add(const unsigned long long* a, const unsigned long long* b, unsigned long long* out) {
    for (int i = 0; i < 5; i++) {
        out[i] = a[i] + b[i];
    }
    f51_reduce(out);
}

__device__ __forceinline__
void f51_sub(const unsigned long long* a, const unsigned long long* b, unsigned long long* out) {
    // Add 2*p to avoid underflow:
    // 2*p in field51: limb0 = 2*(2^51-19-1) isn't clean. Instead use the
    // standard trick: add a large multiple of p that keeps all limbs positive.
    // p in field51 is (2^51-19, 2^51-1, 2^51-1, 2^51-1, 2^51-1)
    // 2*p:           (2^52-38, 2^52-2, 2^52-2, 2^52-2, 2^52-2)
    constexpr unsigned long long bias[5] = {
        2ULL * ((1ULL << 51) - 19ULL),
        2ULL * F51_MASK,
        2ULL * F51_MASK,
        2ULL * F51_MASK,
        2ULL * F51_MASK
    };
    for (int i = 0; i < 5; i++) {
        out[i] = (a[i] + bias[i]) - b[i];
    }
    f51_reduce(out);
}

__device__ __forceinline__
void f51_mul(const unsigned long long* a, const unsigned long long* b, unsigned long long* out) {
    // Multiply using 128-bit intermediates with reduction by 19
    unsigned __int128 t0 = (unsigned __int128)a[0] * b[0]
                         + (unsigned __int128)(a[1] * 19ULL) * b[4]
                         + (unsigned __int128)(a[2] * 19ULL) * b[3]
                         + (unsigned __int128)(a[3] * 19ULL) * b[2]
                         + (unsigned __int128)(a[4] * 19ULL) * b[1];

    unsigned __int128 t1 = (unsigned __int128)a[0] * b[1]
                         + (unsigned __int128)a[1] * b[0]
                         + (unsigned __int128)(a[2] * 19ULL) * b[4]
                         + (unsigned __int128)(a[3] * 19ULL) * b[3]
                         + (unsigned __int128)(a[4] * 19ULL) * b[2];

    unsigned __int128 t2 = (unsigned __int128)a[0] * b[2]
                         + (unsigned __int128)a[1] * b[1]
                         + (unsigned __int128)a[2] * b[0]
                         + (unsigned __int128)(a[3] * 19ULL) * b[4]
                         + (unsigned __int128)(a[4] * 19ULL) * b[3];

    unsigned __int128 t3 = (unsigned __int128)a[0] * b[3]
                         + (unsigned __int128)a[1] * b[2]
                         + (unsigned __int128)a[2] * b[1]
                         + (unsigned __int128)a[3] * b[0]
                         + (unsigned __int128)(a[4] * 19ULL) * b[4];

    unsigned __int128 t4 = (unsigned __int128)a[0] * b[4]
                         + (unsigned __int128)a[1] * b[3]
                         + (unsigned __int128)a[2] * b[2]
                         + (unsigned __int128)a[3] * b[1]
                         + (unsigned __int128)a[4] * b[0];

    // Carry chain
    unsigned long long c;
    c = (unsigned long long)(t0 >> 51); out[0] = (unsigned long long)t0 & F51_MASK;
    t1 += c;
    c = (unsigned long long)(t1 >> 51); out[1] = (unsigned long long)t1 & F51_MASK;
    t2 += c;
    c = (unsigned long long)(t2 >> 51); out[2] = (unsigned long long)t2 & F51_MASK;
    t3 += c;
    c = (unsigned long long)(t3 >> 51); out[3] = (unsigned long long)t3 & F51_MASK;
    t4 += c;
    c = (unsigned long long)(t4 >> 51); out[4] = (unsigned long long)t4 & F51_MASK;
    out[0] += c * 19ULL;
    out[1] += out[0] >> 51; out[0] &= F51_MASK;
}

__device__ __forceinline__
void f51_square(const unsigned long long* a, unsigned long long* out) {
    unsigned long long a0_2 = 2ULL * a[0];
    unsigned long long a1_2 = 2ULL * a[1];
    unsigned long long a2_2 = 2ULL * a[2];
    unsigned long long a3_2 = 2ULL * a[3];

    unsigned __int128 t0 = (unsigned __int128)a[0] * a[0]
                         + (unsigned __int128)(a1_2 * 19ULL) * a[4]
                         + (unsigned __int128)(a2_2 * 19ULL) * a[3];

    unsigned __int128 t1 = (unsigned __int128)a0_2 * a[1]
                         + (unsigned __int128)(a[2] * 19ULL) * a2_2  // wait, need 19*a2*a4 + 19*a3*a3
                         + (unsigned __int128)(a3_2 * 19ULL) * a[3]; // wrong, redo

    // Redo the squaring properly. For a^2 with reduction by 19 on overflow:
    // d0 = a0*a0 + 2*19*a1*a4 + 2*19*a2*a3
    // d1 = 2*a0*a1 + 2*19*a2*a4 + 19*a3*a3
    // d2 = 2*a0*a2 + a1*a1 + 2*19*a3*a4
    // d3 = 2*a0*a3 + 2*a1*a2 + 19*a4*a4
    // d4 = 2*a0*a4 + 2*a1*a3 + a2*a2

    unsigned long long a4_19 = a[4] * 19ULL;
    unsigned long long a3_19 = a[3] * 19ULL;

    t0 = (unsigned __int128)a[0] * a[0]
       + (unsigned __int128)(a1_2) * a4_19
       + (unsigned __int128)(a2_2) * a3_19;

    t1 = (unsigned __int128)a0_2 * a[1]
       + (unsigned __int128)(a2_2) * a4_19
       + (unsigned __int128)a3_19 * a[3];

    unsigned __int128 t2_s = (unsigned __int128)a0_2 * a[2]
                           + (unsigned __int128)a[1] * a[1]
                           + (unsigned __int128)(a3_2) * a4_19;

    unsigned __int128 t3_s = (unsigned __int128)a0_2 * a[3]
                           + (unsigned __int128)a1_2 * a[2]
                           + (unsigned __int128)a[4] * a4_19; // 19*a4*a4

    unsigned __int128 t4_s = (unsigned __int128)a0_2 * a[4]
                           + (unsigned __int128)a1_2 * a[3]
                           + (unsigned __int128)a[2] * a[2];

    unsigned long long c;
    c = (unsigned long long)(t0 >> 51); out[0] = (unsigned long long)t0 & F51_MASK;
    t1 += c;
    c = (unsigned long long)(t1 >> 51); out[1] = (unsigned long long)t1 & F51_MASK;
    t2_s += c;
    c = (unsigned long long)(t2_s >> 51); out[2] = (unsigned long long)t2_s & F51_MASK;
    t3_s += c;
    c = (unsigned long long)(t3_s >> 51); out[3] = (unsigned long long)t3_s & F51_MASK;
    t4_s += c;
    c = (unsigned long long)(t4_s >> 51); out[4] = (unsigned long long)t4_s & F51_MASK;
    out[0] += c * 19ULL;
    out[1] += out[0] >> 51; out[0] &= F51_MASK;
}

__device__ __forceinline__
void f51_copy(const unsigned long long* src, unsigned long long* dst) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3]; dst[4] = src[4];
}

// Fully canonicalize a field51 element to [0, p)
__device__ __forceinline__
void f51_canonicalize(unsigned long long* limbs) {
    f51_reduce(limbs);
    // Subtract p and check for underflow
    // p = (2^51-19, 2^51-1, 2^51-1, 2^51-1, 2^51-1)
    unsigned long long tmp[5];
    long long borrow = 0LL;
    tmp[0] = limbs[0] - (F51_MASK - 18ULL); // subtract (2^51 - 19)
    borrow = (tmp[0] > limbs[0]) ? 1LL : 0LL;
    for (int i = 1; i < 5; i++) {
        unsigned long long sub = F51_MASK + (unsigned long long)borrow;
        tmp[i] = limbs[i] - sub;
        borrow = (tmp[i] > limbs[i]) ? 1LL : 0LL;
    }
    // If no borrow, tmp is the canonical form (value >= p)
    if (borrow == 0) {
        for (int i = 0; i < 5; i++) limbs[i] = tmp[i];
    }
}

__device__ __forceinline__
bool f51_is_zero(const unsigned long long* limbs) {
    unsigned long long tmp[5];
    for (int i = 0; i < 5; i++) tmp[i] = limbs[i];
    f51_canonicalize(tmp);
    return tmp[0] == 0 && tmp[1] == 0 && tmp[2] == 0 && tmp[3] == 0 && tmp[4] == 0;
}

__device__ __forceinline__
bool f51_eq(const unsigned long long* a, const unsigned long long* b) {
    unsigned long long tmp[5];
    f51_sub(a, b, tmp);
    return f51_is_zero(tmp);
}

// Convert a u64 to field51
__device__ __forceinline__
void f51_from_u64(unsigned long long val, unsigned long long* out) {
    out[0] = val & F51_MASK;
    out[1] = val >> 51;
    out[2] = 0; out[3] = 0; out[4] = 0;
}

// ============================================================================
// field51 exponentiation for Tonelli-Shanks
// ============================================================================

// Compute base^exp mod p where exp is given as a u256 (4 x u64 limbs)
__device__
void f51_pow_u256(const unsigned long long* base, const unsigned long long* exp_u256,
                  unsigned long long* result) {
    // Start with 1
    unsigned long long acc[5] = {1, 0, 0, 0, 0};
    unsigned long long tmp[5];
    unsigned long long b[5];
    f51_copy(base, b);

    unsigned int nbits = u256_bits(exp_u256);
    if (nbits == 0) {
        f51_copy(acc, result);
        return;
    }

    // Square-and-multiply, MSB first
    bool started = false;
    for (int bit = (int)nbits - 1; bit >= 0; bit--) {
        if (started) {
            f51_square(acc, tmp);
            f51_copy(tmp, acc);
        }
        unsigned int limb_idx = bit / 64;
        unsigned int bit_idx = bit % 64;
        if ((exp_u256[limb_idx] >> bit_idx) & 1ULL) {
            if (!started) {
                f51_copy(b, acc);
                started = true;
            } else {
                f51_mul(acc, b, tmp);
                f51_copy(tmp, acc);
            }
        }
    }
    f51_copy(acc, result);
}

// ============================================================================
// Modular sqrt for p = 2^255 - 19
// ============================================================================
//
// Since p mod 8 = 5, we use:
//   beta = a^((p+3)/8) mod p
//   If beta^2 == a, return beta
//   If beta^2 == -a, return beta * sqrt(-1)
//   Otherwise, no square root exists
//
// (p+3)/8 as u256:
//   p   = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED
//   p+3 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0
//   (p+3)/8 = 0x0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE
__device__ __constant__ unsigned long long P_PLUS_3_DIV_8[4] = {
    0xFFFFFFFFFFFFFFFeULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0x0FFFFFFFFFFFFFFFULL
};

// Returns true if sqrt was found, and writes it to out_root (field51).
__device__
bool f51_sqrt_neg_d(unsigned long long d_val,
                    const unsigned long long* p_u256,
                    const unsigned long long* sqrt_m1_f51,
                    unsigned long long* out_root) {
    // a = p - d = -d mod p.  Since d < p, this is just p - d.
    // Convert to field51: we need (p - d) in field51 form.
    // p in field51 = (2^51-19, 2^51-1, 2^51-1, 2^51-1, 2^51-1)
    unsigned long long a_f51[5];
    // First make d in field51
    unsigned long long d_f51[5];
    f51_from_u64(d_val, d_f51);
    // p in field51
    unsigned long long p_f51[5] = {
        F51_MASK - 18ULL, F51_MASK, F51_MASK, F51_MASK, F51_MASK
    };
    f51_sub(p_f51, d_f51, a_f51);

    // Check if a is zero
    if (f51_is_zero(a_f51)) {
        for (int i = 0; i < 5; i++) out_root[i] = 0;
        return true;
    }

    // beta = a^((p+3)/8) mod p
    unsigned long long beta[5];
    f51_pow_u256(a_f51, P_PLUS_3_DIV_8, beta);

    // Check beta^2 == a
    unsigned long long beta_sq[5];
    f51_square(beta, beta_sq);
    f51_reduce(beta_sq);

    if (f51_eq(beta_sq, a_f51)) {
        f51_copy(beta, out_root);
        return true;
    }

    // Check beta^2 == -a (i.e. p - a)
    unsigned long long neg_a[5];
    f51_sub(p_f51, a_f51, neg_a);
    if (f51_eq(beta_sq, neg_a)) {
        // root = beta * sqrt(-1)
        unsigned long long tmp[5];
        f51_mul(beta, sqrt_m1_f51, tmp);
        f51_reduce(tmp);
        f51_copy(tmp, out_root);
        return true;
    }

    return false;
}

// ============================================================================
// field51 <-> u256 conversion
// ============================================================================

__device__ __forceinline__
void f51_to_u256(const unsigned long long* f, unsigned long long* out) {
    // Canonicalize first
    unsigned long long tmp[5];
    f51_copy(f, tmp);
    f51_canonicalize(tmp);

    // Reconstruct the 256-bit integer from 5 x 51-bit limbs
    // bit layout: bits [0..50] = tmp[0], [51..101] = tmp[1], ... [204..254] = tmp[4]
    out[0] = tmp[0] | (tmp[1] << 51);        // bits 0..63 (limb1 contributes bits 51..63 = 13 bits)
    out[1] = (tmp[1] >> 13) | (tmp[2] << 38); // bits 64..127
    out[2] = (tmp[2] >> 26) | (tmp[3] << 25); // bits 128..191
    out[3] = (tmp[3] >> 39) | (tmp[4] << 12); // bits 192..255
}

__device__ __forceinline__
void u256_to_f51(const unsigned long long* u, unsigned long long* f) {
    f[0] = u[0] & F51_MASK;
    f[1] = ((u[0] >> 51) | (u[1] << 13)) & F51_MASK;
    f[2] = ((u[1] >> 38) | (u[2] << 26)) & F51_MASK;
    f[3] = ((u[2] >> 25) | (u[3] << 39)) & F51_MASK;
    f[4] = (u[3] >> 12) & F51_MASK;
}

// ============================================================================
// Jacobi symbol in u64
// ============================================================================

__device__ __forceinline__
unsigned long long mul_mod_u64(unsigned long long a, unsigned long long b, unsigned long long m) {
    return (unsigned long long)((unsigned __int128)a * b % (unsigned __int128)m);
}

__device__ __forceinline__
unsigned long long pow_mod_u64(unsigned long long base, unsigned int exp, unsigned long long m) {
    if (m == 1) return 0;
    unsigned long long acc = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) acc = mul_mod_u64(acc, base, m);
        base = mul_mod_u64(base, base, m);
        exp >>= 1;
    }
    return acc;
}

// Compute p = 2^255 - 19 mod a u64 value
__device__ __forceinline__
unsigned long long p_mod_u64(unsigned long long modulus) {
    if (modulus == 0) return 0;
    unsigned long long two_pow = pow_mod_u64(2ULL, 255U, modulus);
    unsigned long long minus_19 = 19ULL % modulus;
    return (two_pow + modulus - minus_19) % modulus;
}

// Standard Jacobi symbol (a/n) for odd n > 0
__device__
int jacobi_symbol_u64(unsigned long long a, unsigned long long n) {
    a %= n;
    int result = 1;
    while (a != 0) {
        while ((a & 1) == 0) {
            a >>= 1;
            unsigned long long n_mod_8 = n & 7;
            if (n_mod_8 == 3 || n_mod_8 == 5) {
                result = -result;
            }
        }
        // Swap
        unsigned long long tmp = a;
        a = n;
        n = tmp;
        if ((a & 3) == 3 && (n & 3) == 3) {
            result = -result;
        }
        a %= n;
    }
    return (n == 1) ? result : 0;
}

// Jacobi symbol of discriminant: jacobi(-D, p).
// Since p mod 4 = 1, jacobi(-1, p) = 1, so jacobi(-D, p) = jacobi(D, p).
// We compute jacobi(D, p) using the Rust algorithm:
//   1. Strip factors of 2 from D (flip sign based on P_MOD_8)
//   2. If a > 1: use quadratic reciprocity -> jacobi(p mod a, a)
__device__
int jacobi_symbol_discriminant(unsigned long long d) {
    if (d == 0) return 0;
    unsigned long long a = d;
    int result = 1;
    // Strip factors of 2
    while ((a & 1) == 0) {
        a >>= 1;
        // P_MOD_8 = 5, which is 3 or 5, so flip
        result = -result;
    }
    if (a == 1) return result;
    // Since P_MOD_4 = 1 (p mod 4 = 1), no sign flip for reciprocity law application
    // Now compute jacobi(p mod a, a)
    return result * jacobi_symbol_u64(p_mod_u64(a), a);
}

// ============================================================================
// Local solve sieve check
// ============================================================================

__device__
bool passes_local_solve_sieve(unsigned long long d) {
    // Must be odd
    if ((d & 1) == 0) return false;

    // Check against the wheel bitfield (primes 7, 11, 13, 17, 19)
    unsigned int residue = (unsigned int)(d % LOCAL_SOLVE_SIEVE_WHEEL_MODULUS);
    unsigned int word_idx = residue / 64;
    unsigned int bit_idx = residue % 64;
    if (!((LOCAL_SOLVE_SIEVE_WHEEL_BITS[word_idx] >> bit_idx) & 1ULL)) {
        return false;
    }

    // Check extended primes
    for (unsigned int i = 0; i < LOCAL_SOLVE_EXTENDED_PRIME_COUNT; i++) {
        if (d % LOCAL_SOLVE_EXTENDED_PRIMES[i] == 0) return false;
    }
    return true;
}

// ============================================================================
// Squarefree check
// ============================================================================

__device__
bool is_squarefree_check(unsigned long long d,
                         const unsigned long long* prime_squares,
                         unsigned int prime_square_count) {
    // Wheel check first
    unsigned int residue = (unsigned int)(d % SQUAREFREE_SIEVE_WHEEL_MODULUS);
    unsigned int word_idx = residue / 64;
    unsigned int bit_idx = residue % 64;
    if (!((SQUAREFREE_SIEVE_WHEEL_BITS[word_idx] >> bit_idx) & 1ULL)) {
        return false;
    }

    // Check remaining prime squares (those beyond the wheel primes 2,3,5,7,11)
    for (unsigned int i = 0; i < prime_square_count; i++) {
        unsigned long long ps = prime_squares[i];
        if (ps > d) break;
        if (d % ps == 0) return false;
    }
    return true;
}

// ============================================================================
// Small prime sieve for q (trial division up to 997)
// ============================================================================

__device__
bool passes_q_sieve(const unsigned long long* q_u256,
                    const unsigned long long* sieve_primes,
                    unsigned int sieve_prime_count) {
    for (unsigned int i = 0; i < sieve_prime_count; i++) {
        unsigned long long p = sieve_primes[i];
        // Check if q == p (then it's prime, passes)
        if (q_u256[1] == 0 && q_u256[2] == 0 && q_u256[3] == 0 && q_u256[0] == p) {
            return true;
        }
        if (u256_mod_u64(q_u256, p) == 0) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Cornacchia / Euclidean reduction
// ============================================================================

// u256 mod u256: compute a mod b using repeated subtraction with shift optimization
// This is O(bits_a - bits_b) iterations of subtract, much faster than bit-by-bit
__device__
void u256_mod(const unsigned long long* a, const unsigned long long* b, unsigned long long* out) {
    if (u256_is_zero(b) || u256_cmp(a, b) < 0) {
        u256_copy(a, out);
        return;
    }
    if (u256_cmp(a, b) == 0) {
        u256_from_u64(0, out);
        return;
    }

    unsigned long long rem[4];
    u256_copy(a, rem);

    unsigned int a_bits = u256_bits(a);
    unsigned int b_bits = u256_bits(b);

    // Shift b left to align with a, then do shifted subtractions
    int shift = (int)a_bits - (int)b_bits;
    if (shift < 0) {
        u256_copy(a, out);
        return;
    }

    unsigned long long shifted_b[4];
    u256_shr(b, 0, shifted_b); // copy
    // Shift b left by 'shift' bits
    if (shift > 0) {
        // left shift
        unsigned int word_shift = shift / 64;
        unsigned int bit_shift = shift % 64;
        unsigned long long tmp[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; i++) {
            int src = i - (int)word_shift;
            if (src >= 0 && src < 4) {
                tmp[i] = b[src] << bit_shift;
                if (bit_shift != 0 && src > 0) {
                    tmp[i] |= b[src - 1] >> (64 - bit_shift);
                }
            }
        }
        u256_copy(tmp, shifted_b);
    }

    for (int s = shift; s >= 0; s--) {
        if (u256_cmp(rem, shifted_b) >= 0) {
            unsigned long long tmp[4];
            u256_sub(rem, shifted_b, tmp);
            u256_copy(tmp, rem);
        }
        // Right shift shifted_b by 1
        unsigned long long tmp[4];
        u256_shr(shifted_b, 1, tmp);
        u256_copy(tmp, shifted_b);
    }

    u256_copy(rem, out);
}

// u256 division: compute quotient and remainder of a / b
__device__
void u256_divmod(const unsigned long long* a, const unsigned long long* b,
                 unsigned long long* quotient, unsigned long long* remainder) {
    u256_from_u64(0, quotient);

    if (u256_is_zero(b) || u256_cmp(a, b) < 0) {
        u256_copy(a, remainder);
        return;
    }
    if (u256_cmp(a, b) == 0) {
        u256_from_u64(1, quotient);
        u256_from_u64(0, remainder);
        return;
    }

    unsigned long long rem[4];
    u256_copy(a, rem);

    unsigned int a_bits = u256_bits(a);
    unsigned int b_bits = u256_bits(b);
    int shift = (int)a_bits - (int)b_bits;

    unsigned long long shifted_b[4];
    // Left-shift b by 'shift' bits
    {
        unsigned int word_shift = shift / 64;
        unsigned int bit_shift = shift % 64;
        for (int i = 3; i >= 0; i--) {
            int src = i - (int)word_shift;
            if (src >= 0 && src < 4) {
                shifted_b[i] = b[src] << bit_shift;
                if (bit_shift != 0 && src > 0) {
                    shifted_b[i] |= b[src - 1] >> (64 - bit_shift);
                }
            } else {
                shifted_b[i] = 0;
            }
        }
    }

    for (int s = shift; s >= 0; s--) {
        if (u256_cmp(rem, shifted_b) >= 0) {
            unsigned long long tmp[4];
            u256_sub(rem, shifted_b, tmp);
            u256_copy(tmp, rem);
            quotient[s / 64] |= 1ULL << (s % 64);
        }
        unsigned long long tmp[4];
        u256_shr(shifted_b, 1, tmp);
        u256_copy(tmp, shifted_b);
    }

    u256_copy(rem, remainder);
}

// Integer square root of a u256 value using Newton's method
__device__
void u256_isqrt(const unsigned long long* n, unsigned long long* out) {
    if (u256_is_zero(n)) {
        u256_from_u64(0, out);
        return;
    }
    unsigned int nbits = u256_bits(n);
    // Initial guess: 2^((nbits+1)/2)
    unsigned long long x[4] = {0, 0, 0, 0};
    unsigned int init_bit = (nbits + 1) / 2;
    if (init_bit < 256) {
        x[init_bit / 64] = 1ULL << (init_bit % 64);
    }

    for (int iter = 0; iter < 300; iter++) {
        // y = (x + n/x) / 2
        unsigned long long q[4], r[4];
        u256_divmod(n, x, q, r);

        unsigned long long sum[4], y[4];
        u256_add(x, q, sum);
        u256_shr(sum, 1, y);

        if (u256_cmp(y, x) >= 0) break;
        u256_copy(y, x);
    }
    u256_copy(x, out);
}

// Check if n is a perfect square; if so, write sqrt to out and return true
__device__
bool u256_is_perfect_square(const unsigned long long* n, unsigned long long* out) {
    if (u256_is_zero(n)) {
        u256_from_u64(0, out);
        return true;
    }
    u256_isqrt(n, out);
    // Verify: out * out == n
    unsigned long long product[8];
    u256_mul_schoolbook(out, out, product);
    // Check high limbs are zero and low 4 match n
    if (product[4] != 0 || product[5] != 0 || product[6] != 0 || product[7] != 0) return false;
    return product[0] == n[0] && product[1] == n[1] && product[2] == n[2] && product[3] == n[3];
}

// Cornacchia reduction on p: given sqrt(-D) mod p, find (t, s) such that
// t^2 + D*s^2 = 4p.
//
// Even path (standard Cornacchia on p):
//   Reduce (p, root) by Euclidean algorithm down to bound = isqrt(p).
//   Then check if (p - b^2) / D is a perfect square.
//   If so, t = 2*b, s = 2*sqrt((p - b^2)/D).
//
// Odd path (d mod 4 == 3, Cornacchia on 4p):
//   Ensure root is odd; reduce (2p, root) down to bound = isqrt(4p).
//   Check if (4p - t^2) / D is a perfect square, and t,s both odd.

// Solve the even path. Returns true if solution found.
// trace_out = 2*b (u256), sets_out[0..3] = s value
__device__
bool cornacchia_even(unsigned long long d_val,
                     const unsigned long long* p_u256,
                     const unsigned long long* sqrt_p_u256,
                     const unsigned long long* root_u256,
                     unsigned long long* trace_out) {
    // Ensure root > p/2; if not, use p - root
    unsigned long long half_p_approx[4];
    u256_shr(p_u256, 1, half_p_approx);

    unsigned long long r0[4];
    if (u256_cmp(root_u256, half_p_approx) <= 0) {
        u256_sub(p_u256, root_u256, r0);
    } else {
        u256_copy(root_u256, r0);
    }

    // Euclidean reduction: a = p, b = r0, reduce until b <= isqrt(p)
    unsigned long long a[4], b[4];
    u256_copy(p_u256, a);
    u256_copy(r0, b);

    while (u256_cmp(b, sqrt_p_u256) > 0) {
        // a, b = b, a mod b
        unsigned long long remainder[4];
        u256_mod(a, b, remainder);
        u256_copy(b, a);
        u256_copy(remainder, b);
    }

    // b is now <= isqrt(p)
    // Check: rem = p - b*b
    unsigned long long b_sq[8];
    u256_mul_schoolbook(b, b, b_sq);
    // b*b should fit in 256 bits since b <= isqrt(p) < p
    unsigned long long b_sq_256[4] = {b_sq[0], b_sq[1], b_sq[2], b_sq[3]};

    unsigned long long rem[4];
    unsigned int borrow = u256_sub(p_u256, b_sq_256, rem);
    if (borrow) return false;

    if (u256_is_zero(rem)) return false;

    // Check rem % d == 0
    unsigned long long q_div[4];
    unsigned long long r = u256_div_mod_u64(rem, d_val, q_div);
    if (r != 0) return false;

    // Check if q_div is a perfect square
    unsigned long long s[4];
    if (!u256_is_perfect_square(q_div, s)) return false;

    // trace = 2 * b
    unsigned long long two_b[4];
    u256_add(b, b, two_b);
    u256_copy(two_b, trace_out);
    return true;
}

// Solve the odd path (d mod 4 == 3). Returns true if found.
__device__
bool cornacchia_odd(unsigned long long d_val,
                    const unsigned long long* p_u256,
                    const unsigned long long* two_p_u256,
                    const unsigned long long* four_p_u320,
                    const unsigned long long* sqrt_four_p_u256,
                    const unsigned long long* root_u256,
                    unsigned long long* trace_out) {
    // Ensure root is odd
    unsigned long long r0[4];
    if ((root_u256[0] & 1) == 0) {
        u256_sub(p_u256, root_u256, r0);
    } else {
        u256_copy(root_u256, r0);
    }
    // Reduce (2p, r0) until b <= isqrt(4p)
    unsigned long long a[4], b[4];
    u256_copy(two_p_u256, a);
    u256_copy(r0, b);

    int steps = 0;
    while (u256_cmp(b, sqrt_four_p_u256) > 0) {
        unsigned long long remainder[4];
        u256_mod(a, b, remainder);
        u256_copy(b, a);
        u256_copy(remainder, b);
        steps++;
    }

    unsigned long long t[4];
    u256_copy(b, t);

    // Check t is odd
    if ((t[0] & 1) == 0) { return false; }

    // rem = 4p - t^2.  We use u320 for 4p.
    unsigned long long t_sq[8];
    u256_mul_schoolbook(t, t, t_sq);
    unsigned long long t_sq_320[5] = {t_sq[0], t_sq[1], t_sq[2], t_sq[3], t_sq[4]};
    unsigned long long rem320[5];
    unsigned int borrow = u320_sub(four_p_u320, t_sq_320, rem320);
    if (borrow) { return false; }

    // rem320 = 4p - t^2, may need all 5 limbs since 4p > 2^256
    // Check rem320 is not zero
    bool rem_zero = (rem320[0] == 0 && rem320[1] == 0 && rem320[2] == 0 && rem320[3] == 0 && rem320[4] == 0);
    if (rem_zero) { return false; }

    // Check rem320 % d == 0, using 5-limb division by u64
    // Divide 320-bit value by u64: process from high limb down
    {
        unsigned __int128 r_div = 0;
        unsigned long long q_div[5]; // quotient could be up to 320 bits
        for (int i = 4; i >= 0; i--) {
            unsigned __int128 wide = (r_div << 64) | (unsigned __int128)rem320[i];
            q_div[i] = (unsigned long long)(wide / (unsigned __int128)d_val);
            r_div = wide % (unsigned __int128)d_val;
        }
        unsigned long long r = (unsigned long long)r_div;
        if (r != 0) { return false; }

        // q_div should fit in 256 bits (since rem < 4p ≈ 2^257 and d >= 3)
        // Actually it could be up to ~2^257 / 3 ≈ 2^255, so it fits in 256 bits
        if (q_div[4] != 0) { return false; }

        // Check if q_div[0..3] is a perfect square
        unsigned long long q256[4] = {q_div[0], q_div[1], q_div[2], q_div[3]};
        unsigned long long s[4];
        if (!u256_is_perfect_square(q256, s)) { return false; }

        // s must be odd
        if ((s[0] & 1) == 0) { return false; }
    }

    u256_copy(t, trace_out);
    return true;
}

// ============================================================================
// Kernel 1: solve_stage_prefilter_range_compact
// ============================================================================

extern "C" __global__
void solve_stage_prefilter_range_compact(
    unsigned long long start_d, unsigned int count,
    const unsigned long long* prime_squares, unsigned int prime_square_count,
    unsigned int* compact_stats, unsigned long long* discriminants_out)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    unsigned long long d = start_d + (unsigned long long)tid;
    if (d < 3) return;

    // 1. Squarefree check
    if (!is_squarefree_check(d, prime_squares, prime_square_count)) return;

    // 2. Jacobi symbol: need jacobi(-D, p) == 1
    int j = jacobi_symbol_discriminant(d);
    if (j != 1) return;

    // 3. Local solve sieve
    if (!passes_local_solve_sieve(d)) return;

    // All filters passed: compact output
    unsigned int idx = atomicAdd(&compact_stats[0], 1U);
    discriminants_out[idx] = d;
}

// ============================================================================
// Kernel 2: solve_verified_qsieve_compact_queued
// ============================================================================

extern "C" __global__
void solve_verified_qsieve_compact_queued(
    const unsigned long long* discriminants, unsigned int count,
    const unsigned long long* primality_sieve_primes, unsigned int primality_sieve_prime_count,
    const unsigned long long* sqrt_m1_f51, const unsigned long long* p_u256,
    const unsigned long long* p_plus_one_u256, const unsigned long long* p_u320,
    const unsigned long long* four_p_u320, const unsigned long long* two_p_u256,
    const unsigned long long* half_p_u256, const unsigned long long* sqrt_p_u256,
    const unsigned long long* sqrt_four_p_u256,
    unsigned int* compact_stats, unsigned long long* compact_records)
{
    __shared__ unsigned long long s_sqrt_m1[5];
    __shared__ unsigned long long s_p[4];
    __shared__ unsigned long long s_p1[4];        // p + 1
    __shared__ unsigned long long s_two_p[4];
    __shared__ unsigned long long s_sqrt_p[4];
    __shared__ unsigned long long s_sqrt_4p[4];
    __shared__ unsigned long long s_four_p_320[5];

    // Load constants into shared memory
    if (threadIdx.x < 5) {
        s_sqrt_m1[threadIdx.x] = sqrt_m1_f51[threadIdx.x];
        s_four_p_320[threadIdx.x] = four_p_u320[threadIdx.x];
    }
    if (threadIdx.x < 4) {
        s_p[threadIdx.x] = p_u256[threadIdx.x];
        s_p1[threadIdx.x] = p_plus_one_u256[threadIdx.x];
        s_two_p[threadIdx.x] = two_p_u256[threadIdx.x];
        s_sqrt_p[threadIdx.x] = sqrt_p_u256[threadIdx.x];
        s_sqrt_4p[threadIdx.x] = sqrt_four_p_u256[threadIdx.x];
    }
    __syncthreads();

    // compact_stats[0] = total work items from prefilter
    // compact_stats[3] = compact record write counter
    unsigned int total_work = compact_stats[0];

    // Simple per-thread work assignment (no work queue to avoid __syncthreads issues)
    unsigned int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int my_idx = global_tid; my_idx < total_work; my_idx += stride) {
        unsigned long long d = discriminants[my_idx];

        // Step 1: Compute sqrt(-D) mod p using field51
        unsigned long long root_f51[5];
        bool has_root = f51_sqrt_neg_d(d, s_p, s_sqrt_m1, root_f51);

        if (has_root) {
            // Convert root to u256 for Cornacchia
            unsigned long long root_u256[4];
            f51_to_u256(root_f51, root_u256);

            unsigned long long trace_even[4] = {0, 0, 0, 0};
            unsigned long long trace_odd[4] = {0, 0, 0, 0};
            bool found_even = false;
            bool found_odd = false;

            // Step 2: Even path (standard Cornacchia on p)
            found_even = cornacchia_even(d, s_p, s_sqrt_p, root_u256, trace_even);

            // Step 3: Odd path if d mod 4 == 3
            if ((d & 3) == 3) {
                found_odd = cornacchia_odd(d, s_p, s_two_p, s_four_p_320,
                                           s_sqrt_4p, root_u256, trace_odd);

                // Check that odd trace is not the same as even trace
                if (found_odd && found_even && u256_cmp(trace_even, trace_odd) == 0) {
                    found_odd = false;
                }
            }

            // For each trace found, compute q = p + 1 - t and q' = p + 1 + t
            // Then sieve both against small primes
            unsigned long long traces[2][4];
            bool trace_found[2] = {found_even, found_odd};
            u256_copy(trace_even, traces[0]);
            u256_copy(trace_odd, traces[1]);

            for (int ti = 0; ti < 2; ti++) {
                if (!trace_found[ti]) continue;

                unsigned long long* t = traces[ti];
                unsigned long long q_pos[4], q_neg[4];
                unsigned int flags = 0;

                // q_pos = p + 1 - t
                u256_sub(s_p1, t, q_pos);

                // q_neg = p + 1 + t
                u256_add(s_p1, t, q_neg);

                // Sieve q_pos
                bool pos_pass = passes_q_sieve(q_pos, primality_sieve_primes,
                                               primality_sieve_prime_count);
                if (pos_pass) flags |= 1U;

                // Sieve q_neg
                bool neg_pass = passes_q_sieve(q_neg, primality_sieve_primes,
                                               primality_sieve_prime_count);
                if (neg_pass) flags |= 2U;

                if (flags != 0) {
                    // Write compact record: [D, t0, t1, t2, t3, flags]
                    unsigned int rec_idx = atomicAdd(&compact_stats[3], 1U);
                    unsigned int base_off = rec_idx * 6;
                    compact_records[base_off + 0] = d;
                    compact_records[base_off + 1] = t[0];
                    compact_records[base_off + 2] = t[1];
                    compact_records[base_off + 3] = t[2];
                    compact_records[base_off + 4] = t[3];
                    compact_records[base_off + 5] = (unsigned long long)flags;
                }
            }
        }
    }
}

// ============================================================================
// Kernel 3: mark_squareful_range_bits
// ============================================================================

extern "C" __global__
void mark_squareful_range_bits(
    unsigned long long start_d, unsigned int count,
    const unsigned long long* prime_squares, unsigned int prime_square_count,
    unsigned int* squareful_bits)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= prime_square_count) return;

    unsigned long long ps = prime_squares[tid];
    if (ps == 0) return;

    // Find first multiple of ps >= start_d
    unsigned long long first;
    unsigned long long rem = start_d % ps;
    if (rem == 0) {
        first = start_d;
    } else {
        first = start_d + (ps - rem);
    }

    unsigned long long end_d = start_d + (unsigned long long)count;

    // Mark all multiples of ps in [start_d, start_d+count)
    for (unsigned long long m = first; m < end_d; m += ps) {
        unsigned int offset = (unsigned int)(m - start_d);
        // Set bit in the squareful_bits bitfield
        unsigned int word_idx = offset / 32;
        unsigned int bit_idx = offset % 32;
        atomicOr(&squareful_bits[word_idx], 1U << bit_idx);
    }
}

// ============================================================================
// Kernel 4: solve_stage_compact_from_squarefree_bits
// ============================================================================

extern "C" __global__
void solve_stage_compact_from_squarefree_bits(
    unsigned long long start_d, unsigned int count,
    const unsigned int* squareful_bits,
    unsigned int* compact_stats, unsigned long long* discriminants_out)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    // Check if bit is set (squareful) -> skip
    unsigned int word_idx = tid / 32;
    unsigned int bit_idx = tid % 32;
    if ((squareful_bits[word_idx] >> bit_idx) & 1U) return;

    unsigned long long d = start_d + (unsigned long long)tid;
    if (d < 3) return;

    // Jacobi symbol check
    int j = jacobi_symbol_discriminant(d);
    if (j != 1) return;

    // Local solve sieve check
    if (!passes_local_solve_sieve(d)) return;

    // All filters passed: compact output
    unsigned int idx = atomicAdd(&compact_stats[0], 1U);
    discriminants_out[idx] = d;
}
