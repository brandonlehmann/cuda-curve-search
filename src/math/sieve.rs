use std::sync::Mutex;

pub const SMALL_PRIME_LIMIT: usize = 1_000_000;
const SQUAREFREE_PRIME_SEGMENT_LEN: u64 = 1_000_000;

lazy_static::lazy_static! {
    pub static ref P: num_bigint::BigInt = num_bigint::BigInt::from(2u64).pow(255) - num_bigint::BigInt::from(19u64);
    pub static ref SMALL_PRIMES: Vec<u64> = sieve_primes(SMALL_PRIME_LIMIT);
    pub static ref PRIMALITY_SIEVE_PRIMES: Vec<u64> = SMALL_PRIMES
        .iter()
        .copied()
        .take_while(|&p| p <= 997)
        .collect();
    static ref SQUAREFREE_PRIME_CACHE: Mutex<SquarefreePrimeCache> =
        Mutex::new(SquarefreePrimeCache::new());
}

pub fn is_squarefree(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let max_prime = isqrt_u64(n);
    let mut cache = SQUAREFREE_PRIME_CACHE
        .lock()
        .expect("squarefree prime cache lock poisoned");
    cache.ensure(max_prime);
    for &p in cache.primes.iter() {
        if p > max_prime {
            break;
        }
        let prime_square = p * p;
        if n % prime_square == 0 {
            return false;
        }
    }
    true
}

pub fn isqrt_u64(n: u64) -> u64 {
    let mut root = (n as f64).sqrt() as u64;
    while ((root as u128) + 1) * ((root as u128) + 1) <= n as u128 {
        root += 1;
    }
    while (root as u128) * (root as u128) > n as u128 {
        root -= 1;
    }
    root
}

struct SquarefreePrimeCache {
    limit: u64,
    primes: Vec<u64>,
}

impl SquarefreePrimeCache {
    fn new() -> Self {
        Self {
            limit: SMALL_PRIME_LIMIT as u64,
            primes: SMALL_PRIMES.clone(),
        }
    }

    fn ensure(&mut self, limit: u64) {
        if limit <= self.limit {
            return;
        }
        let base_limit = isqrt_u64(limit);
        if self.limit < base_limit {
            self.ensure(base_limit);
        }
        let mut segment_start = self.limit.saturating_add(1).max(2);
        while segment_start <= limit {
            let segment_end = segment_start
                .saturating_add(SQUAREFREE_PRIME_SEGMENT_LEN - 1)
                .min(limit);
            let mut is_prime = vec![true; (segment_end - segment_start + 1) as usize];
            let mut prime_index = 0usize;
            while prime_index < self.primes.len() {
                let prime = self.primes[prime_index];
                let prime_square = prime.saturating_mul(prime);
                if prime_square > segment_end {
                    break;
                }
                let mut composite = prime_square.max(segment_start.div_ceil(prime) * prime);
                while composite <= segment_end {
                    is_prime[(composite - segment_start) as usize] = false;
                    composite = composite.saturating_add(prime);
                }
                prime_index += 1;
            }
            for (offset, is_prime) in is_prime.into_iter().enumerate() {
                if !is_prime {
                    continue;
                }
                self.primes.push(segment_start + offset as u64);
            }
            segment_start = segment_end.saturating_add(1);
        }
        self.limit = limit;
    }
}

pub fn sieve_primes(limit: usize) -> Vec<u64> {
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit > 0 {
        is_prime[1] = false;
    }
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    (2..=limit)
        .filter(|&i| is_prime[i])
        .map(|i| i as u64)
        .collect()
}

pub fn prime_squares_host_up_to(max_discriminant: u64) -> Vec<u64> {
    let max_prime = isqrt_u64(max_discriminant.max(4));
    let mut cache = SQUAREFREE_PRIME_CACHE
        .lock()
        .expect("squarefree prime cache lock poisoned");
    cache.ensure(max_prime);
    cache
        .primes
        .iter()
        .copied()
        .take_while(|&prime| prime <= max_prime)
        .map(|prime| prime * prime)
        .collect()
}

pub fn primality_sieve_primes_host() -> Vec<u64> {
    PRIMALITY_SIEVE_PRIMES.clone()
}
