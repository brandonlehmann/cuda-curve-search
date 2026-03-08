# cuda-curve-search

GPU-accelerated elliptic curve cycle search over `p = 2^255 - 19`.

Searches for CM (complex multiplication) discriminant values `D` that produce 2-cycles of elliptic curves suitable for curve cycle and curve ladder constructions. Each candidate cycle consists of two curves (curve A and curve B) whose twist orders have large prime factors, providing strong twist security. Only candidates with fully factored twist orders on both curves are reported.

## Requirements

- **Rust** 1.85+ (edition 2024)
- **CUDA Toolkit** 12.0+ with `nvcc` in `PATH` (CPU-only fallback available)
- **GMP** system library (for the `rug` crate)
- **PARI/GP** (`gp` binary) -- used to factor composite twist orders that Pollard-rho cannot handle; optional but strongly recommended, as candidates with unfactorable twist orders are discarded

## Build

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ROOT="/usr/local/cuda"
cargo build --release
```

The build system auto-detects GPU compute capability via `nvidia-smi`, generates optimized sieve wheels at compile time, and compiles CUDA kernels into a fatbin. Override the detected capability with `CUDA_COMPUTE_CAP=XX`.

## Usage

```
cuda-curve-search [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--start-d` | `3` | Start of discriminant search range |
| `--end-d` | -- | End of discriminant search range |
| `--count` | -- | Number of discriminants to search from `start-d` |
| `--random` | off | Use a random starting discriminant |
| `--min-twist` | `100` | Minimum twist security bits to report |
| `--max-gamma-bits` | `127,128` | Comma-separated allowed gamma bit lengths |
| `--q-mod-8` | `3,5` | Comma-separated allowed `q mod 8` residues |
| `--rho-seconds` | `2.0` | Pollard-rho time budget per twist order (seconds) |
| `--ranking-mode` | `security` | Ranking mode: `security` or `balanced` |
| `--top` | `30` | Number of top results to maintain |
| `--output` | `candidates.jsonl` | Base output filename (see [Output](#output)) |
| `--chunk-size` | `0` (auto) | GPU chunk size; `0` auto-tunes from SM count |
| `--threads` | `0` (auto) | CPU thread count; `0` uses all available cores |
| `--progress-secs` | `2` | Progress reporting interval in seconds |
| `--cpu-only` | off | Skip GPU, use CPU-only path |
| `--profile-stages` | off | Print per-stage timing breakdown |
| `--verified-output` | -- | Path for verified results JSON |

If neither `--end-d` nor `--count` is specified, the search defaults to 100 million discriminants from `--start-d`.

### Examples

Search the first million discriminants with default filters:

```bash
./target/release/cuda-curve-search --end-d 1000000
```

Broad search with relaxed filters:

```bash
./target/release/cuda-curve-search --end-d 10000000 --min-twist 80 --max-gamma-bits 255 --q-mod-8 1,3,5,7
```

CPU-only with profiling:

```bash
./target/release/cuda-curve-search --end-d 100000 --cpu-only --profile-stages
```

Random start with a fixed count:

```bash
./target/release/cuda-curve-search --random --count 50000000
```

## How It Works

### Pipeline

1. **Prefilter (GPU)** -- For each discriminant D in the range:
   - Squarefree check (D must not be divisible by any prime square)
   - Jacobi symbol: `(-D | p) = 1`
   - Local solve sieve: modular conditions for norm equation solvability against small primes

2. **Solve (GPU)** -- For surviving D values:
   - Compute `sqrt(-D) mod p` via Tonelli-Shanks in radix-2^51 field arithmetic
   - Cornacchia's algorithm to solve `t^2 + D*s^2 = 4p`
   - Trial-divide candidate `q = p + 1 - t` against primes up to 997

3. **Finalize (CPU, parallel via rayon)** -- For GPU survivors:
   - Miller-Rabin primality test on q (15 deterministic witnesses)
   - Full twist order factorization for both curves in the cycle
   - Playbook filtering: gamma bits and `q mod 8` constraints
   - Ranking by twist security and gamma efficiency

### Mandatory Full Factorization

Every reported result has both `twist_fully_factored_curve_a: true` and `twist_fully_factored_curve_b: true`. The factorization pipeline is:

1. Trial division against primes up to 1,000,000
2. Pollard-rho with configurable time budget (`--rho-seconds`)
3. PARI/GP `factor()` for remaining composites
4. If all three fail, the candidate is **rejected**

### Resumability

Searches are fully resumable. The checkpoint system tracks scanned discriminant ranges, and existing results are reloaded on restart. Re-filtering with updated parameters (e.g., changing `--min-twist` or `--q-mod-8`) is applied to all previously scanned results without re-scanning.

## Output

All output files are derived from the `--output` base name (default: `candidates`). For example, with `--output candidates.jsonl`:

| File | Format | Content |
|------|--------|---------|
| `candidates_gpu.jsonl` | JSONL | Raw GPU scan records (discriminant + trace + flags) |
| `candidates_scanned.jsonl` | JSONL | All fully processed cycles (pre-filter) |
| `candidates_filtered.jsonl` | JSONL | Cycles passing playbook filters, deduplicated by q |
| `candidates_top.json` | JSON array | Top-N ranked candidates |
| `candidates_winner.json` | JSON object | Single best candidate |
| `candidates_checkpoint.json` | JSON object | Scanned ranges for resume |

### Sample Record

```json
{
    "discriminant": "15203",
    "t": "409295345563421029935987980112490582783",
    "q": "57896044618658097711785492504343953926225696987256860989792804023844074237167",
    "gamma": "409295345563421029935987980112490582801",
    "gamma_bits": 129,
    "q_mod_8": 7,
    "twist_bits_curve_a": 166,
    "twist_bits_curve_b": 177,
    "twist_factor_curve_a": "3^2 * 7 * 103 * 181 * 14082524851 * 52667219477 * ...",
    "twist_factor_curve_b": "3^2 * 192343 * 347072888924580647 * ...",
    "twist_fully_factored_curve_a": true,
    "twist_fully_factored_curve_b": true,
    "min_twist_bits": 166,
    "p": "57896044618658097711785492504343953926634992332820282019728792003956564819949"
}
```

### Field Reference

| Field | Description |
|-------|-------------|
| `discriminant` | CM discriminant D |
| `t` | Trace of Frobenius |
| `q` | Curve order of the second curve (curve B): `p + 1 - t` |
| `gamma` | Crandall c value where `q = 2^k - c` |
| `gamma_bits` | Bit length of gamma |
| `q_mod_8` | `q mod 8` (relevant for square root algorithms and curve properties) |
| `twist_bits_curve_a` | Largest prime factor bits of curve A twist order `(p + 1 + t)` |
| `twist_bits_curve_b` | Largest prime factor bits of curve B twist order `(q + 1 + (2 - t))` |
| `twist_factor_curve_a` | Full factorization of the curve A twist order |
| `twist_factor_curve_b` | Full factorization of the curve B twist order |
| `twist_fully_factored_curve_a` | Whether curve A twist was completely factored |
| `twist_fully_factored_curve_b` | Whether curve B twist was completely factored |
| `min_twist_bits` | `min(twist_bits_curve_a, twist_bits_curve_b)` |
| `p` | Base field prime `2^255 - 19` |
| `crandall_k` | Exponent k in Crandall form `q = 2^k - c` |
| `crandall_c` | Offset c in Crandall form |
| `playbook_tags` | Classification tags from playbook filter matching |

## Project Structure

```
cuda-curve-search/
  Cargo.toml                  # Dependencies and build config
  build.rs                    # Sieve wheel generation + nvcc compilation
  cuda/
    kernels.cu                # CUDA kernels (prefilter, solve, squarefree bitmap)
  src/
    main.rs                   # CLI, orchestration, output management, resume logic
    search.rs                 # CPU-only search path (rayon parallel)
    verify.rs                 # Post-hoc verification (class number, MOV, embedding degree)
    math.rs                   # Math module root
    math/
      types.rs                # CycleResult, Solve4pSolution, RankingMode
      sieve.rs                # Prime sieve, squarefree check, base prime P
      jacobi.rs               # Jacobi symbol, local solve sieve
      cornacchia.rs           # Tonelli-Shanks sqrt, Cornacchia norm equation solver
      primality.rs            # Miller-Rabin primality (BigInt)
      twist.rs                # Twist security: trial division + Pollard-rho + PARI/GP
      u256.rs                 # Fixed-width u256/u320/field51 arithmetic, Montgomery mul
      ranking.rs              # Result construction, ranking, Crandall analysis, tagging
    gpu.rs                    # GPU module root
    gpu/
      context.rs              # CUDA device init, fatbin loading, constant transfer
      workspace.rs            # Device buffer allocation and management
      kernels.rs              # Kernel launch wrappers and result readback
      pipeline.rs             # GPU scan loop with CPU finalization
```

## GPU Architecture

The CUDA code (`cuda/kernels.cu`) implements four kernels:

| Kernel | Purpose |
|--------|---------|
| `solve_stage_prefilter_range_compact` | Combined squarefree + Jacobi + local sieve with atomic compaction (D < 100B) |
| `mark_squareful_range_bits` | Per-prime-square bit marking into a bitmap (D >= 100B) |
| `solve_stage_compact_from_squarefree_bits` | Jacobi + sieve on squarefree survivors from bitmap (D >= 100B) |
| `solve_verified_qsieve_compact_queued` | field51 modular sqrt, Cornacchia reduction, q trial sieve; outputs 6-limb compact records |

All GPU arithmetic uses either u256 (4x u64 limbs) for integer operations or field51 (5x 51-bit limbs) for modular arithmetic mod p. Kernels use grid-stride loops and shared memory for constants.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
