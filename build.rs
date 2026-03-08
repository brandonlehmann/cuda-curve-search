use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const LOCAL_SOLVE_SIEVE_WHEEL_PRIMES: [u64; 5] = [7, 11, 13, 17, 19];
const SQUAREFREE_SIEVE_WHEEL_PRIMES: [u64; 5] = [2, 3, 5, 7, 11];

fn detect_compute_capability_digits() -> String {
    if let Ok(value) = env::var("CUDA_COMPUTE_CAP") {
        let digits: String = value.chars().filter(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return digits;
        }
    }

    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            if let Some(line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                let digits: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
                if !digits.is_empty() {
                    return digits;
                }
            }
        }
    }

    "75".to_string()
}

fn write_local_solve_sieve_wheel_include(out_dir: &PathBuf) -> PathBuf {
    let modulus = LOCAL_SOLVE_SIEVE_WHEEL_PRIMES
        .iter()
        .copied()
        .product::<u64>();
    let word_count = modulus.div_ceil(64) as usize;
    let mut words = vec![0u64; word_count];

    for residue in 0..modulus {
        if LOCAL_SOLVE_SIEVE_WHEEL_PRIMES
            .iter()
            .all(|prime| residue % prime != 0)
        {
            let word_index = (residue / 64) as usize;
            let bit_index = (residue % 64) as u32;
            words[word_index] |= 1u64 << bit_index;
        }
    }

    let mut include = String::new();
    writeln!(
        &mut include,
        "constexpr unsigned int LOCAL_SOLVE_SIEVE_WHEEL_PRIME_PREFIX_COUNT = {}U;",
        LOCAL_SOLVE_SIEVE_WHEEL_PRIMES.len()
    )
    .unwrap();
    writeln!(
        &mut include,
        "constexpr unsigned int LOCAL_SOLVE_SIEVE_WHEEL_MODULUS = {}U;",
        modulus
    )
    .unwrap();
    writeln!(
        &mut include,
        "constexpr unsigned int LOCAL_SOLVE_SIEVE_WHEEL_WORD_COUNT = {}U;",
        word_count
    )
    .unwrap();
    include.push_str(
        "__device__ __constant__ unsigned long long LOCAL_SOLVE_SIEVE_WHEEL_BITS[LOCAL_SOLVE_SIEVE_WHEEL_WORD_COUNT] = {",
    );
    for (idx, word) in words.iter().enumerate() {
        if idx % 4 == 0 {
            include.push_str("\n    ");
        } else {
            include.push(' ');
        }
        write!(&mut include, "0x{word:016X}ULL").unwrap();
        if idx + 1 != words.len() {
            include.push(',');
        }
    }
    include.push_str("\n};\n");

    let path = out_dir.join("local_solve_sieve_wheel.inc");
    fs::write(&path, include).expect("failed to write generated local solve sieve wheel include");
    path
}

fn write_squarefree_sieve_wheel_include(out_dir: &PathBuf) -> PathBuf {
    let modulus = SQUAREFREE_SIEVE_WHEEL_PRIMES
        .iter()
        .copied()
        .map(|prime| prime * prime)
        .product::<u64>();
    let word_count = modulus.div_ceil(64) as usize;
    let mut words = vec![0u64; word_count];

    for residue in 0..modulus {
        if SQUAREFREE_SIEVE_WHEEL_PRIMES
            .iter()
            .all(|prime| residue % (prime * prime) != 0)
        {
            let word_index = (residue / 64) as usize;
            let bit_index = (residue % 64) as u32;
            words[word_index] |= 1u64 << bit_index;
        }
    }

    let mut include = String::new();
    writeln!(
        &mut include,
        "constexpr unsigned int SQUAREFREE_SIEVE_WHEEL_PRIME_PREFIX_COUNT = {}U;",
        SQUAREFREE_SIEVE_WHEEL_PRIMES.len()
    )
    .unwrap();
    writeln!(
        &mut include,
        "constexpr unsigned int SQUAREFREE_SIEVE_WHEEL_MODULUS = {}U;",
        modulus
    )
    .unwrap();
    writeln!(
        &mut include,
        "constexpr unsigned int SQUAREFREE_SIEVE_WHEEL_WORD_COUNT = {}U;",
        word_count
    )
    .unwrap();
    include.push_str(
        "__device__ const unsigned long long SQUAREFREE_SIEVE_WHEEL_BITS[SQUAREFREE_SIEVE_WHEEL_WORD_COUNT] = {",
    );
    for (idx, word) in words.iter().enumerate() {
        if idx % 4 == 0 {
            include.push_str("\n    ");
        } else {
            include.push(' ');
        }
        write!(&mut include, "0x{word:016X}ULL").unwrap();
        if idx + 1 != words.len() {
            include.push(',');
        }
    }
    include.push_str("\n};\n");

    let path = out_dir.join("squarefree_sieve_wheel.inc");
    fs::write(&path, include).expect("failed to write generated squarefree sieve wheel include");
    path
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVCC");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    write_local_solve_sieve_wheel_include(&out_dir);
    write_squarefree_sieve_wheel_include(&out_dir);
    let output = out_dir.join("kernels.fatbin");
    let compute_capability = detect_compute_capability_digits();
    let compute_arch = format!("compute_{compute_capability}");
    let sm_arch = format!("sm_{compute_capability}");
    let gencode_sm = format!("-gencode=arch={compute_arch},code={sm_arch}");
    let gencode_ptx = format!("-gencode=arch={compute_arch},code={compute_arch}");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let status = Command::new(&nvcc)
        .args(["-fatbin", "-std=c++17", "-O3"])
        .arg(format!("-I{}", out_dir.display()))
        .arg(&gencode_sm)
        .arg(&gencode_ptx)
        .arg("cuda/kernels.cu")
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap_or_else(|err| panic!("failed to invoke nvcc at {nvcc}: {err}"));

    if !status.success() {
        panic!("nvcc failed to compile cuda/kernels.cu");
    }
}
