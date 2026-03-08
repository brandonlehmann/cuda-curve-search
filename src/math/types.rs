use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CycleResult {
    pub discriminant: String,
    pub t: String,
    pub q: String,
    pub gamma: String,
    pub gamma_bits: u32,
    pub q_mod_8: u8,
    pub twist_bits_curve_a: u32,
    pub twist_bits_curve_b: u32,
    pub twist_factor_curve_a: String,
    pub twist_factor_curve_b: String,
    pub twist_fully_factored_curve_a: bool,
    pub twist_fully_factored_curve_b: bool,
    pub min_twist_bits: u32,
    pub p: String,
    pub discriminant_u64: u64,
    pub crandall_k: u32,
    #[serde(default)]
    pub crandall_c: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub playbook_tags: Vec<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RankingMode {
    Security,
    Balanced,
}

impl RankingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Security => "security",
            Self::Balanced => "balanced",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

impl ExecutionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchSummary {
    pub total_discriminants: u64,
    pub total_cycles_found: usize,
    pub results: Vec<CycleResult>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchProfile {
    pub discriminants_checked: u64,
    pub squarefree_passes: u64,
    pub jacobi_passes: u64,
    pub solve_4p_passes: u64,
    pub q_candidates: u64,
    pub prime_q_passes: u64,
    pub twist_candidates: u64,
    pub squarefree_time: Duration,
    pub jacobi_time: Duration,
    pub solve_4p_time: Duration,
    pub q_prime_time: Duration,
    pub twist_time: Duration,
}

impl SearchProfile {
    pub fn total_measured_time(&self) -> Duration {
        self.squarefree_time
            + self.jacobi_time
            + self.solve_4p_time
            + self.q_prime_time
            + self.twist_time
    }

    pub fn measured_d_per_second(&self) -> Option<f64> {
        let total = self.total_measured_time().as_secs_f64();
        if total == 0.0 {
            None
        } else {
            Some(self.discriminants_checked as f64 / total)
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.discriminants_checked += other.discriminants_checked;
        self.squarefree_passes += other.squarefree_passes;
        self.jacobi_passes += other.jacobi_passes;
        self.solve_4p_passes += other.solve_4p_passes;
        self.q_candidates += other.q_candidates;
        self.prime_q_passes += other.prime_q_passes;
        self.twist_candidates += other.twist_candidates;
        self.squarefree_time += other.squarefree_time;
        self.jacobi_time += other.jacobi_time;
        self.solve_4p_time += other.solve_4p_time;
        self.q_prime_time += other.q_prime_time;
        self.twist_time += other.twist_time;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfiledSearchSummary {
    pub summary: SearchSummary,
    pub profile: SearchProfile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solve4pSolution {
    pub t: BigInt,
    pub s: BigInt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solve4pWorkItem {
    pub discriminant: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solve4pWorkResult {
    pub discriminant: u64,
    pub solutions: Vec<Solve4pSolution>,
}

pub trait Solve4pKernelBackend: Send + Sync {
    fn label(&self) -> &'static str;
    fn solve_batch(&self, work: &[Solve4pWorkItem]) -> Vec<Solve4pWorkResult>;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct CpuBigIntSolve4pBackend;

pub const CPU_BIGINT_SOLVE4P_BACKEND: CpuBigIntSolve4pBackend = CpuBigIntSolve4pBackend;
