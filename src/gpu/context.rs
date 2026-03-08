use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaModule, CudaSlice};

use crate::math::sieve::{prime_squares_host_up_to, primality_sieve_primes_host};
use crate::math::u256::*;

const DEFAULT_SQUAREFREE_COVERAGE_END: u64 = 1_000_000_000_000;

const KERNELS_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/kernels.fatbin"));

pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub module: Arc<CudaModule>,
    pub prime_squares: CudaSlice<u64>,
    pub prime_square_count: u32,
    pub primality_sieve_primes: CudaSlice<u64>,
    pub primality_sieve_prime_count: u32,
    pub sqrt_m1: CudaSlice<u64>,
    pub p_u256: CudaSlice<u64>,
    pub p_plus_one_u256: CudaSlice<u64>,
    pub p_u320: CudaSlice<u64>,
    pub four_p_u320: CudaSlice<u64>,
    pub two_p_u256: CudaSlice<u64>,
    pub half_p_u256: CudaSlice<u64>,
    pub sqrt_p_u256: CudaSlice<u64>,
    pub sqrt_four_p_u256: CudaSlice<u64>,
    pub sm_count: u32,
    pub device_name: String,
}

impl GpuContext {
    pub fn try_new() -> Option<Self> {
        Self::new().ok()
    }

    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Load the fatbin
        let ptx = cudarc::nvrtc::Ptx::from_binary(KERNELS_FATBIN.to_vec());
        let module = ctx.load_module(ptx)?;

        let prime_squares_host = prime_squares_host_up_to(DEFAULT_SQUAREFREE_COVERAGE_END);
        let prime_square_count = prime_squares_host.len() as u32;
        let prime_squares = stream.clone_htod(&prime_squares_host)?;

        let psp_host = primality_sieve_primes_host();
        let primality_sieve_prime_count = psp_host.len() as u32;
        let primality_sieve_primes = stream.clone_htod(&psp_host)?;

        let sqrt_m1_host = sqrt_m1_field51_host();
        let sqrt_m1 = stream.clone_htod(&sqrt_m1_host)?;

        let p_u256_data = curve_prime_u256();
        let p_u256 = stream.clone_htod(&p_u256_data)?;
        let p_plus_one_data = curve_prime_plus_one_u256();
        let p_plus_one_u256 = stream.clone_htod(&p_plus_one_data)?;
        let p_u320_data = curve_prime_u320();
        let p_u320 = stream.clone_htod(&p_u320_data)?;
        let four_p_data = curve_four_p_u320();
        let four_p_u320 = stream.clone_htod(&four_p_data)?;
        let two_p_data = curve_two_p_u256();
        let two_p_u256 = stream.clone_htod(&two_p_data)?;
        let half_p_data = curve_half_floor_u256();
        let half_p_u256 = stream.clone_htod(&half_p_data)?;
        let sqrt_p_data = curve_sqrt_p_u256();
        let sqrt_p_u256 = stream.clone_htod(&sqrt_p_data)?;
        let sqrt_four_p_data = curve_sqrt_four_p_u256();
        let sqrt_four_p_u256 = stream.clone_htod(&sqrt_four_p_data)?;

        // Get SM count and device name
        let sm_count = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let device_name = ctx.name()?;

        Ok(Self {
            ctx,
            stream,
            module,
            prime_squares,
            prime_square_count,
            primality_sieve_primes,
            primality_sieve_prime_count,
            sqrt_m1,
            p_u256,
            p_plus_one_u256,
            p_u320,
            four_p_u320,
            two_p_u256,
            half_p_u256,
            sqrt_p_u256,
            sqrt_four_p_u256,
            sm_count,
            device_name,
        })
    }
}
