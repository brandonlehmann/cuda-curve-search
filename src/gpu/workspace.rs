use std::sync::Arc;
use cudarc::driver::{CudaSlice, CudaStream};

use super::context::GpuContext;

pub const COMPACT_RECORD_LIMBS: usize = 6;
pub const COMPACT_STATS_COUNT: usize = 4;
pub const COMPACT_STAT_SOLVE_STAGE: usize = 0;
pub const COMPACT_STAT_VERIFIED: usize = 1;
pub const COMPACT_STAT_Q_SURVIVORS: usize = 2;
pub const COMPACT_STAT_RECORD_COUNT: usize = 3;
pub const COMPACT_FLAG_POSITIVE_Q: u64 = 1;
pub const COMPACT_FLAG_NEGATIVE_Q: u64 = 2;

pub struct GpuWorkspace {
    pub stream: Arc<CudaStream>,
    pub discriminants_device: CudaSlice<u64>,
    pub squareful_bits_device: CudaSlice<u32>,
    pub compact_stats_device: CudaSlice<u32>,
    pub compact_records_device: CudaSlice<u64>,
    pub max_chunk_size: usize,
    pub max_compact_records: usize,
}

impl GpuWorkspace {
    pub fn new(gpu: &GpuContext, max_chunk_size: usize) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let stream = gpu.ctx.new_stream()?;
        let max_compact_records = max_chunk_size * 2;
        let squareful_bit_words = max_chunk_size.div_ceil(32);

        let discriminants_device = stream.alloc_zeros::<u64>(max_chunk_size)?;
        let squareful_bits_device = stream.alloc_zeros::<u32>(squareful_bit_words)?;
        let compact_stats_device = stream.alloc_zeros::<u32>(COMPACT_STATS_COUNT)?;
        let compact_records_device = stream.alloc_zeros::<u64>(max_compact_records * COMPACT_RECORD_LIMBS)?;

        Ok(Self {
            stream,
            discriminants_device,
            squareful_bits_device,
            compact_stats_device,
            compact_records_device,
            max_chunk_size,
            max_compact_records,
        })
    }
}
