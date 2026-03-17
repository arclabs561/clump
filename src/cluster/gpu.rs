//! GPU-accelerated k-means assignment via Metal compute shaders.
//!
//! Enabled by the `gpu` feature flag (macOS only). Implements the
//! FlashAssign pattern: online argmin over tiled centroids, avoiding
//! materialization of the N*K distance matrix.
//!
//! The single `unsafe` block is required for reading GPU buffer contents
//! back to CPU -- Metal's `contents()` returns a raw pointer. This is
//! sound because the buffer is allocated with `StorageModeShared` and
//! we wait for GPU completion before reading.

#[cfg(feature = "gpu")]
use metal::*;

/// Metal shader source for squared Euclidean k-means assignment.
///
/// Each thread processes one point: iterates over all centroids,
/// computing squared Euclidean distance on-the-fly and tracking the
/// running minimum (online argmin). Only the final assignment index
/// is written to output -- no intermediate N*K matrix.
#[cfg(feature = "gpu")]
const KMEANS_ASSIGN_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kmeans_assign(
    device const float* data       [[buffer(0)]],  // N x D, row-major
    device const float* centroids  [[buffer(1)]],  // K x D, row-major
    device uint*        labels     [[buffer(2)]],  // N output labels
    constant uint&      n          [[buffer(3)]],
    constant uint&      k          [[buffer(4)]],
    constant uint&      d          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    device const float* point = data + tid * d;
    float best_dist = INFINITY;
    uint  best_k    = 0;

    for (uint c = 0; c < k; c++) {
        device const float* centroid = centroids + c * d;
        float dist = 0.0;
        for (uint j = 0; j < d; j++) {
            float diff = point[j] - centroid[j];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = c;
        }
    }

    labels[tid] = best_k;
}
"#;

/// GPU-accelerated k-means assignment context.
///
/// Holds Metal device, pipeline, and command queue. Reusable across
/// iterations to avoid per-iteration setup overhead.
#[cfg(feature = "gpu")]
pub(crate) struct GpuAssigner {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

#[cfg(feature = "gpu")]
impl GpuAssigner {
    /// Create a new GPU assigner. Returns `None` if no Metal device is available.
    pub(crate) fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(KMEANS_ASSIGN_SHADER, &options)
            .ok()?;
        let function = library.get_function("kmeans_assign", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;

        Some(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Run GPU assignment: find nearest centroid for each point.
    ///
    /// `data` is N*D flattened row-major, `centroids` is K*D flattened.
    /// Returns N labels.
    #[allow(unsafe_code)]
    pub(crate) fn assign(
        &self,
        data_flat: &[f32],
        centroids_flat: &[f32],
        n: usize,
        k: usize,
        d: usize,
    ) -> Vec<usize> {
        let data_buf = self.device.new_buffer_with_data(
            data_flat.as_ptr() as *const _,
            (data_flat.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cent_buf = self.device.new_buffer_with_data(
            centroids_flat.as_ptr() as *const _,
            (centroids_flat.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let label_buf = self
            .device
            .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let d_u32 = d as u32;

        let n_buf = self.device.new_buffer_with_data(
            &n_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let k_buf = self.device.new_buffer_with_data(
            &k_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let d_buf = self.device.new_buffer_with_data(
            &d_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&cent_buf), 0);
        encoder.set_buffer(2, Some(&label_buf), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);
        encoder.set_buffer(4, Some(&k_buf), 0);
        encoder.set_buffer(5, Some(&d_buf), 0);

        let thread_group_size = self.pipeline.max_total_threads_per_threadgroup().min(256);
        let grid_size = MTLSize::new(n as u64, 1, 1);
        let group_size = MTLSize::new(thread_group_size, 1, 1);
        encoder.dispatch_threads(grid_size, group_size);
        encoder.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        // Read back labels.
        let ptr = label_buf.contents() as *const u32;
        let labels_u32 = unsafe { std::slice::from_raw_parts(ptr, n) };
        labels_u32.iter().map(|&l| l as usize).collect()
    }
}

/// Flatten `&[Vec<f32>]` into a contiguous row-major `Vec<f32>`.
#[cfg(feature = "gpu")]
pub(crate) fn flatten(data: &[Vec<f32>]) -> Vec<f32> {
    let d = data.first().map_or(0, |v| v.len());
    let mut flat = Vec::with_capacity(data.len() * d);
    for row in data {
        flat.extend_from_slice(row);
    }
    flat
}
