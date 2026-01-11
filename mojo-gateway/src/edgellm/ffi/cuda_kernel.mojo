"""
CUDA Kernel FFI Wrapper

Mojo wrapper for the high-performance CUDA kernel.
Provides GPU acceleration for BitNet inference on NVIDIA devices.

Target: Jetson Nano/Orin, RTX GPUs (80-400 tok/s)
"""

from sys.ffi import DLHandle, c_char
from memory import UnsafePointer


# Kernel library handle (lazy loaded)
var _cuda_handle: DLHandle = DLHandle()
var _cuda_loaded: Bool = False
var _cuda_available: Bool = False
var _cuda_checked: Bool = False


fn _load_cuda_kernel() raises -> DLHandle:
    """Load the CUDA kernel shared library."""
    var paths = List[String](
        "./lib/libtmac_kernel_cuda.dylib",
        "./lib/libtmac_kernel_cuda.so",
        "/usr/local/lib/libtmac_kernel_cuda.dylib",
        "/usr/local/lib/libtmac_kernel_cuda.so",
        "libtmac_kernel_cuda.dylib",
        "libtmac_kernel_cuda.so",
    )

    for i in range(len(paths)):
        try:
            return DLHandle(paths[i])
        except:
            continue

    raise Error("CUDA kernel not found. Build with 'make' in src/kernels/cuda/")


fn get_cuda_kernel() raises -> DLHandle:
    """Get or load the CUDA kernel library."""
    if not _cuda_loaded:
        _cuda_handle = _load_cuda_kernel()
        _cuda_loaded = True
    return _cuda_handle


fn cuda_available() -> Bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA device is available and kernel is loaded.
    """
    if _cuda_checked:
        return _cuda_available

    try:
        var kernel = get_cuda_kernel()
        _cuda_available = kernel.call["cuda_available", Int]() == 1
    except:
        _cuda_available = False

    _cuda_checked = True
    return _cuda_available


fn cuda_init(
    max_weights_bytes: Int,
    max_activations: Int,
    max_output: Int,
) raises -> Bool:
    """
    Initialize CUDA context and allocate device memory.

    Args:
        max_weights_bytes: Maximum size of weight buffer in bytes
        max_activations: Maximum number of activation elements
        max_output: Maximum number of output elements

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["cuda_init", Int](
        max_weights_bytes, max_activations, max_output
    )
    return result == 0


fn cuda_cleanup() raises:
    """Cleanup CUDA resources."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_cleanup", NoneType]()


fn cuda_device_name() raises -> String:
    """
    Get CUDA device name.

    Returns:
        Device name string.
    """
    if not cuda_available():
        return "No CUDA device"

    var kernel = get_cuda_kernel()
    var name_ptr = kernel.call["cuda_device_name", UnsafePointer[c_char]]()
    # Convert C string to Mojo String
    return String(name_ptr)


fn cuda_sync() raises:
    """Synchronize CUDA device."""
    if _cuda_loaded:
        var kernel = get_cuda_kernel()
        kernel.call["cuda_sync", NoneType]()


fn tmac_matmul_cuda(
    output: UnsafePointer[Float32],
    weights: UnsafePointer[Int8],
    activations: UnsafePointer[Float32],
    scales: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
) raises -> Bool:
    """
    T-MAC Matrix Multiplication using CUDA.

    GPU-accelerated BitNet inference using table lookups.

    Args:
        output: Output buffer [M * N]
        weights: Packed ternary weights [M * (K/4)]
        activations: Input activations [K * N]
        scales: Per-row scaling factors [M]
        M: Number of output rows
        N: Number of output columns (batch size)
        K: Inner dimension

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["tmac_matmul_cuda", Int](
        weights, activations, output, scales, M, N, K
    )
    return result == 0


fn rmsnorm_cuda(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    batch_size: Int,
    size: Int,
    eps: Float32 = 1e-6,
) raises -> Bool:
    """
    CUDA-accelerated RMSNorm.

    output[i] = (input[i] / rms) * weight[i]
    where rms = sqrt(mean(input^2) + eps)

    Args:
        output: Output buffer [batch_size * size]
        input: Input buffer [batch_size * size]
        weight: Weight buffer [size]
        batch_size: Number of batches
        size: Vector size per batch
        eps: Epsilon for numerical stability

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["rmsnorm_cuda", Int](
        output, input, weight, batch_size, size, eps
    )
    return result == 0


fn softmax_cuda(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    batch_size: Int,
    size: Int,
) raises -> Bool:
    """
    CUDA-accelerated Softmax.

    Numerically stable: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        output: Output buffer [batch_size * size]
        input: Input buffer [batch_size * size]
        batch_size: Number of batches
        size: Vector size per batch

    Returns:
        True on success, False on failure.
    """
    if not cuda_available():
        return False

    var kernel = get_cuda_kernel()
    var result = kernel.call["softmax_cuda", Int](
        output, input, batch_size, size
    )
    return result == 0


# ============================================================================
# Device Info Utilities
# ============================================================================

struct CUDADeviceInfo:
    """CUDA device information."""
    var name: String
    var total_memory_gb: Float64
    var sm_count: Int
    var compute_major: Int
    var compute_minor: Int
    var available: Bool

    fn __init__(out self):
        self.name = "Unknown"
        self.total_memory_gb = 0.0
        self.sm_count = 0
        self.compute_major = 0
        self.compute_minor = 0
        self.available = False


fn get_cuda_device_info() raises -> CUDADeviceInfo:
    """
    Get comprehensive CUDA device information.

    Returns:
        CUDADeviceInfo struct with device details.
    """
    var info = CUDADeviceInfo()

    if not cuda_available():
        return info

    info.available = True
    info.name = cuda_device_name()

    var kernel = get_cuda_kernel()

    # Get device properties
    var total_mem = UnsafePointer[Int].alloc(1)
    var sm_count = UnsafePointer[Int].alloc(1)
    var major = UnsafePointer[Int].alloc(1)
    var minor = UnsafePointer[Int].alloc(1)

    var result = kernel.call["cuda_device_info", Int](
        total_mem, sm_count, major, minor
    )

    if result == 0:
        info.total_memory_gb = Float64(total_mem[0]) / (1024.0 * 1024.0 * 1024.0)
        info.sm_count = sm_count[0]
        info.compute_major = major[0]
        info.compute_minor = minor[0]

    total_mem.free()
    sm_count.free()
    major.free()
    minor.free()

    return info


fn print_cuda_info() raises:
    """Print CUDA device information."""
    var info = get_cuda_device_info()

    if not info.available:
        print("CUDA: Not available")
        return

    print("CUDA Device Information:")
    print("  Name:", info.name)
    print("  Memory:", info.total_memory_gb, "GB")
    print("  SM Count:", info.sm_count)
    print("  Compute:", info.compute_major, ".", info.compute_minor)
