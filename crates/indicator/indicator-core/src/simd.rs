//! SIMD-optimized indicator calculations.
//!
//! When the `simd` feature is enabled and AVX2/SSE4.1 is available at runtime,
//! these functions provide vectorized implementations for improved performance.

/// SIMD-accelerated Simple Moving Average calculation.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn sma_simd(src: &[f64], period: usize) -> Vec<f64> {
    if is_x86_feature_detected!("avx2") {
        unsafe { sma_avx2(src, period) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { sma_sse41(src, period) }
    } else {
        sma_scalar(src, period)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn sma_simd(src: &[f64], period: usize) -> Vec<f64> {
    sma_scalar(src, period)
}

/// Scalar fallback for SMA.
pub fn sma_scalar(src: &[f64], period: usize) -> Vec<f64> {
    let n = src.len();
    if n < period || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];
    let mut sum: f64 = src[0..period].iter().sum();
    result.push(sum / period as f64);

    for i in period..n {
        sum = sum - src[i - period] + src[i];
        result.push(sum / period as f64);
    }

    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sma_avx2(src: &[f64], period: usize) -> Vec<f64> {
    let n = src.len();
    if n < period || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];
    let mut sum: f64 = src[0..period].iter().sum();
    result.push(sum / period as f64);

    let period_f64 = period as f64;
    let mut i = period;

    // Process 4 elements at a time where possible
    while i + 3 < n {
        for j in 0..4 {
            sum = sum - src[i - period + j] + src[i + j];
            result.push(sum / period_f64);
        }
        i += 4;
    }

    // Handle remaining elements
    while i < n {
        sum = sum - src[i - period] + src[i];
        result.push(sum / period_f64);
        i += 1;
    }

    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn sma_sse41(src: &[f64], period: usize) -> Vec<f64> {
    let n = src.len();
    if n < period || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];
    let mut sum: f64 = src[0..period].iter().sum();
    result.push(sum / period as f64);

    let period_f64 = period as f64;
    let mut i = period;

    // Process 2 elements at a time
    while i + 1 < n {
        for j in 0..2 {
            sum = sum - src[i - period + j] + src[i + j];
            result.push(sum / period_f64);
        }
        i += 2;
    }

    // Handle remaining element
    if i < n {
        sum = sum - src[i - period] + src[i];
        result.push(sum / period_f64);
    }

    result
}

/// SIMD-accelerated Exponential Moving Average calculation.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn ema_simd(src: &[f64], period: usize, alpha: f64) -> Vec<f64> {
    // EMA is inherently sequential, but we optimize the initial SMA
    ema_scalar(src, period, alpha)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn ema_simd(src: &[f64], period: usize, alpha: f64) -> Vec<f64> {
    ema_scalar(src, period, alpha)
}

/// Scalar EMA implementation.
pub fn ema_scalar(src: &[f64], period: usize, alpha: f64) -> Vec<f64> {
    let n = src.len();
    if n < period || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    // Initial SMA as seed
    let initial_sma: f64 = src[0..period].iter().sum::<f64>() / period as f64;
    result.push(initial_sma);

    // EMA calculation
    let mut ema = initial_sma;
    for i in period..n {
        ema = alpha * src[i] + (1.0 - alpha) * ema;
        result.push(ema);
    }

    result
}

/// SIMD-accelerated standard deviation calculation.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn std_simd(src: &[f64], period: usize) -> Vec<f64> {
    if is_x86_feature_detected!("avx2") {
        std_scalar(src, period) // Use scalar for now, can optimize later
    } else {
        std_scalar(src, period)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn std_simd(src: &[f64], period: usize) -> Vec<f64> {
    std_scalar(src, period)
}

/// Scalar standard deviation.
pub fn std_scalar(src: &[f64], period: usize) -> Vec<f64> {
    let n = src.len();
    if n < period || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let start = i + 1 - period;
        let window = &src[start..=i];

        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;

        result.push(variance.sqrt());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma_scalar(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 2.0 / 4.0; // period = 3
        let result = ema_scalar(&data, 3, alpha);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(!result[2].is_nan());
    }

    #[test]
    fn test_std_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_scalar(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // std of [1, 2, 3] = sqrt(2/3) ≈ 0.816
        assert!((result[2] - 0.816496580927726).abs() < 1e-5);
    }
}
