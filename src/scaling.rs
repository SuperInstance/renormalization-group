//! # Finite-Size Scaling and Scaling Relations
//!
//! Critical exponents, scaling collapse, and finite-size corrections
//! for bounded fleets.

use serde::{Deserialize, Serialize};

/// A critical exponent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalExponent {
    /// Name of the exponent
    pub name: String,
    /// Numerical value
    pub value: f64,
}

impl CriticalExponent {
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// Scaling function result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingResult {
    /// Scaled x values
    pub x_scaled: Vec<f64>,
    /// Scaled y values
    pub y_scaled: Vec<f64>,
    /// Exponents used for scaling
    pub exponents_used: Vec<CriticalExponent>,
}

/// Scaling collapse: given data at different system sizes, collapse
/// onto a universal curve using finite-size scaling.
///
/// For an observable O(t, L) where t = (T - Tc)/Tc and L is system size:
///   O(t, L) = L^(γ/ν) * f(t * L^(1/ν))
///
/// This function computes the scaled coordinates.
pub fn scaling_collapse(
    temperatures: &[f64],
    observables: &[f64],
    system_sizes: &[usize],
    tc: f64,
    nu: f64,
    gamma_over_nu: f64,
) -> ScalingResult {
    let mut x_scaled = Vec::new();
    let mut y_scaled = Vec::new();

    for (i, &t) in temperatures.iter().enumerate() {
        if i >= observables.len() || i >= system_sizes.len() {
            break;
        }
        let l = system_sizes[i] as f64;
        let reduced_t = (t - tc) / tc;

        // x = t * L^(1/ν)
        let x = reduced_t * l.powf(1.0 / nu);
        // y = O / L^(γ/ν)
        let y = observables[i] / l.powf(gamma_over_nu);

        x_scaled.push(x);
        y_scaled.push(y);
    }

    ScalingResult {
        x_scaled,
        y_scaled,
        exponents_used: vec![
            CriticalExponent::new("nu", nu),
            CriticalExponent::new("gamma/nu", gamma_over_nu),
        ],
    }
}

/// Compute the finite-size scaling form of the order parameter.
/// M(L) ~ L^(-β/ν) at criticality.
pub fn finite_size_order_parameter(
    system_size: usize,
    beta: f64,
    nu: f64,
) -> f64 {
    let l = system_size as f64;
    l.powf(-beta / nu)
}

/// Compute the finite-size scaling form of the susceptibility.
/// χ(L) ~ L^(γ/ν) at criticality.
pub fn finite_size_susceptibility(
    system_size: usize,
    gamma: f64,
    nu: f64,
) -> f64 {
    let l = system_size as f64;
    l.powf(gamma / nu)
}

/// Compute the finite-size scaling form of the correlation length.
/// ξ(L) ~ L at criticality (trivially bounded by system size).
pub fn finite_size_correlation_length(system_size: usize) -> f64 {
    system_size as f64
}

/// Compute the Binder cumulant for finite-size analysis.
/// U_L = 1 - <m^4> / (3 * <m^2>^2)
/// At criticality, U_L approaches a universal value independent of L.
pub fn binder_cumulant(m2: f64, m4: f64) -> f64 {
    if m2.abs() < 1e-15 {
        return 0.0;
    }
    1.0 - m4 / (3.0 * m2 * m2)
}

/// Verify hyperscaling: 2 - α = dν (valid for d < 4).
pub fn verify_hyperscaling(d: usize, alpha: f64, nu: f64) -> bool {
    ((2.0 - alpha) - d as f64 * nu).abs() < 0.05
}

/// Extract critical temperature from finite-size data using Binder cumulant crossing.
/// The Binder cumulant curves for different L cross at Tc.
pub fn find_tc_from_crossing(
    temperatures: &[f64],
    binder_l1: &[f64],
    binder_l2: &[f64],
) -> Option<f64> {
    if temperatures.len() < 2 || binder_l1.len() != temperatures.len() || binder_l2.len() != temperatures.len() {
        return None;
    }

    // Find where binder_l1 and binder_l2 cross
    for i in 0..temperatures.len() - 1 {
        let diff1 = binder_l1[i] - binder_l2[i];
        let diff2 = binder_l1[i + 1] - binder_l2[i + 1];

        if diff1 * diff2 < 0.0 {
            // Linear interpolation
            let t = diff1.abs() / (diff1.abs() + diff2.abs());
            let tc = temperatures[i] + t * (temperatures[i + 1] - temperatures[i]);
            return Some(tc);
        }
    }
    None
}

/// Compute the correlation function for a 1D system.
/// C(r) = <s_i s_{i+r}> - <s_i>^2
pub fn correlation_function(spins: &[f64], max_r: usize) -> Vec<f64> {
    let n = spins.len();
    let mean: f64 = spins.iter().sum::<f64>() / n as f64;
    let mut corr = Vec::with_capacity(max_r);

    for r in 0..max_r {
        if r >= n {
            corr.push(0.0);
            continue;
        }
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..(n - r) {
            sum += spins[i] * spins[i + r];
            count += 1;
        }
        let c = if count > 0 { sum / count as f64 - mean * mean } else { 0.0 };
        corr.push(c);
    }
    corr
}

/// Extract correlation length from a correlation function.
/// Fits C(r) ~ exp(-r/ξ) and returns ξ.
pub fn extract_correlation_length(corr: &[f64]) -> f64 {
    if corr.len() < 2 {
        return 0.0;
    }
    // Use first two nonzero points
    let mut r0 = 0;
    while r0 < corr.len() && corr[r0].abs() < 1e-15 {
        r0 += 1;
    }
    if r0 >= corr.len() - 1 {
        return 0.0;
    }
    let c0 = corr[r0].abs();
    let r1 = r0 + 1;
    let c1 = corr[r1].abs();

    if c0 < 1e-15 || c1 < 1e-15 {
        return 1.0; // Very short correlation
    }
    // ln(C(r+1)/C(r)) = -1/ξ
    let ratio = c1 / c0;
    if ratio <= 0.0 {
        return 1.0;
    }
    let xi = -1.0 / ratio.ln();
    if xi.is_nan() || xi.is_infinite() || xi < 0.0 {
        return f64::MAX;
    }
    xi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_collapse_mean_field() {
        // Mean field: ν=0.5, γ=1.0, γ/ν=2.0
        let temps = vec![0.9, 0.95, 1.05, 1.1];
        let obs = vec![100.0, 400.0, 400.0, 100.0];
        let sizes = vec![10, 20, 20, 10];
        let result = scaling_collapse(&temps, &obs, &sizes, 1.0, 0.5, 2.0);
        assert_eq!(result.x_scaled.len(), 4);
        assert_eq!(result.y_scaled.len(), 4);
    }

    #[test]
    fn test_finite_size_order_parameter_2d_ising() {
        // 2D Ising: β = 1/8, ν = 1
        // M(L) ~ L^(-1/8)
        let m10 = finite_size_order_parameter(10, 1.0 / 8.0, 1.0);
        let m100 = finite_size_order_parameter(100, 1.0 / 8.0, 1.0);
        // Ratio should be (100/10)^(-1/8) = 10^(-1/8) ≈ 0.7499
        let ratio = m100 / m10;
        let expected = 10.0_f64.powf(-1.0 / 8.0);
        assert!((ratio - expected).abs() < 0.01);
    }

    #[test]
    fn test_finite_size_susceptibility_mean_field() {
        // Mean field: γ=1, ν=0.5, γ/ν=2
        let chi10 = finite_size_susceptibility(10, 1.0, 0.5);
        let chi100 = finite_size_susceptibility(100, 1.0, 0.5);
        // Ratio: (100/10)^2 = 100
        let ratio = chi100 / chi10;
        assert!((ratio - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_binder_cumulant_uniform() {
        // All spins aligned: m = ±1, m^2 = 1, m^4 = 1
        let u = binder_cumulant(1.0, 1.0);
        assert!((u - (1.0 - 1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_binder_cumulant_gaussian() {
        // Gaussian distribution: U = 0 for Gaussian
        // <m^4> = 3<m^2>^2 for Gaussian → U = 1 - 3<m^2>^2/(3<m^2>^2) = 0
        let u = binder_cumulant(1.0, 3.0);
        assert!(u.abs() < 1e-10);
    }

    #[test]
    fn test_hyperscaling_2d_ising() {
        // d=2, α=0, ν=1: 2-0=2, 2*1=2 ✓
        assert!(verify_hyperscaling(2, 0.0, 1.0));
    }

    #[test]
    fn test_hyperscaling_mean_field() {
        // d=4, α=0, ν=0.5: 2-0=2, 4*0.5=2 ✓
        assert!(verify_hyperscaling(4, 0.0, 0.5));
    }

    #[test]
    fn test_correlation_function_constant() {
        let spins = vec![1.0; 100];
        let corr = correlation_function(&spins, 10);
        // C(0) should be <s^2> - <s>^2 = 1 - 1 = 0
        assert!(corr[0].abs() < 1e-10);
    }

    #[test]
    fn test_correlation_function_alternating() {
        let spins: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let corr = correlation_function(&spins, 5);
        // C(1) should be negative (anticorrelated)
        assert!(corr[1] < 0.0);
        // C(2) should be positive
        assert!(corr[2] > 0.0);
    }

    #[test]
    fn test_find_tc_from_crossing() {
        // binder_l1 increases with T, binder_l2 decreases with T → they cross
        let temps = vec![0.8, 0.9, 1.0, 1.1, 1.2];
        let binder_l1 = vec![0.3, 0.45, 0.55, 0.65, 0.75]; // increasing
        let binder_l2 = vec![0.7, 0.55, 0.45, 0.35, 0.25]; // decreasing
        // They cross between T=0.9 and T=1.0
        let tc = find_tc_from_crossing(&temps, &binder_l1, &binder_l2);
        assert!(tc.is_some());
        let tc = tc.unwrap();
        assert!((tc - 0.95).abs() < 0.1);
    }

    #[test]
    fn test_extract_correlation_length() {
        // Exponentially decaying correlation with ξ = 2.0
        let corr: Vec<f64> = (0..20).map(|r| (-r as f64 / 2.0).exp()).collect();
        let xi = extract_correlation_length(&corr);
        assert!((xi - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_finite_size_correlation_length() {
        assert_eq!(finite_size_correlation_length(100), 100.0);
    }
}
