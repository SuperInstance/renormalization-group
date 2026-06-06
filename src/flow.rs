//! # RG Flow Equations
//!
//! Track how coupling parameters evolve under repeated coarse-graining.
//! The RG flow is the trajectory in parameter space as scale increases.

use crate::linalg;
use serde::{Deserialize, Serialize};

/// State of the RG flow at a particular scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGFlow {
    /// Current parameter values (coupling constants, fields, etc.)
    pub parameters: Vec<f64>,
    /// Current scale level
    pub scale: usize,
}

/// A single step in the RG flow trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowStep {
    /// Scale at this step
    pub scale: usize,
    /// Parameter values
    pub parameters: Vec<f64>,
    /// Change in parameters from previous step
    pub delta: Vec<f64>,
}

/// The RG flow equation: maps parameters at scale ℓ to scale ℓ+1.
/// This is the beta function in RG language.
pub trait FlowEquation: Send + Sync {
    /// Compute parameters at the next scale given current parameters.
    fn flow_step(&self, params: &[f64], scale: usize) -> Vec<f64>;

    /// Dimensionality of parameter space
    fn dim(&self) -> usize;
}

/// 1D Ising RG flow (decimation).
/// Parameters: [K = J/kT, h = H/kT]
/// Under decimation by factor b=2:
///   K' = atanh(tanh(K)^2) (simplified form)
///   h' = 0 (simplification for zero field)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ising1DFlow {
    /// Decimation factor
    pub b: f64,
}

impl Ising1DFlow {
    pub fn new(b: f64) -> Self {
        Self { b }
    }
}

impl FlowEquation for Ising1DFlow {
    fn flow_step(&self, params: &[f64], _scale: usize) -> Vec<f64> {
        let k = params[0];
        // K' = atanh(tanh(K)^2) for b=2 decimation
        let tanh_k = k.tanh();
        let new_k = tanh_k * tanh_k;
        let k_prime = new_k.atanh();
        // h remains 0 for simplicity (add field evolution if needed)
        let h = if params.len() > 1 { params[1] } else { 0.0 };
        let h_prime = if h.abs() < 1e-15 {
            0.0
        } else {
            // Field renormalization for 1D Ising
            2.0 * h // Simplified: field doubles under decimation
        };
        vec![k_prime, h_prime]
    }

    fn dim(&self) -> usize {
        2
    }
}

/// Gaussian (free) RG flow.
/// Parameters: [r (mass), u (coupling)]
/// r' = b^2 * r  (relevant, eigenvalue b^2)
/// u' = b^(4-d) * u  (marginal in d=4)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianFlow {
    /// Spatial dimension
    pub d: usize,
    /// Scale factor
    pub b: f64,
}

impl GaussianFlow {
    pub fn new(d: usize, b: f64) -> Self {
        Self { d, b }
    }
}

impl FlowEquation for GaussianFlow {
    fn flow_step(&self, params: &[f64], _scale: usize) -> Vec<f64> {
        let r = params[0];
        let u = if params.len() > 1 { params[1] } else { 0.0 };
        let r_prime = self.b * self.b * r;
        let u_prime = self.b.powi(4 - self.d as i32) * u;
        vec![r_prime, u_prime]
    }

    fn dim(&self) -> usize {
        2
    }
}

/// Wilson-Fisher flow near d=4.
/// Beta functions: dr/dl = 2r + (n+2)u/(6π²)
///                 du/dl = ε*u - (n+8)u²/(6π²)
/// where ε = 4-d
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilsonFisherFlow {
    /// Spatial dimension
    pub d: f64,
    /// Number of field components (n=1 for Ising)
    pub n: usize,
    /// Step size for discrete flow
    pub dl: f64,
}

impl WilsonFisherFlow {
    pub fn new(d: f64, n: usize, dl: f64) -> Self {
        Self { d, n, dl }
    }
}

impl FlowEquation for WilsonFisherFlow {
    fn flow_step(&self, params: &[f64], _scale: usize) -> Vec<f64> {
        let r = params[0];
        let u = params[1];
        let epsilon = 4.0 - self.d;
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;

        let dr = 2.0 * r + (self.n as f64 + 2.0) * u / (6.0 * pi2);
        let du = epsilon * u - (self.n as f64 + 8.0) * u * u / (6.0 * pi2);

        vec![r + self.dl * dr, u + self.dl * du]
    }

    fn dim(&self) -> usize {
        2
    }
}

impl RGFlow {
    /// Create a new RG flow state.
    pub fn new(parameters: Vec<f64>) -> Self {
        Self {
            parameters,
            scale: 0,
        }
    }

    /// Evolve the flow for `n_steps` steps.
    pub fn evolve(&mut self, flow_eq: &dyn FlowEquation, n_steps: usize) -> Vec<FlowStep> {
        let mut trajectory = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let old_params = self.parameters.clone();
            let new_params = flow_eq.flow_step(&self.parameters, self.scale);
            let delta: Vec<f64> = new_params
                .iter()
                .zip(old_params.iter())
                .map(|(n, o)| n - o)
                .collect();
            self.scale += 1;
            self.parameters = new_params;
            trajectory.push(FlowStep {
                scale: self.scale,
                parameters: self.parameters.clone(),
                delta,
            });
        }
        trajectory
    }

    /// Compute the stability matrix (Jacobian of the flow equation)
    /// at the current parameter values.
    pub fn stability_matrix(
        &self,
        flow_eq: &dyn FlowEquation,
    ) -> Vec<Vec<f64>> {
        linalg::jacobian(
            &|p| flow_eq.flow_step(p, self.scale),
            &self.parameters,
            1e-7,
        )
    }
}

/// Compute the flow length (total parameter displacement) of a trajectory.
pub fn flow_length(trajectory: &[FlowStep]) -> f64 {
    trajectory
        .iter()
        .map(|step| step.delta.iter().map(|d| d * d).sum::<f64>().sqrt())
        .sum()
}

/// Check if a trajectory has converged (delta < tolerance for last steps).
pub fn has_converged(trajectory: &[FlowStep], tolerance: f64, n_check: usize) -> bool {
    let n = trajectory.len();
    if n < n_check {
        return false;
    }
    trajectory[(n - n_check)..]
        .iter()
        .all(|step| step.delta.iter().map(|d| d * d).sum::<f64>().sqrt() < tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_1d_flow_toward_zero() {
        // 1D Ising has no phase transition at finite T;
        // K = J/kT should flow to 0 (high-T fixed point)
        let flow_eq = Ising1DFlow::new(2.0);
        let mut flow = RGFlow::new(vec![0.5, 0.0]);
        let _traj = flow.evolve(&flow_eq, 20);
        // K should decrease toward 0
        assert!(flow.parameters[0] < 0.5);
        // After many steps, K should be very small
        assert!(flow.parameters[0].abs() < 0.1);
    }

    #[test]
    fn test_ising_1d_zero_k_fixed_point() {
        // K=0 is a fixed point of the 1D Ising flow
        let flow_eq = Ising1DFlow::new(2.0);
        let result = flow_eq.flow_step(&[0.0, 0.0], 0);
        assert!(result[0].abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_r_grows_relevant() {
        // r is a relevant operator: it grows under RG flow
        let flow_eq = GaussianFlow::new(3, 2.0); // d=3, b=2
        let mut flow = RGFlow::new(vec![0.1, 0.0]);
        let _traj = flow.evolve(&flow_eq, 5);
        // r should have grown: r' = 4*r each step
        let expected = 0.1 * 4.0_f64.powi(5);
        assert!((flow.parameters[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_u_irrelevant_d3() {
        // In d=3, u is irrelevant: u' = b^(4-3)*u = b*u... wait b^(4-d)
        // b=2, d=3: u' = 2^1 * u = 2u (relevant in d=3!)
        // Let's use d=5: u' = b^(4-5)*u = b^(-1)*u = u/2 (irrelevant)
        let flow_eq = GaussianFlow::new(5, 2.0);
        let mut flow = RGFlow::new(vec![0.0, 1.0]);
        let _traj = flow.evolve(&flow_eq, 3);
        // u should shrink: u' = 2^(-1)*u each step = u/2 each step
        let expected = 1.0 * 0.5_f64.powi(3);
        assert!((flow.parameters[1] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_zero_is_fixed_point() {
        let flow_eq = GaussianFlow::new(3, 2.0);
        let result = flow_eq.flow_step(&[0.0, 0.0], 0);
        assert!(result[0].abs() < 1e-15);
        assert!(result[1].abs() < 1e-15);
    }

    #[test]
    fn test_wilson_fisher_fixed_point() {
        // Wilson-Fisher fixed point at u* = 6π²ε/(n+8), r* from beta_r=0
        let flow_eq = WilsonFisherFlow::new(3.99, 1, 0.01); // ε=0.01, n=1
        let epsilon = 0.01;
        let n = 1.0;
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        let u_star = 6.0 * pi2 * epsilon / (n + 8.0);

        // Start near the Wilson-Fisher fixed point
        let mut flow = RGFlow::new(vec![0.0, u_star]);
        let _traj = flow.evolve(&flow_eq, 500);
        // Should stay near fixed point (not diverge wildly)
        assert!(flow.parameters[1] > 0.0);
        assert!(flow.parameters[1] < u_star * 10.0);
    }

    #[test]
    fn test_flow_convergence_detection() {
        let flow_eq = GaussianFlow::new(3, 2.0);
        let mut flow = RGFlow::new(vec![0.0, 0.0]); // At fixed point
        let traj = flow.evolve(&flow_eq, 5);
        assert!(has_converged(&traj, 1e-10, 3));
    }

    #[test]
    fn test_flow_length_positive() {
        let flow_eq = GaussianFlow::new(3, 2.0);
        let mut flow = RGFlow::new(vec![1.0, 0.0]);
        let traj = flow.evolve(&flow_eq, 3);
        assert!(flow_length(&traj) > 0.0);
    }
}
