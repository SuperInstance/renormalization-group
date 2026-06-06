//! # Fixed-Point Detection and Stability Analysis
//!
//! Linearize the RG flow around fixed points. Eigenvalues of the
//! linearized flow determine relevant vs irrelevant directions.

use crate::flow::FlowEquation;
use crate::linalg;
use serde::{Deserialize, Serialize};

/// A fixed point of the RG flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint {
    /// Location in parameter space
    pub location: Vec<f64>,
    /// Eigenvalues of the stability matrix at this fixed point
    pub eigenvalues: Vec<f64>,
    /// Whether each direction is stable (irrelevant = true, relevant = false)
    pub stable: Vec<bool>,
    /// Names/labels for the parameters
    pub param_names: Vec<String>,
}

impl FixedPoint {
    /// Classify a direction as relevant (eigenvalue > 1), irrelevant (< 1),
    /// or marginal (≈ 1).
    pub fn classify_direction(&self, index: usize) -> DirectionClass {
        if index >= self.eigenvalues.len() {
            return DirectionClass::Marginal;
        }
        let ev = self.eigenvalues[index];
        if ev > 1.0 + 1e-6 {
            DirectionClass::Relevant
        } else if ev < 1.0 - 1e-6 {
            DirectionClass::Irrelevant
        } else {
            DirectionClass::Marginal
        }
    }

    /// Count the number of relevant directions.
    pub fn n_relevant(&self) -> usize {
        self.eigenvalues.iter().filter(|&&ev| ev > 1.0 + 1e-6).count()
    }

    /// Count the number of irrelevant directions.
    pub fn n_irrelevant(&self) -> usize {
        self.eigenvalues.iter().filter(|&&ev| ev < 1.0 - 1e-6).count()
    }

    /// The number of relevant directions equals the codimension of the
    /// critical surface.
    pub fn codimension(&self) -> usize {
        self.n_relevant()
    }
}

/// Classification of RG flow directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectionClass {
    /// Eigenvalue > 1: grows under RG (destabilizes at large scale)
    Relevant,
    /// Eigenvalue < 1: shrinks under RG (irrelevant at large scale)
    Irrelevant,
    /// Eigenvalue ≈ 1: needs higher-order analysis
    Marginal,
}

/// Find a fixed point by iterating the flow equation.
/// Returns None if the iteration doesn't converge.
pub fn find_fixed_point(
    flow_eq: &dyn FlowEquation,
    initial_guess: &[f64],
    max_iter: usize,
    tolerance: f64,
) -> Option<FixedPoint> {
    let mut current = initial_guess.to_vec();

    for _ in 0..max_iter {
        let next = flow_eq.flow_step(&current, 0);
        let delta: f64 = next
            .iter()
            .zip(current.iter())
            .map(|(n, c)| (n - c).powi(2))
            .sum::<f64>()
            .sqrt();

        if delta < tolerance {
            // Found fixed point; compute stability matrix
            let jac = linalg::jacobian(
                &|p| flow_eq.flow_step(p, 0),
                &current,
                1e-7,
            );
            // Symmetrize for eigenvalue computation
            let n = jac.len();
            let mut sym_jac = jac.clone();
            for i in 0..n {
                for j in 0..n {
                    sym_jac[i][j] = 0.5 * (jac[i][j] + jac.get(j).map(|r| r[i]).unwrap_or(0.0));
                    sym_jac[j][i] = sym_jac[i][j];
                }
            }
            let eigenvalues = linalg::eigenvalues_symmetric(&sym_jac);
            let stable: Vec<bool> = eigenvalues.iter().map(|&ev| ev.abs() < 1.0).collect();
            let param_names = (0..current.len())
                .map(|i| format!("param_{}", i))
                .collect();

            return Some(FixedPoint {
                location: current,
                eigenvalues,
                stable,
                param_names,
            });
        }
        current = next;
    }
    None
}

/// Analyze stability around a known fixed point.
/// Computes eigenvalues and classifies directions.
pub fn analyze_stability(
    flow_eq: &dyn FlowEquation,
    fixed_point: &[f64],
) -> FixedPoint {
    let jac = linalg::jacobian(
        &|p| flow_eq.flow_step(p, 0),
        fixed_point,
        1e-7,
    );
    let n = jac.len();
    let mut sym_jac = jac.clone();
    for i in 0..n {
        for j in 0..n {
            sym_jac[i][j] = 0.5 * (jac[i][j] + jac.get(j).map(|r| r[i]).unwrap_or(0.0));
            sym_jac[j][i] = sym_jac[i][j];
        }
    }
    let eigenvalues = linalg::eigenvalues_symmetric(&sym_jac);
    let stable: Vec<bool> = eigenvalues.iter().map(|&ev| ev.abs() < 1.0).collect();
    let param_names = (0..fixed_point.len())
        .map(|i| format!("param_{}", i))
        .collect();

    FixedPoint {
        location: fixed_point.to_vec(),
        eigenvalues,
        stable,
        param_names,
    }
}

/// Test if a given parameter vector is near a fixed point.
pub fn is_near_fixed_point(
    flow_eq: &dyn FlowEquation,
    params: &[f64],
    tolerance: f64,
) -> bool {
    let next = flow_eq.flow_step(params, 0);
    let delta: f64 = next
        .iter()
        .zip(params.iter())
        .map(|(n, c)| (n - c).powi(2))
        .sum::<f64>()
        .sqrt();
    delta < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{GaussianFlow, Ising1DFlow, RGFlow};

    #[test]
    fn test_ising_1d_zero_fixed_point() {
        let flow_eq = Ising1DFlow::new(2.0);
        let fp = find_fixed_point(&flow_eq, &[0.1, 0.0], 100, 1e-10);
        assert!(fp.is_some());
        let fp = fp.unwrap();
        assert!(fp.location[0].abs() < 0.01);
    }

    #[test]
    fn test_gaussian_zero_fixed_point() {
        // The origin is trivially a fixed point (maps to itself)
        let flow_eq = GaussianFlow::new(3, 2.0);
        assert!(is_near_fixed_point(&flow_eq, &[0.0, 0.0], 1e-10));
        // Also verify via direct stability analysis
        let fp = analyze_stability(&flow_eq, &[0.0, 0.0]);
        assert!(fp.location[0].abs() < 1e-10);
        assert!(fp.location[1].abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_r_is_relevant() {
        // At the Gaussian fixed point, r has eigenvalue b^2 = 4 (relevant)
        let flow_eq = GaussianFlow::new(3, 2.0);
        let fp = analyze_stability(&flow_eq, &[0.0, 0.0]);
        // Both eigenvalues should be > 1 (r: b^2=4, u: b^(4-d)=b^1=2)
        let max_eigenvalue = fp.eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_eigenvalue > 1.0);
        assert!(fp.n_relevant() >= 1);
    }

    #[test]
    fn test_uniform_fleet_trivial_fixed_point() {
        // Uniform fleet (all identical agents) → all parameters equal → trivial fixed point
        let flow_eq = GaussianFlow::new(3, 2.0);
        assert!(is_near_fixed_point(&flow_eq, &[0.0, 0.0], 1e-10));
    }

    #[test]
    fn test_direction_classification() {
        let fp = FixedPoint {
            location: vec![0.0, 0.0],
            eigenvalues: vec![4.0, 0.5],
            stable: vec![false, true],
            param_names: vec!["r".into(), "u".into()],
        };
        assert_eq!(fp.classify_direction(0), DirectionClass::Relevant);
        assert_eq!(fp.classify_direction(1), DirectionClass::Irrelevant);
        assert_eq!(fp.n_relevant(), 1);
        assert_eq!(fp.n_irrelevant(), 1);
        assert_eq!(fp.codimension(), 1);
    }

    #[test]
    fn test_relevant_operator_grows() {
        // Start with small perturbation in relevant direction → should grow
        let flow_eq = GaussianFlow::new(3, 2.0);
        let mut flow = RGFlow::new(vec![0.01, 0.0]); // small r
        flow.evolve(&flow_eq, 5);
        // r should have grown significantly
        assert!(flow.parameters[0] > 0.01);
    }

    #[test]
    fn test_irrelevant_operator_shrinks() {
        // In d=5, u is irrelevant for Gaussian flow
        let flow_eq = GaussianFlow::new(5, 2.0);
        let mut flow = RGFlow::new(vec![0.0, 0.1]); // small u, zero r
        flow.evolve(&flow_eq, 5);
        // u should have shrunk
        assert!(flow.parameters[1].abs() < 0.1);
    }

    #[test]
    fn test_1d_ising_no_finite_t_critical() {
        // 1D Ising: all nonzero K flow to 0 (no finite-T critical point)
        let flow_eq = Ising1DFlow::new(2.0);
        let mut flow = RGFlow::new(vec![2.0, 0.0]); // Large K (low T)
        flow.evolve(&flow_eq, 50);
        // Should flow toward K=0
        assert!(flow.parameters[0] < 1.0);
    }
}
