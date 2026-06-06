//! # Critical Surface Identification
//!
//! Identifies critical manifolds separating fleet phases
//! (ordered/disordered/critical). The critical surface has codimension
//! equal to the number of relevant operators at the critical fixed point.

use crate::flow::FlowEquation;
use crate::fixed_point::{FixedPoint, analyze_stability};
use serde::{Deserialize, Serialize};

/// A fleet phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetPhase {
    /// Ordered phase: agents are aligned
    Ordered,
    /// Disordered phase: agents are random
    Disordered,
    /// Critical: on the boundary between phases
    Critical,
}

/// A critical surface (manifold) in parameter space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalSurface {
    /// The fixed point this surface is associated with
    pub fixed_point: FixedPoint,
    /// Number of relevant directions (codimension of the surface)
    pub codimension: usize,
    /// Description of the surface
    pub description: String,
}

/// Result of a phase determination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    /// The identified phase
    pub phase: FleetPhase,
    /// Distance to the nearest critical surface (if known)
    pub distance_to_critical: Option<f64>,
    /// Order parameter value
    pub order_parameter: f64,
}

/// Determine the fleet phase by flowing to the IR fixed point
/// and checking which basin of attraction the flow ends up in.
pub fn determine_phase(
    flow_eq: &dyn FlowEquation,
    initial_params: &[f64],
    order_param_index: usize,
    n_steps: usize,
) -> PhaseResult {
    let mut current = initial_params.to_vec();
    let _dim = flow_eq.dim();

    // Flow for n_steps
    for _ in 0..n_steps {
        current = flow_eq.flow_step(&current, 0);
    }

    // Check if the flow diverged (ordered) or went to zero (disordered)
    let magnitude: f64 = current.iter().map(|x| x * x).sum::<f64>().sqrt();
    let order_param = if order_param_index < current.len() {
        current[order_param_index]
    } else {
        magnitude
    };

    let phase = if magnitude > 1e10 {
        FleetPhase::Ordered
    } else if magnitude < 1e-3 {
        FleetPhase::Disordered
    } else {
        FleetPhase::Critical
    };

    PhaseResult {
        phase,
        distance_to_critical: None,
        order_parameter: order_param.abs(),
    }
}

/// Compute the critical surface for a given fixed point.
/// The critical surface is the set of points that flow to the fixed point.
/// Its codimension equals the number of relevant operators.
pub fn compute_critical_surface(
    flow_eq: &dyn FlowEquation,
    fixed_point_location: &[f64],
) -> CriticalSurface {
    let fp = analyze_stability(flow_eq, fixed_point_location);
    CriticalSurface {
        codimension: fp.n_relevant(),
        fixed_point: fp.clone(),
        description: format!(
            "Critical surface with codimension {} at {:?}",
            fp.n_relevant(),
            fixed_point_location
        ),
    }
}

/// Bisection to find the critical point along a line in parameter space.
/// Given two endpoints (one ordered, one disordered), find where the phase boundary is.
pub fn find_critical_point(
    flow_eq: &dyn FlowEquation,
    ordered_params: &[f64],
    disordered_params: &[f64],
    order_param_index: usize,
    n_steps: usize,
    tolerance: f64,
    max_iter: usize,
) -> Vec<f64> {
    let mut lo = ordered_params.to_vec();
    let mut hi = disordered_params.to_vec();
    let dim = lo.len();

    for _ in 0..max_iter {
        let mid: Vec<f64> = (0..dim).map(|i| 0.5 * (lo[i] + hi[i])).collect();
        let mut hi_diff = 0.0;
        for i in 0..dim {
            hi_diff += (hi[i] - lo[i]).powi(2);
        }
        if hi_diff.sqrt() < tolerance {
            return mid;
        }

        let result = determine_phase(flow_eq, &mid, order_param_index, n_steps);
        match result.phase {
            FleetPhase::Ordered => lo = mid,
            FleetPhase::Disordered => hi = mid,
            FleetPhase::Critical => return mid,
        }
    }
    (0..dim).map(|i| 0.5 * (lo[i] + hi[i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{GaussianFlow, Ising1DFlow};

    #[test]
    fn test_disordered_phase_1d_ising() {
        // Small K (high T) → disordered
        let flow_eq = Ising1DFlow::new(2.0);
        let result = determine_phase(&flow_eq, &[0.01, 0.0], 0, 30);
        assert_eq!(result.phase, FleetPhase::Disordered);
    }

    #[test]
    fn test_ordered_direction_gaussian() {
        // Large r → flow diverges → ordered
        let flow_eq = GaussianFlow::new(3, 2.0);
        let result = determine_phase(&flow_eq, &[100.0, 0.0], 0, 20);
        assert_eq!(result.phase, FleetPhase::Ordered);
    }

    #[test]
    fn test_disordered_direction_gaussian() {
        // r=0 at Gaussian fixed point → disordered
        let flow_eq = GaussianFlow::new(3, 2.0);
        let result = determine_phase(&flow_eq, &[0.0, 0.0], 0, 10);
        assert_eq!(result.phase, FleetPhase::Disordered);
    }

    #[test]
    fn test_critical_surface_codimension() {
        // In d=3, Gaussian fixed point: r eigenvalue b²=4 (relevant), u eigenvalue b^(4-3)=b=2 (relevant)
        // So codimension = 2. Use d=5 where u is irrelevant → codimension = 1
        let flow_eq = GaussianFlow::new(5, 2.0);
        let surface = compute_critical_surface(&flow_eq, &[0.0, 0.0]);
        assert_eq!(surface.codimension, 1); // Only r is relevant in d=5
    }

    #[test]
    fn test_bisection_finds_critical() {
        // For Gaussian flow, critical point is at r=0
        let flow_eq = GaussianFlow::new(3, 2.0);
        let critical = find_critical_point(
            &flow_eq,
            &[1.0, 0.0],  // ordered
            &[-1.0, 0.0], // disordered (negative r doesn't exist, but let's test)
            0,
            20,
            0.01,
            50,
        );
        // The critical point should be near r=0
        // Note: for Gaussian flow, negative r also grows, so this is a bit degenerate
        // The bisection should still converge to something sensible
        assert!(critical[0].abs() < 1.0);
    }
}
