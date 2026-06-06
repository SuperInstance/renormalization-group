//! # Universality Classes
//!
//! Different fleet topologies can have the same RG fixed point → same
//! large-scale behavior. This module classifies universality classes
//! based on critical exponents.

use serde::{Deserialize, Serialize};

/// A single critical exponent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalExponent {
    /// Name of the exponent (α, β, γ, δ, ν, η)
    pub name: String,
    /// Numerical value
    pub value: f64,
    /// Description of what this exponent characterizes
    pub description: String,
}

/// A universality class defined by its critical exponents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalityClass {
    /// Name of the universality class
    pub name: String,
    /// Critical exponents
    pub exponents: Vec<CriticalExponent>,
    /// Spatial dimension
    pub dimension: usize,
    /// Number of order parameter components (n)
    pub n_components: usize,
    /// Description
    pub description: String,
}

impl UniversalityClass {
    /// Get a critical exponent by name.
    pub fn exponent(&self, name: &str) -> Option<f64> {
        self.exponents.iter().find(|e| e.name == name).map(|e| e.value)
    }

    /// Check if two universality classes are the same by comparing exponents.
    pub fn is_same_universality_class(&self, other: &UniversalityClass) -> bool {
        if self.exponents.len() != other.exponents.len() {
            return false;
        }
        for e in &self.exponents {
            if let Some(other_val) = other.exponent(&e.name) {
                if (e.value - other_val).abs() > 0.01 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

/// Mean-field universality class (d ≥ 4).
pub fn mean_field_class() -> UniversalityClass {
    UniversalityClass {
        name: "Mean Field".into(),
        dimension: 4,
        n_components: 1,
        description: "Mean-field universality class valid for d ≥ 4".into(),
        exponents: vec![
            CriticalExponent {
                name: "alpha".into(),
                value: 0.0,
                description: "Specific heat: C ~ |t|^(-α)".into(),
            },
            CriticalExponent {
                name: "beta".into(),
                value: 0.5,
                description: "Order parameter: M ~ |t|^β".into(),
            },
            CriticalExponent {
                name: "gamma".into(),
                value: 1.0,
                description: "Susceptibility: χ ~ |t|^(-γ)".into(),
            },
            CriticalExponent {
                name: "delta".into(),
                value: 3.0,
                description: "Critical isotherm: M ~ h^(1/δ)".into(),
            },
            CriticalExponent {
                name: "nu".into(),
                value: 0.5,
                description: "Correlation length: ξ ~ |t|^(-ν)".into(),
            },
            CriticalExponent {
                name: "eta".into(),
                value: 0.0,
                description: "Correlation function decay".into(),
            },
        ],
    }
}

/// 2D Ising universality class (exact solution).
pub fn ising_2d_class() -> UniversalityClass {
    UniversalityClass {
        name: "2D Ising".into(),
        dimension: 2,
        n_components: 1,
        description: "2D Ising universality class (Onsager solution)".into(),
        exponents: vec![
            CriticalExponent {
                name: "alpha".into(),
                value: 0.0, // log divergence
                description: "Specific heat: log divergence".into(),
            },
            CriticalExponent {
                name: "beta".into(),
                value: 1.0 / 8.0,
                description: "Order parameter: M ~ |t|^(1/8)".into(),
            },
            CriticalExponent {
                name: "gamma".into(),
                value: 7.0 / 4.0,
                description: "Susceptibility: χ ~ |t|^(-7/4)".into(),
            },
            CriticalExponent {
                name: "delta".into(),
                value: 15.0,
                description: "Critical isotherm: M ~ h^(1/15)".into(),
            },
            CriticalExponent {
                name: "nu".into(),
                value: 1.0,
                description: "Correlation length: ξ ~ |t|^(-1)".into(),
            },
            CriticalExponent {
                name: "eta".into(),
                value: 0.25,
                description: "Correlation function decay".into(),
            },
        ],
    }
}

/// 3D Ising universality class (numerical).
pub fn ising_3d_class() -> UniversalityClass {
    UniversalityClass {
        name: "3D Ising".into(),
        dimension: 3,
        n_components: 1,
        description: "3D Ising universality class (numerical estimates)".into(),
        exponents: vec![
            CriticalExponent {
                name: "alpha".into(),
                value: 0.110,
                description: "Specific heat exponent".into(),
            },
            CriticalExponent {
                name: "beta".into(),
                value: 0.326,
                description: "Order parameter exponent".into(),
            },
            CriticalExponent {
                name: "gamma".into(),
                value: 1.237,
                description: "Susceptibility exponent".into(),
            },
            CriticalExponent {
                name: "delta".into(),
                value: 4.789,
                description: "Critical isotherm exponent".into(),
            },
            CriticalExponent {
                name: "nu".into(),
                value: 0.630,
                description: "Correlation length exponent".into(),
            },
            CriticalExponent {
                name: "eta".into(),
                value: 0.036,
                description: "Anomalous dimension".into(),
            },
        ],
    }
}

/// XY universality class (n=2, includes superfluid transition).
pub fn xy_class() -> UniversalityClass {
    UniversalityClass {
        name: "3D XY".into(),
        dimension: 3,
        n_components: 2,
        description: "3D XY universality class (superfluid transition)".into(),
        exponents: vec![
            CriticalExponent {
                name: "alpha".into(),
                value: -0.015,
                description: "Specific heat exponent".into(),
            },
            CriticalExponent {
                name: "beta".into(),
                value: 0.349,
                description: "Order parameter exponent".into(),
            },
            CriticalExponent {
                name: "gamma".into(),
                value: 1.316,
                description: "Susceptibility exponent".into(),
            },
            CriticalExponent {
                name: "nu".into(),
                value: 0.671,
                description: "Correlation length exponent".into(),
            },
            CriticalExponent {
                name: "eta".into(),
                value: 0.038,
                description: "Anomalous dimension".into(),
            },
            CriticalExponent {
                name: "delta".into(),
                value: 4.77,
                description: "Critical isotherm exponent".into(),
            },
        ],
    }
}

/// Heisenberg universality class (n=3).
pub fn heisenberg_class() -> UniversalityClass {
    UniversalityClass {
        name: "3D Heisenberg".into(),
        dimension: 3,
        n_components: 3,
        description: "3D Heisenberg universality class".into(),
        exponents: vec![
            CriticalExponent {
                name: "alpha".into(),
                value: -0.115,
                description: "Specific heat exponent".into(),
            },
            CriticalExponent {
                name: "beta".into(),
                value: 0.366,
                description: "Order parameter exponent".into(),
            },
            CriticalExponent {
                name: "gamma".into(),
                value: 1.386,
                description: "Susceptibility exponent".into(),
            },
            CriticalExponent {
                name: "nu".into(),
                value: 0.706,
                description: "Correlation length exponent".into(),
            },
            CriticalExponent {
                name: "eta".into(),
                value: 0.037,
                description: "Anomalous dimension".into(),
            },
            CriticalExponent {
                name: "delta".into(),
                value: 4.78,
                description: "Critical isotherm exponent".into(),
            },
        ],
    }
}

/// Verify the Rushbrooke scaling relation: α + 2β + γ = 2
pub fn verify_rushbrooke(class: &UniversalityClass) -> bool {
    let alpha = class.exponent("alpha").unwrap_or(0.0);
    let beta = class.exponent("beta").unwrap_or(0.0);
    let gamma = class.exponent("gamma").unwrap_or(0.0);
    (alpha + 2.0 * beta + gamma - 2.0).abs() < 0.05
}

/// Verify the Widom scaling relation: γ = β(δ - 1)
pub fn verify_widom(class: &UniversalityClass) -> bool {
    let gamma = class.exponent("gamma").unwrap_or(0.0);
    let beta = class.exponent("beta").unwrap_or(0.0);
    let delta = class.exponent("delta").unwrap_or(0.0);
    (gamma - beta * (delta - 1.0)).abs() < 0.05
}

/// Verify the Fisher scaling relation: γ = ν(2 - η)
pub fn verify_fisher(class: &UniversalityClass) -> bool {
    let gamma = class.exponent("gamma").unwrap_or(0.0);
    let nu = class.exponent("nu").unwrap_or(0.0);
    let eta = class.exponent("eta").unwrap_or(0.0);
    (gamma - nu * (2.0 - eta)).abs() < 0.05
}

/// Verify the Josephson scaling relation: 2 - α = dν
pub fn verify_josephson(class: &UniversalityClass) -> bool {
    let alpha = class.exponent("alpha").unwrap_or(0.0);
    let nu = class.exponent("nu").unwrap_or(0.0);
    (2.0 - alpha - class.dimension as f64 * nu).abs() < 0.05
}

/// A "microscopic" fleet model that can be classified into a universality class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetModel {
    /// Name of the model
    pub name: String,
    /// Parameters
    pub parameters: Vec<f64>,
    /// Topology type
    pub topology: String,
}

impl FleetModel {
    pub fn new(name: &str, parameters: Vec<f64>, topology: &str) -> Self {
        Self {
            name: name.into(),
            parameters,
            topology: topology.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_field_exponents() {
        let mf = mean_field_class();
        assert!((mf.exponent("alpha").unwrap() - 0.0).abs() < 1e-10);
        assert!((mf.exponent("beta").unwrap() - 0.5).abs() < 1e-10);
        assert!((mf.exponent("gamma").unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_2d_ising_exact_exponents() {
        let ising = ising_2d_class();
        assert!((ising.exponent("beta").unwrap() - 1.0 / 8.0).abs() < 1e-10);
        assert!((ising.exponent("gamma").unwrap() - 7.0 / 4.0).abs() < 1e-10);
        assert!((ising.exponent("nu").unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rushbrooke_mean_field() {
        assert!(verify_rushbrooke(&mean_field_class()));
    }

    #[test]
    fn test_rushbrooke_2d_ising() {
        assert!(verify_rushbrooke(&ising_2d_class()));
    }

    #[test]
    fn test_widom_mean_field() {
        assert!(verify_widom(&mean_field_class()));
    }

    #[test]
    fn test_widom_2d_ising() {
        assert!(verify_widom(&ising_2d_class()));
    }

    #[test]
    fn test_fisher_mean_field() {
        assert!(verify_fisher(&mean_field_class()));
    }

    #[test]
    fn test_josephson_mean_field() {
        assert!(verify_josephson(&mean_field_class()));
    }

    #[test]
    fn test_universality_same_class() {
        // Two different microscopic models with same exponents → same class
        let class1 = ising_3d_class();
        let mut class2 = ising_3d_class();
        class2.name = "Fleet Ising 3D".into();
        assert!(class1.is_same_universality_class(&class2));
    }

    #[test]
    fn test_different_universality_classes() {
        let ising = ising_3d_class();
        let xy = xy_class();
        assert!(!ising.is_same_universality_class(&xy));
    }

    #[test]
    fn test_fleet_model_creation() {
        let model = FleetModel::new("test", vec![1.0, 2.0], "1D chain");
        assert_eq!(model.name, "test");
        assert_eq!(model.parameters.len(), 2);
    }
}
