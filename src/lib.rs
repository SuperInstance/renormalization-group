//! # Renormalization Group for Multi-Scale Fleet Analysis
//!
//! This library applies the renormalization group (RG) from statistical physics
//! to multi-scale fleet analysis. Agents are coarse-grained into block variables
//! at increasing scale; the library tracks how constraint parameters flow under
//! this transformation.
//!
//! ## Modules
//!
//! - [`block_spin`] — Coarse-graining transformations
//! - [`flow`] — RG flow equations for coupling constants
//! - [`fixed_point`] — Fixed-point detection and stability analysis
//! - [`critical_surface`] — Critical manifolds separating fleet phases
//! - [`universality`] — Universality classes of fleet behavior
//! - [`scaling`] — Finite-size scaling and scaling relations

pub mod block_spin;
pub mod critical_surface;
pub mod fixed_point;
pub mod flow;
pub mod scaling;
pub mod universality;

pub use block_spin::*;
pub use critical_surface::*;
pub use fixed_point::*;
pub use flow::*;
pub use scaling::{
    scaling_collapse, finite_size_order_parameter, finite_size_susceptibility,
    finite_size_correlation_length, binder_cumulant, verify_hyperscaling,
    find_tc_from_crossing, correlation_function, extract_correlation_length,
    ScalingResult,
};
pub use universality::*;

/// Gaussian elimination solver for dense linear algebra.
/// Implemented from scratch with no external math dependencies.
pub mod linalg {
    /// Solve a linear system Ax = b via Gaussian elimination with partial pivoting.
    /// Returns `None` if the system is singular.
    pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = b.len();
        if a.len() != n {
            return None;
        }
        for row in a {
            if row.len() != n {
                return None;
            }
        }

        // Build augmented matrix [A|b]
        let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = a[i].clone();
            row.push(b[i]);
            aug.push(row);
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                let val = aug[row][col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return None; // singular
            }
            // Swap rows
            aug.swap(col, max_row);

            // Eliminate below
            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=(n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if aug[i][i].abs() < 1e-14 {
                return None;
            }
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }
        Some(x)
    }

    /// Compute eigenvalues of a real symmetric matrix using the QR algorithm
    /// with Wilkinson shifts. Returns eigenvalues sorted by absolute value descending.
    pub fn eigenvalues_symmetric(mat: &[Vec<f64>]) -> Vec<f64> {
        let n = mat.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![mat[0][0]];
        }

        // Use Jacobi eigenvalue algorithm for robustness
        let mut a: Vec<Vec<f64>> = mat.to_vec();
        let max_iter = 100 * n * n;
        let tol = 1e-12;

        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < tol {
                break;
            }

            // Compute rotation angle
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];

            let theta = if (app - aqq).abs() < 1e-15 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * apq / (app - aqq)).atan()
            };
            let c = theta.cos();
            let s = theta.sin();

            // Apply Jacobi rotation
            // Update rows/cols p and q
            let mut new_ap = vec![0.0; n];
            let mut new_aq = vec![0.0; n];
            for i in 0..n {
                if i != p && i != q {
                    new_ap[i] = c * a[i][p] + s * a[i][q];
                    new_aq[i] = -s * a[i][p] + c * a[i][q];
                }
            }
            new_ap[p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            new_aq[q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            new_ap[q] = 0.0;
            new_aq[p] = 0.0;

            for i in 0..n {
                a[i][p] = new_ap[i];
                a[p][i] = new_ap[i];
                a[i][q] = new_aq[i];
                a[q][i] = new_aq[i];
            }
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }

        let mut eigenvals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        eigenvals.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
        eigenvals
    }

    /// Compute the Jacobian of a vector function numerically.
    /// `f` takes a parameter vector and returns a vector of the same dimension.
    /// Uses central differences with step size `h`.
    pub fn jacobian(
        f: &dyn Fn(&[f64]) -> Vec<f64>,
        x: &[f64],
        h: f64,
    ) -> Vec<Vec<f64>> {
        let n = x.len();
        let fx = f(x);
        let m = fx.len();
        let mut jac = vec![vec![0.0; n]; m];

        for j in 0..n {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[j] += h;
            xm[j] -= h;
            let fp = f(&xp);
            let fm = f(&xm);
            for i in 0..m {
                jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        jac
    }

    /// Matrix-vector multiplication
    pub fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Transpose a matrix
    pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if a.is_empty() {
            return vec![];
        }
        let m = a.len();
        let n = a[0].len();
        (0..n)
            .map(|j| (0..m).map(|i| a[i][j]).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::linalg::*;

    #[test]
    fn test_gaussian_solve_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 4.0];
        let x = solve(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_solve_3x3() {
        let a = vec![vec![2.0, 1.0, -1.0], vec![-3.0, -1.0, 2.0], vec![-2.0, 1.0, 2.0]];
        let b = vec![8.0, -11.0, -3.0];
        let x = solve(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        assert!(solve(&a, &b).is_none());
    }

    #[test]
    fn test_eigenvalues_2x2() {
        let mat = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let eigs = eigenvalues_symmetric(&mat);
        // Eigenvalues of [[4,1],[1,3]] are (7 ± sqrt(5))/2 ≈ 4.618, 2.382
        let e1 = (7.0 + 5.0_f64.sqrt()) / 2.0;
        let e2 = (7.0 - 5.0_f64.sqrt()) / 2.0;
        assert!((eigs[0] - e1).abs() < 0.01 || (eigs[0] - e2).abs() < 0.01);
        assert!((eigs[1] - e1).abs() < 0.01 || (eigs[1] - e2).abs() < 0.01);
    }
}
