//! Information geometry module — Fisher information, natural gradient, Fisher-Rao distance.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Information geometry navigator.
pub struct InformationGeometry {
    /// Number of parameters (capabilities).
    pub n_params: usize,
    /// Cached Fisher information matrix.
    pub fisher_matrix: Option<DMatrix<f64>>,
}

impl Default for InformationGeometry {
    fn default() -> Self {
        Self::new(8)
    }
}

impl InformationGeometry {
    /// Create a new geometry navigator for `n_params` dimensions.
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            fisher_matrix: None,
        }
    }

    /// Compute Fisher information matrix from performance history.
    pub fn compute_fisher_information(
        &mut self,
        performances: &[HashMap<String, f64>],
    ) -> DMatrix<f64> {
        if performances.len() < 2 {
            let fisher = DMatrix::from_diagonal_element(self.n_params, self.n_params, 0.1);
            self.fisher_matrix = Some(fisher.clone());
            return fisher;
        }

        let window = performances.len().min(10);
        let recent = &performances[performances.len() - window..];
        let keys: Vec<String> = recent[0].keys().cloned().take(self.n_params).collect();
        let n = keys.len();

        let mut perf_matrix = DMatrix::zeros(window, n);
        for (i, perf) in recent.iter().enumerate() {
            for (j, key) in keys.iter().enumerate() {
                perf_matrix[(i, j)] = perf.get(key).copied().unwrap_or(0.0);
            }
        }

        let fisher = if window >= 2 {
            // gradients: row differences
            let mut gradients = DMatrix::zeros(window - 1, n);
            for i in 0..(window - 1) {
                for j in 0..n {
                    gradients[(i, j)] = perf_matrix[(i + 1, j)] - perf_matrix[(i, j)];
                }
            }
            let mut fisher = gradients.transpose() * &gradients;
            let denom = gradients.nrows() as f64;
            fisher /= denom;
            // Regularize
            for i in 0..n {
                fisher[(i, i)] += 0.01;
            }
            fisher
        } else {
            DMatrix::from_diagonal_element(n, n, 0.1)
        };

        self.fisher_matrix = Some(fisher.clone());
        fisher
    }

    /// Compute natural gradient: F⁻¹ × ∇L.
    pub fn natural_gradient(&self, gradient: &[f64]) -> Vec<f64> {
        let fisher = match &self.fisher_matrix {
            Some(f) => f,
            None => return gradient.to_vec(),
        };

        let n = gradient.len().min(fisher.nrows());
        let grad = DVector::from_vec(gradient[..n].to_vec());
        // Extract sub-matrix manually for compatibility
        let mut sub = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                sub[(i, j)] = fisher[(i, j)];
            }
        }

        match sub.try_inverse() {
            Some(inv) => (inv * grad).iter().copied().collect(),
            None => gradient.to_vec(),
        }
    }

    /// Compute Fisher-Rao distance between two performance states.
    pub fn fisher_rao_distance(
        &self,
        a: &HashMap<String, f64>,
        b: &HashMap<String, f64>,
    ) -> f64 {
        let keys: Vec<&String> = a.keys().filter(|k| b.contains_key(*k)).collect();
        if keys.is_empty() {
            return 0.0;
        }

        let diff: DVector<f64> = DVector::from_vec(
            keys.iter().map(|k| a.get(*k).copied().unwrap_or(0.0) - b.get(*k).copied().unwrap_or(0.0)).collect()
        );

        match &self.fisher_matrix {
            Some(fisher) if fisher.nrows() >= diff.len() => {
                let n = diff.len();
                // Extract sub-matrix manually
                let mut sub = DMatrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        sub[(i, j)] = fisher[(i, j)];
                    }
                }
                match sub.try_inverse() {
                    Some(inv) => {
                        let mahal = &diff.transpose() * &inv * &diff;
                        mahal[(0, 0)].sqrt()
                    }
                    None => diff.norm(),
                }
            }
            _ => diff.norm(),
        }
    }
}

/// Fisher information summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherSummary {
    /// Trace of the Fisher matrix.
    pub trace: f64,
    /// Determinant.
    pub determinant: f64,
    /// Condition number (ratio of largest to smallest eigenvalue).
    pub condition_number: f64,
    /// Effective dimensionality.
    pub effective_dim: f64,
}

impl InformationGeometry {
    /// Get a summary of the Fisher information.
    pub fn fisher_summary(&self) -> FisherSummary {
        match &self.fisher_matrix {
            Some(f) => {
                let trace = f.trace();
                let n = f.nrows();
                // Approximate determinant via diagonal product
                let mut det = 1.0_f64;
                for i in 0..n {
                    det *= f[(i, i)];
                }
                let max_diag = (0..n).map(|i| f[(i, i)]).fold(f64::NEG_INFINITY, f64::max);
                let min_diag = (0..n).map(|i| f[(i, i)]).fold(f64::INFINITY, f64::min);
                let cond = if min_diag > 1e-10 { max_diag / min_diag } else { f64::INFINITY };

                FisherSummary {
                    trace,
                    determinant: det,
                    condition_number: cond,
                    effective_dim: if trace > 1e-10 { (trace * trace) / det.abs().max(1e-20) } else { 0.0 },
                }
            }
            None => FisherSummary {
                trace: 0.0,
                determinant: 0.0,
                condition_number: 0.0,
                effective_dim: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_perf(vals: &[f64]) -> HashMap<String, f64> {
        let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
        vals.iter().enumerate()
            .map(|(i, &v)| (names[i % names.len()].to_string(), v))
            .collect()
    }

    #[test]
    fn test_creation() {
        let ig = InformationGeometry::new(4);
        assert_eq!(ig.n_params, 4);
        assert!(ig.fisher_matrix.is_none());
    }

    #[test]
    fn test_default() {
        let ig = InformationGeometry::default();
        assert_eq!(ig.n_params, 8);
    }

    #[test]
    fn test_fisher_single_performance() {
        let mut ig = InformationGeometry::new(4);
        let perf = make_perf(&[1.0, 2.0, 3.0, 4.0]);
        let fisher = ig.compute_fisher_information(&[perf]);
        assert_eq!(fisher.nrows(), 4);
        assert_eq!(fisher.ncols(), 4);
    }

    #[test]
    fn test_fisher_multiple_performances() {
        let mut ig = InformationGeometry::new(3);
        let perfs: Vec<HashMap<String, f64>> = (0..5)
            .map(|i| make_perf(&[i as f64, (i + 1) as f64, (i + 2) as f64]))
            .collect();
        let fisher = ig.compute_fisher_information(&perfs);
        // Diagonal should be > 0 (regularized)
        for i in 0..3 {
            assert!(fisher[(i, i)] > 0.0);
        }
    }

    #[test]
    fn test_natural_gradient_without_fisher() {
        let ig = InformationGeometry::new(3);
        let grad = vec![1.0, 2.0, 3.0];
        let nat = ig.natural_gradient(&grad);
        assert_eq!(nat, grad);
    }

    #[test]
    fn test_natural_gradient_with_fisher() {
        let mut ig = InformationGeometry::new(3);
        let perfs: Vec<HashMap<String, f64>> = (0..5)
            .map(|i| make_perf(&[i as f64, (i + 1) as f64, (i + 2) as f64]))
            .collect();
        ig.compute_fisher_information(&perfs);
        let grad = vec![1.0, 0.0, 0.0];
        let nat = ig.natural_gradient(&grad);
        assert_eq!(nat.len(), 3);
        // Natural gradient should differ from regular gradient
        // (unless Fisher is identity, which it won't be with varied data)
    }

    #[test]
    fn test_fisher_rao_distance_no_fisher() {
        let ig = InformationGeometry::new(3);
        let a = make_perf(&[0.0, 0.0, 0.0]);
        let b = make_perf(&[3.0, 4.0, 0.0]);
        let dist = ig.fisher_rao_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_distance_identical() {
        let ig = InformationGeometry::new(3);
        let a = make_perf(&[1.0, 2.0, 3.0]);
        let dist = ig.fisher_rao_distance(&a, &a);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_distance_with_fisher() {
        let mut ig = InformationGeometry::new(3);
        let perfs: Vec<HashMap<String, f64>> = (0..5)
            .map(|i| make_perf(&[i as f64, (i + 1) as f64, (i + 2) as f64]))
            .collect();
        ig.compute_fisher_information(&perfs);
        let a = make_perf(&[0.0, 0.0, 0.0]);
        let b = make_perf(&[1.0, 1.0, 1.0]);
        let dist = ig.fisher_rao_distance(&a, &b);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_fisher_summary() {
        let mut ig = InformationGeometry::new(4);
        let perfs: Vec<HashMap<String, f64>> = (0..5)
            .map(|i| make_perf(&[i as f64, (i + 1) as f64, (i + 2) as f64, (i + 3) as f64]))
            .collect();
        ig.compute_fisher_information(&perfs);
        let summary = ig.fisher_summary();
        assert!(summary.trace > 0.0);
        assert!(summary.determinant > 0.0);
    }

    #[test]
    fn test_fisher_summary_no_matrix() {
        let ig = InformationGeometry::new(4);
        let summary = ig.fisher_summary();
        assert_eq!(summary.trace, 0.0);
    }
}
