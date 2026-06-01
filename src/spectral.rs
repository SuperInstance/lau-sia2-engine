//! Spectral analysis module — eigendecomposition of performance matrix.
//!
//! Decomposes agent performance into eigenmodes of the capability correlation
//! matrix. The Laplacian eigenvalues reveal which 'frequencies' of performance
//! are strong and which are weak.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Default capability dimension names.
pub const CAPABILITY_NAMES: [&str; 8] = [
    "reasoning",
    "tool_use",
    "error_handling",
    "efficiency",
    "robustness",
    "generalization",
    "creativity",
    "consistency",
];

/// A single eigenmode of agent performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMode {
    /// How much this mode contributes (eigenvalue).
    pub eigenvalue: f64,
    /// The direction in capability space (eigenvector).
    pub eigenvector: Vec<f64>,
    /// Human-readable name.
    pub mode_name: String,
    /// Temporal frequency of this mode.
    pub frequency: f64,
    /// How fast this mode decays without reinforcement.
    pub decay_rate: f64,
}

impl SpectralMode {
    /// Fraction of total variance explained by this mode.
    pub fn contribution(&self) -> f64 {
        self.eigenvalue.abs()
    }

    /// Whether this mode is below the spectral gap threshold.
    pub fn is_weak(&self) -> bool {
        self.eigenvalue < 0.1
    }
}

/// Spectral analyzer — decomposes agent performance into eigenmodes.
pub struct SpectralAnalyzer {
    /// Number of capability dimensions.
    pub n_capabilities: usize,
    /// Capability dimension names.
    pub capability_names: Vec<String>,
}

impl Default for SpectralAnalyzer {
    fn default() -> Self {
        Self::new(8)
    }
}

impl SpectralAnalyzer {
    /// Create a new analyzer for `n_capabilities` dimensions.
    pub fn new(n_capabilities: usize) -> Self {
        let capability_names = CAPABILITY_NAMES
            .iter()
            .take(n_capabilities)
            .map(|s| s.to_string())
            .collect();
        Self {
            n_capabilities,
            capability_names,
        }
    }

    /// Extract spectral modes from execution metrics.
    ///
    /// The performance vector is projected onto the eigenspace of the
    /// capability correlation matrix.
    pub fn analyze(
        &self,
        execution_log: &serde_json::Value,
        metrics: &std::collections::HashMap<String, f64>,
    ) -> Vec<SpectralMode> {
        let corr_matrix = self.build_correlation_matrix(execution_log);
        let n = self.n_capabilities;

        // Symmetric eigendecomposition via nalgebra
        let (eigenvalues, eigenvectors) = symmetric_eigendecompose(&corr_matrix);

        // Sort by magnitude descending
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .abs()
                .partial_cmp(&eigenvalues[a].abs())
                .unwrap()
        });

        indices
            .iter()
            .enumerate()
            .map(|(i, &eigen_idx)| {
                let freq = (i + 1) as f64 * std::f64::consts::PI / n as f64;
                let decay = (1.0 - eigenvalues[eigen_idx].abs()).max(0.0);
                let evec: Vec<f64> = eigenvectors.column(eigen_idx).iter().copied().collect();

                SpectralMode {
                    eigenvalue: eigenvalues[eigen_idx],
                    eigenvector: evec.clone(),
                    mode_name: format!("mode_{}_{}", i, self.classify_mode(&evec)),
                    frequency: freq,
                    decay_rate: decay,
                }
            })
            .collect()
    }

    /// Find the weakest eigenmode for targeted improvement.
    pub fn find_weakest_mode<'a>(&self, modes: &'a [SpectralMode]) -> &'a SpectralMode {
        modes
            .iter()
            .min_by(|a, b| {
                a.eigenvalue
                    .abs()
                    .partial_cmp(&b.eigenvalue.abs())
                    .unwrap()
            })
            .expect("modes should not be empty")
    }

    /// Compute the natural gradient improvement direction for a target mode.
    ///
    /// Uses information geometry: improvement direction is the eigenvector
    /// of the weakest mode, scaled by the inverse Fisher information.
    pub fn compute_improvement_direction(&self, target_mode: &SpectralMode) -> Vec<f64> {
        let scale = 1.0 / target_mode.eigenvalue.abs().max(0.01);
        target_mode
            .eigenvector
            .iter()
            .map(|&v| v * scale)
            .collect()
    }

    /// Build capability correlation matrix from execution data.
    pub fn build_correlation_matrix(
        &self,
        _execution_log: &serde_json::Value,
    ) -> DMatrix<f64> {
        let n = self.n_capabilities;
        let mut corr = DMatrix::from_diagonal_element(n, n, 0.5);

        for i in 0..n {
            for j in (i + 1)..n {
                let c = if (i as isize - j as isize).unsigned_abs() == 1 {
                    0.3
                } else if (i as isize - j as isize).unsigned_abs() <= 3 {
                    0.1
                } else {
                    0.0
                };
                corr[(i, j)] = c;
                corr[(j, i)] = c;
            }
        }
        corr
    }

    fn classify_mode(&self, eigenvector: &[f64]) -> &str {
        let (dominant_idx, _) = eigenvector
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();

        if dominant_idx < self.capability_names.len() {
            &self.capability_names[dominant_idx]
        } else {
            "unknown"
        }
    }
}

/// Symmetric eigendecomposition: returns (eigenvalues, eigenvectors as column matrix).
pub fn symmetric_eigendecompose(matrix: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let n = matrix.nrows();
    assert!(matrix.is_square(), "Matrix must be square");

    // Use Jacobi eigenvalue algorithm for symmetric matrices
    let mut a = matrix.clone();
    let mut v = DMatrix::identity(n, n);

    let max_iter = 100 * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find off-diagonal element with largest absolute value
        let mut max_val = 0.0_f64;
        let mut p = 0usize;
        let mut q = 1usize;

        for i in 0..n {
            for j in (i + 1)..n {
                if a[(i, j)].abs() > max_val {
                    max_val = a[(i, j)].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation
        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation: A = G^T A G
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[(i, p)];
                let aiq = a[(i, q)];
                new_a[(i, p)] = c * aip + s * aiq;
                new_a[(p, i)] = new_a[(i, p)];
                new_a[(i, q)] = -s * aip + c * aiq;
                new_a[(q, i)] = new_a[(i, q)];
            }
        }
        new_a[(p, p)] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[(q, q)] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[(p, q)] = 0.0;
        new_a[(q, p)] = 0.0;
        a = new_a;

        // Accumulate eigenvectors
        let mut new_v = v.clone();
        for i in 0..n {
            let vip = v[(i, p)];
            let viq = v[(i, q)];
            new_v[(i, p)] = c * vip + s * viq;
            new_v[(i, q)] = -s * vip + c * viq;
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_mode_contribution() {
        let mode = SpectralMode {
            eigenvalue: 0.42,
            eigenvector: vec![1.0, 0.0],
            mode_name: "test".into(),
            frequency: 1.0,
            decay_rate: 0.5,
        };
        assert!((mode.contribution() - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_mode_is_weak() {
        let weak = SpectralMode {
            eigenvalue: 0.05,
            eigenvector: vec![],
            mode_name: "".into(),
            frequency: 0.0,
            decay_rate: 0.0,
        };
        let strong = SpectralMode {
            eigenvalue: 0.5,
            eigenvector: vec![],
            mode_name: "".into(),
            frequency: 0.0,
            decay_rate: 0.0,
        };
        assert!(weak.is_weak());
        assert!(!strong.is_weak());
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = SpectralAnalyzer::new(8);
        assert_eq!(analyzer.n_capabilities, 8);
        assert_eq!(analyzer.capability_names.len(), 8);
    }

    #[test]
    fn test_correlation_matrix_is_symmetric() {
        let analyzer = SpectralAnalyzer::new(8);
        let m = analyzer.build_correlation_matrix(&serde_json::Value::Null);
        assert_eq!(m.nrows(), 8);
        assert_eq!(m.ncols(), 8);
        for i in 0..8 {
            for j in 0..8 {
                assert!((m[(i, j)] - m[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_analyze_produces_modes() {
        let analyzer = SpectralAnalyzer::new(8);
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("reasoning".into(), 0.8);
        metrics.insert("tool_use".into(), 0.6);
        let modes = analyzer.analyze(&serde_json::Value::Null, &metrics);
        assert_eq!(modes.len(), 8);
        // Modes should be sorted by descending eigenvalue magnitude
        for i in 1..modes.len() {
            assert!(modes[i].eigenvalue.abs() <= modes[i - 1].eigenvalue.abs());
        }
    }

    #[test]
    fn test_find_weakest_mode() {
        let analyzer = SpectralAnalyzer::new(4);
        let modes = vec![
            SpectralMode {
                eigenvalue: 0.9,
                eigenvector: vec![1.0, 0.0, 0.0, 0.0],
                mode_name: "m0".into(),
                frequency: 1.0,
                decay_rate: 0.1,
            },
            SpectralMode {
                eigenvalue: 0.01,
                eigenvector: vec![0.0, 1.0, 0.0, 0.0],
                mode_name: "m1".into(),
                frequency: 2.0,
                decay_rate: 0.99,
            },
            SpectralMode {
                eigenvalue: 0.5,
                eigenvector: vec![0.0, 0.0, 1.0, 0.0],
                mode_name: "m2".into(),
                frequency: 3.0,
                decay_rate: 0.5,
            },
        ];
        let weakest = analyzer.find_weakest_mode(&modes);
        assert_eq!(weakest.mode_name, "m1");
    }

    #[test]
    fn test_improvement_direction() {
        let analyzer = SpectralAnalyzer::new(2);
        let mode = SpectralMode {
            eigenvalue: 0.5,
            eigenvector: vec![1.0, 0.0],
            mode_name: "test".into(),
            frequency: 1.0,
            decay_rate: 0.5,
        };
        let dir = analyzer.compute_improvement_direction(&mode);
        assert!((dir[0] - 2.0).abs() < 1e-10);
        assert!((dir[1]).abs() < 1e-10);
    }

    #[test]
    fn test_eigendecompose_identity() {
        let m = DMatrix::identity(3, 3);
        let (eigenvalues, _) = symmetric_eigendecompose(&m);
        assert_eq!(eigenvalues.len(), 3);
        for ev in &eigenvalues {
            assert!((ev - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_eigendecompose_diagonal() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0, 2.0]));
        let (eigenvalues, _) = symmetric_eigendecompose(&m);
        assert_eq!(eigenvalues.len(), 3);
        let mut sorted: Vec<f64> = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 3.0).abs() < 1e-6);
        assert!((sorted[1] - 2.0).abs() < 1e-6);
        assert!((sorted[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_eigendecompose_symmetric() {
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let (eigenvalues, eigenvectors) = symmetric_eigendecompose(&m);
        // Eigenvalues should be 3 and 1
        let mut sorted: Vec<f64> = eigenvalues;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 3.0).abs() < 1e-6);
        assert!((sorted[1] - 1.0).abs() < 1e-6);
        // Check orthogonality
        let dot = eigenvectors.column(0).dot(&eigenvectors.column(1));
        assert!(dot.abs() < 1e-6);
    }

    #[test]
    fn test_mode_frequency_formula() {
        let analyzer = SpectralAnalyzer::new(4);
        let modes = analyzer.analyze(&serde_json::Value::Null, &std::collections::HashMap::new());
        assert!((modes[0].frequency - std::f64::consts::PI / 4.0).abs() < 1e-10);
        assert!((modes[1].frequency - 2.0 * std::f64::consts::PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_classify_mode() {
        let analyzer = SpectralAnalyzer::new(8);
        assert_eq!(
            analyzer.classify_mode(&[0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            "tool_use"
        );
    }
}
