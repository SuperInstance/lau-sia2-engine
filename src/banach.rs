//! Banach convergence tracker — contraction ratio, convergence prediction, fixed point detection.
//!
//! The improvement operator T maps agent_i to agent_{i+1}.
//! If ||T(x) - T(y)|| ≤ q·||x - y|| for some q < 1,
//! then the sequence MUST converge to a unique fixed point (Banach's theorem).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Banach convergence tracker.
pub struct BanachConvergence {
    /// Names of the metrics being tracked.
    pub metric_names: Vec<String>,
    /// History of performance measurements.
    pub performance_history: Vec<HashMap<String, f64>>,
}

impl BanachConvergence {
    /// Create a new tracker with empty metric names (will be inferred from data).
    pub fn new() -> Self {
        Self {
            metric_names: Vec::new(),
            performance_history: Vec::new(),
        }
    }

    /// Create a new tracker for the given metric names.
    pub fn with_metric_names(metric_names: Vec<String>) -> Self {
        Self {
            metric_names,
            performance_history: Vec::new(),
        }
    }

    /// Compute the contraction ratio q.
    ///
    /// q = ||perf_n - perf_{n-1}|| / ||perf_{n-1} - perf_{n-2}||
    ///
    /// If q < 1, the improvement operator is a contraction mapping
    /// and convergence is GUARANTEED by Banach's theorem.
    pub fn compute_contraction_ratio(&mut self, current: &HashMap<String, f64>) -> f64 {
        self.performance_history.push(current.clone());

        if self.performance_history.len() < 3 {
            return 1.0;
        }

        let d_current = self.metric_distance(
            &self.performance_history[self.performance_history.len() - 1],
            &self.performance_history[self.performance_history.len() - 2],
        );
        let d_previous = self.metric_distance(
            &self.performance_history[self.performance_history.len() - 2],
            &self.performance_history[self.performance_history.len() - 3],
        );

        if d_previous == 0.0 {
            return 0.0; // Already at fixed point
        }

        let contraction = d_current / d_previous;
        contraction.clamp(0.0, 2.0)
    }

    /// Predict when convergence will be reached (generation number).
    ///
    /// If q < 1, the fixed point is reached in O(log(ε) / log(q)) steps.
    pub fn predict_convergence_generation(&self) -> Option<usize> {
        if self.performance_history.len() < 3 {
            return None;
        }

        let n = self.performance_history.len();
        let q = {
            let d_curr = self.metric_distance(&self.performance_history[n - 1], &self.performance_history[n - 2]);
            let d_prev = self.metric_distance(&self.performance_history[n - 2], &self.performance_history[n - 3]);
            if d_prev == 0.0 {
                return Some(n);
            }
            d_curr / d_prev
        };

        if q >= 1.0 {
            return None;
        }

        let current_dist = self.metric_distance(
            &self.performance_history[n - 2],
            &self.performance_history[n - 1],
        );

        if current_dist == 0.0 || q == 0.0 {
            return Some(n);
        }

        let epsilon = 0.001;
        let remaining = (epsilon / current_dist).ln() / q.ln();
        Some(n + 1.max(remaining as usize))
    }

    /// Whether the improvement operator is a contraction mapping.
    pub fn is_contraction(&self) -> bool {
        if self.performance_history.len() < 3 {
            return true; // Assume contraction until proven otherwise
        }
        let n = self.performance_history.len();
        let d_curr = self.metric_distance(&self.performance_history[n - 1], &self.performance_history[n - 2]);
        let d_prev = self.metric_distance(&self.performance_history[n - 2], &self.performance_history[n - 3]);
        if d_prev == 0.0 {
            return true;
        }
        d_curr / d_prev < 1.0
    }

    /// Check if a fixed point has been reached (no movement between steps).
    pub fn is_fixed_point(&self) -> bool {
        if self.performance_history.len() < 2 {
            return false;
        }
        let n = self.performance_history.len();
        self.metric_distance(&self.performance_history[n - 1], &self.performance_history[n - 2]) < 1e-6
    }

    /// Compute contraction ratio (alias for orchestrator compatibility).
    pub fn compute_contraction(&mut self, current: &HashMap<String, f64>) -> f64 {
        // Auto-detect metric names from first call
        if self.metric_names.is_empty() && !current.is_empty() {
            self.metric_names = current.keys().cloned().collect();
        }
        self.compute_contraction_ratio(current)
    }

    /// Get the latest performance snapshot.
    pub fn performance(&self) -> Option<&HashMap<String, f64>> {
        self.performance_history.last()
    }

    /// Get the last computed contraction ratio.
    pub fn last_contraction(&self) -> Option<f64> {
        if self.performance_history.len() < 3 {
            return None;
        }
        let n = self.performance_history.len();
        let d_curr = self.metric_distance(&self.performance_history[n - 1], &self.performance_history[n - 2]);
        let d_prev = self.metric_distance(&self.performance_history[n - 2], &self.performance_history[n - 3]);
        if d_prev == 0.0 { return Some(0.0); }
        Some((d_curr / d_prev).clamp(0.0, 2.0))
    }

    fn metric_distance(&self, a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
        let names = if self.metric_names.is_empty() {
            a.keys().collect::<Vec<_>>()
        } else {
            self.metric_names.iter().collect::<Vec<_>>()
        };
        let sum: f64 = names
            .iter()
            .map(|name| {
                let va = a.get(*name).copied().unwrap_or(0.0);
                let vb = b.get(*name).copied().unwrap_or(0.0);
                (va - vb).powi(2)
            })
            .sum();
        sum.sqrt()
    }
}

/// Convergence status summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStatus {
    /// Whether the operator is a contraction.
    pub is_contraction: bool,
    /// Latest contraction ratio.
    pub contraction_ratio: f64,
    /// Predicted convergence generation (if applicable).
    pub predicted_generation: Option<usize>,
    /// Whether fixed point is reached.
    pub at_fixed_point: bool,
    /// Number of generations tracked.
    pub generations: usize,
}

impl BanachConvergence {
    /// Get a summary of convergence status.
    pub fn status(&self) -> ConvergenceStatus {
        let q = if self.performance_history.len() >= 3 {
            let n = self.performance_history.len();
            let d_curr = self.metric_distance(&self.performance_history[n - 1], &self.performance_history[n - 2]);
            let d_prev = self.metric_distance(&self.performance_history[n - 2], &self.performance_history[n - 3]);
            if d_prev == 0.0 { 0.0 } else { (d_curr / d_prev).clamp(0.0, 2.0) }
        } else {
            1.0
        };

        ConvergenceStatus {
            is_contraction: q < 1.0,
            contraction_ratio: q,
            predicted_generation: self.predict_convergence_generation(),
            at_fixed_point: self.is_fixed_point(),
            generations: self.performance_history.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_initial_contraction_is_one() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        let q = banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        assert!((q - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_with_converging_sequence() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.5)]));
        let q = banach.compute_contraction_ratio(&make_metrics(&[("a", 5.7)]));
        // d_curr = 0.2, d_prev = 0.5, q = 0.4
        assert!(q < 1.0, "Expected q < 1.0, got {}", q);
    }

    #[test]
    fn test_contraction_diverging() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.5)]));
        let q = banach.compute_contraction_ratio(&make_metrics(&[("a", 6.5)]));
        // d_curr = 1.0, d_prev = 0.5, q = 2.0
        assert!(q >= 1.0);
    }

    #[test]
    fn test_contraction_at_fixed_point() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.5)]));
        let q = banach.compute_contraction_ratio(&make_metrics(&[("a", 5.5)]));
        assert!((q - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_convergence() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 1.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 0.5)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 0.25)]));
        let pred = banach.predict_convergence_generation();
        assert!(pred.is_some());
        assert!(pred.unwrap() >= 3);
    }

    #[test]
    fn test_predict_no_convergence_when_diverging() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.5)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 6.5)]));
        let pred = banach.predict_convergence_generation();
        assert!(pred.is_none());
    }

    #[test]
    fn test_is_contraction() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.5)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.7)]));
        assert!(banach.is_contraction());
    }

    #[test]
    fn test_is_fixed_point() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        assert!(!banach.is_fixed_point());
        banach.compute_contraction_ratio(&make_metrics(&[("a", 5.0)]));
        assert!(banach.is_fixed_point());
    }

    #[test]
    fn test_status_summary() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 1.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 0.5)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 0.25)]));
        let status = banach.status();
        assert!(status.is_contraction);
        assert_eq!(status.generations, 3);
    }

    #[test]
    fn test_multi_metric_distance() {
        let mut banach = BanachConvergence::with_metric_names(vec!["a".into(), "b".into()]);
        banach.compute_contraction_ratio(&make_metrics(&[("a", 0.0), ("b", 0.0)]));
        banach.compute_contraction_ratio(&make_metrics(&[("a", 3.0), ("b", 4.0)]));
        // distance = sqrt(9 + 16) = 5
        banach.compute_contraction_ratio(&make_metrics(&[("a", 3.0), ("b", 4.0)]));
        // d_curr = 0, so q = 0
        assert!(banach.is_fixed_point());
    }
}
