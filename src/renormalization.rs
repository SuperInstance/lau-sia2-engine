//! Renormalization group tracker — RG beta function, fixed point detection, universality classification.
//!
//! RG flow describes how a system changes as you "zoom out".
//! For agents: as generations progress, the improvement trajectory
//! flows toward fixed points (universality classes).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Universality class of improvement flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniversalityClass {
    /// β(g) ≈ 0 — improvement is trivial / at fixed point.
    Gaussian,
    /// β(g) has non-trivial zero — near phase transition.
    WilsonFisher,
    /// β(g) → 0 as g → ∞ — gets better at getting better.
    AsymptoticFreedom,
    /// Strong flow (not yet classified).
    RelevantOperator,
    /// Insufficient data.
    Unknown,
}

impl std::fmt::Display for UniversalityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gaussian => write!(f, "gaussian"),
            Self::WilsonFisher => write!(f, "wilson-fisher"),
            Self::AsymptoticFreedom => write!(f, "asymptotic_freedom"),
            Self::RelevantOperator => write!(f, "relevant_operator"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Renormalization group tracker.
pub struct RenormalizationTracker {
    /// Scale levels (coarse-grained performance at each generation).
    pub scales: Vec<HashMap<String, f64>>,
}

impl Default for RenormalizationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RenormalizationTracker {
    /// Create a new RG tracker.
    pub fn new() -> Self {
        Self { scales: Vec::new() }
    }

    /// Add a scale level (coarse-grained performance).
    pub fn add_scale(&mut self, metrics: &HashMap<String, f64>) {
        self.scales.push(metrics.clone());
    }

    /// Compute RG beta function: β(g) = dg/d(ln μ).
    ///
    /// At fixed points: β(g*) = 0.
    pub fn compute_beta_function(&self) -> HashMap<String, f64> {
        if self.scales.len() < 2 {
            return HashMap::new();
        }

        let mut beta = HashMap::new();
        let keys: Vec<&String> = self.scales.last().unwrap().keys().collect();

        for key in keys {
            let window = self.scales.len().min(5);
            let recent = &self.scales[self.scales.len() - window..];
            if recent.len() >= 2 {
                let v_last = recent.last().unwrap().get(key).copied().unwrap_or(0.0);
                let v_prev = recent[recent.len() - 2].get(key).copied().unwrap_or(0.0);
                beta.insert(key.clone(), v_last - v_prev);
            }
        }

        beta
    }

    /// Detect if improvement has reached a fixed point (β(g*) = 0).
    pub fn find_fixed_point(&self) -> Option<HashMap<String, f64>> {
        let beta = self.compute_beta_function();
        if beta.is_empty() {
            return None;
        }

        if beta.values().all(|v| v.abs() < 0.01) {
            Some(self.scales.last().unwrap().clone())
        } else {
            None
        }
    }

    /// Classify the universality class of the improvement flow.
    pub fn classify_universality(&self) -> UniversalityClass {
        let beta = self.compute_beta_function();
        if beta.is_empty() {
            return UniversalityClass::Unknown;
        }

        let avg_beta = beta.values().map(|v| v.abs()).sum::<f64>() / beta.len() as f64;

        if avg_beta < 0.01 {
            return UniversalityClass::Gaussian;
        }

        if avg_beta < 0.1 {
            return UniversalityClass::WilsonFisher;
        }

        // Check for asymptotic freedom: beta decreasing over time
        if self.scales.len() >= 4 {
            let mut recent_betas = Vec::new();
            let start = self.scales.len().saturating_sub(4);
            for i in start..self.scales.len() - 1 {
                for key in self.scales[i].keys() {
                    let v1 = self.scales[i].get(key).copied().unwrap_or(0.0);
                    let v2 = self.scales[i + 1].get(key).copied().unwrap_or(0.0);
                    recent_betas.push((v2 - v1).abs());
                }
            }

            if recent_betas.len() >= 2 && recent_betas.last().unwrap() < recent_betas.first().unwrap() {
                return UniversalityClass::AsymptoticFreedom;
            }
        }

        UniversalityClass::RelevantOperator
    }

    /// Compute the correlation length (distance to fixed point).
    pub fn correlation_length(&self) -> f64 {
        let beta = self.compute_beta_function();
        if beta.is_empty() {
            return f64::INFINITY;
        }

        let avg_beta = beta.values().map(|v| v.abs()).sum::<f64>() / beta.len() as f64;
        if avg_beta == 0.0 {
            return f64::INFINITY;
        }

        1.0 / avg_beta
    }

    /// Check if the flow is approaching a fixed point (IR attractive).
    pub fn is_approaching_fixed_point(&self) -> bool {
        if self.scales.len() < 4 {
            return false;
        }

        let n = self.scales.len();
        let early_avg = self.compute_avg_beta_for_range(0, n / 2);
        let late_avg = self.compute_avg_beta_for_range(n / 2, n);

        late_avg < early_avg
    }

    fn compute_avg_beta_for_range(&self, start: usize, end: usize) -> f64 {
        if end <= start + 1 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for i in start..end.saturating_sub(1) {
            if i + 1 < self.scales.len() {
                for key in self.scales[i].keys() {
                    let v1 = self.scales[i].get(key).copied().unwrap_or(0.0);
                    let v2 = self.scales[i + 1].get(key).copied().unwrap_or(0.0);
                    total += (v2 - v1).abs();
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
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
    fn test_empty_tracker() {
        let rg = RenormalizationTracker::new();
        assert!(rg.compute_beta_function().is_empty());
        assert!(rg.find_fixed_point().is_none());
        assert_eq!(rg.classify_universality(), UniversalityClass::Unknown);
    }

    #[test]
    fn test_beta_function_two_scales() {
        let mut rg = RenormalizationTracker::new();
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.5)]));
        let beta = rg.compute_beta_function();
        assert!((beta["a"] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_point_detected() {
        let mut rg = RenormalizationTracker::new();
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.001)]));
        let fp = rg.find_fixed_point();
        assert!(fp.is_some());
    }

    #[test]
    fn test_fixed_point_not_detected() {
        let mut rg = RenormalizationTracker::new();
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 2.0)]));
        let fp = rg.find_fixed_point();
        assert!(fp.is_none());
    }

    #[test]
    fn test_gaussian_classification() {
        let mut rg = RenormalizationTracker::new();
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.001)]));
        assert_eq!(rg.classify_universality(), UniversalityClass::Gaussian);
    }

    #[test]
    fn test_wilson_fisher_classification() {
        let mut rg = RenormalizationTracker::new();
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.05)]));
        assert_eq!(rg.classify_universality(), UniversalityClass::WilsonFisher);
    }

    #[test]
    fn test_relevant_operator_classification() {
        let mut rg = RenormalizationTracker::new();
        // Strong, increasing flow
        for i in 0..6 {
            rg.add_scale(&make_metrics(&[("a", i as f64)]));
        }
        assert_eq!(rg.classify_universality(), UniversalityClass::RelevantOperator);
    }

    #[test]
    fn test_asymptotic_freedom_classification() {
        let mut rg = RenormalizationTracker::new();
        // Decreasing beta: big changes early, small later
        rg.add_scale(&make_metrics(&[("a", 0.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.5)]));
        rg.add_scale(&make_metrics(&[("a", 1.6)]));
        rg.add_scale(&make_metrics(&[("a", 1.65)]));
        let class = rg.classify_universality();
        assert!(
            matches!(class, UniversalityClass::AsymptoticFreedom | UniversalityClass::WilsonFisher),
            "Expected asymptotic_freedom or wilson-fisher, got {:?}",
            class
        );
    }

    #[test]
    fn test_correlation_length() {
        let mut rg = RenormalizationTracker::new();
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.5)]));
        let xi = rg.correlation_length();
        assert!(xi.is_finite());
        assert!(xi > 0.0);
    }

    #[test]
    fn test_approaching_fixed_point() {
        let mut rg = RenormalizationTracker::new();
        // Decreasing changes
        rg.add_scale(&make_metrics(&[("a", 0.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.0)]));
        rg.add_scale(&make_metrics(&[("a", 1.5)]));
        rg.add_scale(&make_metrics(&[("a", 1.7)]));
        rg.add_scale(&make_metrics(&[("a", 1.75)]));
        rg.add_scale(&make_metrics(&[("a", 1.76)]));
        assert!(rg.is_approaching_fixed_point());
    }

    #[test]
    fn test_universality_display() {
        assert_eq!(UniversalityClass::Gaussian.to_string(), "gaussian");
        assert_eq!(UniversalityClass::WilsonFisher.to_string(), "wilson-fisher");
        assert_eq!(UniversalityClass::AsymptoticFreedom.to_string(), "asymptotic_freedom");
    }
}
