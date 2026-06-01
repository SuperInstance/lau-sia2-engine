//! PDE improvement dynamics — heat equation solver for performance prediction.
//!
//! Models agent improvement as a PDE on the performance manifold:
//! ∂u/∂t = D·Δu + R(u)
//!
//! - u(x,t) = agent performance at capability x, generation t
//! - D = diffusion coefficient (exploration rate)
//! - Δu = Laplacian (spreads improvements across capabilities)
//! - R(u) = reaction term (task-specific improvement)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// PDE improvement dynamics model.
pub struct PDEImprovementDynamics {
    /// Number of capability dimensions.
    pub n_capabilities: usize,
    /// Diffusion coefficient (exploration rate).
    pub diffusion: f64,
    /// Time step (one generation per step).
    pub dt: f64,
}

impl Default for PDEImprovementDynamics {
    fn default() -> Self {
        Self::new(8, 0.1)
    }
}

impl PDEImprovementDynamics {
    /// Create a new PDE dynamics model.
    pub fn new(n_capabilities: usize, diffusion: f64) -> Self {
        Self {
            n_capabilities,
            diffusion,
            dt: 1.0,
        }
    }

    /// Predict next generation's performance using PDE dynamics (explicit Euler).
    pub fn predict_next_state(
        &self,
        current: &HashMap<String, f64>,
        reaction_rate: f64,
    ) -> HashMap<String, f64> {
        let keys: Vec<&String> = current.keys().collect();
        let u: Vec<f64> = keys.iter().map(|k| current[*k]).collect();

        // Discrete 1D Laplacian: Δu_i = u_{i+1} - 2u_i + u_{i-1}
        let laplacian: Vec<f64> = u
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let left = if i > 0 { u[i - 1] } else { u[i] };
                let right = if i < u.len() - 1 { u[i + 1] } else { u[i] };
                left - 2.0 * u[i] + right
            })
            .collect();

        let mean = u.iter().sum::<f64>() / u.len().max(1) as f64;

        // Reaction: drives below-average capabilities upward
        let reaction: Vec<f64> = u
            .iter()
            .map(|&v| reaction_rate * (mean - v).max(0.0))
            .collect();

        // PDE step: u_new = u + dt * (D * Δu + R(u))
        let mut result = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            let u_new = u[i] + self.dt * (self.diffusion * laplacian[i] + reaction[i]);
            result.insert((*key).clone(), u_new);
        }
        result
    }

    /// Compute L² energy of the performance state.
    ///
    /// Energy estimate: ||u(t)||₂ ≤ ||u(0)||₂ · e^{-2Dt}
    pub fn energy_estimate(&self, current: &HashMap<String, f64>) -> f64 {
        current.values().map(|v| v * v).sum()
    }

    /// Verify maximum principle: performance doesn't decrease below minimum.
    ///
    /// For parabolic PDEs: min performance doesn't decrease.
    pub fn maximum_principle_check(
        &self,
        before: &HashMap<String, f64>,
        after: &HashMap<String, f64>,
        tolerance: f64,
    ) -> bool {
        let min_before = before.values().copied().fold(f64::MAX, f64::min);
        let min_after = after.values().copied().fold(f64::MAX, f64::min);
        min_after >= min_before - tolerance
    }

    /// Compute energy decay rate.
    ///
    /// Returns the ratio of current energy to previous energy.
    pub fn energy_decay_rate(
        &self,
        before: &HashMap<String, f64>,
        after: &HashMap<String, f64>,
    ) -> f64 {
        let e_before = self.energy_estimate(before);
        let e_after = self.energy_estimate(after);
        if e_before == 0.0 {
            return 0.0;
        }
        e_after / e_before
    }

    /// Run multiple PDE steps to get a long-range prediction.
    pub fn predict_n_steps(
        &self,
        initial: &HashMap<String, f64>,
        reaction_rate: f64,
        n_steps: usize,
    ) -> HashMap<String, f64> {
        let mut state = initial.clone();
        for _ in 0..n_steps {
            state = self.predict_next_state(&state, reaction_rate);
        }
        state
    }
}

/// PDE state summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDEStateSummary {
    /// L² energy.
    pub energy: f64,
    /// Minimum performance value.
    pub min_value: f64,
    /// Maximum performance value.
    pub max_value: f64,
    /// Mean performance value.
    pub mean_value: f64,
    /// Variance of performance.
    pub variance: f64,
}

impl PDEImprovementDynamics {
    /// Get a summary of the current PDE state.
    pub fn state_summary(&self, current: &HashMap<String, f64>) -> PDEStateSummary {
        let values: Vec<f64> = current.values().copied().collect();
        let energy: f64 = values.iter().map(|v| v * v).sum();
        let min_value = values.iter().copied().fold(f64::MAX, f64::min);
        let max_value = values.iter().copied().fold(f64::MIN, f64::max);
        let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / values.len().max(1) as f64;

        PDEStateSummary {
            energy,
            min_value,
            max_value,
            mean_value: mean,
            variance,
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
    fn test_default_creation() {
        let pde = PDEImprovementDynamics::default();
        assert_eq!(pde.n_capabilities, 8);
        assert!((pde.diffusion - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_predict_next_state() {
        let pde = PDEImprovementDynamics::new(4, 0.1);
        let current = make_metrics(&[("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]);
        let next = pde.predict_next_state(&current, 0.1);
        assert_eq!(next.len(), 4);
        // Values should change
        assert!(next["a"] != 1.0 || next["d"] != 4.0);
    }

    #[test]
    fn test_energy_estimate() {
        let pde = PDEImprovementDynamics::new(3, 0.1);
        let state = make_metrics(&[("a", 3.0), ("b", 4.0)]);
        let energy = pde.energy_estimate(&state);
        assert!((energy - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_maximum_principle_holds() {
        let pde = PDEImprovementDynamics::new(3, 0.1);
        let before = make_metrics(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        let after = make_metrics(&[("a", 1.1), ("b", 2.1), ("c", 3.1)]);
        assert!(pde.maximum_principle_check(&before, &after, 0.01));
    }

    #[test]
    fn test_maximum_principle_violated() {
        let pde = PDEImprovementDynamics::new(3, 0.1);
        let before = make_metrics(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        let after = make_metrics(&[("a", 0.5), ("b", 2.0), ("c", 3.0)]);
        assert!(!pde.maximum_principle_check(&before, &after, 0.01));
    }

    #[test]
    fn test_energy_decay_rate() {
        let pde = PDEImprovementDynamics::new(3, 0.1);
        let before = make_metrics(&[("a", 2.0), ("b", 2.0)]);
        let after = make_metrics(&[("a", 1.0), ("b", 1.0)]);
        let rate = pde.energy_decay_rate(&before, &after);
        assert!((rate - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_predict_n_steps() {
        let pde = PDEImprovementDynamics::new(3, 0.1);
        let initial = make_metrics(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        let result = pde.predict_n_steps(&initial, 0.1, 5);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_state_summary() {
        let pde = PDEImprovementDynamics::new(4, 0.1);
        let state = make_metrics(&[("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]);
        let summary = pde.state_summary(&state);
        assert!((summary.energy - 30.0).abs() < 1e-10);
        assert!((summary.min_value - 1.0).abs() < 1e-10);
        assert!((summary.max_value - 4.0).abs() < 1e-10);
        assert!((summary.mean_value - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_reaction_drives_below_average_up() {
        let pde = PDEImprovementDynamics::new(3, 0.0); // No diffusion
        let current = make_metrics(&[("a", 1.0), ("b", 5.0), ("c", 5.0)]);
        let next = pde.predict_next_state(&current, 0.5);
        // "a" is below mean (3.67), reaction should push it up
        assert!(next["a"] > 1.0);
        // "b" and "c" are above mean, no reaction for them
    }

    #[test]
    fn test_diffusion_smooths_values() {
        let pde = PDEImprovementDynamics::new(4, 1.0); // High diffusion
        let current = make_metrics(&[("a", 0.0), ("b", 10.0), ("c", 0.0), ("d", 10.0)]);
        let next = pde.predict_next_state(&current, 0.0); // No reaction
        // High values should decrease, low should increase
        assert!(next["b"] < 10.0);
        assert!(next["a"] > 0.0);
    }

    #[test]
    fn test_uniform_state_stable_without_reaction() {
        let pde = PDEImprovementDynamics::new(3, 0.1);
        let current = make_metrics(&[("a", 5.0), ("b", 5.0), ("c", 5.0)]);
        let next = pde.predict_next_state(&current, 0.0);
        // Uniform state: Laplacian = 0, reaction = 0
        for v in next.values() {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }
}
