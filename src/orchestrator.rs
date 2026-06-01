//! Full SIA² improvement loop orchestrator.

use crate::banach::BanachConvergence;
use crate::conservation::ConservationChecker;
use crate::fisher::InformationGeometry;
use crate::pde::PDEImprovementDynamics;
use crate::renormalization::RenormalizationTracker;
use crate::spectral::SpectralAnalyzer;
use crate::trajectory::{ImprovementStep, ImprovementTrajectory};

use std::collections::HashMap;

/// The full SIA² spectral improvement orchestrator.
pub struct SIA2Orchestrator {
    /// Spectral analyzer.
    pub spectral: SpectralAnalyzer,
    /// Conservation checker.
    pub conservation: Option<ConservationChecker>,
    /// Banach convergence tracker.
    pub banach: BanachConvergence,
    /// Information geometry.
    pub info_geom: InformationGeometry,
    /// PDE dynamics.
    pub pde: PDEImprovementDynamics,
    /// Renormalization tracker.
    pub rg: RenormalizationTracker,
    /// Improvement trajectory.
    pub trajectory: ImprovementTrajectory,
}

impl SIA2Orchestrator {
    /// Create a new orchestrator.
    pub fn new(n_capabilities: usize) -> Self {
        Self {
            spectral: SpectralAnalyzer::new(n_capabilities),
            conservation: None,
            banach: BanachConvergence::new(),
            info_geom: InformationGeometry::new(n_capabilities),
            pde: PDEImprovementDynamics::new(n_capabilities, 0.1),
            rg: RenormalizationTracker::new(),
            trajectory: ImprovementTrajectory::new("default"),
        }
    }

    /// Initialize with starting metrics.
    pub fn initialize(&mut self, initial_metrics: &HashMap<String, f64>, task_name: &str) {
        self.conservation = Some(ConservationChecker::new(initial_metrics));
        self.banach = BanachConvergence::new();
        self.trajectory = ImprovementTrajectory::new(task_name);
        self.rg.add_scale(initial_metrics);
    }

    /// Run one step of the spectral improvement loop.
    pub fn analyze_and_plan(
        &mut self,
        metrics: &HashMap<String, f64>,
    ) -> ImprovementStep {
        // 1. Spectral decomposition
        let modes = self.spectral.analyze(&serde_json::Value::Null, metrics);
        let weakest = self.spectral.find_weakest_mode(&modes);

        // 2. Improvement direction
        let _direction = self.spectral.compute_improvement_direction(weakest);

        // 3. Conservation check
        let laws_hold = self.conservation
            .as_mut()
            .map(|c| c.check(metrics).iter().all(|l| l.is_conserved()))
            .unwrap_or(true);

        // 4. Banach contraction (auto-detects metric names)
        let contraction = self.banach.compute_contraction(metrics);

        // 5. Fisher distance
        let fisher_dist = self.banach.performance()
            .map(|prev| self.info_geom.fisher_rao_distance(prev, metrics))
            .unwrap_or(0.0);

        // 6. Update Fisher information
        if self.banach.performance_history.len() >= 2 {
            self.info_geom.compute_fisher_information(&self.banach.performance_history);
        }

        // 7. Spectral gap
        let spectral_gap = if modes.len() >= 2 {
            let mut eigs: Vec<f64> = modes.iter().map(|m| m.eigenvalue.abs()).collect();
            eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
            eigs[0] - eigs[1]
        } else {
            0.0
        };

        // 8. RG tracking
        self.rg.add_scale(metrics);

        // Build step
        let gen = self.trajectory.len() + 1;
        let before = self.banach.performance()
            .cloned()
            .unwrap_or_default();
        let step = ImprovementStep {
            generation: gen,
            target_mode: weakest.mode_name.clone(),
            performance_before: before,
            performance_after: metrics.clone(),
            banach_contraction: contraction,
            conservation_holds: laws_hold,
            information_gain: fisher_dist,
            spectral_gap,
        };

        self.trajectory.add_step(step.clone());
        step
    }

    /// Whether the improvement loop has converged.
    pub fn is_converged(&self) -> bool {
        self.trajectory.is_converged()
    }

    /// Get the current contraction ratio.
    pub fn contraction_ratio(&self) -> f64 {
        self.banach.last_contraction().unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metrics(vals: &[f64]) -> HashMap<String, f64> {
        let names = ["reasoning", "tool_use", "error_handling", "efficiency",
                     "robustness", "generalization", "creativity", "consistency"];
        vals.iter().enumerate()
            .map(|(i, &v)| (names[i % names.len()].to_string(), v))
            .collect()
    }

    #[test]
    fn test_orchestrator_creation() {
        let orch = SIA2Orchestrator::new(8);
        assert!(orch.trajectory.is_empty());
    }

    #[test]
    fn test_orchestrator_initialize() {
        let mut orch = SIA2Orchestrator::new(8);
        let metrics = test_metrics(&[0.5; 8]);
        orch.initialize(&metrics, "test_task");
        assert!(orch.conservation.is_some());
    }

    #[test]
    fn test_orchestrator_single_step() {
        let mut orch = SIA2Orchestrator::new(8);
        let metrics = test_metrics(&[0.5; 8]);
        orch.initialize(&metrics, "test");

        let step = orch.analyze_and_plan(&test_metrics(&[0.6; 8]));
        assert_eq!(step.generation, 1);
        assert!(!step.target_mode.is_empty());
    }

    #[test]
    fn test_orchestrator_multiple_steps() {
        let mut orch = SIA2Orchestrator::new(8);
        let metrics = test_metrics(&[0.5; 8]);
        orch.initialize(&metrics, "test");

        for i in 0..5 {
            let v = 0.5 + i as f64 * 0.05;
            let step = orch.analyze_and_plan(&test_metrics(&[v; 8]));
            assert_eq!(step.generation, i + 1);
        }
        assert_eq!(orch.trajectory.len(), 5);
    }

    #[test]
    fn test_orchestrator_convergence() {
        let mut orch = SIA2Orchestrator::new(4);
        let metrics = test_metrics(&[0.5; 4]);
        orch.initialize(&metrics, "test");

        for i in 0..10 {
            let v = 0.5 + (1.0 - 0.8_f64.powi(i + 1)) * 0.3;
            orch.analyze_and_plan(&test_metrics(&[v; 4]));
        }
        assert_eq!(orch.trajectory.len(), 10);
    }

    #[test]
    fn test_orchestrator_not_converged_initially() {
        let orch = SIA2Orchestrator::new(8);
        assert!(!orch.is_converged());
    }

    #[test]
    fn test_orchestrator_contraction_ratio_default() {
        let orch = SIA2Orchestrator::new(4);
        assert!((orch.contraction_ratio() - 1.0).abs() < 1e-10);
    }
}
