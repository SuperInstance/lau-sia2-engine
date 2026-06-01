//! Trajectory tracking for SIA² improvement loop.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single step in the spectral improvement loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementStep {
    /// Generation number.
    pub generation: usize,
    /// Target mode being improved.
    pub target_mode: String,
    /// Performance before improvement.
    pub performance_before: HashMap<String, f64>,
    /// Performance after improvement.
    pub performance_after: HashMap<String, f64>,
    /// Banach contraction ratio.
    pub banach_contraction: f64,
    /// Whether conservation laws hold.
    pub conservation_holds: bool,
    /// Information gain (Fisher distance).
    pub information_gain: f64,
    /// Spectral gap (largest - second largest eigenvalue).
    pub spectral_gap: f64,
}

impl ImprovementStep {
    /// Whether this step is converging (Banach contraction < 1).
    pub fn is_converging(&self) -> bool {
        self.banach_contraction < 1.0
    }
}

/// Full trajectory of agent improvement over generations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImprovementTrajectory {
    /// Task name.
    pub task_name: String,
    /// When the trajectory started.
    pub started_at: String,
    /// When it converged (if it did).
    pub converged_at: Option<String>,
    /// All improvement steps.
    pub steps: Vec<ImprovementStep>,
}

impl ImprovementTrajectory {
    /// Create a new trajectory.
    pub fn new(task_name: &str) -> Self {
        Self {
            task_name: task_name.to_string(),
            started_at: chrono_now(),
            ..Default::default()
        }
    }

    /// Whether the trajectory has converged.
    pub fn is_converged(&self) -> bool {
        self.steps.last().map_or(false, |s| s.banach_contraction < 0.5)
    }

    /// Total Fisher information gained.
    pub fn total_information_gain(&self) -> f64 {
        self.steps.iter().map(|s| s.information_gain).sum()
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Add a step.
    pub fn add_step(&mut self, step: ImprovementStep) {
        self.steps.push(step);
    }

    /// Predict next generation's performance using linear extrapolation.
    pub fn predict_next(&self) -> Option<HashMap<String, f64>> {
        if self.steps.len() < 2 {
            return None;
        }
        let last = &self.steps[self.steps.len() - 1];
        let prev = &self.steps[self.steps.len() - 2];

        let mut predicted = HashMap::new();
        for key in last.performance_after.keys() {
            let curr = last.performance_after.get(key).copied().unwrap_or(0.0);
            let prev_val = prev.performance_after.get(key).copied().unwrap_or(0.0);
            let delta = curr - prev_val;
            predicted.insert(key.clone(), curr + delta * last.banach_contraction);
        }
        Some(predicted)
    }
}

fn chrono_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step(gen: usize, contraction: f64) -> ImprovementStep {
        let mut before = HashMap::new();
        before.insert("perf".to_string(), 0.5 + gen as f64 * 0.1);
        let mut after = HashMap::new();
        after.insert("perf".to_string(), 0.5 + (gen + 1) as f64 * 0.1);
        ImprovementStep {
            generation: gen,
            target_mode: "mode_0".to_string(),
            performance_before: before,
            performance_after: after,
            banach_contraction: contraction,
            conservation_holds: true,
            information_gain: 0.1,
            spectral_gap: 0.2,
        }
    }

    #[test]
    fn test_step_converging() {
        let step = make_step(1, 0.8);
        assert!(step.is_converging());
        let step2 = make_step(1, 1.2);
        assert!(!step2.is_converging());
    }

    #[test]
    fn test_trajectory_creation() {
        let traj = ImprovementTrajectory::new("test_task");
        assert_eq!(traj.task_name, "test_task");
        assert!(traj.is_empty());
    }

    #[test]
    fn test_trajectory_add_steps() {
        let mut traj = ImprovementTrajectory::new("test");
        traj.add_step(make_step(1, 0.8));
        traj.add_step(make_step(2, 0.6));
        assert_eq!(traj.len(), 2);
    }

    #[test]
    fn test_trajectory_information_gain() {
        let mut traj = ImprovementTrajectory::new("test");
        traj.add_step(make_step(1, 0.8));
        traj.add_step(make_step(2, 0.6));
        assert!((traj.total_information_gain() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_convergence() {
        let mut traj = ImprovementTrajectory::new("test");
        traj.add_step(make_step(1, 0.8));
        assert!(!traj.is_converged());
        traj.add_step(make_step(2, 0.3));
        assert!(traj.is_converged());
    }

    #[test]
    fn test_trajectory_prediction() {
        let mut traj = ImprovementTrajectory::new("test");
        traj.add_step(make_step(1, 0.8));
        // Need at least 2 steps
        assert!(traj.predict_next().is_none());
        traj.add_step(make_step(2, 0.8));
        let pred = traj.predict_next();
        assert!(pred.is_some());
        let p = pred.unwrap();
        assert!(p.contains_key("perf"));
    }

    #[test]
    fn test_trajectory_serialization() {
        let mut traj = ImprovementTrajectory::new("test");
        traj.add_step(make_step(1, 0.5));
        let json = serde_json::to_string(&traj).unwrap();
        let back: ImprovementTrajectory = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back.steps[0].generation, 1);
    }

    #[test]
    fn test_empty_trajectory_not_converged() {
        let traj = ImprovementTrajectory::new("test");
        assert!(!traj.is_converged());
    }

    #[test]
    fn test_step_serialization() {
        let step = make_step(3, 0.7);
        let json = serde_json::to_string(&step).unwrap();
        let back: ImprovementStep = serde_json::from_str(&json).unwrap();
        assert_eq!(back.generation, 3);
        assert!((back.banach_contraction - 0.7).abs() < 1e-10);
    }
}
