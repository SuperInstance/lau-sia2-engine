//! # SIA² Engine — Spectral Improvement Architecture
//!
//! High-performance Rust engine for SIA² (Self-Improving AI with Spectral Architecture).
//!
//! Mathematical foundations:
//! - Banach fixed point theorem → improvement MUST converge
//! - Spectral decomposition → targeted improvement of weakest frequencies
//! - Information geometry → Riemannian structure on improvement landscape
//! - Conservation laws → no capability lost during improvement
//! - PDE dynamics → improvement follows diffusion equation on belief space
//! - Noether's theorem → symmetries produce conserved quantities
//!
//! # Modules
//! - [`spectral`] — Eigendecomposition of performance matrix, eigenmode ranking
//! - [`conservation`] — Noether conservation law checking
//! - [`banach`] — Contraction ratio, convergence prediction, fixed point detection
//! - [`fisher`] — Fisher information matrix, natural gradient, Fisher-Rao distance
//! - [`pde`] — Heat equation solver, energy estimates, maximum principle
//! - [`renormalization`] — RG beta function, fixed point detection, universality classification
//! - [`trajectory`] — Full improvement trajectory tracking and serialization
//! - [`feedback`] — Enhanced feedback prompt generation with spectral analysis
//! - [`orchestrator`] — The full SIA² improvement loop

pub mod banach;
pub mod conservation;
pub mod feedback;
pub mod fisher;
pub mod orchestrator;
pub mod pde;
pub mod renormalization;
pub mod spectral;
pub mod trajectory;

pub use banach::BanachConvergence;
pub use conservation::ConservationChecker;
pub use fisher::InformationGeometry;
pub use orchestrator::SIA2Orchestrator;
pub use pde::PDEImprovementDynamics;
pub use renormalization::RenormalizationTracker;
pub use spectral::SpectralAnalyzer;
pub use trajectory::{ImprovementStep, ImprovementTrajectory};
