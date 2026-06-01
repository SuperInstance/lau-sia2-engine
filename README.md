# lau-sia2-engine

**High-performance Rust engine for SIA² (Self-Improving AI with Spectral Architecture).**

A mathematically rigorous framework that guarantees AI self-improvement converges, using the Banach fixed-point theorem, spectral decomposition, information geometry, PDE dynamics, renormalization-group flow, and Noether conservation laws.

---

## What This Does

This crate implements the *computational core* of the SIA² loop — the theorem-backed machinery that takes an AI agent's performance metrics, decides *what* to improve and *how fast* it will converge, and verifies that no capability is lost along the way.

Given a snapshot of performance across N capability dimensions (reasoning, tool use, error handling, efficiency, robustness, generalization, creativity, consistency), the engine:

1. **Decomposes** performance into spectral eigenmodes via eigendecomposition of the capability correlation matrix.
2. **Identifies** the weakest eigenmode — the "frequency" of performance most in need of reinforcement.
3. **Computes** a natural-gradient improvement direction scaled by the inverse Fisher information.
4. **Tracks** convergence via the Banach contraction ratio, predicting *when* the agent will reach a fixed point.
5. **Verifies** four conservation laws (capability conservation, Landauer bound, continuity, monotonicity) so improvement never destroys existing capability.
6. **Models** multi-step dynamics as a reaction–diffusion PDE on the performance manifold.
7. **Classifies** the improvement trajectory into a renormalization-group universality class (Gaussian, Wilson–Fisher, asymptotic freedom, or relevant operator).

Everything is pure Rust, zero unsafe, serializable with `serde`.

---

## Key Idea

> **Self-improvement is a contraction mapping on a Banach space.**

If the improvement operator T satisfies ‖T(x) − T(y)‖ ≤ q‖x − y‖ for some q < 1, then Banach's fixed-point theorem guarantees the agent converges to a unique optimal state. The engine monitors q in real time and raises an alarm when the operator ceases to be a contraction.

---

## Install

```toml
# Cargo.toml
[dependencies]
lau-sia2-engine = "0.1"
```

Requires Rust 2021 edition (MSRV 1.56+).

### Dependencies

| crate | purpose |
|---|---|
| `nalgebra` | linear algebra (eigen-decomposition, matrix ops) |
| `num-complex` | complex number support |
| `serde` / `serde_json` | serialization of trajectories, steps, summaries |
| `chrono` | timestamping |

Dev dependency: `approx` for floating-point assertions.

---

## Quick Start

```rust
use lau_sia2_engine::SIA2Orchestrator;
use std::collections::HashMap;

fn main() {
    let mut orch = SIA2Orchestrator::new(8);

    // Initial performance across 8 capability dimensions
    let initial: HashMap<String, f64> = [
        ("reasoning", 0.5), ("tool_use", 0.5), ("error_handling", 0.5),
        ("efficiency", 0.5), ("robustness", 0.5), ("generalization", 0.5),
        ("creativity", 0.5), ("consistency", 0.5),
    ].iter().map(|(k, v)| (k.to_string(), *v)).collect();

    orch.initialize(&initial, "my_task");

    // Simulate improvement over generations
    for gen in 1..=10 {
        let improved: HashMap<String, f64> = initial.keys()
            .map(|k| (k.clone(), 0.5 + gen as f64 * 0.05))
            .collect();
        let step = orch.analyze_and_plan(&improved);
        println!(
            "gen={} mode={} q={:.4} conserved={} gap={:.4}",
            step.generation, step.target_mode,
            step.banach_contraction, step.conservation_holds,
            step.spectral_gap,
        );
    }

    println!("Converged? {}", orch.is_converged());
}
```

---

## API Reference

### `SIA2Orchestrator` — the top-level loop

| method | description |
|---|---|
| `new(n_capabilities)` | create orchestrator for N dimensions |
| `initialize(&metrics, task_name)` | set baseline, reset trackers |
| `analyze_and_plan(&metrics) → ImprovementStep` | run one full cycle (spectral → conservation → Banach → Fisher → RG → PDE) |
| `is_converged() → bool` | has the contraction ratio dropped below 0.5? |
| `contraction_ratio() → f64` | latest Banach q value |

Fields are public: `.spectral`, `.conservation`, `.banach`, `.info_geom`, `.pde`, `.rg`, `.trajectory`.

### `SpectralAnalyzer` — eigenmode decomposition

| method | description |
|---|---|
| `new(n)` | N-dimension analyzer (default 8) |
| `analyze(&log, &metrics) → Vec<SpectralMode>` | full eigendecomposition, sorted descending by eigenvalue magnitude |
| `find_weakest_mode(&modes) → &SpectralMode` | smallest-eigenvalue mode |
| `compute_improvement_direction(&mode) → Vec<f64>` | natural-gradient direction (eigenvector / eigenvalue) |
| `build_correlation_matrix(&log) → DMatrix` | tridiagonal-adjacent capability correlation |

`SpectralMode` fields: `eigenvalue`, `eigenvector`, `mode_name`, `frequency`, `decay_rate`.

### `BanachConvergence` — contraction tracking

| method | description |
|---|---|
| `new()` / `with_metric_names(names)` | create tracker |
| `compute_contraction_ratio(&metrics) → f64` | q = ‖Δ_n‖ / ‖Δ_{n-1}‖ |
| `predict_convergence_generation() → Option<usize>` | O(log ε / log q) estimate |
| `is_contraction() → bool` | q < 1? |
| `is_fixed_point() → bool` | ‖Δ_n‖ < 1e-6? |
| `status() → ConvergenceStatus` | serializable summary |

### `ConservationChecker` — Noether law verification

Returns a `Vec<ConservationLaw>` with four entries per cycle:

| law | invariant |
|---|---|
| `capability_conservation` | total score ≥ initial (±5%) |
| `landauer_bound` | improvement cost bounded |
| `continuity` | no metric jumps > 0.8 |
| `monotonicity` | no metric decreases > 0.05 |

### `InformationGeometry` — Fisher information

| method | description |
|---|---|
| `compute_fisher_information(&perfs) → DMatrix` | F = (1/n) Σ ∇p ∇pᵀ + λI |
| `natural_gradient(&grad) → Vec<f64>` | F⁻¹ ∇L |
| `fisher_rao_distance(&a, &b) → f64` | Mahalanobis distance under F |
| `fisher_summary() → FisherSummary` | trace, det, condition number, effective dimensionality |

### `PDEImprovementDynamics` — reaction-diffusion model

| method | description |
|---|---|
| `predict_next_state(&state, reaction_rate) → HashMap` | explicit Euler: u += dt(D Δu + R(u)) |
| `predict_n_steps(&initial, rate, n) → HashMap` | multi-step rollout |
| `energy_estimate(&state) → f64` | L² energy |
| `maximum_principle_check(&before, &after, tol) → bool` | parabolic min-principle |
| `state_summary(&state) → PDEStateSummary` | energy, min, max, mean, variance |

### `RenormalizationTracker` — RG flow

| method | description |
|---|---|
| `add_scale(&metrics)` | record a coarse-grained level |
| `compute_beta_function() → HashMap` | β(g) = dg/d(ln μ) |
| `find_fixed_point() → Option` | β ≈ 0? |
| `classify_universality() → UniversalityClass` | Gaussian / WilsonFisher / AsymptoticFreedom / RelevantOperator |
| `correlation_length() → f64` | ξ = 1/|β| |
| `is_approaching_fixed_point() → bool` | late β < early β? |

### `ImprovementTrajectory` — step history

Serializable via `serde`. Methods: `add_step`, `is_converged`, `total_information_gain`, `predict_next` (linear extrapolation weighted by contraction ratio).

---

## How It Works

### The Improvement Loop (per generation)

```
┌──────────────────────────────────────────────────┐
│  metrics in                                      │
│  ├── SpectralAnalyzer: eigen-decompose           │
│  │   └── find weakest mode                       │
│  ├── ConservationChecker: verify 4 laws          │
│  ├── BanachConvergence: compute q                │
│  ├── InformationGeometry: Fisher distance        │
│  ├── PDEImprovementDynamics: predict next state  │
│  └── RenormalizationTracker: classify flow       │
│  → ImprovementStep (serialized to trajectory)    │
└──────────────────────────────────────────────────┘
```

1. **Spectral analysis** builds a correlation matrix over capabilities, performs Jacobi eigen-decomposition, and ranks modes by eigenvalue magnitude.
2. **Conservation checking** compares current metrics against the baseline to ensure no law is violated.
3. **Banach tracking** computes the ratio of successive improvements; if q < 1, convergence is guaranteed.
4. **Fisher information** accumulates gradient outer products from the performance history to form a Riemannian metric tensor.
5. **PDE dynamics** models the next-generation state as a diffusion + reaction step.
6. **Renormalization** classifies the trajectory's long-range behavior into a universality class.

### Jacobi Eigenvalue Algorithm

The engine uses the classical Jacobi rotation method for symmetric eigendecomposition — numerically stable, no unsafe code, and guaranteed to converge for real symmetric matrices. Off-diagonal elements are annihilated one at a time via Givens rotations until the matrix is diagonal (tolerance 1e-12).

---

## The Math

### Banach Fixed-Point Theorem

For a complete metric space (X, d) and contraction T: X → X with d(Tx, Ty) ≤ q·d(x, y), q ∈ [0, 1):

- T has a **unique** fixed point x*.
- For any x₀, the sequence xₙ₊₁ = T(xₙ) converges: d(xₙ, x*) ≤ qⁿ/(1−q) · d(x₀, x₁).

The engine tracks q in real time and predicts convergence generation via ⌈log(ε)/log(q)⌉.

### Spectral Decomposition

The capability correlation matrix C ∈ ℝ^{N×N} is decomposed: C = V Λ Vᵀ. Eigenvalues λ_i measure how much each orthogonal mode contributes to total performance variance. The **spectral gap** λ₁ − λ₂ indicates how dominant the leading mode is.

### Information Geometry

Performance lives on a statistical manifold with Fisher metric F_ij = E[∂ᵢ log p · ∂ⱼ log p]. The Fisher-Rao distance between states a, b is:

d_FR(a, b) = √((a−b)ᵀ F⁻¹ (a−b))

The natural gradient F⁻¹∇L corrects the steepest-descent direction for the curvature of the parameter space.

### PDE Dynamics

Improvement follows the reaction–diffusion equation:

∂u/∂t = D Δu + R(u)

where D is the diffusion coefficient (cross-capability spreading), Δu is the discrete Laplacian, and R(u) = r · max(mean − u, 0) drives below-average capabilities upward.

**Energy estimate:** ‖u(t)‖₂ ≤ ‖u(0)‖₂ · e^{−2Dt}.  
**Maximum principle:** min u(x, t) ≥ min u(x, 0) (performance floor never drops).

### Renormalization Group

As generations act as coarse-graining steps, the RG beta function β(g) = dg/d(ln μ) characterizes flow:

| class | β behavior | meaning |
|---|---|---|
| Gaussian | β ≈ 0 | at fixed point (trivial flow) |
| Wilson–Fisher | β has non-trivial zero | near phase transition |
| Asymptotic Freedom | β → 0 as g → ∞ | self-improving improvement rate |
| Relevant Operator | β large | strong, unclassifed flow |

### Conservation Laws (Noether)

By analogy with Noether's theorem (symmetries → conserved quantities):

| symmetry | conserved quantity |
|---|---|
| translational invariance in capability space | total capability score |
| information erasure symmetry | Landauer bound on improvement cost |
| temporal smoothness | continuity (no jumps) |
| monotonic improvement | individual metric non-decrease |

---

## Test Suite

**90 tests** across all 9 source files:

| module | tests |
|---|---|
| `spectral` | 12 |
| `conservation` | 11 |
| `banach` | 10 |
| `fisher` | 11 |
| `pde` | 11 |
| `renormalization` | 11 |
| `feedback` | 8 |
| `trajectory` | 9 |
| `orchestrator` | 7 |

Run: `cargo test`

---

## License

MIT
