# lau-sia2-engine

> SIA² spectral engine: Banach fixed-point learning, conservation laws, Fisher information, and renormalization in Rust

## What This Does

lau-sia2-engine is the Rust core of the SIA² (Spectral Improvement Architecture squared) framework. It provides ten interlocking modules that implement contraction-based learning (Banach fixed-point theorem), spectral conservation laws, feedback control loops, Fisher information geometry, orchestration of spectral operations, PDE solvers, renormalization group flow, spectral decompositions, and trajectory optimization — all with rigorous mathematical foundations.

## The Key Idea

Traditional learning uses gradient descent. SIA² uses **contraction mappings**: the learning update is provably a contraction in a Banach space, guaranteeing convergence to a unique fixed point. Combined with spectral conservation laws (total spectral energy is preserved across transformations) and renormalization (coarse-graining that preserves fixed points), you get a learning system with provable convergence, stability, and scale-invariance.

## Install

```toml
[dependencies]
lau-sia2-engine = { git = "https://github.com/SuperInstance/lau-sia2-engine" }
```

## Quick Start

```rust
use lau_sia2_engine::banach::ContractionMap;
use lau_sia2_engine::spectral::SpectralDecomposer;
use lau_sia2_engine::fisher::FisherInformation;
use lau_sia2_engine::orchestrator::Orchestrator;
use nalgebra::DVector;

// Create a contraction mapping for learning
let contraction = ContractionMap::new(0.5); // Lipschitz constant < 1
let initial = DVector::from_vec(vec![1.0, 2.0, 3.0]);
let fixed_point = contraction.iterate_to_fixed_point(&initial, 1e-10);
println!("Fixed point: {:?}", fixed_point);

// Spectral decomposition
let decomposer = SpectralDecomposer::new();
let matrix = /* your matrix */;
let (eigenvalues, eigenvectors) = decomposer.decompose(&matrix);

// Fisher information matrix from parameter samples
let fisher = FisherInformation::from_samples(&samples);
let natural_gradient = fisher.natural_gradient(&euclidean_gradient);
```

## API Reference

### `banach` — Contraction Mappings

| Type | Description |
|------|-------------|
| `ContractionMap::new(lipschitz)` | Create with Lipschitz constant < 1. |
| `iterate_to_fixed_point(x0, tol)` | Iterate until ‖f(x) - x‖ < tol. Guaranteed convergence. |
| `lipschitz_constant()` | Returns the contraction rate. |

### `conservation` — Spectral Conservation Laws

| Type | Description |
|------|-------------|
| `ConservationLaw` | Verifies total spectral energy is preserved. |
| `verify_conservation(before, after)` | Check that Σλᵢ is unchanged. |

### `feedback` — Feedback Control

| Type | Description |
|------|-------------|
| `FeedbackController` | PD controller for spectral trajectory stabilization. |
| `compute_correction(current, target)` | Returns correction vector. |

### `fisher` — Fisher Information

| Type | Description |
|------|-------------|
| `FisherInformation::from_samples(samples)` | Estimate Fisher information matrix. |
| `natural_gradient(euclidean_grad)` | Transform to natural gradient via F⁻¹∇. |

### `orchestrator` — Spectral Orchestration

| Type | Description |
|------|-------------|
| `Orchestrator` | Coordinates spectral operations across modules. |
| `run_step(state)` | One orchestration step: spectral → feedback → update. |

### `pde` — PDE Solvers

| Type | Description |
|------|-------------|
| `HeatSolver` | Implicit heat equation solver (spectral diffusion). |
| `WaveSolver` | Wave equation via spectral decomposition. |

### `renormalization` — Renormalization Group

| Type | Description |
|------|-------------|
| `RGFlow` | Coarse-graining that preserves fixed points. |
| `renormalize(matrix, scale)` | Apply one RG step. |

### `spectral` — Spectral Methods

| Type | Description |
|------|-------------|
| `SpectralDecomposer` | Eigenvalue/eigenvector computation. |
| `decompose(matrix)` | Full eigendecomposition. |

### `trajectory` — Trajectory Optimization

| Type | Description |
|------|-------------|
| `TrajectoryOptimizer` | Geodesic path planning on manifolds. |
| `optimize(start, end, steps)` | Compute optimal spectral trajectory. |

## How It Works

The engine operates in cycles:
1. **Spectral Decomposition**: Extract eigenstructure of the current state.
2. **Conservation Check**: Verify spectral energy is conserved.
3. **Fisher Update**: Compute natural gradient using Fisher information.
4. **Contraction Step**: Apply contraction mapping (guaranteed convergence).
5. **Feedback Correction**: PD controller stabilizes the trajectory.
6. **Renormalization**: Coarse-grain if spectral scale changes significantly.

Each cycle is a Banach contraction, so the entire process converges provably.

## The Math

- **Banach Fixed-Point Theorem**: If f: X→X has Lipschitz constant L < 1, then f has a unique fixed point x* and xₙ → x* for any starting x₀.
- **Fisher Information**: I(θ) = E[(∂ log p(x|θ)/∂θ)(∂ log p(x|θ)/∂θ)ᵀ]. Natural gradient: ∇̃ = I⁻¹∇.
- **Conservation Law**: Σλᵢ is invariant under unitary transformations.

## Testing

90 tests covering:
- Contraction mapping convergence proofs
- Conservation law verification
- Feedback controller stability
- Fisher information estimation accuracy
- Orchestrator full-cycle runs
- PDE solver convergence
- Renormalization fixed-point preservation
- Spectral decomposition correctness

## License

MIT
