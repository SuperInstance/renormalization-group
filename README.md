# renormalization-group

A Rust library applying the **renormalization group (RG)** from statistical physics to **multi-scale fleet analysis**.

Agents are coarse-grained into block variables at increasing scale; the library tracks how constraint parameters flow under this transformation. **Fixed points** correspond to self-similar fleet configurations; **relevant/irrelevant operators** classify which perturbations destabilize the fleet at large scales.

## Why RG for Fleet Analysis?

A fleet of autonomous agents has structure at every scale: individual agents form local clusters, clusters form divisions, divisions form the whole fleet. The renormalization group provides the mathematical machinery to systematically integrate out short-scale degrees of freedom and understand which parameters control large-scale behavior.

Key insights from RG thinking:

- **Fixed points** = self-similar fleet configurations that look the same at every scale
- **Relevant operators** = perturbations that grow under coarse-graining and dominate large-scale behavior (things you *must* get right)
- **Irrelevant operators** = perturbations that wash out at large scales (details that don't matter in the aggregate)
- **Universality** = different microscopic fleet topologies can produce identical large-scale behavior, meaning the details of individual agent interactions may matter less than you think

## Modules

### `block_spin` — Coarse-Graining Transformations

Group nearby agents into blocks and compute block properties (mean belief, aggregate constraint). This is the fundamental RG operation: integrating out short-scale degrees of freedom.

```rust
use renormalization_group::block_spin::*;

let agents: Vec<Agent> = (0..64)
    .map(|i| Agent {
        id: i,
        position: vec![i as f64],
        belief: 1.0,
        constraint: 1.0,
    })
    .collect();

let bt = BlockTransform::new(1, 4);
let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
assert_eq!(blocks.len(), 16); // 64/4 = 16 blocks
```

Three blocking strategies are supported:
- **Sequential**: agents 0..b, b..2b, etc.
- **SpatialNearest**: sort by position, then block
- **MajorityRule**: block spin = sign of mean belief (±1)

### `flow` — RG Flow Equations

Track how coupling parameters evolve under repeated coarse-graining.

```rust
use renormalization_group::flow::*;

// 1D Ising decimation flow
let flow_eq = Ising1DFlow::new(2.0);
let mut flow = RGFlow::new(vec![0.5, 0.0]); // K=0.5, h=0
let trajectory = flow.evolve(&flow_eq, 20);
// K flows toward 0 (high-temperature fixed point for 1D)
```

Built-in flow equations:
- **`Ising1DFlow`**: 1D Ising decimation — K' = atanh(tanh²(K))
- **`GaussianFlow`**: Gaussian (free) theory — r' = b²r, u' = b^(4-d)u
- **`WilsonFisherFlow`**: Wilson-Fisher ε-expansion near d=4

### `fixed_point` — Fixed-Point Detection & Stability

Linearize the RG flow around fixed points. Eigenvalues of the linearized flow determine relevant vs irrelevant directions.

```rust
use renormalization_group::fixed_point::*;

let flow_eq = GaussianFlow::new(3, 2.0);
let fp = analyze_stability(&flow_eq, &[0.0, 0.0]);
// Eigenvalues: b²=4 (relevant), b^(4-3)=2 (relevant in d=3)
assert_eq!(fp.n_relevant(), 2);
```

### `critical_surface` — Phase Boundaries

Identify critical manifolds separating fleet phases (ordered/disordered/critical).

```rust
use renormalization_group::critical_surface::*;

let flow_eq = GaussianFlow::new(3, 2.0);
let result = determine_phase(&flow_eq, &[0.01, 0.0], 0, 30);
assert_eq!(result.phase, FleetPhase::Disordered);
```

### `universality` — Universality Classes

Classify universality classes of fleet behavior. Different fleet topologies with the same RG fixed point exhibit the same large-scale behavior.

Pre-built universality classes with verified critical exponents:
- **Mean Field** (d ≥ 4)
- **2D Ising** (exact Onsager solution)
- **3D Ising** (numerical)
- **3D XY** (superfluid transition)
- **3D Heisenberg** (n=3)

Scaling relations verified:
- Rushbrooke: α + 2β + γ = 2
- Widom: γ = β(δ − 1)
- Fisher: γ = ν(2 − η)
- Josephson: 2 − α = dν

### `scaling` — Finite-Size Scaling

Critical exponents, scaling collapse, and finite-size corrections for bounded fleets.

```rust
use renormalization_group::scaling::*;

// Scaling collapse for susceptibility data
let result = scaling_collapse(
    &temperatures, &susceptibilities, &system_sizes,
    tc, nu, gamma_over_nu,
);

// Binder cumulant crossing to locate Tc
let tc = find_tc_from_crossing(&temps, &binder_l1, &binder_l2);
```

## Core Types

```rust
struct BlockTransform { scale: usize, block_size: usize }
struct RGFlow { parameters: Vec<f64>, scale: usize }
struct FixedPoint { location: Vec<f64>, eigenvalues: Vec<f64>, stable: Vec<bool> }
struct CriticalExponent { name: String, value: f64 }
struct UniversalityClass { name: String, exponents: Vec<CriticalExponent> }
```

## Linear Algebra

All matrix operations (Gaussian elimination, Jacobi eigenvalue algorithm, numerical Jacobians) are implemented from scratch with **no external math dependencies**. Only `serde` is required for serialization.

## Installation

```toml
[dependencies]
renormalization-group = "0.1"
```

## Testing

61 tests covering:
- Block transformation reduces agent count by expected factor
- Flow converges to known fixed points for simple systems
- Relevant operators grow under RG flow
- Irrelevant operators shrink under RG flow
- Uniform fleet → trivial fixed point
- 1D nearest-neighbor fleet → solvable (like 1D Ising)
- Critical exponents match analytical predictions for known models
- Scaling relations (Rushbrooke, Widom, Fisher, Josephson)
- Universality: two different microscopic models with same fixed point
- Conservation laws preserved under coarse-graining
- Scaling collapse for finite-size data
- Binder cumulant crossing for Tc extraction

```bash
cargo test
```

## License

MIT
