# TrajectoryBundles

[![Build Status](https://github.com/aarontrowbridge/TrajectoryBundles.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aarontrowbridge/TrajectoryBundles.jl/actions/workflows/CI.yml?query=branch%4Amain)

## Description

TrajectoryBundles is a Julia package designed for creating, manipulating, and optimizing trajectory bundles. It is particularly useful for control and optimization problems where multiple trajectories need to be considered simultaneously. The package provides tools for defining trajectories, setting up optimization problems, and solving them efficiently.

### Features

- **Trajectory Definition**: Easily define trajectories with initial and goal states, control inputs, and constraints.
- **Optimization**: Set up and solve trajectory optimization problems using various solvers and scheduling strategies.
- **Visualization**: Plot and visualize trajectories and optimization results.
- **Integration**: Compatible with other Julia packages such as `NamedTrajectories`, `OrdinaryDiffEq`, and `CairoMakie` for enhanced functionality.

## Installation

To install TrajectoryBundles, use the following command in the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/aarontrowbridge/TrajectoryBundles.jl")
```

## Usage

Here is a basic example of how to use TrajectoryBundles:

```julia
using TrajectoryBundles
using NamedTrajectories
using OrdinaryDiffEq
using LinearAlgebra 
import PiccoloQuantumObjects as PQO

# Define Pauli matrices
X = PQO.Isomorphisms.G(PAULIS.X)
Y = PQO.Isomorphisms.G(PAULIS.Y)
Z = PQO.Isomorphisms.G(PAULIS.Z)

# Define initial and goal states
x_init = [1.0, 0.0, 0.0, 0.0]
x_goal = [0.0, 1.0, 0.0, 0.0]

# Define control inputs
N = 50  # Number of time steps
M = 10  # Number of bundle samples per knot point 
Δt = 1.0  # Time step duration

# Define random controls
u_initial = 2rand(2, N) .- 1

# Build initial trajectory
x_initial = stack([prod(exp(Matrix(G(u) * Δt)) for u ∈ eachcol(u_initial[:, 1:k])) * x_init for k = 1:N])

# Create a named trajectory
traj = NamedTrajectory((
    x = x_initial,
    u = u_initial 
    );
    controls = (:u,),
    timestep = Δt,
    bounds = (
        u = ([-1.0, -1.0], [1.0, 1.0]),
    ),
    initial = (
        x = x_init,
        u = zeros(2)
    ),
    final = (
        u = zeros(2),
    ),
    goal = (
        x = x_goal,
    )
)

# Define dynamics function
G(u) = X * u[1] + Y * u[2]

function f(x, u, t)
    return G(u) * x
end

# Define reward term
r_term = x -> 100.0 * (x - traj.goal.x)

# Define running costs
rs = Function[(x, u) -> [1e-3 * u;] for k = 1:N-1]

# Define constraints
cs = Function[(x, u) -> [u - traj.bounds.u[1]; traj.bounds.u[2] - u] for k = 1:N]

# Initial constraints
cs[1] = (x, u) -> [
    x - traj.initial.x;
    traj.initial.x - x;
    u - traj.initial.u;
    traj.initial.u - u;
]

# Final constraints
cs[end] = (x, u) -> [u - traj.final.u; traj.final.u - u]

# Create and solve trajectory bundle problem
bundle = TrajectoryBundle(traj, M, f, r_term, rs, cs)
prob = TrajectoryBundleProblem(bundle)

TrajectoryBundles.solve!(prob; max_iter=100)

# Plot results
NamedTrajectories.plot(prob.bundle.Z̄)
```

For more detailed usage and examples, please refer to the documentation.

