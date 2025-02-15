# TrajectoryBundles

[![Build Status](https://github.com/aarontrowbridge/TrajectoryBundles.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aarontrowbridge/TrajectoryBundles.jl/actions/workflows/CI.yml?query=branch%3Amain)

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

# Define initial and goal states
x_init = [1.0, 0.0, 0.0, 0.0]
x_goal = [0.0, 1.0, 0.0, 0.0]

# Define control inputs
N = 50
u_initial = 2rand(2, N) .- 1

# Create a named trajectory
traj = NamedTrajectory((
    x = x_initial,
    u = u_initial 
    );
    controls = (:u,),
    timestep = 1.0,
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
G(a) = X * a[1] + Y * a[2]
function f(x, p, t)
    return G(p) * x
end

# Create and solve trajectory bundle problem
bundle = TrajectoryBundle(traj, 10, f, x -> 100.0 * (x - traj.goal.x), rs, cs)
prob = TrajectoryBundleProblem(bundle)
TrajectoryBundles.solve!(prob)

# Plot results
NamedTrajectories.plot(prob.bundle.Z̄)
```

For more detailed usage and examples, please refer to the documentation.
