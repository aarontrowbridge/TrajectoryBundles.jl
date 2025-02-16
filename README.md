# TrajectoryBundles.jl

[![Build Status](https://github.com/aarontrowbridge/TrajectoryBundles.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aarontrowbridge/TrajectoryBundles.jl/actions/workflows/CI.yml?query=branch%4Amain)

## Description

The *trajectory bundle method* is a gradient-free, parallelizable optimization algorithm for solving trajectory optimization problems. 
<!-- TrajectoryBundles.jl is a Julia package that provides a high-level interface for defining, solving, and visualizing trajectory optimization problems using the trajectory bundle method. -->

**TrajectoryBundles.jl** uses
 - [NamedTrajectories.jl](https://github.com/kestrelquantum/NamedTrajectories.jl) to define, manipulate, and plot trajectories
 - [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), [DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl), and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) to solve the underlying ODEs
 - [Convex.jl](https://github.com/jump-dev/Convex.jl) and [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) for solving the underlying quadratic program

## Installation

To install TrajectoryBundles.jl, enter package mode in the Julia REPL:

```
$ julia --project
```

and run the following command:

```julia
julia> ]
pkg> add https://github.com/aarontrowbridge/TrajectoryBundles.jl.git 
```

## Usage

:construction: Interface is changing rapidly :construction:

See the example script [/examples/linear_dynamis.jl](./examples/linear_dynamics.jl) for a the most up-to-date usage.
