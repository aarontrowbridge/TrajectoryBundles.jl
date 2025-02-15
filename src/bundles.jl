module Bundles 

using DiffEqGPU
using OrdinaryDiffEq
using CUDA
using LinearAlgebra
using NamedTrajectories 

mutable struct TrajectoryBundle
    Z̄::NamedTrajectory
    M::Int
    f::Function
end

function step!(bundle::TrajectoryBundle)

end

end