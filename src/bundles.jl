module Bundles

export TrajectoryBundle
export evolve!

using LinearAlgebra
using OrdinaryDiffEq
using DiffEqGPU
using CUDA
using NamedTrajectories
using TrajectoryIndexingUtils
using Distributions

using ..Utils

mutable struct TrajectoryBundle
    Z̄::NamedTrajectory
    N::Int
    M::Int
    f!::Function
    r_term::Function
    rs::Vector{Function}
    cs::Vector{Function}
    Wxs::Vector{Matrix{Float64}}
    Wus::Vector{Matrix{Float64}}
    Wfs::Vector{Matrix{Float64}}
    Wrs::Vector{Matrix{Float64}}
    Wcs::Vector{Matrix{Float64}}

    function TrajectoryBundle(
        Z̄::NamedTrajectory,
        M::Int,
        f!::Function,
        r_term::Function,
        rs::Vector{Function},
        cs::Vector{Function}
    )
        N = Z̄.T

        @assert length(rs) == N - 1
        @assert length(cs) == N

        x_dim = Z̄.dims.x
        u_dim = Z̄.dims.u

        r_dims = [
            [length(rs[k](zeros(x_dim), zeros(u_dim))) for k = 1:N - 1];
            length(r_term(zeros(x_dim)))
        ]

        c_dims = [length(cs[k](zeros(x_dim), zeros(u_dim))) for k = 1:N]

        Wxs = [zeros(x_dim, M) for k = 1:N]
        Wus = [zeros(u_dim, M) for k = 1:N]
        Wfs = [zeros(x_dim, M) for k = 1:N-1]
        Wrs = [zeros(r_dims[k], M) for k = 1:N]
        Wcs = [zeros(c_dims[k], M) for k = 1:N]

        new(
            copy(Z̄),
            N,
            M,
            f!,
            r_term,
            rs,
            cs,
            Wxs,
            Wus,
            Wfs,
            Wrs,
            Wcs,
        )
    end
end

function update_bundle_matrices!(
    bundle::TrajectoryBundle
)
    N = bundle.N
    M = bundle.M

    for k = 1:N-1
        for j = 1:M
            xₖⱼ = @view bundle.Wxs[k][:, j]
            uₖⱼ = @view bundle.Wus[k][:, j]

            bundle.Wrs[k][:, j] = bundle.rs[k](xₖⱼ, uₖⱼ)
            bundle.Wcs[k][:, j] = bundle.cs[k](xₖⱼ, uₖⱼ)
        end
    end

    bundle.Wxs[end] .= bundle.Wfs[end]

    bundle.Wrs[end] .= mapslices(bundle.r_term, bundle.Wxs[end], dims = 1)

    bundle.Wcs[end] .= stack(
        bundle.cs[end](xⱼ, uⱼ)
            for (xⱼ, uⱼ) ∈ zip(eachcol(bundle.Wxs[end]), eachcol(bundle.Wus[end]))
    )

    return nothing
end

function evolve!(
    bundle::TrajectoryBundle;
    normalize_states = false,
    integrator = Tsit5(),
    gpu = false,
    gpu_backend = CUDA.CUDABackend(),
    dist = Normal(0.0, 1.0)
)
    N = bundle.N
    M = bundle.M

    # NOTE: can't change tspan with EnsembleGPUArray

    for k = 1:N-1
        # ode problem to sample over
        prob = ODEProblem(
            bundle.f!,
            bundle.Z̄[k].x,
            (k - 1, k) .* bundle.Z̄.timestep,
            bundle.Z̄[k].u
        )

        # ode problem sampler
        prob_func = (prob, j, repeat) -> begin
            x = get_sample(bundle.Z̄[k].x; dist = dist, normalize = normalize_states)
            u = get_sample(bundle.Z̄[k].u; dist = dist)
            return remake(prob, u0 = x, p = u)
        end

        # ensemble problem
        prob_ensemble = EnsembleProblem(prob; prob_func = prob_func, safetycopy = false)

        # TODO: take another look at EnsembleGPUKernel (will require in-place f)
        sol_ensemble = solve(
            prob_ensemble,
            integrator,
            gpu ? EnsembleGPUArray(gpu_backend) : EnsembleThreads();
            trajectories = M,
        )

        bundle.Wxs[k] = stack(sol_ensemble[j][1] for j = 1:M)
        bundle.Wus[k] = stack(sol_ensemble[j].prob.p for j = 1:M)
        bundle.Wfs[k] = stack(sol_ensemble[j][end] for j = 1:M)
    end

    update_bundle_matrices!(bundle)

    return nothing
end

end
