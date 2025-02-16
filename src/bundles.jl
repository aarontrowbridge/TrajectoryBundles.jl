module Bundles

export TrajectoryBundle
export evolve!

using LinearAlgebra
using OrdinaryDiffEq
using DiffEqGPU
using CUDA
using NamedTrajectories
using TrajectoryIndexingUtils

function get_sample(z::AbstractVector; σ = 0.1, normalize = false)
    z_sampled = z + σ * randn(length(z))
    if normalize
        normalize!(z_sampled)
    end
    return z_sampled
end

function get_samples(z::AbstractVector, n::Int; kwargs...)
    return hcat([get_sample(z; kwargs...) for i = 1:n]...)
end

mutable struct TrajectoryBundle
    Z̄::NamedTrajectory
    N::Int
    M::Int
    f::Function
    r_term::Function
    rs::Vector{Function}
    cs::Vector{Function}
    Wxs::Vector{Matrix{Float64}}
    Wus::Vector{Matrix{Float64}}
    Wrs::Vector{Matrix{Float64}}
    Wfs::Vector{Matrix{Float64}}
    Wcs::Vector{Matrix{Float64}}

    function TrajectoryBundle(
        Z̄::NamedTrajectory,
        M::Int,
        f::Function,
        r_term::Function,
        rs::Vector{Function},
        cs::Vector{Function}
    )
        N = Z̄.T

        Wxs = Vector{Matrix{Float64}}(undef, N)
        Wus = Vector{Matrix{Float64}}(undef, N)
        Wrs = Vector{Matrix{Float64}}(undef, N)
        Wfs = Vector{Matrix{Float64}}(undef, N - 1)
        Wcs = Vector{Matrix{Float64}}(undef, N)

        new(
            copy(Z̄),
            N,
            M,
            f,
            r_term,
            rs,
            cs,
            Wxs,
            Wus,
            Wrs,
            Wfs,
            Wcs,
        )
    end
end

function get_bundle_matrices(
    ensemble::EnsembleSolution,
    bundle::TrajectoryBundle;
    σ = 0.1,
    normalize_states = true,
)
    N = bundle.N
    M = bundle.M
    r_term = bundle.r_term
    rs = bundle.rs
    cs = bundle.cs

    @assert length(rs) == N - 1
    @assert length(cs) == N

    Wxs = Vector{Matrix{Float64}}(undef, N)
    Wus = Vector{Matrix{Float64}}(undef, N)
    Wfs = Vector{Matrix{Float64}}(undef, N - 1)
    Wrs = Vector{Matrix{Float64}}(undef, N)
    Wcs = Vector{Matrix{Float64}}(undef, N)

    for k = 1:N-1
        Wxs[k] = hcat([ensemble[index(k, j, M)].u[1] for j = 1:M]...)
        Wus[k] = hcat([ensemble[index(k, j, M)].prob.p for j = 1:M]...)
        Wfs[k] = hcat([ensemble[index(k, j, M)].u[end] for j = 1:M]...)
    end

    Wxs[end] = get_samples(bundle.Z̄[end].x, M; σ = σ, normalize = normalize_states)
    Wus[end] = get_samples(bundle.Z̄[end].u, M; σ = σ)

    for k = 1:N-1
        Wrₖs = Vector{Float64}[]
        for j = 1:M
            push!(Wrₖs, rs[k](Wxs[k][:, j], Wus[k][:, j]))
        end
        Wrs[k] = hcat(Wrₖs...)
    end

    Wrs[end] = mapslices(r_term, Wxs[end], dims = 1)

    for k = 1:N
        Wcₖs = Vector{Float64}[]
        for j = 1:M
            push!(Wcₖs, cs[k](Wxs[k][:, j], Wus[k][:, j]))
        end
        Wcs[k] = hcat(Wcₖs...)
    end

    return Wxs, Wus, Wrs, Wfs, Wcs
end

function evolve!(
    bundle::TrajectoryBundle;
    normalize_states = true,
    σ = 0.1
)
    prob = ODEProblem(
        bundle.f,
        bundle.Z̄.initial.x,
        (0.0, bundle.Z̄.timestep),
        bundle.Z̄.initial.u
    )

    prob_func = (prob, j, repeat) -> begin
        k = (j - 1) ÷ bundle.M + 1
        x = get_sample(bundle.Z̄[k].x; σ = σ, normalize = normalize_states)
        u = get_sample(bundle.Z̄[k].u; σ = σ)
        remake(prob,
            u0 = x,
            p = u,
            tspan = ((k - 1) * bundle.Z̄.timestep, k * bundle.Z̄.timestep)
        )
    end

    prob_ensemble = EnsembleProblem(
        prob;
        prob_func = prob_func
    )

    sol_ensemble = solve(
        prob_ensemble,
        Tsit5(),
        EnsembleThreads(),
        trajectories = (bundle.N - 1) * bundle.M,
    )

    Wxs, Wus, Wrs, Wfs, Wcs = get_bundle_matrices(sol_ensemble, bundle;
        σ = σ,
        normalize_states = normalize_states
    )

    bundle.Wxs .= Wxs
    bundle.Wus .= Wus
    bundle.Wrs .= Wrs
    bundle.Wfs .= Wfs
    bundle.Wcs .= Wcs

    return nothing
end

end
