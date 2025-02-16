module TrajectoryBundles

export TrajectoryBundle
export TrajectoryBundleProblem
export evolve!
export step!
export rollout
export rollout!

export linear_scheduler
export cosine_annealing
export exponential_decay

using DiffEqGPU
using OrdinaryDiffEq
using CUDA
using LinearAlgebra
using NamedTrajectories
using TrajectoryIndexingUtils
using Convex
using Clarabel
import MathOptInterface as MOI

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

function get_bundle_matrices(
    ensemble::EnsembleSolution,
    bundle::TrajectoryBundle;
    σ = 0.1
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

    Wxs[end] = get_samples(bundle.Z̄[end].x, M; σ = σ, normalize = true)
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

    Wxs, Wus, Wrs, Wfs, Wcs = get_bundle_matrices(sol_ensemble, bundle; σ = σ)

    bundle.Wxs .= Wxs
    bundle.Wus .= Wus
    bundle.Wrs .= Wrs
    bundle.Wfs .= Wfs
    bundle.Wcs .= Wcs

    return nothing
end

function NamedTrajectories.update!(
    bundle::TrajectoryBundle,
    α::AbstractMatrix{Float64}
)
    X_new = similar(bundle.Z̄.x)
    X_new[:, 1] = bundle.Z̄.initial.x
    for k = 2:bundle.N
        X_new[:, k] = bundle.Wfs[k - 1] * α[:, k - 1]
    end

    U_new = similar(bundle.Z̄.u)
    U_new[:, 1] = bundle.Z̄.initial.u
    for k = 2:bundle.N-1
        U_new[:, k] = bundle.Wus[k] * α[:, k]
    end
    U_new[:, end] = bundle.Z̄.final.u

    update!(bundle.Z̄, :x, X_new)
    update!(bundle.Z̄, :u, U_new)

    return nothing
end

function step!(bundle::TrajectoryBundle;
    σ = 0.1,
    ρ = 1.0e5,
    silent_solve = false,
    slacks = true,
    normalize_states = true
)
    evolve!(bundle; σ = σ, normalize_states = normalize_states)

    α = Variable(bundle.M, bundle.N, Positive())

    if slacks
        s = Variable(bundle.Z̄.dims.x, bundle.N - 1)
        ws = [Variable(size(Wcₖ, 1)) for Wcₖ in bundle.Wcs]
    end

    obj = sum(sumsquares(bundle.Wrs[k] * α[:, k]) for k = 1:bundle.N-1) +
        sumsquares(bundle.Wrs[bundle.N] * α[:, bundle.N]) +
        (slacks ? ρ * (
            norm(s, 1) +
            sum(norm(w, 1) for w ∈ ws)
        ) : 0.0)

    constraints = Constraint[sum(α[:, k]) == 1 for k = 1:bundle.N]

    for k = 1:bundle.N-1
        push!(constraints,
            bundle.Wxs[k + 1] * α[:, k + 1] - bundle.Wfs[k] * α[:, k] ==
                (slacks ? s[:, k] : 0.0)
        )
        push!(constraints,
            bundle.Wcs[k] * α[:, k] >= (slacks ? ws[k] : 0.0)
        )
    end

    prob = minimize(obj, constraints)

    Convex.solve!(prob, Clarabel.Optimizer; silent=silent_solve)

    # display(MOI.TerminationStatusCode)

    if prob.status ∈ [
        MOI.TerminationStatusCode(1), # OPTIMAL
        MOI.TerminationStatusCode(7), # ALMOST_OPTIMAL
    ]
        update!(bundle, α.value)
    else
        println("    Subproblem optimization failed with status: $(prob.status)")
    end

    # if prob.status !=
    #     @info "Optimization failed with status: $(prob.status)" prob.status
    #     println("Optimization failed with status: $(prob.status)")
    #     return nothing
    # else

    return nothing
end

linear_scheduler(σ₀, σ_min, i, max_iter) = max(σ_min, σ₀ * (1.0 - i/max_iter))

function cosine_annealing(σ₀, σ_min, i, max_iter)
    return σ_min + 0.5 * (σ₀ - σ_min) * (1 + cos(i * π / max_iter))
end

exponential_decay(σ₀, σ_min, i, max_iter; γ=0.9, T=5) = σ₀ * γ^(i / T)

mutable struct TrajectoryBundleProblem
    bundle::TrajectoryBundle
    σ_scheduler::Function
    Js::Vector{Float64}

    function TrajectoryBundleProblem(
        bundle::TrajectoryBundle;
        σ_scheduler = linear_scheduler
    )
        new(bundle, σ_scheduler, Float64[])
    end
end

function eval_objective(bundle::TrajectoryBundle)
    return sum(
        norm(bundle.rs[k](bundle.Z̄[k].x, bundle.Z̄[k].u))^2
            for k = 1:bundle.N - 1
    ) + norm(bundle.r_term(bundle.Z̄[end].x))^2
end

# function eval_constraint_norm(bundle::TrajectoryBundle)
#     return sum(
#         norm(bundle.cs[k](bundle.Z̄[k].x, bundle.Z̄[k].u), 1) +
#         norm(bundle)
#             for k = 1:bundle.N
#     )
# end

function rollout(
    x_init::AbstractVector,
    u_traj::AbstractMatrix,
    f::Function,
    Δt::Float64,
    N::Int;
    alg = Tsit5(),
    return_full_solution = false
)
    f_full = (x, us, t) -> begin
        k = Int(t ÷ Δt) + 1
        uₖ = us[:, k]
        return f(x, uₖ, t)
    end

    prob_rollout = ODEProblem(
        f_full,
        x_init,
        (0.0, Δt * (N - 1)),
        u_traj
    )

    sol = solve(prob_rollout, alg, saveat = Δt)

    if return_full_solution
        return sol
    else
        return stack(sol.u)
    end
end

function rollout(bundle::TrajectoryBundle; kwargs...)
    return rollout(
        bundle.Z̄.initial.x,
        bundle.Z̄.u,
        bundle.f,
        bundle.Z̄.timestep,
        bundle.N;
        kwargs...
    )
end

function rollout!(bundle::TrajectoryBundle)
    bundle.Z̄.x = rollout(bundle)
    return nothing
end

function solve!(prob::TrajectoryBundleProblem;
    σ₀ = 0.1,
    σ_min = 0.0,
    ρ = 1.0e6,
    silent_solve = true,
    slack_tol = 1.0e-4,
    normalize_states = true,
    max_iter = 100,
    manifold_projection = true
)
    J⁰ = eval_objective(prob.bundle)
    prob.Js = Float64[J⁰]

    println("Iteration 0: J = $J⁰, σ = $σ₀")

    for i = 1:max_iter

        σ = prob.σ_scheduler(σ₀, σ_min, i, max_iter)

        step!(prob.bundle;
            σ = σ,
            ρ = ρ,
            silent_solve = silent_solve,
            slacks = i > 1 ? prob.Js[end] > slack_tol : true,
            normalize_states = normalize_states
        )

        if manifold_projection
            rollout!(prob.bundle)
        end

        Jⁱ = eval_objective(prob.bundle)

        push!(prob.Js, Jⁱ)

        println("Iteration $i: J = $Jⁱ, σ = $σ")
    end

    return nothing
end

end
