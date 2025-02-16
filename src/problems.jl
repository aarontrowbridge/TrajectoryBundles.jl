module Problems

export TrajectoryBundleProblem
export eval_objective
export step!
export solve!

export linear_scheduler
export cosine_annealing
export exponential_decay

using LinearAlgebra
using Convex
using Clarabel
import MathOptInterface as MOI
using NamedTrajectories

using ..Bundles
using ..Rollouts

linear_scheduler(σ₀, σ_min, i, max_iter) = max(σ_min, σ₀ * (1.0 - i/max_iter))

function cosine_annealing(σ₀, σ_min, i, max_iter)
    return σ_min + 0.5 * (σ₀ - σ_min) * (1 + cos(i * π / max_iter))
end

exponential_decay(σ₀, σ_min, i, max_iter; γ=0.9, T=5) = σ₀ * γ^(i * max_iter / T)

mutable struct TrajectoryBundleProblem
    bundle::TrajectoryBundle
    σ_scheduler::Function
    Js::Vector{Float64}
    best_traj::NamedTrajectory

    function TrajectoryBundleProblem(
        bundle::TrajectoryBundle;
        σ_scheduler::Function = linear_scheduler
    )
        new(bundle, σ_scheduler, Float64[], copy(bundle.Z̄))
    end

    function TrajectoryBundleProblem(
        Z_init::NamedTrajectory,
        f::Function,
        r_loss::Function,
        rs::Vector{Function},
        cs::Vector{Function};
        M::Int = 2 * Z_init.dim + 1,
        kwargs...
    )
        bundle = TrajectoryBundle(Z_init, M, f, r_loss, rs, cs)
        TrajectoryBundleProblem(bundle; kwargs...)
    end
end

function eval_objective(traj::NamedTrajectory, r_term::Function, rs::Vector{Function})
    J = sum(
        norm(rs[k](traj[k].x, traj[k].u))^2
            for k = 1:traj.T - 1
    )
    J += norm(r_term(traj[end].x))^2
    return J
end

eval_objective(bundle::TrajectoryBundle) =
    eval_objective(bundle.Z̄, bundle.r_term, bundle.rs)

# function eval_constraint_norm(bundle::TrajectoryBundle)
#     return sum(
#         norm(bundle.cs[k](bundle.Z̄[k].x, bundle.Z̄[k].u), 1) +
#         norm(bundle)
#             for k = 1:bundle.N
#     )
# end

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

function step!(prob::TrajectoryBundleProblem;
    σ = 0.1,
    ρ = 1.0e5,
    silent_solve = false,
    slacks = true,
    normalize_states = true
)
    evolve!(prob.bundle; σ = σ, normalize_states = normalize_states)

    α = Variable(prob.bundle.M, prob.bundle.N, Positive())

    if slacks
        s = Variable(prob.bundle.Z̄.dims.x, prob.bundle.N - 1)
        ws = [Variable(size(Wcₖ, 1)) for Wcₖ in prob.bundle.Wcs]
    end

    obj = sum(sumsquares(prob.bundle.Wrs[k] * α[:, k]) for k = 1:prob.bundle.N-1) +
        sumsquares(prob.bundle.Wrs[prob.bundle.N] * α[:, prob.bundle.N]) +
        (slacks ? ρ * (
            norm(s, 1) +
            sum(norm(w, 1) for w ∈ ws)
        ) : 0.0)

    constraints = Constraint[sum(α[:, k]) == 1 for k = 1:prob.bundle.N]

    for k = 1:prob.bundle.N-1
        push!(constraints,
            prob.bundle.Wxs[k + 1] * α[:, k + 1] - prob.bundle.Wfs[k] * α[:, k] ==
                (slacks ? s[:, k] : 0.0)
        )
        push!(constraints,
            prob.bundle.Wcs[k] * α[:, k] >= (slacks ? ws[k] : 0.0)
        )
    end

    subprob = minimize(obj, constraints)

    Convex.solve!(subprob, Clarabel.Optimizer; silent=silent_solve)

    # display(MOI.TerminationStatusCode)

    if subprob.status ∈ [
        MOI.TerminationStatusCode(1), # OPTIMAL
        MOI.TerminationStatusCode(7), # ALMOST_OPTIMAL
    ]
        update!(prob.bundle, α.value)
    else
        println("    Subproblem optimization failed with status: $(subprob.status)")
    end

    # if prob.status !=
    #     @info "Optimization failed with status: $(prob.status)" prob.status
    #     println("Optimization failed with status: $(prob.status)")
    #     return nothing
    # else

    return nothing
end

function solve!(prob::TrajectoryBundleProblem;
    max_iter = 100,
    σ₀ = 1.0,
    σ_min = 0.0,
    ρ = 1.0e6,
    slack_tol = 1.0e-1,
    normalize_states = false,
    feasibility_projection = false,
    silent_solve = true,
)
    J⁰ = eval_objective(prob.bundle)
    prob.Js = Float64[J⁰]

    println("Iteration 0: J = $J⁰, σ = $σ₀")

    for i = 1:max_iter

        σ = prob.σ_scheduler(σ₀, σ_min, i, max_iter)

        step!(prob;
            σ = σ,
            ρ = ρ,
            silent_solve = silent_solve,
            slacks = i > 1 ? prob.Js[end] > slack_tol : true,
            normalize_states = normalize_states
        )

        if feasibility_projection
            rollout!(prob.bundle)
        end

        Jⁱ = eval_objective(prob.bundle)

        if Jⁱ < minimum(prob.Js)
            prob.best_traj = copy(prob.bundle.Z̄)
        end

        push!(prob.Js, Jⁱ)

        println("Iteration $i: J = $Jⁱ, σ = $σ")
    end

    prob.bundle.Z̄ = copy(prob.best_traj)

    return nothing
end




end
