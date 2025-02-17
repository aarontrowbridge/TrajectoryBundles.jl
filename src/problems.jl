module Problems

export TrajectoryBundleProblem
export eval_objective
export step!
export solve!

using LinearAlgebra
using Convex
using Clarabel
using OrdinaryDiffEq
using CUDA
import MathOptInterface as MOI
using NamedTrajectories
using Distributions

using ..Utils
using ..Bundles
using ..Rollouts

mutable struct TrajectoryBundleProblem
    bundle::TrajectoryBundle
    σ_scheduler::Function
    Js::Vector{Float64}
    Cs::Vector{Float64}
    best_traj::NamedTrajectory

    function TrajectoryBundleProblem(
        bundle::TrajectoryBundle;
        σ_scheduler::Function = linear_scheduler
    )
        new(bundle, σ_scheduler, Float64[], Float64[], copy(bundle.Z̄))
    end

    function TrajectoryBundleProblem(
        Z_init::NamedTrajectory,
        f!::Function,
        r_loss::Function,
        rs::Vector{Function},
        cs::Vector{Function};
        M::Int = 2 * Z_init.dim + 1,
        kwargs...
    )
        bundle = TrajectoryBundle(Z_init, M, f!, r_loss, rs, cs)
        return TrajectoryBundleProblem(bundle; kwargs...)
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
    ρ = 1.0e5,
    silent_solve = false,
    dist = Normal(0.0, 1.0),
    slacks = true,
    normalize_states = false,
    integrator = Tsit5(),
    gpu = false,
    gpu_backend = CUDA.CUDABackend(),
)
    evolve!(prob.bundle;
        dist = dist,
        normalize_states = normalize_states,
        integrator = integrator,
        gpu = gpu,
        gpu_backend = gpu_backend
    )

    α = Variable(prob.bundle.M, prob.bundle.N, Positive())

    base_obj = sum(sumsquares(prob.bundle.Wrs[k] * α[:, k]) for k = 1:prob.bundle.N)

    if slacks
        s = Variable(prob.bundle.Z̄.dims.x, prob.bundle.N - 1)
        ws = [Variable(size(Wcₖ, 1)) for Wcₖ ∈ prob.bundle.Wcs]
        slack_obj = ρ * (norm(s, 1) + sum(norm(w, 1) for w ∈ ws))
        obj = base_obj + slack_obj
    else
        obj = base_obj
    end

    constraints = Constraint[sum(α[:, k]) == 1 for k = 1:prob.bundle.N]

    for k = 1:prob.bundle.N-1
        fₖ_con = prob.bundle.Wxs[k + 1] * α[:, k + 1] ==
            prob.bundle.Wfs[k] * α[:, k] + (slacks ? s[:, k] : 0.0)

        push!(constraints, fₖ_con)
    end

    for k = 1:prob.bundle.N
        push!(constraints,
            prob.bundle.Wcs[k] * α[:, k] >= (slacks ? ws[k] : 0.0)
        )
    end

    subprob = minimize(obj, constraints)

    Convex.solve!(subprob, Clarabel.Optimizer; silent=silent_solve)

    if subprob.status ∈ [
        MOI.TerminationStatusCode(1), # OPTIMAL
        MOI.TerminationStatusCode(7), # ALMOST_OPTIMAL
    ]
        update!(prob.bundle, α.value)

        push!(prob.Js, evaluate(base_obj))

        if slacks
            push!(prob.Cs, evaluate(slack_obj))
        else
            push!(prob.Cs, 0.0)
        end

        return :solved
    else
        println("    subproblem solve failed with status: $(subprob.status)")
        return :failed
    end
end

function solve!(prob::TrajectoryBundleProblem;
    max_iter = 100,
    σ₀ = 1.0,
    σ_min = 0.0,
    ρ = 1.0e5,
    slack_tol = 1.0e-1,
    normalize_states = false,
    feasibility_projection = false,
    silent_solve = true,
    integrator = Tsit5(),
    gpu = false,
    gpu_backend = CUDA.CUDABackend()
)
    println("-------------------------------------------------------------")
    println("| iteration |    J value    |    C value    |    σ value    |")
    println("-------------------------------------------------------------")

    J⁰ = eval_objective(prob.bundle)
    C⁰ = Inf

    push!(prob.Js, J⁰)
    push!(prob.Cs, C⁰)

    println("| $(rpad(0, 9)) | $(rpad(round(J⁰, digits=6), 12)) | $(rpad(round(C⁰, digits=6), 12)) | $(rpad(round(σ₀, digits=6), 12)) |")

    status = nothing

    for i = 1:max_iter

        σ = prob.σ_scheduler(σ₀, σ_min, i, max_iter)

        status = step!(prob;
            ρ = ρ,
            dist = Normal(0.0, σ),
            silent_solve = silent_solve,
            slacks = prob.Cs[end] > slack_tol || status == :failed,
            normalize_states = normalize_states,
            integrator = integrator,
            gpu = gpu,
            gpu_backend = gpu_backend
        )

        if status == :failed
            continue
        end

        if feasibility_projection
            rollout!(prob.bundle)
        end

        Jⁱ = prob.Js[end]
        Cⁱ = prob.Cs[end]

        if Cⁱ < slack_tol && Jⁱ < minimum(prob.Js)
            prob.best_traj = copy(prob.bundle.Z̄)
        end

        println("| $(rpad(i, 9)) | $(rpad(round(Jⁱ, digits=6), 12)) | $(rpad(round(Cⁱ, digits=6), 12)) | $(rpad(round(σ, digits=6), 12)) |")
    end

    println("---------------------------------------------")

    # prob.bundle.Z̄ = copy(prob.best_traj)

    return nothing
end

end
