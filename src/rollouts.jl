module Rollouts

export rollout
export rollout!

using OrdinaryDiffEq

using ..Bundles

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


end
