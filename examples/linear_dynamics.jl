using Revise
using NamedTrajectories
using TrajectoryIndexingUtils
using TrajectoryBundles
using OrdinaryDiffEq
using LinearAlgebra
using SparseArrays
using CairoMakie
using Plots
using Convex
using Clarabel
import PiccoloQuantumObjects as PQO

X = PQO.Isomorphisms.G(PQO.PAULIS.X)
Y = PQO.Isomorphisms.G(PQO.PAULIS.Y)
Z = PQO.Isomorphisms.G(PQO.PAULIS.Z)

X = sparse(X)
Y = sparse(Y)
Z = sparse(Z)

ω = 1.0e1

G(u) = ω * Z + X * u[1] + Y * u[2]

function f(x, u, t)
    return G(u) * cos(ω * t) * x
end

x_init = [1.0, 0.0, 0.0, 0.0]
x_goal = [0.0, 1.0, 0.0, 0.0]

N = 100
M = 2 * (4 + 2) + 1
# M = 8
Δt = 0.05

u_bound = 1.0

u_initial = u_bound * (2rand(2, N) .- 1)

x_initial = rollout(x_init, u_initial, f, Δt, N)

traj = NamedTrajectory((
        x = x_initial,
        u = u_initial
    );
    controls = (:u,),
    timestep = Δt,
    bounds = (
        u = u_bound,
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

NamedTrajectories.plot(traj)

r_term = x -> 100.0 * (x - traj.goal.x)

rs = Function[(x, u) -> [2e-1 * u;] for k = 1:N-1]

cs = Function[(x, u) -> [u - traj.bounds.u[1]; traj.bounds.u[2] - u] for k = 1:N]

cs[1] = (x, u) -> [
    x - traj.initial.x;
    traj.initial.x - x;
    u - traj.initial.u;
    traj.initial.u - u;
]

cs[end] = (x, u) -> [u - traj.final.u; traj.final.u - u]

bundle = TrajectoryBundle(traj, M, f, r_term, rs, cs)

prob = TrajectoryBundleProblem(bundle;
    # σ_scheduler = (args...) -> exponential_decay(args...; γ=0.9)
    σ_scheduler = cosine_annealing
    # σ_scheduler = linear_scheduler
)

TrajectoryBundles.solve!(prob;
    max_iter = 200,
    σ₀ = 0.1,
    ρ = 1.0e6,
    slack_tol = 1.0e1,
    silent_solve = true,
    normalize_states = false,
    manifold_projection = false
)

NamedTrajectories.plot(prob.bundle.Z̄)

lines(log.(prob.Js[2:end]))

rollout(bundle)[:, end]

bundle.Z̄.x[:, end]

rollout!(bundle)
