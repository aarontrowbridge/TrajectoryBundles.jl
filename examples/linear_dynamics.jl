using Revise
using NamedTrajectories
using TrajectoryIndexingUtils
using TrajectoryBundles
using OrdinaryDiffEq
using DiffEqGPU
using LinearAlgebra
using StaticArrays
using CairoMakie
using Plots
using CUDA
using Convex
using Clarabel
using PiccoloQuantumObjects
using SparseArrays
import PiccoloQuantumObjects as PQO

X = PQO.Isomorphisms.G(PAULIS.X)
Y = PQO.Isomorphisms.G(PAULIS.Y)
Z = PQO.Isomorphisms.G(PAULIS.Z)

X = sparse(X)
Y = sparse(Y)
Z = sparse(Z)

x_init = [1.0, 0.0, 0.0, 0.0]
x_goal = [0.0, 1.0, 0.0, 0.0]

N = 50
M = 10
Δt = 1.0

u_initial = 2rand(2, N) .- 1

x_initial = stack([prod(exp(Matrix(G(u) * Δt)) for u ∈ eachcol(u_initial[:, 1:k])) * x_init for k = 1:N])

traj = NamedTrajectory((
    # iterpolate between x_init and x_goal
        x = x_initial,
        u = u_initial 
    );
    controls = (:u,),
    timestep = Δt,
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

NamedTrajectories.plot(traj)

G(a) = X * a[1] + Y * a[2]

function f(x, p, t)
    return G(p) * x
end

traj.u


r_term = x -> 100.0 * (x - traj.goal.x)

rs = Function[(x, u) -> [1e-2 * u;] for k = 1:N-1]

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
    σ_scheduler = linear_scheduler
)

TrajectoryBundles.solve!(prob;
    max_iter = 10,
    σ₀ = 0.0001,
    ρ = 1.0e7,
    silent_solve = false
)

NamedTrajectories.plot(prob.bundle.Z̄)

prob.bundle.Z̄[1].x

NamedTrajectories.plot(traj)
