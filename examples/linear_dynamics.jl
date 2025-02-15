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
using PiccoloQuantumObjects
import PiccoloQuantumObjects as PQO

X = PQO.Isomorphisms.G(PAULIS.X)
Y = PQO.Isomorphisms.G(PAULIS.Y)
Z = PQO.Isomorphisms.G(PAULIS.Z)

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
Δt = 0.01

u_bound = 0.02

u_initial = u_bound * (2rand(2, N) .- 1)

x_initial = stack([prod(exp(Matrix(G(u) * Δt)) for u ∈ eachcol(u_initial[:, 1:k])) * x_init for k = 1:N])

traj = NamedTrajectory((
    # iterpolate between x_init and x_goal
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

rs = Function[(x, u) -> [1e-3 * u;] for k = 1:N-1]

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
)

TrajectoryBundles.solve!(prob;
    max_iter = 100,
    σ₀ = 0.1,
    ρ = 1.0e7,
    silent_solve = true
)

NamedTrajectories.plot(prob.bundle.Z̄)

prob.bundle.Wcs[31]



for k = 1:bundle.N
    println(cs[k](bundle.Z̄[k].x, bundle.Z̄[k].u))
end


for Wc ∈ prob.bundle.Wcs
    println(Wc)
end
