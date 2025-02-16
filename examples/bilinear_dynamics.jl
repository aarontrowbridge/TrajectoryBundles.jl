using Revise
using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using CairoMakie
using NamedTrajectories
using TrajectoryBundles

# construct dynamical generators: elements of real isomorphism of the Lie algebra 𝔰𝔲(2)

Gx = sparse([
     0  0 0 1;
     0  0 1 0;
     0 -1 0 0;
    -1  0 0 0
])

Gy = sparse([
    0 -1 0  0;
    1  0 0  0;
    0  0 0 -1;
    0  0 1  0
])

Gz = sparse([
     0 0 1  0;
     0 0 0 -1;
    -1 0 0  0;
     0 1 0  0
])

# drift and control generators
G_drift = Gz
G_drives = [Gx, Gy]

# drift frequency
ω = 1.0e0 * 2π

# carrier waves
carrier = t -> [cos(ω * t), sin(ω * t)]

# bilinear dynamics equation: ẋ = (G₀ + ∑(uᵢ * carrier(t) * Gᵢ)) x
function f(x, u, t)
    G = ω * Gz + sum((u .* carrier(t)) .* G_drives)
    return G * x
end

# initial and goal states
x_init = [1.0, 0.0, 0.0, 0.0]
x_goal = [0.0, 1.0, 0.0, 0.0]

# number of time steps
N = 100

# number of bundle samples at each knot point
M = 2 * (4 + 2) + 1
# M = 8

# time step
Δt = 0.05

# control bounds
u_bound = 1.0

# initial control sequence
u_initial = u_bound * (2rand(2, N) .- 1)

# initial control trajectory, via rollout with initial control sequence
x_initial = rollout(x_init, u_initial, f, Δt, N)

# construct initial trajectory
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

# plot initial trajectory
NamdTrajectories.plot(traj)
# NamedTrajectories.plot(joinpath(pwd(), "TrajectoryBundles.jl/examples/plots/initial.png"), traj)

# goal loss weight
Q = 1.0e3

# control regularization weight
R = 1.0e-2

# terminal loss residual
r_loss = x -> √Q * (x - traj.goal.x)

# control regularization residual
r_reg = (x, u) -> √R * u

# control bounds residual
c_bound = (x, u) -> [u - traj.bounds.u[1]; traj.bounds.u[2] - u]

# initial and final state constraints
c_initial = (x, u) -> [
    x - traj.initial.x;
    traj.initial.x - x;
    u - traj.initial.u;
    traj.initial.u - u;
]

c_final = (x, u) -> [
    u - traj.final.u;
    traj.final.u - u
]

# assemble costs and constraints
rs = Function[fill(r_reg, N-1)...]
cs = Function[c_initial; fill(c_bound, N-2); c_final]

bundle = TrajectoryBundle(traj, M, f, r_term, rs, cs)

# construct bundle problem
prob = TrajectoryBundleProblem(bundle;
    # σ_scheduler = (args...) -> exponential_decay(args...; γ=0.9)
    σ_scheduler = cosine_annealing
    # σ_scheduler = linear_scheduler
)

# solve bundle problem
TrajectoryBundles.solve!(prob;
    max_iter = 200,
    σ₀ = 1.0,
    ρ = 1.0e6,
    slack_tol = 1.0e0,
    silent_solve = true,
    normalize_states = false,
    manifold_projection = false
)

# plot bundle solution
NamedTrajectories.plot(prob.bundle.Z̄)

# save
# NamedTrajectories.plot(joinpath(pwd(), "TrajectoryBundles.jl/examples/plots/final.png"), prob.bundle.Z̄)

# plot loss
lines(log.(prob.Js[2:end]))
