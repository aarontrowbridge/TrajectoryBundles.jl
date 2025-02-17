begin
    using Revise
    using LinearAlgebra
    using SparseArrays
    using OrdinaryDiffEq
    using DiffEqGPU
    using CUDA
    using CairoMakie
    using NamedTrajectories
    using Symbolics
    using Distributions
    using TrajectoryBundles
end

# dynamical generators: elements of real isomorphism of the Lie algebra 𝔰𝔲(2)
begin
    Gx = sparse(Float64[
        0  0 0 1;
        0  0 1 0;
        0 -1 0 0;
        -1  0 0 0
    ])

    Gy = sparse(Float64[
        0 -1 0  0;
        1  0 0  0;
        0  0 0 -1;
        0  0 1  0
    ])

    Gz = sparse(Float64[
        0 0 1  0;
        0 0 0 -1;
        -1 0 0  0;
        0 1 0  0
    ])


    x_init = [1.0, 0.0, 0.0, 0.0]
    x_goal = [0.0, 1.0, 0.0, 0.0]

    # drift and drive generators
    G_drift = Gz
    G_drives = [Gx, Gy]
end

# set up dynamics
begin
    # drift frequency
    ω = 1.0e0 * 2π

    # carrier waves
    carrier = t -> [cos(ω * t), sin(ω * t)]

    # generator for bilinear dynamics: ẋ = G(u(t), t) x
    G(u, t) = ω * Gz + sum((u .* carrier(t)) .* G_drives)

    # GPU compatible kernel function
    f! = build_kernel_function((dx, x, u, t) -> mul!(dx, G(u, t), x), 4, 2)
end

# initial trajectory
begin
    # number of time steps
    N = 100

    # time step
    Δt = 0.05

    # control bounds
    u_bound = 1.0

    # initial control sequence
    u_initial = u_bound * (2rand(2, N) .- 1)

    # initial control trajectory, via rollout with initial control sequence
    x_initial = rollout(x_init, u_initial, f!, Δt, N)

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
    NamedTrajectories.plot(traj)
    # NamedTrajectories.plot(joinpath(pwd(), "TrajectoryBundles.jl/examples/plots/initial.png"), traj)
end

# objective and constraints
begin
    # goal loss weight
    Q = 1.0e3

    # control regularization weight
    R = 1.0e-0

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
end

prob = TrajectoryBundleProblem(traj, f!, r_loss, rs, cs;
    # M = 8,
    σ_scheduler = cosine_annealing
    # σ_scheduler = linear_scheduler
);


TrajectoryBundles.solve!(prob;
    max_iter = 200,
    σ₀ = 1.0e-1,
    σ_min = 1e-2,
    ρ = 1.0e5,
    slack_tol = 1.0e-1,
    silent_solve = true,
    normalize_states = false,
    feasibility_projection = false,
    gpu = true 
)

# plot bundle solution
NamedTrajectories.plot(prob.bundle.Z̄)

eval_objective(prob.bundle)

rollout!(prob.bundle)

eval_objective(prob.bundle)

# save
NamedTrajectories.plot(joinpath(pwd(), "TrajectoryBundles.jl/examples/plots/final.png"), prob.bundle.Z̄)

# plot loss

begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "Loss over Iterations", xlabel = "Iteration", ylabel = "Log Loss", yscale = log10)
    lines!(ax, 2:length(prob.Js), log.(prob.Js[2:end]), color = :blue, linewidth = 2)
    fig
end

# save log loss plot
# save(joinpath(pwd(), "TrajectoryBundles.jl/examples/plots/loss.png"), fig)
