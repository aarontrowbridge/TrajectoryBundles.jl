module Utils 

export get_sample
export get_samples

export linear_scheduler
export cosine_annealing
export exponential_decay

export build_kernel_function

using Symbolics
using Distributions


linear_scheduler(σ₀, σ_min, i, max_iter) = max(σ_min, σ₀ * (1.0 - i/max_iter))

function cosine_annealing(σ₀, σ_min, i, max_iter)
    return σ_min + 0.5 * (σ₀ - σ_min) * (1 + cos(i * π / max_iter))
end

exponential_decay(σ₀, σ_min, i, max_iter; γ=0.9, T=5) = σ₀ * γ^(i * max_iter / T)


function get_sample(
    z::AbstractVector;
    normalize = false,
    dist = Normal(0, 1.0)
)
    ξ = rand(dist, length(z))
    z_sampled = z + ξ
    if normalize
        normalize!(z_sampled)
    end
    return z_sampled
end

function get_samples(z::AbstractVector, n::Int; kwargs...)
    return stack(get_sample(z; kwargs...) for i = 1:n)
end


function build_kernel_function(f!::Function, x_dim::Int, u_dim::Int)
    x = collect(@variables(x[1:x_dim])...)
    u = collect(@variables(u[1:u_dim])...)
    t = @variables(t)[1]
    f = similar(x)
    f!(f, x, u, t)
    return eval(build_function(f, x, u, t)[2])
end

end