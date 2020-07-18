using StatsPlots


"""
    coinflip_discrete_bayes_update(p_θ, data)

Given a discrete (possibly unnormalized) prior and some binary data
returns the likelihood and posterior.
"""
function coinflip_discrete_bayes_update(p_θ, data; plot_result=true)
    n_vals = length(p_θ) # number of candidates for θ
    θ = collect(0 : 1/(n_vals -1) : 1) # equally spaced bins
    p_θ = p_θ / sum(p_θ)     # Makes sure that beliefs sum to 1.
    heads = sum(data)
    tails = length(data) - heads
    likelihood = θ.^heads .* (1 .- θ).^tails
    p_data = sum(likelihood .* p_θ)
    posterior = (likelihood .* p_θ) ./ p_data
    return (likelihood = likelihood, posterior = posterior)
end

"""
    plot_coinflip_discrete_bayes_update(p_θ, data)

Reproduces figure 5.1 with custom prior and data.
To reproduce the figure exactly use:
```julia-repl
julia> θ = range(0.1,1, step=0.1)
julia> p_θ = min.(θ, 1 .- θ)
julia> plot_coinflip_discrete_bayes_update(p_θ, [1])
```
"""
function plot_coinflip_discrete_bayes_update(p_θ, data)
    n_vals = length(p_θ)
    θ = collect(0 : 1/(n_vals -1) : 1)
    p_θ = p_θ / sum(p_θ)
    likelihood, posterior = coinflip_discrete_bayes_update(p_θ, data)
    l = @layout (3,1)
    # Plot the prior
    plot1 = bar(θ, p_θ, title="Prior", leg=false, bar_width = 0.01)

    # Plot likelihood
    plot2 = bar(θ, likelihood, title="Likelihood", leg=false, bar_width = 0.01)

    # Plot the posterior
    plot3 = bar(θ, posterior, title="Posterior", leg=false, bar_width = 0.01)
    plt = plot(plot1, plot2, plot3, layout=l, xlims=(-0.02,1.02), xticks=round.(θ,digits=2))
    display(plt)
end

