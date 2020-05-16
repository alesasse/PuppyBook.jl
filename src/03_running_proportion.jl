using StatsBase
using StatsPlots
using Random

"""
    compute_running_proportion(p_heads::Float64, N::Int)

Computes the running proportion of heads when 0 < p_heads < 1 is the probability of getting an head
and N is the number of trials.
"""
function compute_running_proportion(p_heads::Float64, N::Int)
    @assert p_heads >= 0 && p_heads <= 1
    flip_sequence = sample(0:1,pweights([1-p_heads,p_heads]), N)
    r = cumsum(flip_sequence)
    n = 1:N
    run_prop = r ./ n
    return run_prop
end

"""
    plot_running_proportion(p_heads::Float64, N::Int) 

Reproduces figure 4.1 with custom probability of heads and number of flips.
"""
function plot_running_proportion(p_heads::Float64, N::Int)
    run_prop = compute_running_proportion(p_heads, N)
    n = 1:N
    plot(n,run_prop,
        title="Running Proportion of Heads",
        legend=false,
        marker = :dot,
        xscale = :log10)
    plot!(ylims=(0,1))
    # Add annotation with end proportion
    annotation = "End Proportion = " * string(run_prop[end])
    annotate!(N/4,0.7,annotation)
    # Add reference line (true probability)
    plot!(n, fill(p_heads, N), linestyle = :dot)
    xlabel!("Flip Number")
    ylabel!("Proportion Heads")
end