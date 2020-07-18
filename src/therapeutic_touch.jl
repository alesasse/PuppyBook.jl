using CSV, DataFrames
using Distributions
using Turing, MCMCChains
using StatsPlots


"""
    load_therapeutic_touch_data(;plot = false)

Loads the TherapeuticTouchData into a DataFrame. If plot is true, it also
displays the histogram of the observed accuracy of practitioners (figure 9.9).
"""
function load_therapeutic_touch_data(;plot = false)
    df = CSV.read("./data/TherapeuticTouchData.csv")
    if plot
        freqtable = combine(groupby(df, :s), :y => mean) # compute the accuracy of each practitioner
        fig = histogram(freqtable[:y_mean], nbins = 8, xlabel = "Observed accuracy", ylabel = "# Practitioners", legend = false)
        display(fig)
    end
    f = x -> parse(Int,x[2:end]) # parse the string col 
    df[:subject_n] = f.(df[:s])
    return df
end


"""
    tt_sample_basic_model(;iterations = 1000, nchains = 3)

Estimates the accuracy separately for each subject of the TherapeuticTouchData with ``\\beta(1,1)`` prior. Returns an MCMCChain.
"""
function tt_sample_basic_model(;iterations = 1000, nchains = 3)
    df = load_therapeutic_touch_data()
    # model definition
    @model tt_unpooled(obs, subj) = begin
        # Assumptions: one θ for each subject
        n_subjects = length(unique(subj))
        θ = Vector{Real}(undef, n_subjects)
        for i=1:n_subjects
            θ[i] ~ Beta(1,1) # unifor priors
        end
        # Observations
        for j=1:length(obs)
            obs[j] ~ Bernoulli(θ[subj[j]])
        end
    end
    obs, subj = (df[:y], df[:subject_n])
    ϵ = 0.05
    τ = 10
    chains = mapreduce(c -> sample(tt_unpooled(obs, subj),HMC(ϵ, τ), iterations), chainscat, 1:nchains)
    return chains
end


"""
    tt_sample_hierarchical_model(;iterations = 1000, nchains = 3)

Estimates the accuracy for each subject of the TherapeuticTouchData in a hierarchical model. 
This model is shown in the book in figure 9.13.
Returns an MCMCChain.
"""
function tt_sample_hierarchical_model(;iterations = 1000, nchains = 3)
    df = load_therapeutic_touch_data()
    # model definition
    @model therapeutic_touch(obs, subj) = begin
        # Assumptions: one θ for each subject
        n_subjects = length(unique(subj))
        ω ~ Beta(1,1)
        κ ~ Gamma(0.01, 100) # notice the different parametrization wrt PYMC3
        a = ω * κ + 1
        b = (1-ω) * κ + 1
        θ = Vector{Real}(undef, n_subjects)
        for i=1:n_subjects
            θ[i] ~ Beta(a,b)
        end
        # Observations
        for j=1:length(obs)
            obs[j] ~ Bernoulli(θ[subj[j]])
        end
    end
    obs, subj = (df[:y], df[:subject_n])
    ϵ = 0.05
    τ = 10
    chains = mapreduce(c -> sample(therapeutic_touch(obs, subj),HMC(ϵ, τ), iterations), chainscat, 1:nchains)
    return chains
end


"""
    plot_tt_shrinkage_effect(basic, hierarchical)

Demonstrates the effect of shrinkage obtained using a hierarchical model instead of the basic model for the TherapeuticTouchData.
Usage:
```julia-repl
julia> basic = tt_sample_basic_model(nchains=1);
julia> hierarchical = tt_sample_hierarchical_model(nchains=1);
julia> plot_shrinkage_effect(basic, hierarchical)
```
"""
function plot_tt_shrinkage_effect(basic, hierarchical)
    basic_means = mean(Array(basic),dims=1)
    hierarchical_means = reshape(mean(Array(hierarchical),dims=1)[1:end-2], (1,28));
    labels = reshape(["θ[$(i)]" for i=1:28],(1,28))
    p = plot(["basic", "hierarchical"], 
        [basic_means; hierarchical_means], 
        markershape = :circle, 
        legend = :outertopright, 
        labels = labels, 
        size = (700, 500),
        xlabel = "Model",
        ylabel = "Estimated accuracy"
        )
    display(p)
end