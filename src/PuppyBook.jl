module PuppyBook

    include("03_running_proportion.jl")
    include("04_bayes_update.jl")
    include("therapeutic_touch.jl")

    export compute_running_proportion, plot_running_proportion,
        coinflip_discrete_bayes_update, plot_coinflip_discrete_bayes_update,
        load_therapeutic_touch_data, tt_sample_basic_model, tt_sample_hierarchical_model,
        plot_tt_shrinkage_effect
    
    greet() = print("Hello Puppies!")

end # module
