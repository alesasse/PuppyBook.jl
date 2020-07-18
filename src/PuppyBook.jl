module PuppyBook

    include("03_running_proportion.jl")
    include("04_bayes_update.jl")

    export compute_running_proportion, plot_running_proportion,
        coinflip_discrete_bayes_update, plot_coinflip_discrete_bayes_update
    
    greet() = print("Hello Puppies!")

end # module
