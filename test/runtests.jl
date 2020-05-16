using PuppyBook
using Test
using Logging

function test_running_proportion_length(ntests=10)
    n_trials = rand(1:100,ntests)
    has_correct_length = Bool[]
    for n in n_trials
        is_correct = length(compute_running_proportion(0.1,n)) == n
        is_correct || @error "Test failed with N = $n"
        append!(has_correct_length, is_correct)
    end
    return all(has_correct_length)
end

function test_running_proportion_convergence(;tol=1e-1, ntrials=1000)
    p_heads = rand(0.01:0.01:0.99, 10)
    converged = []
    for p in p_heads
        convd = isapprox(compute_running_proportion(p, ntrials)[end],p,atol=tol)
        convd || @error "Test failed with p_heads=$p and N=$ntrials"
        append!(converged, convd)
    end
    return all(converged)
end

@testset "PuppyBook.jl" begin
    @testset "03_running_proportion.jl" begin
        @test_throws AssertionError compute_running_proportion(-2.0, 10)
        @test_throws AssertionError compute_running_proportion(1.01, 10)
        @test test_running_proportion_length()
        @test test_running_proportion_convergence()
    end
end
