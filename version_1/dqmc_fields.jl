using Random

mutable struct HSField_CDW
    γ::Float64
    α::Float64
    
    elements::Vector{Float64}
    Δs::Vector{Float64}
    boson_ratios::Vector{Float64}

    Δ::Float64
    boson_ratio::Float64
    det_ratio::Float64
    ratio::Float64

    accepted_ratio::Float64

    conf::Matrix{Int8}
    temp_conf::Matrix{Int8}  # for global update
end

function HSField_CDW(;delta_tau::Float64,slice_num::Int, n_sites::Int, U::Float64)
    γ = 1 / 2 * exp(-0.25 * delta_tau * abs(U))
    
    x = exp(0.5 * delta_tau * abs(U))
    α = acosh(x)
    y = exp(delta_tau * abs(U))

    elements = [x + sqrt(y-1), x - sqrt(y-1)]
    # elements = [exp(α), exp(-α)]

    boson_ratios = [elements[1]^2, elements[2]^2]
    # boson_energy = [exp(2α), exp(-2α)]

    Δs = [boson_ratios[2]-1, boson_ratios[1]-1]
    # delta = [exp(-2α)-1, exp(2α)-1]
    
    Δ = Δs[1]
    boson_ratio = boson_ratios[1]
    det_ratio = Float64(1)
    ratio = Float64(1)
    accepted_ratio = Float64(0)

    conf = rand!(Matrix{Int8}(undef, n_sites, slice_num), (Int8(-1), Int8(1)))
    temp_conf = conf
    
    HSField_CDW(γ, α, elements, Δs, boson_ratios, Δ, boson_ratio, det_ratio, ratio, accepted_ratio, conf, temp_conf)
end
