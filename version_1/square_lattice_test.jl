using Revise, MonteCarlo, Printf, LinearAlgebra

mcs = []

# ALF defines our I - G as the measured Greens function
function mygreens(mc, m, ij, G)
    i, j = ij; N = length(lattice(mc))
    swapop(G)[i, j] + swapop(G)[i+N, j+N]
end

# The interaction energy needs to be adjusted to ALF's Hamiltonian
function my_intE(mc, m, G)
    E = 0.0; N = length(lattice(mc))
    Gup, Gdown = G.val.blocks
    for i in 1:N
        E += (1 - Gup[i, i]) * (1 - Gdown[i, i])
    end
    m.U * E
end

# ALF includes 0 and β in the time displaced greens function
myGk(mc, m, ij, Gs) = begin G00, G0l, Gl0, Gll = Gs; i, j = ij; Gl0[i, j] end

# ALF subtracts the uncorrelated part
function myDenDen(mc, m, ij, G)
    i, j = ij; N = length(lattice(mc))
    swapop(G)[i, j] * G[i, j] + swapop(G)[i+N, j+N] * G[i+N, j+N]
end

function myDenDenTau(mc, m, ij, Gs)
    i, j = ij; N = length(lattice(mc))
    G00, G0l, Gl0, Gll = Gs
    swapop(G0l)[i, j] * Gl0[i, j] + swapop(G0l)[i+N, j+N] * Gl0[i+N, j+N]
end


@time for beta in [1.0, 6.0, 12.0]
    m = HubbardModel(4, 2, U = -4)
    mc = DQMC(
        m, beta=beta, thermalization=5_000, sweeps=15_000, 
        print_rate=5_000, delta_tau = 0.05#, measure_rate=5
    )
    
    # our default versions
    mc[:G] = greens_measurement(mc, m)
    mc[:SDCz] = spin_density_correlation(mc, m, :z)
    mc[:SDSz] = spin_density_susceptibility(mc, m, :z)
    mc[:T] = noninteracting_energy(mc, m)
    
    # mc[:Gr] = MonteCarlo.Measurement(mc, m, Greens, EachSitePairByDistance(), mygreens)
    
    # mc[:V] = MonteCarlo.Measurement(mc, m, Greens, nothing, my_intE)
    
    # mc[:IGk] = MonteCarlo.Measurement(
    #     mc, m, CombinedGreensIterator, EachSitePairByDistance(), myGk
    # )
    
    # mc[:DenDen] = MonteCarlo.Measurement(mc, m, Greens, EachSitePairByDistance(), myDenDen)
    
    # mc[:DenDenTau] = MonteCarlo.Measurement(mc, m, CombinedGreensIterator, EachSitePairByDistance(), myDenDenTau)
    
    run!(mc)
    push!(mcs, mc)
end