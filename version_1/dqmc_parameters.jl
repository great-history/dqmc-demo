struct dqmc_parameters
    beta::Float64
    delta_tau::Float64
    slices::Int
    wrap_num::Int
    
    thermalization_sweeps::Int
    measurement_sweeps::Int
    measure_rate::Int
end

function dqmc_parameters(;beta::Float64 = 2.0, delta_tau::Float64 = 0.125, wrap_num::Int = 8, 
                        th_sweeps::Int64 = 1000, meas_sweeps::Int64 = 1000, measure_rate::Int = 1)

    slices = round(Int, beta / delta_tau)
    delta_tau = beta / slices
    
    dqmc_parameters(beta, delta_tau, slices, wrap_num, 
                    th_sweeps, meas_sweeps, measure_rate)
end