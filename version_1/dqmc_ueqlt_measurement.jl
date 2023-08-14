## 说明:对于非等时测量，需要用到等时格林函数Gll,Gl0,G0l
mutable struct dqmc_ueqlt_vector_real
    lattice_iterator::String
    bins::Vector{Float64}
    func::Function
end

function dqmc_ueqlt_vector_real()
    
end

# 测量结果存放在一个矩阵中，一般用于计算双粒子格林函数，比如配对,流流，磁化率等
mutable struct dqmc_ueqlt_matrix_real
    lattice_iterator::String
    greens_iterator::Nothing
    bins::Vector{Float64}
    func::Function
end

function dqmc_ueqlt_matrix_real()
    
end