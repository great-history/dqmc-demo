## 说明:对于等时测量，只需要用到等时格林函数G_c(0,0)即可
# 测量结果存放在一个矢量中，一般用于计算on_site型的单粒子格林函数，比如占据数，局域自旋磁矩
abstract type Abstract_Eqlt_Meas end

mutable struct dqmc_eqlt_vector_real <: Abstract_Eqlt_Meas
    kernel::Function
    lattice_iterator::String
    bins::Vector{Float64}
    mean_val::Float64
    std_error::Float64
    n_bins::Int64
end

function dqmc_eqlt_vector_real(lattice_iterator::String, n_bins::Int64, kernel::Function)
    bins = zeros(Float64, n_bins)
    dqmc_eqlt_vector_real(lattice_iterator, bins, n_bins, kernel)
end

# 测量结果存放在一个矩阵中，一般用于计算双粒子格林函数，比如配对,流流，磁化率等
mutable struct dqmc_eqlt_matrix_real <: Abstract_Eqlt_Meas
    kernel::Function
    lattice_iterator::String
    bins::Matrix{Float64}
    mean_val::Float64
    std_error::Float64
    n_bins::Int64
end

function dqmc_eqlt_matrix_real(lattice_iterator::String, n_bins::Int64, dims::Int64, kernel::Function)
    bins = Matrix{Float64}(undef, dims, n_bins)
    dqmc_eqlt_matrix_real(lattice_iterator, bins, n_bins, kernel)
end

mutable struct dqmc_eqlt_tensor_real <: Abstract_Eqlt_Meas
    func::Function
    lattice_iterator::String
    bins::Vector{Matrix{Float64}}
    mean_val::Float64
    std_error::Float64
    n_bins::Int64
end

function dqmc_eqlt_tensor_real(lattice_iterator::String, n_bins::Int64, dims::Int64, kernel::Function)
    bins = [Matrix{Float64}(I, dims, dims) for _ in 1:n_bins]
    dqmc_eqlt_tensor_real(lattice_iterator, bins, kernel)
end

# 单粒子型: occupation
# 格林函数迭代器:只需提供等时格林函数G_c(0,0)
function occupation_average_kernel(bins::Vector{Float64}, greens_temp::Matrix{Float64}, n_meas::Int64, sites::Int64 = size(greens_temp, 1))
    x = zero(Float64)
    @inbounds begin
        @turbo for i in 1:sites
            x += greens_temp[i,i]
        end
    end
    bins[n_meas] = x
    nothing
end

function occupation_kernel(bins::Matrix{Float64}, greens_temp::Matrix{Float64}, n_meas::Int64, sites::Int64 = size(greens_temp, 1))
    @inbounds begin
        @turbo for i in 1:sites
            bins[i, n_meas] = greens_temp[i,i]
        end
    end
    nothing
end

# 双粒子型: density-density correlation
function density_density_correlation_kernel()
    
end