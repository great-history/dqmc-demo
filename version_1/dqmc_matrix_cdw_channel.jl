using LinearAlgebra
using LoopVectorization
include("dqmc_linalg.jl")
include("dqmc_stable.jl")

"""
stack就是仓库的意思
dqmc_matrix_stack_real_qr 是指所有矩阵都为实数而非复数, 且适用QR分解的DQMC情形 TODO::real_svd // complex_qr // complex_svd

"""
mutable struct dqmc_matrices_stack_real_qr
    # immutable
    n_stacks::Int
    n_slices::Int
    dims::Int
    slice_rem::Vector{Int}
    ranges::Vector{UnitRange{Int64}}

    ## 跃迁矩阵相关
    hopping_matrix::Matrix{Float64}
    hopping_matrix_exp::Matrix{Float64}
    hopping_matrix_exp_inv::Matrix{Float64}
    hopping_matrix_exp_half::Matrix{Float64}
    hopping_matrix_exp_inv_half::Matrix{Float64}
    
    # mutable
    current_slice::Int
    current_range::Int
    current_direction::Int

    current_B::Matrix{Float64}
    current_B_inv::Matrix{Float64}
    # TODO::用StructArray来存储
    Ul::Matrix{Float64}
    Dl::Vector{Float64}
    Tl::Matrix{Float64}
    Ur::Matrix{Float64}
    Dr::Vector{Float64}
    Tr::Matrix{Float64}
    greens::Matrix{Float64}
    greens_temp::Matrix{Float64}  ## 主要用于全局更新和交叉

    # TODO::用StructArray来存储
    u_stack::Vector{Matrix{Float64}}
    d_stack::Vector{Vector{Float64}}
    t_stack::Vector{Matrix{Float64}}
    # pivot_stack::Vector{Vector{Float64}}

    matrix_tmp1::Matrix{Float64}
    matrix_tmp2::Matrix{Float64}
    matrix_tmp3::Matrix{Float64}
    matrix_tmp4::Matrix{Float64}
    vector_tmp1::Vector{Float64}
    vector_tmp2::Vector{Float64}
    pivot_tmp::Vector{Int64}
    diagonal_tmp::Diagonal{Float64, Vector{Float64}}
end

## 初始化dqmc_matrices_stack_real_qr
function dqmc_matrices_stack_real_qr(model::Single_Band_Hubbard_Model, parameters::dqmc_parameters)
    n_slices = parameters.slices
    wrap_num = parameters.wrap_num
    n_sites = model.lattice.sites
    n_flavors = model.n_flavors
    dims = n_sites * n_flavors

    n_chunks = cld(n_slices, wrap_num)
    slices_per_chunk = cld(n_slices, n_chunks)
    ranges = get_chunks_ranges(n_chunks, slices_per_chunk)
    slice_rem = [i for i in 1:n_slices]
    slice_rem = slice_rem .% wrap_num
    n_stacks = length(ranges) + 1
    
    u_stack = [Matrix{Float64}(I, dims, dims) for _ in 1:n_stacks]
    d_stack = [ones(Float64, dims) for _ in 1:n_stacks]
    t_stack = [Matrix{Float64}(I, dims, dims) for _ in 1:n_stacks]

    Ul = Matrix{Float64}(undef, dims, dims)
    Ur = Matrix{Float64}(undef, dims, dims)
    Dl = ones(Float64, dims)
    Dr = ones(Float64, dims)
    Tl = Matrix{Float64}(undef, dims, dims)
    Tr = Matrix{Float64}(undef, dims, dims)

    greens = Matrix{Float64}(undef, dims, dims)
    greens_temp = Matrix{Float64}(undef, dims, dims)  ## 主要用于全局更新和交叉

    current_slice = 0
    current_range = 1
    current_direction = 1
    current_B = Matrix{Float64}(I, dims, dims)
    current_B_inv = Matrix{Float64}(I, dims, dims)

    matrix_tmp1 = Matrix{Float64}(I, dims, dims)
    matrix_tmp2 = Matrix{Float64}(I, dims, dims)
    matrix_tmp3 = Matrix{Float64}(I, dims, dims)
    matrix_tmp4 = Matrix{Float64}(I, dims, dims)
    vector_tmp1 = ones(Float64, dims)
    vector_tmp2 = ones(Float64, dims)
    pivot_tmp = zeros(Int64, dims)
    diagonal_tmp = Diagonal(vector_tmp1)

    # 跃迁矩阵相关
    hopping_matrix, hopping_matrix_exp, hopping_matrix_exp_inv, hopping_matrix_exp_half, hopping_matrix_exp_inv_half = get_nn_hopping_matrices(parameters.delta_tau, model)

    dqmc_matrices_stack_real_qr(n_stacks, n_slices, dims, slice_rem, ranges,
                                hopping_matrix, hopping_matrix_exp, hopping_matrix_exp_inv, hopping_matrix_exp_half, hopping_matrix_exp_inv_half,
                                current_slice, current_range, current_direction,
                                current_B, current_B_inv, Ul, Dl, Tl, Ur, Dr, Tr, greens, greens_temp,
                                u_stack, d_stack, t_stack,
                                matrix_tmp1, matrix_tmp2, matrix_tmp3, matrix_tmp4, vector_tmp1, vector_tmp2, pivot_tmp, diagonal_tmp)
end

function get_chunks_ranges(n_chunks::Int, slices_per_chunk::Int)
    [round(Int, (i-1)*slices_per_chunk) + 1 : round(Int, i * slices_per_chunk) for i in 1:n_chunks]
end

# B_slice的相关计算
## hopping matrices
function get_nn_hopping_matrix(model::Single_Band_Hubbard_Model)
    N = model.lattice.sites
    T = diagm(0 => fill(-model.mu, N))
    @inbounds begin
        for i = 1:model.lattice.sites
            for j = 1:model.lattice.n_coord
                T[i, model.lattice.nearest_neighbors[j,i]] = - model.t
            end
        end 
    end

    return T
end

function get_nn_hopping_matrices(delta_tau::Float64, model::Single_Band_Hubbard_Model)
    hopping_matrix = get_nn_hopping_matrix(model)
    hopping_matrix_exp = exp(-delta_tau * hopping_matrix)
    hopping_matrix_exp_inv = exp(delta_tau * hopping_matrix)
    hopping_matrix_exp_half = exp(-(delta_tau / 2) * hopping_matrix)
    hopping_matrix_exp_inv_half = exp((delta_tau / 2) * hopping_matrix)

    return hopping_matrix, hopping_matrix_exp, hopping_matrix_exp_inv, hopping_matrix_exp_half, hopping_matrix_exp_inv_half
end

## interaction matrices
@inline function get_interaction_matrix_exp!(field::HSField_CDW, slice_index::Int, vector_tmp1::Vector{Float64})  ## 待修改：matrices_stack →→→→→ vector{Float64}
    @inbounds for i in eachindex(vector_tmp1)
        vector_tmp1[i] = field.conf[i, slice_index] < 0 ? field.elements[2] : field.elements[1]
    end
    nothing
end

@inline function get_interaction_matrix_exp_inv!(field::HSField_CDW, slice_index::Int, vector_tmp2::Vector{Float64})
    @inbounds for i in eachindex(vector_tmp2)
        vector_tmp2[i] = field.conf[i, slice_index] < 0 ? field.elements[1] : field.elements[2]
    end
    nothing
end

## type-I B matrix: exp(V(Sl(C)))exp(-ΔτT) and its inverse exp(ΔτT)exp(-V(Sl(C)))

# Complex matrix power for upper triangular factor, see:
#   Higham and Lin, "A Schur-Padé algorithm for fractional powers of a Matrix",
#     SIAM J. Matrix Anal. & Appl., 32 (3), (2011) 1056–1078.
#   Higham and Lin, "An improved Schur-Padé algorithm for fractional powers of
#     a matrix and their Fréchet derivatives", SIAM. J. Matrix Anal. & Appl.,
#     34(3), (2013) 1341–1360.

"""
    LowerTriangular(A::AbstractMatrix)

Construct a `LowerTriangular` view of the matrix `A`.

# Examples
```jldoctest
julia> A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
3×3 Matrix{Float64}:
 1.0  2.0  3.0
 4.0  5.0  6.0
 7.0  8.0  9.0

julia> LowerTriangular(A)
3×3 LowerTriangular{Float64, Matrix{Float64}}:
 1.0   ⋅    ⋅
 4.0  5.0   ⋅
 7.0  8.0  9.0
```
"""
### test: @time (for i = 1:1000; get_single_B_slice!(hs_field, 1, stack.vector_tmp1, stack.matrix_tmp1, stack.current_B);end)  用时:0.011s左右，足够了
@inline function get_single_B_slice!(field::HSField_CDW, slice_index::Int, vector_tmp1::Vector{Float64}, hopping_matrix_exp::Matrix{Float64}, B_slice::Matrix{Float64})  # get_property 占了 25 / 60,我的建议是在for循环调用get_single_B_slice前先赋值两个矩阵
    @inbounds begin
        
        for i in eachindex(vector_tmp1)  # 12 / 60
            vector_tmp1[i] = field.conf[i, slice_index] < 0 ? field.elements[2] : field.elements[1]  # 千万不要把field.conf[:, slice_index]单独定义出来，这样会花不少时间
        end
        
        mul_diag_left!(B_slice, vector_tmp1, hopping_matrix_exp) # 9 / 60
        nothing
    end
end

### test: @time (for i = 1:1000; get_single_B_slice_inv!(hs_field, 1, stack.hopping_matrix_exp_inv,  stack.vector_tmp2, stack.current_B_inv);end)
@inline function get_single_B_slice_inv!(field::HSField_CDW, slice_index::Int, hopping_matrix_exp_inv::Matrix{Float64}, vector_tmp2::Vector{Float64}, B_slice_inv::Matrix{Float64})
    @inbounds begin
        
        for i in eachindex(vector_tmp2)  # 12 / 60
            vector_tmp2[i] = field.conf[i, slice_index] < 0 ? field.elements[1] : field.elements[2]
        end
        
        mul_diag_right!(B_slice_inv, hopping_matrix_exp_inv, vector_tmp2)
        nothing
    end
end

### test: @time (for i = 1:1000; get_single_B_slices!(hs_field, 1, stack.vector_tmp1, stack.hopping_matrix_exp, stack.current_B, stack.vector_tmp2, stack.hopping_matrix_exp_inv, stack.current_B_inv); end)
@inline function get_single_B_slices!(field::HSField_CDW, slice_index::Int, vector_tmp1::Vector{Float64}, hopping_matrix_exp::Matrix{Float64}, B_slice::Matrix{Float64}, 
                                        vector_tmp2::Vector{Float64}, hopping_matrix_exp_inv::Matrix{Float64}, B_slice_inv::Matrix{Float64})  # get_property 占了 3 / 11
    @inbounds begin

        for i in eachindex(vector_tmp1)  # 2 / 11
            field.conf[i, slice_index] < 0 ? (vector_tmp1[i] = field.elements[2] ; vector_tmp2[i] = field.elements[1]) : (vector_tmp1[i] = field.elements[1] ; vector_tmp2[i] = field.elements[2])  # 千万不要把field.conf[:, slice_index]单独定义出来，这样会花不少时间
        end
        
        mul_diag_left!(B_slice, vector_tmp1, hopping_matrix_exp) # 2 / 11
        mul_diag_right!(B_slice_inv, hopping_matrix_exp_inv, vector_tmp2) # 2 / 11
        nothing
    end
end

# 只会用到build_stack中,一般的过程不会用到这个函数
## @time multiply_B_right_slices_per_chunk!(hs_field, stack, 2)   0.007580 seconds
function multiply_B_right_slices_per_chunk!(field::HSField_CDW, stack::dqmc_matrices_stack_real_qr, range_idx::Int)
    @inbounds begin
        eT = stack.hopping_matrix_exp
        eV = stack.vector_tmp1
        B_slice = stack.current_B
        Ur = stack.u_stack[range_idx + 1]   ## TODO::规定Ur存放的是adjoint之后的U矩阵
        matrix_tmp1 = stack.matrix_tmp1

        for slice_index in reverse(stack.ranges[range_idx])
            get_single_B_slice!(field, slice_index, eV, eT, B_slice)
            mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
        end
        
        mul_diag_right!(stack.Tr, matrix_tmp1, stack.d_stack[range_idx])

        qr_real_pivot!(stack.Ur, stack.Tr, stack.pivot_tmp, stack.Dr)

        vmul!(stack.t_stack[range_idx], stack.Tr, stack.t_stack[range_idx + 1])
        
        nothing
    end
end

function multiply_B_right_slices_per_chunk!(::Val{true}, field::HSField_CDW, stack::dqmc_matrices_stack_real_qr, range_idx::Int)
    @inbounds begin
        eT = stack.hopping_matrix_exp
        eV = stack.vector_tmp1
        B_slice = stack.current_B
        Ur = stack.u_stack[range_idx + 1]   ## TODO::规定Ur存放的是adjoint之后的U矩阵
        matrix_tmp1 = stack.matrix_tmp1

        for slice_index in reverse(stack.ranges[range_idx])
            get_single_B_slice!(field, slice_index, eV, eT, B_slice)
            mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
        end
        
        mul_diag_right!(stack.Tr, matrix_tmp1, stack.d_stack[range_idx])

        qr_real_pivot!(stack.Ur, stack.Tr, stack.pivot_tmp, stack.Dr)

        permute_columns!(stack.t_stack[2], stack.Tr, stack.pivot_tmp, stack.dims)
        
        nothing
    end
end

# B_slices 数值稳定性专用
## 适用于sweep up, 我把整个stack都传入是因为要用到u/d/t_stack,它们是比较大的 TODO::能不能把u/d/t_stack 和 其他单独的矩阵 分开到两个类型中存储？？可以定义一个专门存放UDT的类型和相应的类型数组
function B_left_slices_stable_by_qr!(stack::dqmc_matrices_stack_real_qr, current_range = stack.current_range)
    mul!(stack.matrix_tmp1, stack.current_B, stack.Ul)
    stack.diagonal_tmp = Diagonal(stack.Dl)
    mul!(stack.matrix_tmp2, stack.matrix_tmp1, stack.diagonal_tmp)

    udt_real!(stack.matrix_tmp2, stack.Ul, stack.Dl, stack.matrix_tmp1)
    copyto!(stack.u_stack[current_range], stack.Ul)
    copyto!(stack.d_stack[current_range], stack.Dl)
    mul!(stack.matrix_tmp2, stack.matrix_tmp1, stack.Tl)
    copyto!(stack.t_stack[current_range], stack.matrix_tmp2)
    copyto!(stack.Tl, stack.matrix_tmp2)

    nothing
end

## 适用于sweep down, 我把整个stack都传入是因为要用到u/d/t_stack,它们是比较大的
function B_right_slices_stable_by_qr!(stack::dqmc_matrices_stack_real_qr, current_range = stack.current_range)
    @inbounds begin
        mul!(stack.matrix_tmp2, stack.Ur, stack.current_B)
        stack.diagonal_tmp = Diagonal(stack.Dr)
        # 作qr分解: Ur^{†}*Bc(;) ⟹ Bc^{†}(;)Ur
        stack.matrix_tmp1 = adjoint(stack.matrix_tmp2)
        rmul!(stack.matrix_tmp1, stack.diagonal_tmp)
        
        udt_real!(stack.matrix_tmp1, stack.Ur, stack.Dr, stack.matrix_tmp2)
        copyto!(stack.u_stack[current_range], stack.Ur)
        stack.Ur = adjoint(stack.Ur)
        copyto!(stack.d_stack[current_range], stack.Dr)
        mul!(stack.matrix_tmp1, stack.matrix_tmp2, stack.Tr)

        copyto!(stack.t_stack[current_range], stack.matrix_tmp2)
        copyto!(stack.Tr, stack.matrix_tmp2)
    end

    nothing
end

# type-II B matrix: exp(-ΔτT)exp(V(Sl(C))) and its inverse exp(-V(Sl(C)))exp(ΔτT)
## test: @time (for i = 1:1000; get_single_B_slice_modified!(hs_field, 1, stack.vector_tmp1, stack.matrix_tmp1, stack.current_B);end)  用时:0.011s左右，足够了
@inline function get_single_B_slice_modified!(field::HSField_CDW, slice_index::Int, hopping_matrix_exp::Matrix{Float64}, vector_tmp1::Vector{Float64}, B_slice::Matrix{Float64})
    @inbounds begin
        for i in eachindex(vector_tmp1)  # 12 / 60
            vector_tmp1[i] = field.conf[i, slice_index] < 0 ? field.elements[2] : field.elements[1]  # 千万不要把field.conf[:, slice_index]单独定义出来，这样会花不少时间
        end
        
        mul_diag_right!(B_slice, hopping_matrix_exp, vector_tmp1) # 9 / 60
        nothing
    end
end

## @time (for i = 1:1000000; get_single_B_slice_inv_modified!(hs_field, 1, stack.vector_tmp2, stack.hopping_matrix_exp_inv, stack.current_B_inv);end)
@inline function get_single_B_slice_inv_modified!(field::HSField_CDW, slice_index::Int, vector_tmp2::Vector{Float64}, hopping_matrix_exp_inv::Matrix{Float64}, B_slice_inv::Matrix{Float64})
    @inbounds begin
        for i in eachindex(vector_tmp2)  # 12 / 60
            vector_tmp2[i] = field.conf[i, slice_index] < 0 ? field.elements[1] : field.elements[2]
        end
        
        mul_diag_left!(B_slice_inv, vector_tmp2, hopping_matrix_exp_inv)
        nothing
    end
end

## @time (for i = 1:1000; get_single_B_slices_modified!(hs_field, 1, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B, stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv); end)
@inline function get_single_B_slices_modified!(field::HSField_CDW, slice_index::Int, hopping_matrix_exp::Matrix{Float64}, vector_tmp1::Vector{Float64}, B_slice::Matrix{Float64}, 
    hopping_matrix_exp_inv::Matrix{Float64}, vector_tmp2::Vector{Float64}, B_slice_inv::Matrix{Float64})
    @inbounds begin

        for i in eachindex(vector_tmp1)
            field.conf[i, slice_index] < 0 ? (vector_tmp1[i] = field.elements[2] ; vector_tmp2[i] = field.elements[1]) : (vector_tmp1[i] = field.elements[1] ; vector_tmp2[i] = field.elements[2])
        end

        mul_diag_right!(B_slice, hopping_matrix_exp, vector_tmp1)
        mul_diag_left!(B_slice_inv, vector_tmp2, hopping_matrix_exp_inv)
        nothing
    end
end

function multiply_B_right_slices_per_chunk!(::Val{true}, field::HSField_CDW, stack::dqmc_matrices_stack_real_qr, range_idx::Int)
    @inbounds begin
        eT = stack.hopping_matrix_exp
        eV = stack.vector_tmp1
        B_slice = stack.current_B
        Ur = stack.u_stack[range_idx + 1]   ## TODO::规定Ur存放的是adjoint之后的U矩阵
        matrix_tmp1 = stack.matrix_tmp1

        for slice_index in reverse(stack.ranges[range_idx])
            get_single_B_slice!(field, slice_index, eV, eT, B_slice)
            mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
        end
        
        mul_diag_right!(stack.Tr, matrix_tmp1, stack.d_stack[range_idx])

        qr_real_pivot!(stack.Ur, stack.Tr, stack.pivot_tmp, stack.Dr)

        permute_columns!(stack.t_stack[2], stack.Tr, stack.pivot_tmp, stack.dims)
        
        nothing
    end
end

# for B_c(l) = exp(-ΔτT)exp(V(C))
function multiply_B_right_slices_per_chunk_modified!(field::HSField_CDW, stack::dqmc_matrices_stack_real_qr, range_idx::Int)
    @inbounds begin
        eT = stack.hopping_matrix_exp
        eV = stack.vector_tmp1
        B_slice = stack.current_B
        Ur = stack.u_stack[range_idx + 1]   ## TODO::规定Ur存放的是adjoint之后的U矩阵
        matrix_tmp1 = stack.matrix_tmp1

        for slice_index in reverse(stack.ranges[range_idx])
            get_single_B_slice_modified!(field, slice_index, eT, eV, B_slice)
            mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
        end
        
        mul_diag_right!(stack.Tr, matrix_tmp1, stack.d_stack[range_idx + 1])

        qr_real_pivot!(stack.Ur, stack.Tr, stack.pivot_tmp, stack.Dr)

        vmul!(stack.t_stack[range_idx], stack.Tr, stack.t_stack[range_idx + 1])
        
        nothing
    end
end

# 数值稳定性一下: advanced up
function advanced_up_by_qr!(stack::dqmc_matrices_stack_real_qr, current_range = stack.current_range)
    B_left_slices_stable_by_qr!(stack::dqmc_matrices_stack_real_qr)

    range = current_range + 1
    copyto!(stack.Ur, stack.u_stack[range])
    copyto!(stack.Dr, stack.d_stack[range])
    copyto!(stack.Tr, stack.t_stack[range])
end

# 数值稳定性一下: advanced down, 每当current_slice % safe_mult == 0时整一波, 当current_slice == n_slices不需要更新了，因为sweep up的时候已经计算过了G(L,L)
function advanced_down_by_qr!(stack::dqmc_matrices_stack_real_qr, current_range = stack.current_range)
    B_right_slices_stable_by_qr!(stack::dqmc_matrices_stack_real_qr)
    
    range = current_range - 1
    copyto!(stack.Ul, stack.u_stack[range])
    copyto!(stack.Dl, stack.d_stack[range])
    copyto!(stack.Tl, stack.t_stack[range])
end

function build_udt_stacks!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    @inbounds begin
        # end的情况
        copyto!(stack.u_stack[end], I)
        copyto!(stack.Ur, stack.u_stack[end])
        stack.d_stack[end] .= one(Float64)
        copyto!(stack.t_stack[end], I)

        ## build stack from end-1 to 2
        eT = stack.hopping_matrix_exp
        eV = stack.vector_tmp1
        B_slice = stack.current_B
        matrix_tmp1 = stack.matrix_tmp1
        Ur = stack.Ur
        pivot = stack.pivot_tmp
        dims = stack.dims

        # 先把end - 1的情形单独算掉
        copyto!(Ur,stack.u_stack[end])  ## TODO::规定Ur存放的是adjoint之后的U矩阵

        for slice_index in reverse(stack.ranges[end])
            get_single_B_slice!(hs_field, slice_index, eT, eV, B_slice)
            mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
            copyto!(Ur, matrix_tmp1)
        end

        qr_real_pivot!(Val(true), stack.u_stack[end-1], Ur, stack.d_stack[end - 1], pivot, dims)
        permute_columns!(stack.t_stack[end-1], Ur, pivot, dims)

        # 把end - 2 到 2的情形算掉
        for range_idx in (stack.n_stacks-2):-1:1
            copyto!(Ur, stack.u_stack[range_idx + 1]) ## TODO::规定Ur存放的是adjoint之后的U矩阵
    
            for slice_index in reverse(stack.ranges[range_idx])
                get_single_B_slice!(hs_field, slice_index, eT, eV, B_slice)
                mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
                copyto!(Ur, matrix_tmp1)
            end
            
            # mul_diag_right!(matrix_tmp1, Ur, stack.d_stack[range_idx + 1])
            mul_diag_right!(matrix_tmp1, stack.d_stack[range_idx + 1])
            qr_real_pivot!(stack.u_stack[range_idx], matrix_tmp1, stack.d_stack[range_idx], pivot, dims)

            permute_rows!(Ur, stack.t_stack[range_idx + 1], pivot, dims)
            # vmul!(stack.t_stack[range_idx], matrix_tmp1, Ur)
            # mul_triang_left!(stack.t_stack[range_idx], matrix_tmp1, Ur)
            lmul!(UpperTriangular(matrix_tmp1), Ur)  # 用lmul!是最快的选择,用vmul和mul_triang_left这两种自己写的比较慢
            copyto!(stack.t_stack[range_idx], Ur)
        end

        # 把1的情形算掉
        copyto!(Ur, stack.u_stack[1])
        copyto!(stack.Dr, stack.d_stack[1])
        copyto!(stack.Tr, stack.t_stack[1])

        copyto!(stack.Ul, I)
        stack.Dl .= one(Float64)
        copyto!(stack.Tl, I)

        # 计算等时格林函数
        get_G00_by_right!(stack)

        # 最后处理一下1
        copyto!(stack.u_stack[1], I)
        stack.d_stack[1] .= one(Float64)
        copyto!(stack.t_stack[1], I)
        
        nothing
    end
end

# @time build_udt_stacks_modified!(hs_field, stack) 0.020704 seconds
# @time (for i = 1:100; build_udt_stacks_modified!(hs_field, stack); end) 2.000329 seconds
# @time reverse_build_stack(dqmc) 0.029740 seconds (127.64 k allocations: 6.703 MiB, 68.58% compilation time)
# @time (for i = 1:100; reverse_build_stack(dqmc); end) 2.824629 seconds
function build_udt_stacks_modified!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    @inbounds begin
        # end的情况
        copyto!(stack.u_stack[end], I)
        copyto!(stack.Ur, stack.u_stack[end])
        stack.d_stack[end] .= one(Float64)
        copyto!(stack.t_stack[end], I)

        ## build stack from end-1 to 2
        eT = stack.hopping_matrix_exp
        eV = stack.vector_tmp1
        B_slice = stack.current_B
        matrix_tmp1 = stack.matrix_tmp1
        Ur = stack.Ur
        pivot = stack.pivot_tmp
        dims = stack.dims

        # 先把end - 1的情形单独算掉
        copyto!(Ur,stack.u_stack[end])  ## TODO::规定Ur存放的是adjoint之后的U矩阵

        for slice_index in reverse(stack.ranges[end])
            get_single_B_slice_modified!(hs_field, slice_index, eT, eV, B_slice)
            mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
            copyto!(Ur, matrix_tmp1)
        end

        qr_real_pivot!(Val(true), stack.u_stack[end-1], Ur, stack.d_stack[end - 1], pivot, dims)
        permute_columns!(stack.t_stack[end-1], Ur, pivot, dims)

        # 把end - 2 到 2的情形算掉
        for range_idx in (stack.n_stacks-2):-1:1
            copyto!(Ur, stack.u_stack[range_idx + 1]) ## TODO::规定Ur存放的是adjoint之后的U矩阵
    
            for slice_index in reverse(stack.ranges[range_idx])
                get_single_B_slice_modified!(hs_field, slice_index, eT, eV, B_slice)
                mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
                copyto!(Ur, matrix_tmp1)
            end
            
            # mul_diag_right!(matrix_tmp1, Ur, stack.d_stack[range_idx + 1])
            mul_diag_right!(matrix_tmp1, stack.d_stack[range_idx + 1])
            qr_real_pivot!(stack.u_stack[range_idx], matrix_tmp1, stack.d_stack[range_idx], pivot, dims)

            permute_rows!(Ur, stack.t_stack[range_idx + 1], pivot, dims)
            # vmul!(stack.t_stack[range_idx], matrix_tmp1, Ur)
            # mul_triang_left!(stack.t_stack[range_idx], matrix_tmp1, Ur)
            lmul!(UpperTriangular(matrix_tmp1), Ur)  # 用lmul!是最快的选择,用vmul和mul_triang_left这两种自己写的比较慢
            copyto!(stack.t_stack[range_idx], Ur)
        end

        # 把1的情形算掉
        copyto!(Ur, stack.u_stack[1])
        copyto!(stack.Dr, stack.d_stack[1])
        copyto!(stack.Tr, stack.t_stack[1])

        copyto!(stack.Ul, I)
        stack.Dl .= one(Float64)
        copyto!(stack.Tl, I)

        # 计算等时格林函数
        get_G00_by_right!(stack)

        # 因为在get_G00_by_right!中Ur,Dr,Tr都变了
        copyto!(Ur, stack.u_stack[1])
        copyto!(stack.Dr, stack.d_stack[1])
        copyto!(stack.Tr, stack.t_stack[1])

        # 最后处理一下1
        # copyto!(stack.u_stack[1], I)
        # stack.d_stack[1] .= one(Float64)
        # copyto!(stack.t_stack[1], I)

        nothing
    end
end