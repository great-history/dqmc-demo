# 等时格林函数的计算以及更新
include("dqmc_matrix_cdw_channel.jl")
# 得到真正的格林函数:exp(-Δτ*T/2)G(0,0)exp(Δτ*T/2) for B = exp(V)exp(-ΔτT) ; exp(Δτ*T/2)G(0,0)exp(-Δτ*T/2) for B = exp(-ΔτT)exp(V)
@inline function get_greens_temp!(stack::dqmc_matrices_stack_real_qr)
    vmul!(stack.matrix_tmp, stack.hopping_mat_exp_half, stack.greens)
    vmul!(stack.greens_temp, stack.matrix_tmp, stack.hopping_mat_exp_inv_half)
end

@inline function get_greens_temp_modified!(stack::dqmc_matrices_stack_real_qr)
    vmul!(stack.matrix_tmp, stack.hopping_mat_exp_inv_half, stack.greens)
    vmul!(stack.greens_temp, stack.matrix_tmp, stack.hopping_mat_exp_half)
end

## wrap up & wrap down 涉及:直接的矩阵乘法
## wrap up : G(l,l) = B(l)G(l-1,l-1)B^{-1}(l)
@inline function get_eqlt_green_by_wrap_up!(stack::dqmc_matrices_stack_real_qr)
    get_eqlt_green_by_wrap_up!(stack.greens, stack.current_B, stack.current_B_inv, stack.matrix_tmp1)
end

@inline function get_eqlt_green_by_wrap_up!(greens::Matrix{Float64}, current_B::Matrix{Float64}, current_B_inv::Matrix{Float64}, matrix_tmp::Matrix{Float64})
    vmul!(matrix_tmp, greens, current_B_inv)
    vmul!(greens, current_B, matrix_tmp)
end

# wrap down : G(l,l) = B^{-1}(l)G(l+1,l+1)B(l)
@inline function get_eqlt_green_by_wrap_down!(stack::dqmc_matrices_stack_real_qr)
    get_eqlt_green_by_wrap_down!(stack.greens, stack.current_B, stack.current_B_inv, stack.matrix_tmp1)
end

@inline function get_eqlt_green_by_wrap_down!(greens::Matrix{Float64}, current_B_inv::Matrix{Float64}, current_B::Matrix{Float64}, matrix_tmp::Matrix{Float64})
    vmul!(matrix_tmp, current_B_inv, greens)
    vmul!(greens, matrix_tmp, current_B)
end

## from ffreyer's codes
## TODO::到时候可以补充ffreyer的代码

## from Gaopei Pan's notes
# eqlt_greens数值稳定性专用 G(l,l) = (1 + B(l;l_min1)B(L;l+1))^{-1} where l % safe_mult == 0 & (l != 0 || l != L)
function get_eqlt_green_by_qr!(greens::Matrix{Float64}, 
    Ul::Matrix{Float64}, Dl::Vector{Float64}, Tl::Matrix{Float64}, 
    Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64},
    D_min::Vector{Float64}, pivot::Vector{Int64}, n::Int = size(greens,1))

    # part 1
    diag_separator!(Dl, D_min, n)
    # mul_adjoint_right!(greens, Ul, Dl, n)  有错误,应该是除以Dl
    mul_diag_inv_left_adjoint_right!(greens, Dl, Ul, n)
    mul_diag_left!(D_min, Tl, n)
    
    # part 2
    diag_separator!(Dr, D_min, n)
    mul_diag_inv_right!(Ur, Dr, n)
    mul_adjoint_left!(Ul, Tr, D_min)

    # part 3
    vmul!(Tr, Tl, Ul)
    vmul!(Ul, greens, Ur)
    add_right!(Tr, Ul)

    # part 4
    qr_real_pivot!(Ul, Tr, Dl, pivot, n)
    permute_columns_inverse!(Tl, Ur, pivot, n)
    div_uppertriang_right!(Tl, Tr, n)
    mul_adjoint_left!(Ur, Ul, greens, n)
    mul_diag_inv_right!(Tl, Dl, n)
    vmul!(greens, Tl, Ur)
end

## 与G(0,0)计算相关的计算 涉及:数值稳定性
# @time get_G00_by_right!(stack.greens, stack.Ur, stack.Dr, stack.Tr, stack.pivot_tmp, stack.matrix_tmp1, stack.vector_tmp1, stack.matrix_tmp2) 0.004255 seconds
## from Gaopei Pan's notes
@inline function get_G00_by_right!(stack::dqmc_matrices_stack_real_qr, n::Int = stack.dims)
    get_G00_by_right!(stack.greens, stack.Ur, stack.Dr, stack.Tr, stack.pivot_tmp, stack.matrix_tmp1, stack.vector_tmp1, stack.matrix_tmp2, n)
end

## 用来计算G(0,0) = (1 + B(L;1))^{-1} or G(L,L)
# 这个版本是全程可以改变Ur/Dr/Tr和U_tmp/D_tmp/T_tmp/matrix_tmp/greens的值
function get_G00_by_right!(greens::Matrix{Float64}, Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64}, pivot::Vector{Int64},
    U_tmp::Matrix{Float64}, D_min::Vector{Float64}, T_tmp::Matrix{Float64}, n::Int = size(pivot, 1))
    # Dr ⟹ D_min + D_max
    diag_separator!(Dr, D_min, n)
    # Ur = Ur * Diagonal(D_max)
    mul_diag_inv_right!(Ur, Dr)
    
    # T_tmp = Tr^{dagger} * Diagonal(D_min)
    # 不需要用到pivot,并且Tr往往不再是上三角矩阵或者置换上三角矩阵的列之后的矩阵
    mul_adjoint_left!(T_tmp, Tr, D_min)
    # T_tmp = T_tmp + Ur
    add_right!(T_tmp, Ur)

    # qr分解: T_tmp ⟹ UDT ~ matrix_tmp * Digonal(D_max) * T_tmp
    qr_real_pivot!(U_tmp, T_tmp, Dr, pivot)
    # pivot : (U_tmp * P) * T^{-1} * ((1 / D_max) * matrix_tmp^{dagger})
    permute_columns_inverse!(Tr, Ur, pivot, n)
    # div_uppertriang_right!(greens, U_tmp, T_tmp, n)
    div_uppertriang_right!(Tr, T_tmp, n)
    mul_diag_inv_right!(Tr, Dr)
    mul_adjoint_right!(greens, Tr, U_tmp)

    nothing
end

function get_G00_by_left!(stack::dqmc_matrices_stack_real_qr, n::Int = stack.dims)
    get_G00_by_right!(stack.greens, stack.Ul, stack.Dl, stack.Tl, stack.pivot_tmp, stack.matrix_tmp1, stack.vector_tmp1, n)
end

function get_G00_by_left!(greens::Matrix{Float64}, Ul::Matrix{Float64}, Dl::Vector{Float64}, Tl::Matrix{Float64}, pivot::Vector{Int64},
    matrix_tmp::Matrix{Float64}, D_min::Vector{Float64}, n::Int = size(pivot, 1))
    diag_separator!(Dl, D_min, n)
    
    # get 1 / Dl_max * Ul^{dagger}
    mul_diag_inv_right!(Ul, Dl, n)
    adjoint_real_matrix!(greens, Ul, n)

    # get D_min * Tl
    mul_diag_left!(D_min, Tl, n)

    # Tl + Ur
    add_right!(Tl, greens, n)
    # get qr decomposition of Tl 
    qr_real_pivot!(Ul, Tl, Dl, pivot, n)
    mul_adjoint_left!(matrix_tmp, Ul, greens, n)

    # 1 / Dl * greens
    mul_diag_inv_left!(Dl, matrix_tmp, n)

    # Tl^{-1} * greens
    div_uppertriang_left!(Tl, matrix_tmp, n)

    # get greens
    permute_rows_inverse!(greens, matrix_tmp, pivot, n)

end

@inline function get_greens_temp_modified!(stack::dqmc_matrices_stack_real_qr)
    vmul!(stack.matrix_tmp4, stack.hopping_matrix_exp_inv_half, stack.greens)
    vmul!(stack.greens_temp, stack.matrix_tmp4, stack.hopping_matrix_exp_half)
end

# 可用来计算非等时格林函数
@inline function get_greens_temp_modified!(stack::dqmc_matrices_stack_real_qr, greens::Matrix{Float64})
    vmul!(stack.matrix_tmp4, stack.hopping_matrix_exp_inv_half, greens)
    vmul!(greens, stack.matrix_tmp4, stack.hopping_matrix_exp_half)
end