# 非等时格林函数的计算以及更新
include("dqmc_eqlt_greens.jl")

# get unequal time greens function by wrap (valid for both normal and modified)
@inline function get_uneqlt_green_by_wrap_up!(Gll::Matrix{Float64}, Gl0::Matrix{Float64}, G0l::Matrix{Float64}, 
    matrix_tmp::Matrix{Float64}, Ul::Matrix{Float64},
    current_B::Matrix{Float64}, current_B_inv::Matrix{Float64})

    # get G(l,l)
    vmul!(matrix_tmp, Gll, current_B_inv)
    vmul!(Gll, current_B, matrix_tmp)

    # get G(l,0)
    vmul!(matrix_tmp, current_B, Gl0)
    copyto!(Gl0, matrix_tmp)

    # get G(0,l)
    vmul!(matrix_tmp, G0l, current_B_inv)
    copyto!(G0l, matrix_tmp)
end

@inline function get_uneqlt_green_by_qr!(Gll::Matrix{Float64}, Gl0::Matrix{Float64}, G0l::Matrix{Float64}, stack::dqmc_matrices_stack_real_qr)
    get_uneqlt_green_by_qr!(Gll, Gl0, G0l, stack.Ul, stack.Dl, stack.Tl, stack.Ur, stack.Dr, stack.Tr, stack.vector_tmp1, stack.pivot_tmp, stack.dims)
end

function get_uneqlt_green_by_qr!(Gll::Matrix{Float64}, Gl0::Matrix{Float64}, G0l::Matrix{Float64}, 
    Ul::Matrix{Float64}, Dl::Vector{Float64}, Tl::Matrix{Float64}, 
    Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64},
    D_min::Vector{Float64}, pivot::Vector{Int64}, n::Int = size(Gll,1))
    
    # part 1
    diag_separator!(Dl, D_min, n)
    mul_diag_inv_left_adjoint_right!(G0l, Dl, Ul, n)
    mul_diag_left!(D_min, Tl, n)

    # part 2
    diag_separator!(Dr, D_min, n)
    mul_diag_inv_right!(Ur, Dr, n)
    mul_adjoint_left!(Ul, Tr, D_min)

    # part 3
    vmul!(Gll, G0l, Ur)
    vmul!(Gl0, Tl, Ul)
    add_right!(Gll, Gl0)

    # part 4
    qr_real_pivot!(Gl0, Gll, Dl, pivot, n)
    
    permute_columns_inverse!(Tr, Ur, pivot, n)
    permute_columns_inverse!(Ur, Ul, pivot, n)
    
    div_uppertriang_right!(Tr, Gll, n)
    div_uppertriang_right!(Ur, Gll, n)

    mul_diag_inv_right!(Tr, Dl, n)
    mul_diag_inv_right!(Ur, Dl, n)

    # part 5
    mul_adjoint_left!(Ul, Gl0, G0l, n)
    vmul!(Gll, Tr, Ul)
    vmul!(G0l, Ur, Ul)
    @turbo for i = 1:n, j = 1:n
        G0l[i,j] = - G0l[i,j]
    end

    mul_adjoint_left!(Ur, Gl0, Tl, n)
    vmul!(Gl0, Tr, Ur)
end

## 这是一个类型，在迭代的过程中需要被用到，里面存放的是迭代过程中需要用到的变量
struct CombinedGreensIterator_CDW_channel
    stack::dqmc_matrices_stack_real_qr
    hs_field::HSField_CDW

    U_tmp::Matrix{Float64}
    D_tmp::Vector{Float64}
    T_tmp::Matrix{Float64}

    G0l::Matrix{Float64}
    Gl0::Matrix{Float64}
    Gll::Matrix{Float64}
end

function CombinedGreensIterator(stack::dqmc_matrices_stack_real_qr, hs_field::HSField_CDW)
    U_tmp = Matrix{Float64}(I, dims, dims)
    D_tmp = ones(Float64, dims)
    T_tmp = Matrix{Float64}(I, dims, dims)

    G0l = Matrix{Float64}(I, dims, dims)
    Gl0 = Matrix{Float64}(I, dims, dims)
    Gll = Matrix{Float64}(I, dims, dims)

    CombinedGreensIterator_CDW_channel(stack, hs_field, U_tmp, D_tmp, T_tmp, G0l, Gl0, Gll)
end

## 这是一个迭代器，迭代得到不同τ(0,1,2,...,L)下对应的不等时格林函数G(τ,0), G(0,τ)和等时格林函数G(τ,τ)
@def greens_iterator_shortcuts begin
    Gll = it.Gll
    Gl0 = it.Gl0
    G0l = it.G0l
    
    U_tmp = it.Ur
    D_tmp = it.Dr
    T_tmp = it.Tr
end

@def stack_greens_iterator_shortcuts begin
    greens = it.stack.greens
    Ul = it.stack.Ul
    Dl = it.stack.Dl
    Tl = it.stack.Tl
    Ur = it.stack.Ur
    Dr = it.stack.Dr
    Tr = it.stack.Tr
    pivot = it.stack.pivot_tmp
end

@def stack_bmats_greens_iterator_shortcuts begin
    hopping_matrix_exp = it.stack.hopping_matrix_exp
    hopping_matrix_exp_inv = it.stack.hopping_matrix_exp_inv
    B_slice = it.stack.current_B
    B_slice_inv = it.stack.current_B_inv

    vector_tmp1 = it.stack.vector_tmp1
    matrix_tmp = it.stack.matrix_tmp1
end

# 可以写一个迭代器，但我不打算写
@inline function get_uneqlt_greens(stack::dqmc_matrices_stack_real_qr, hs_field::HSField_CDW,
    Gll::Matrix{Float64}, Gl0::Matrix{Float64}, G0l::Matrix{Float64},
    U_tmp::Matrix{Float64}, D_tmp::Vector{Float64}, T_tmp::Matrix{Float64},
    slice::Int64, dims::Int64)

    get_single_B_slices_modified!(hs_field, slice, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B, 
        stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv)
    vmul!(stack.matrix_tmp1, stack.current_B, stack.Ul)

    if stack.slice_rem[slice] == 0 # 做一次数值稳定
        mul_diag_right!(stack.matrix_tmp1, stack.Dl, dims)
        stack.current_range += 1

        # println("做了一次数值稳定性:", stack.current_range)
        copyto!(stack.Ur, stack.u_stack[stack.current_range])
        copyto!(stack.Dr, stack.d_stack[stack.current_range])
        copyto!(stack.Tr, stack.t_stack[stack.current_range])

        qr_real_pivot!(stack.Ul, stack.matrix_tmp1, stack.Dl, stack.pivot_tmp, dims)
        permute_rows!(T_tmp, stack.Tl, stack.pivot_tmp, dims)
        lmul!(UpperTriangular(stack.matrix_tmp1), T_tmp)

        copyto!(stack.Tl, T_tmp)
        copyto!(U_tmp, stack.Ul)
        copyto!(D_tmp, stack.Dl)

        get_uneqlt_green_by_qr!(Gll, Gl0, G0l, U_tmp, D_tmp, T_tmp, stack.Ur, stack.Dr, stack.Tr, 
            stack.vector_tmp1, stack.pivot_tmp, stack.dims)
    else
        copyto!(stack.Ul, stack.matrix_tmp1)
        get_uneqlt_green_by_wrap_up!(Gll, Gl0, G0l, stack.matrix_tmp1, stack.Ul, stack.current_B, stack.current_B_inv)
    end
    
end