include("dqmc_matrix_cdw_channel.jl")

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
### test: @time (for i = 1:1000000; get_single_B_slice!(hs_field, 1, stack.vector_tmp1, stack.matrix_tmp1, stack.current_B);end)  用时:0.011s左右，足够了
@inline function get_single_B_slice!(field::HSField_CDW, slice_index::Int, vector_tmp1::Vector{Float64}, hopping_matrix_exp::Matrix{Float64}, B_slice::Matrix{Float64})  # get_property 占了 25 / 60,我的建议是在for循环调用get_single_B_slice前先赋值两个矩阵
    @inbounds begin
        
        for i in eachindex(vector_tmp1)  # 12 / 60
            vector_tmp1[i] = field.conf[i, slice_index] < 0 ? field.elements[2] : field.elements[1]  # 千万不要把field.conf[:, slice_index]单独定义出来，这样会花不少时间
        end
        
        mul_diag_left!(B_slice, vector_tmp1, hopping_matrix_exp) # 9 / 60
        nothing
    end
end

### test: @time (for i = 1:1000000; get_single_B_slice_inv!(hs_field, 1, stack.hopping_matrix_exp_inv,  stack.vector_tmp2, stack.current_B_inv);end)
@inline function get_single_B_slice_inv!(field::HSField_CDW, slice_index::Int, hopping_matrix_exp_inv::Matrix{Float64}, vector_tmp2::Vector{Float64}, B_slice_inv::Matrix{Float64})
    @inbounds begin
        
        for i in eachindex(vector_tmp2)  # 12 / 60
            vector_tmp2[i] = field.conf[i, slice_index] < 0 ? field.elements[1] : field.elements[2]
        end
        
        mul_diag_right!(B_slice_inv, hopping_matrix_exp_inv, vector_tmp2)
        nothing
    end
end

### test: @time (for i = 1:100000; get_single_B_slices!(hs_field, 1, stack.vector_tmp1, stack.hopping_matrix_exp, stack.current_B, stack.vector_tmp2, stack.hopping_matrix_exp_inv, stack.current_B_inv); end)
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

        qr_real_pivot!(stack.u_stack[end-1], Ur, pivot, stack.d_stack[end - 1])
        permute_columns!(stack.t_stack[end-1], Ur, pivot, dims)

        # 把end - 2 到 2的情形算掉
        for range_idx in (stack.n_stacks-2):1
            copyto!(Ur, stack.u_stack[range_idx + 1]) ## TODO::规定Ur存放的是adjoint之后的U矩阵
    
            for slice_index in reverse(stack.ranges[range_idx])
                get_single_B_slice_modified!(hs_field, slice_index, eT, eV, B_slice)
                mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
                copyto!(Ur, matrix_tmp1)
            end
            
            mul_diag_right!(matrix_tmp1, Ur, stack.d_stack[range_idx + 1])
            qr_real_pivot!(stack.u_stack[range_idx], matrix_tmp1, pivot, stack.d_stack[range_idx])

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
        

        copyto!(stack.u_stack[1], I)
        stack.d_stack[1] .= one(Float64)
        copyto!(stack.t_stack[1], I)

        nothing
    end
end

## 分割线-----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
    diag_separator!(D::Vector{Float64}, n::Int = size(D,1))

将D中的大奇异值与小奇异值区分开来

# Examples
```jldoctest
julia> D = similar(stack.d_stack[2]);
julia> copyto!(D, stack.d_stack[2]);
julia> diag_separator(D, stack.vector_tmp1, stack.vector_tmp2);
3×3 Matrix{Float64}:
 1.0  2.0  3.0
 4.0  5.0  6.0
 7.0  8.0  9.0
```r
@time (for i = 1:1000; diag_separator(D, stack.vector_tmp1, stack.vector_tmp2); end) 0.000210 seconds
@time (for i = 1:1000; stack.vector_tmp1 = max.(D, 1);stack.vector_tmp2 = min.(D,1); end)  0.002255 seconds
"""
function diag_separator!(D::Vector{Float64}, vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64}, n::Int = size(D,1))
    m = round(Int64, n / 2)
    left = 1
    right = n
    # 寻找分水岭
    @inbounds while !(D[m] >= 1.0 && D[m+1] <= 1.0)
        
        if D[m] > 1.0
            left = m
            m = left + right
            m = round(Int64, m / 2)            
        else
            right = m
            m = left + right
            m = round(Int64, m / 2)
        end
    end

    @turbo for i = 1:n
        vector_tmp1[i] = 1.0
        vector_tmp2[i] = 1.0
    end

    @turbo for i = 1:m
        vector_tmp1[i] = D[i]
    end
    
    @turbo for i = (m+1):n
        vector_tmp2[i] = D[i]
    end

end

# 这个版本是全程不改变Ur/Dr/Tr的值,只能改变U_tmp/D_tmp/T_tmp/matrix_tmp/greens的值
function get_G00_right!(greens::Matrix{Float64}, Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64}, pivot::Matrix{Int64},
    U_tmp::Matrix{Float64}, D_max::Vector{Float64}, D_min::Vector{Float64}, T_tmp::Matrix{Float64}, matrix_tmp::Matrix{Float64})
    n = size(pivot, 1)
    # Dr ⟹ D_min + D_max
    diag_separator!(Dr, D_max, D_min)
    # greens = Ur * Diagonal(D_max)
    mul_diag_inv_right!(greens, Ur, D_max)
    
    # T_tmp = Tr^{dagger} * Diagonal(D_min) = (Trp * pivot)^{dagger} * Diagonal(D_min) = pivot^{dagger} Trp^{dagger} * Diagonal(D_min)
    # pivot:右乘时,将第i列换到第pivot[i]列;左乘时,将第pivot[i]行换到第i行, pivot的性质: pivot^{-1} = pivot^{dagger}
    # mul_adjoint_left!(T_tmp, Tr, D_min) 太慢了
    mul_adjoint_uppertri_diag!(matrix_tmp, Tr, D_min)
    permute_rows_inverse!(T_tmp, matrix_tmp, pivot, n)
    # matrix_tmp += U_tmp
    add_right!(T_tmp, greens)

    # qr分解: T_tmp ⟹ UDT ~ matrix_tmp * Digonal(D_max) * T_tmp
    qr_real_pivot!(matrix_tmp, T_tmp, D_max, pivot)
    # pivot : (U_tmp * P) * T^{-1} * ((1 / D_max) * matrix_tmp^{dagger})
    permute_columns_inverse!(U_tmp, greens, pivot, n)
    div_uppertriang_right!(greens, U_tmp, T_tmp, n)
    mul_diag_inv_right!(U_tmp, greens, D_max)
    mul_adjoint_right!(greens, U_tmp, matrix_tmp)

    nothing
end

# 这个版本是可以改变Ur/Dr/Tr的值