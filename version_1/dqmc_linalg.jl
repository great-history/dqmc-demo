# LinearAlgebra(矩阵乘法)
function vmul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}
    @turbo for m in 1:size(A,1), n in 1:size(B,2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
    nothing
end

# adjoint_real_matrix!比直接adjoint是要快一点的.
@inline function adjoint_real_matrix!(B::Matrix{Float64}, A::Matrix{Float64}, n::Int = size(A,1))
    @turbo for i in 1:n
        for j in 1:n
            B[i,j] = A[j,i]
        end
    end 
    nothing
end

function mul_adjoint_left!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}, dims::Int = size(A, 1)) where {T <: Real}
    @turbo for m in 1:dims, n in 1:dims
        Cmn = zero(eltype(C))
        for k in 1:size(A, 1)
            Cmn += A[k,m] * B[k,n]
        end
        C[m,n] = Cmn
    end
    nothing
end

# 普通矩阵转置 与 对角矩阵 的乘积
function mul_adjoint_left!(C::Matrix{T}, A::Matrix{T}, D::Vector{T}, n::Int = size(A, 1)) where {T <: Real}
    @turbo for i in 1:n
        for j in 1:n
            C[i,j]= D[j] * A[j,i]
        end
    end
    nothing
end

function mul_adjoint_right!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}, n::Int=size(C, 1)) where {T <: Real}
    @turbo for i in 1:n, j in 1:n
        Cij = zero(eltype(C))
        for k in 1:n
            Cij += A[i, k] * B[j, k]
        end
        C[i,j] = Cij
    end
    nothing
end

function mul_adjoint_right!(C::Matrix{T}, A::Matrix{T}, D::Vector{T}, n::Int = size(A, 1)) where {T <: Real}
    @turbo for i in 1:n, j in 1:n
            C[i,j]= A[j,i] * D[i]
    end
    nothing
end

function mul_diag_inv_left_adjoint_right!(C::Matrix{T}, D::Vector{T}, A::Matrix{T}, n::Int = size(A, 1)) where {T <: Real}
    D_inv = zero(T)
    @turbo for i in 1:n
        D_inv = 1 / D[i]
        for j in 1:n
            C[i,j]= A[j,i] * D_inv
        end
    end
    nothing
end

# 上三角矩阵转置 与 对角矩阵的乘积
function mul_adjoint_uppertri_diag!(up_T::Matrix{Float64}, D::Vector{Float64}, n::Int = size(D,1))
    @inbounds for i in 1:n
        @turbo for j in 1:i
            up_T[i,j] = up_T[j,i] * D[j]
        end
    end
    nothing 
end

## @time (for i = 1:1000; copyto!(matrix_tmp, Tr);rmul!(matrix_tmp, Diagonal(Dr)); end) 0.079825 seconds
## @time (for i = 1:1000; copyto!(matrix_tmp, Tr);mul_diag_right!(greens, matrix_tmp, Dr); end) 0.061976 seconds
## 比较:@time mul_diag_right!(greens, Ur, D_max) 0.000061 seconds
##      @time mul_diag_right!(Ur, D_max) 0.000032 seconds 几乎是两倍的关系
function mul_diag_right!(C::Matrix{T}, A::Matrix{T}, D::Vector{T}, n::Int = size(D,1)) where {T <: Real}
    @turbo for i in 1:n, j in 1:n
        C[i,j] = A[i,j] * D[j]
    end
    nothing
end

function mul_diag_right!(A::Matrix{T}, D::Vector{T}, n::Int = size(D,1)) where {T <: Real}
    @turbo for i in 1:n, j in 1:n
        A[i,j] = A[i,j] * D[j]
    end
    nothing
end

function mul_diag_inv_right!(C::Matrix{T}, A::Matrix{T}, D::Vector{T}, n::Int = size(D,1)) where {T <: Real}
    D_inv = zero(T)
    @turbo for i in 1:n
        D_inv = 1 / D[i]
        for j in 1:n
            C[j,i] = A[j,i] * D_inv
        end
    end
    nothing
end

function mul_diag_inv_right!(A::Matrix{T}, D::Vector{T}, n::Int = size(D,1)) where {T <: Real}
    D_inv = zero(T)
    @turbo for i in 1:n
        D_inv = 1 / D[i]
        for j in 1:n
            A[j,i] = A[j,i] * D_inv
        end
    end
    nothing
end

function mul_diag_left!(C::Matrix{T}, D::Vector{T}, B::Matrix{T}, n::Int = size(D,1)) where {T <: Real}
    @turbo for i in 1:n, j in 1:n
        C[i,j] = D[i] * B[i, j]
    end
    nothing
end

function mul_diag_left!(D::Vector{T}, B::Matrix{T}, n::Int = size(D,1)) where {T <: Real}
    @turbo for j in 1:n
        for i in 1:n
        B[i,j] = D[i] * B[i, j]
        end
    end
    nothing
end

function mul_diag_inv_left!(C::Matrix{T}, D::Vector{T}, A::Matrix{T}, n::Int = size(D,1)) where {T <: Real}
    D_inv = zero(T)
    @turbo for i in 1:n
        D_inv = 1 / D[i]
        for j in 1:n
            C[i,j] = A[i,j] * D_inv
        end
    end
    nothing
end

function mul_diag_inv_left!(D::Vector{T}, A::Matrix{T}, n::Int = size(D,1)) where {T <: Real}
    D_inv = zero(T)
    @turbo for i in 1:n
        D_inv = 1 / D[i]
        for j in 1:n
            A[i,j] = A[i,j] * D_inv
        end
    end
    nothing
end

@inline function permute_rows!(M_tmp::Matrix{Float64}, M::Matrix{Float64}, pivot::Vector{Int}, n::Int)
    row = zero(eltype(n))
    @inbounds for i in 1:n
        row = pivot[i]
        @turbo for j in 1:n
            M_tmp[i, j] = M[row, j] 
        end
    end
    nothing
end

@inline function permute_rows_inverse!(M_tmp::Matrix{Float64}, M::Matrix{Float64}, pivot::Vector{Int}, n::Int=size(pivot, 1))
    row = zero(eltype(n))
    @inbounds for i in 1:n
        row = pivot[i]
        @turbo for j in 1:n
            M_tmp[row, j] = M[i, j] 
        end
    end
    nothing
end

@inline function permute_columns!(M_tmp::Matrix{Float64}, M::Matrix{Float64}, pivot::Vector{Int}, n::Int)
    column = zero(eltype(n))
    @inbounds for i in 1:n
        column = pivot[i]
        @turbo for j in 1:n
            M_tmp[j, column] = M[j, i] 
        end
    end
    nothing
end

@inline function permute_columns_inverse!(M_tmp::Matrix{Float64}, M::Matrix{Float64}, pivot::Vector{Int}, n::Int=size(pivot, 1))
    column = zero(eltype(n))
    @inbounds for i in 1:n
        column = pivot[i]
        @turbo for j in 1:n
            M_tmp[j, i] = M[j, column] 
        end
    end
    nothing
end

function mul_triang_left!(M_tmp::Matrix{C}, T::Matrix{C}, M::Matrix{C}) where {C <: Real}
    inner_dot = zero(eltype(T))
    n = size(M_tmp,1)
    @inbounds for i in 1:n
        # i 控制了矢量的维数
        @turbo for j in 1:n
            inner_dot = 0.0
            for k in i:n
                inner_dot += T[i, k] * M[k, j]
            end
            M_tmp[i, j] = inner_dot
        end
    end

    nothing
end

# @time lmul!(UpperTriangular(T), M); 0.000072 seconds (1 allocation: 16 bytes)
# @time mul_triang_left!(T, vector_tmp, M, 100); 0.000058 seconds , 这里测试的T,M都是100*100的矩阵
# @time (for i = 1:1000; mul_triang_left!(T, vector_tmp, M, 100); end) 0.039636 seconds
# @time (for i = 1:1000; lmul!(UpperTriangular(T), M); end) 0.029746 seconds (1000 allocations: 15.625 KiB)
function mul_triang_left!(T::Matrix{C}, vector_tmp::Vector{C}, M::Matrix{C}, n::Int = size(M_tmp,1)) where {C <: Real}
    inner_dot = zero(eltype(T))
    @inbounds for i in 1:n
        # i 控制了矢量的维数
        @turbo for k = i:n
            vector_tmp[k] = T[i, k]               
        end

        @turbo for j in 1:n
            inner_dot = 0.0
            for k in i:n
                inner_dot += vector_tmp[k] * M[k, j]
            end
            T[i, j] = inner_dot
        end
    end

    nothing
end

function mul_triang_right!(M_tmp::Matrix{C}, M::Matrix{C}, T::Matrix{C}) where {C <: Real}
    inner_dot = zero(eltype(T))
    n = size(M_tmp,1)

    @inbounds for j in 2:n
        # i 控制了矢量的维数
        @turbo for i in 1:n
            inner_dot = 0.0
            for k in 1:(j-1)
                inner_dot += M[i, k] * T[k, j]
            end
            M_tmp[i, j] = inner_dot
        end
    end

    @inbounds for i in 1:n
        if T[i,i] < 0
            @turbo for j in 1:n
                M_tmp[j, i] += (-M[j, i])
            end
        else
            @turbo for j in 1:n
                M_tmp[j, i] += M[j, i]
            end
        end
    end

    nothing
end

"""
    add_right!(A::Matrix{T}, B::Matrix{T}) where {T <: Real}

将两个矩阵相加 (A + B)ij = Aij + Bij

# Examples
```jldoctest
julia> A = rand(10,10); B = rand(10,10);
julia> @time A += B;    0.000007 seconds
julia> @time add_right!(A, B);  0.000004 seconds
3×3 Matrix{Float64}:
```r
@time (for i = 1:1000; A = rand(10,10); B = rand(10,10); A += B; end) 0.001035 seconds
@time (for i = 1:1000; A = rand(10,10); B = rand(10,10); add_right!(A, B); end)  0.000739 seconds
"""
function add_right!(A::Matrix{T}, B::Matrix{T}, n::Int = size(A,1)) where {T <: Real}
    @turbo for i in 1:n, j in 1:n
        A[i, j] = A[i, j] + B[i, j]
    end
    nothing
end

function add_right!(A::Matrix{T}, B::Vector{T}, n::Int = size(A,1)) where {T <: Real}
    @turbo for i in 1:n
        A[i, i] = A[i, i] + B[i]
    end
    nothing
end

"""
    div_uppertriang_right!(B::Matrix{Float64}, A::Matrix{Float64}, T::Matrix{Float64})

一个矩阵B与一个上三角矩阵的逆相乘 B = A * T^{-1}

# Examples
"""
function div_uppertriang_right!(B::Matrix{Float64}, A::Matrix{Float64}, T::Matrix{Float64}, n::Int = size(A,1))
    @inbounds begin
        # 单独作第一列
        x = zero(eltype(T))
        x = 1 / T[1,1]
        @turbo for i in 1:n
            B[i, 1] = A[i, 1] * x
        end

        # 第2列到第n列
        for j in 2:n
            @turbo for i in 1:n
                x = A[i, j]
                for k in 1:j-1
                    x -= B[i, k] * T[k, j]
                end
                B[i, j] = x / T[j, j]
            end
        end
    end
    nothing
end

function div_uppertriang_right!(A::Matrix{Float64}, T::Matrix{Float64}, n::Int = size(A,1))
    @inbounds begin
        # 单独作第一列
        @turbo for i in 1:n
            A[i, 1] = A[i, 1] / T[1, 1]
        end

        # 第2列到第n列
        for j in 2:n
            @turbo for i in 1:n
                x = A[i, j]
                for k in 1:j-1
                    x -= A[i, k] * T[k, j]
                end
                A[i, j] = x / T[j, j]
            end
        end
    end
    nothing
end

function div_uppertriang_left!(B::Matrix{Float64}, T::Matrix{Float64}, A::Matrix{Float64}, n::Int = size(A,1))
    @inbounds begin
        # 单独作第一列
        @turbo for i in 1:n
            B[n, i] = A[n, i] / T[n, n]
        end

        # 第2列到第n列
        for j in (n-1):-1:1
            @turbo for i in 1:n
                x = A[j, i]
                for k in (j+1):n
                    x -= B[k, i] * T[j, k]
                end
                B[j, i] = x / T[j, j]
            end
        end
    end
    nothing
end

function div_uppertriang_left!(T::Matrix{Float64}, A::Matrix{Float64}, n::Int = size(A,1))
    @inbounds begin
        # 单独作第一列
        @turbo for i in 1:n
            A[n, i] = A[n, i] / T[n, n]
        end

        # 第2列到第n列
        for j in (n-1):-1:1
            @turbo for i in 1:n
                x = A[j, i]
                for k in (j+1):n
                    x -= A[k, i] * T[j, k]
                end
                A[j, i] = x / T[j, j]
            end
        end
    end
    nothing
end

function inv_uppertriang!(T_inv::Matrix{Float64}, T::Matrix{Float64}, n::Int = size(T,1))
    @inbounds begin
        # 单独作第一列
        @turbo for i in 1:n
            for j in 1:n
                T_inv[i,j] = 0.0
            end
        end
        T_inv[1,1] = 1 / T[1,1]

        # 第2列到第n列
        for j in 2:n
            T_inv[j,j] = 1 / T[j,j]
            for i in 1:j-1  ## 计算 T_inv[i,j]
                x = zero(Float64)
                @turbo for k in i:j-1
                    x -= T_inv[i, k] * T[k, j]
                end
                T_inv[i, j] = x * T_inv[j,j]
            end
        end
    end
    nothing
end

# function diag_separator(D::Vector{Float64}, vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64}, dims::Int = size(D, 1))
#     @inbounds for i in eachindex(D)  ## TODO::函数屏障
#         D[i] > 1 ? (vector_tmp1[i] = 1 / D[i] ; vector_tmp2[i] = 1) : (vector_tmp2[i] = D[i] ; vector_tmp1[i] = 1)
#     end
# end
## 上面这种diag_separator很慢
function diag_separator!(D::Vector{Float64}, vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64}, n::Int = size(D,1))
    m = round(Int64, n / 2)
    left = 1
    right = n
    # 寻找分水岭
    @inbounds while !(D[m] >= 1.0 && D[m+1] <= 1.0) || !(left == right)
        
        if D[m] > 1.0
            left = m
            m = left + right
            m = round(Int64, m / 2)            
        else
            right = m
            m = left + right
            m = round(Int64, m / 2) - 1
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

## 比较:@time (for i = 1:1000; copyto!(Dr, stack.Dr); diag_separator!(Dr, stack.vector_tmp1, stack.vector_tmp2); end) 0.000188 seconds
##      @time (for i = 1:1000; copyto!(Dr, stack.Dr); diag_separator!(Dr, stack.vector_tmp1); end) 0.000160 seconds
function diag_separator!(D::Vector{Float64}, vector_tmp::Vector{Float64}, n::Int = size(D,1))
    m = round(Int64, n / 2)
    left = 1
    right = n
    # 寻找分水岭
    # println(D)
    @inbounds begin
        if D[1] <= 1.0
            m = 0
        elseif  D[end] >= 1.0
            m = n
        else
            while !(D[m] >= 1.0 && D[m+1] <= 1.0)
            
                if D[m] > 1.0
                    left = m
                    m = left + right
                    m = round(Int64, m / 2)            
                else
                    right = m
                    m = left + right
                    # 与上面情况不同的是这里还要减1，因为round(3/2)=2而不是1
                    m = round(Int64, m / 2) - 1
                end
            end
        end
    end
    @turbo for i = 1:m
        vector_tmp[i] = 1.0
    end
    
    @turbo for i = (m+1):n
        vector_tmp[i] = D[i]
        D[i] = 1.0
    end
end