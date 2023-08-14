# 这个函数用来交换矩阵的列(初等变换矩阵之一),使得作Householder变换时奇异值是从小到大排列
# @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; qr_real_pivot!(U_test, input, pivot, squared_norms); end)
## input::待排序对象, pivot::次序, squared_norms::模长平方, vector_tmp::存放临时的数组方便交换, n::矩阵维数
function qr_real_pivot!(U::Matrix{C}, input::Matrix{C}, squared_norms::Vector{C}, pivot::Vector{Int}, n::Int = size(input,1)) where {C <: Real}
    @inbounds begin
        @turbo for i = 1:n
            pivot[i] = i
        end
    
        # 对n个k维矢量进行排序 与 householder 交叉进行        
        # 找到第1个向量
        current_index = 1
        tmp_pos = 0

        max_norm2 = 0.0
        tmp_norm2 = 0.0
        find_max_column!(input, squared_norms, pivot, tmp_norm2, max_norm2, current_index, tmp_pos, n)
        squared_norms[1] = house_holder!(input, squared_norms[1], tmp_norm2, max_norm2, 1, n)
        # 找第2到n-1个向量
        for i in 2:n-1
            max_norm2 = 0.0
            current_index = i
            
            find_max_column!(input, squared_norms, pivot, tmp_norm2, max_norm2, current_index, tmp_pos, n, i)
            squared_norms[i] = house_holder!(input, squared_norms[i], tmp_norm2, max_norm2, i, n)
        end
        # squared_norms[n] = abs(input[n,n])  # 实际上vector_β只有(n-1)个元素用的上, TODO::将squared_norms用vector_β取代
        # 至此已经得到上三角矩阵T(对角矩阵和归一上三角很容易得到了)

        # 接下来开始寻找幺正矩阵U
        copyto!(U, I)

        tmp_norm2 = input[n,n]
        input[n,n] = sign(tmp_norm2)
        squared_norms[n] = abs(tmp_norm2)
        
        for k = (n-1):-1:1
            tmp_norm2 = squared_norms[k]
            get_unitary!(input, U, max_norm2, tmp_norm2, k, n)
            
            # TODO::也许可以把这个去掉,只要我们心中知道它是零即可
            # @turbo for i in (k+1):n
            #     input[i,k] = 0.0
            # end

            tmp_norm2 = input[k,k]
            input[k,k] = sign(tmp_norm2)
            tmp_norm2 = abs(tmp_norm2)
            squared_norms[k] = tmp_norm2
            tmp_norm2 = 1 / tmp_norm2

            @turbo for j in k+1:n
                input[k,j] *= tmp_norm2
            end
            
        end
        nothing
    end
end

function qr_real_pivot!(::Val{true}, U::Matrix{C}, input::Matrix{C}, squared_norms::Vector{C}, pivot::Vector{Int}, n::Int = size(input,1)) where {C <: Real}
    @inbounds begin
        @turbo for i = 1:n
            pivot[i] = i
        end
    
        # 对n个k维矢量进行排序 与 householder 交叉进行        
        # 找到第1个向量
        current_index = 1
        tmp_pos = 0

        max_norm2 = 0.0
        tmp_norm2 = 0.0
        find_max_column!(input, squared_norms, pivot, tmp_norm2, max_norm2, current_index, tmp_pos, n)
        squared_norms[1] = house_holder!(input, squared_norms[1], tmp_norm2, max_norm2, 1, n)
        # 找第2到n-1个向量
        for i in 2:n-1
            max_norm2 = 0.0
            current_index = i
            
            find_max_column!(input, squared_norms, pivot, tmp_norm2, max_norm2, current_index, tmp_pos, n, i)
            squared_norms[i] = house_holder!(input, squared_norms[i], tmp_norm2, max_norm2, i, n)
        end
        # squared_norms[n] = abs(input[n,n])  # 实际上vector_β只有(n-1)个元素用的上, TODO::将squared_norms用vector_β取代
        # 至此已经得到上三角矩阵T(对角矩阵和归一上三角很容易得到了)

        # 接下来开始寻找幺正矩阵U
        copyto!(U, I)

        tmp_norm2 = input[n,n]
        input[n,n] = sign(tmp_norm2)
        squared_norms[n] = abs(tmp_norm2)
        
        for k = (n-1):-1:1
            tmp_norm2 = squared_norms[k]
            get_unitary!(input, U, max_norm2, tmp_norm2, k, n)
            
            # TODO::也许可以把这个去掉,只要我们心中知道它是零即可
            @turbo for i in (k+1):n
                input[i,k] = 0.0
            end

            tmp_norm2 = input[k,k]
            input[k,k] = sign(tmp_norm2)
            tmp_norm2 = abs(tmp_norm2)
            squared_norms[k] = tmp_norm2
            tmp_norm2 = 1 / tmp_norm2

            @turbo for j in k+1:n
                input[k,j] *= tmp_norm2
            end
            
        end
        nothing
    end
end

@inline function find_max_column!(input::Matrix{Float64}, squared_norms::Vector{Float64}, pivot::Vector{Int}, tmp_norm2::Float64, max_norm2::Float64, current_index::Int, tmp_pos::Int, n::Int) # true 代表就是对第一个向量
    for i in 1:n
        @turbo for j = 1:n
            tmp_norm2 += abs2(input[j,i])
        end
        
        squared_norms[i] = tmp_norm2
        if tmp_norm2 > max_norm2
            max_norm2 = tmp_norm2
            current_index = i
        end
        tmp_norm2 = 0.0
    end
    
    if current_index != 1
        pivot[1] = pivot[current_index]
        pivot[current_index] = 1
        
        ## 下面这样会比较慢, 建议用@turbo来进行加速
        # vector_tmp = input[:, current_index]
        # input[:, current_index] = input[:, 1]
        # input[:, 1] = vector_tmp
        @turbo for j in 1:n
            tmp_norm2 = input[j, current_index]
            input[j, current_index] = input[j, 1]
            input[j, 1] = tmp_norm2
        end
        squared_norms[current_index] = squared_norms[1]
    end
    squared_norms[1] = sqrt(max_norm2)
    nothing
end

@inline function find_max_column!(input::Matrix{Float64}, squared_norms::Vector{Float64}, pivot::Vector{Int}, tmp_norm2::Float64, max_norm2::Float64, current_index::Int, tmp_pos::Int, n::Int, i::Int) # false 代表就是对第一个向量
    for k in i:n
        tmp_norm2 = abs2(input[i-1, k])
        squared_norms[k] -= tmp_norm2

        if squared_norms[k] > max_norm2
            max_norm2 = squared_norms[k]
            current_index = k
        end
    end

    if current_index != i
        tmp_pos = pivot[current_index]
        pivot[current_index] = pivot[i]
        pivot[i] = tmp_pos

        squared_norms[current_index] = squared_norms[i]

        ## 下面这样会比较慢, 建议用@turbo来进行加速
        # vector_tmp = input[:, current_index]
        # input[:, current_index] = input[:, i]
        # input[:, i] = vector_tmp
        @turbo for j in 1:n
            tmp_norm2 = input[j, current_index]
            input[j, current_index] = input[j, i]
            input[j, i] = tmp_norm2
        end
    end
    # 把模长平方重新算一遍，因为上面直接减去一个数可能会有误差(应该至少有e-13的量级)
    max_norm2 = 0.0
    for j in i:n  ## TODO:去掉@turbo看看有没有加速
        max_norm2 += abs2(input[j,i])
    end
    squared_norms[i] = sqrt(max_norm2)
end

# 仅对实数方阵进行操作 TODO::对非方阵如何进行操作???
@inline function house_holder!(input::Matrix{Float64}, norm::Float64, inner_dot::Float64, β::Float64, i::Int, n::Int = size(input, 1))
    @inbounds begin
        x1 = input[i,i]
        if iszero(norm)
            return zero(x1)
        end

        v = LinearAlgebra.copysign(norm, x1)
        x1 += v
        input[i,i] = -v # 为对角元
        ## TODO::回答一个问题:是先加后除还是先除后加?? 这里采用先除后加(虽然会有点误差)
        # 对第一列向量进行操作
        inner_dot = 1 / x1
        @turbo for j in (i+1):n ## TODO::去掉@turbo看看有没有加速
            input[j, i] *= inner_dot
        end

        # 对第二列到第n列向量进行操作  ##TODO::能不能把精度再提高一下，用先加后除??
        β = x1 / v
        for j in (i+1):n
            inner_dot = input[i, j]
            @turbo for k in (i+1):n
                inner_dot += input[k,i] * input[k, j]
            end
            inner_dot *= β
            inner_dot = - inner_dot
        
            input[i, j] += inner_dot
            @turbo for k in (i+1):n
                input[k, j] += inner_dot * input[k,i]
            end
        end

        return β
    end
end

# 仅对实数方阵进行操作 TODO::对非方阵如何进行操作???
# @inline function house_holder_test!(input::Matrix{Float64}, norm::Float64, inner_dot::Float64, β::Float64, i::Int, n::Int = size(input, 1))
#     @inbounds begin
#         x1 = input[i,i]
#         if iszero(norm)
#             return zero(x1)
#         end

#         v = LinearAlgebra.copysign(norm, x1)
#         x1 += v
        
#         # 对第二列到第n列向量进行操作  ##先加后除
#         β = x1 / v
#         for j in (i+1):n
#             inner_dot = 0.0
#             @turbo for k in (i+1):n
#                 inner_dot += input[k,i] * input[k, j]
#             end
#             inner_dot = inner_dot / x1
#             inner_dot += input[i, j]

#             inner_dot = - inner_dot * β
        
#             input[i, j] += inner_dot
#             @turbo for k in (i+1):n
#                 input[k, j] += inner_dot * input[k,i]
#             end
#         end

#         # 对第一列向量进行操作
#         input[i,i] = -v # 为对角元
#         inner_dot = 1 / x1
#         @turbo for j in (i+1):n ## TODO::去掉@turbo看看有没有加速
#             input[j, i] *= inner_dot
#         end

#         return β
#     end
# end

## U = H_k U
@inline function get_unitary!(input::Matrix{Float64}, U_test::Matrix{Float64}, max_norm2::Float64, tmp_norm2::Float64, k::Int, n::Int)
    # tmp_norm2::存放β
    ## 首先计算U[k:n,k]
    ## 然后计算U[k, (k+1):n]
    ## 这两个for-loop放在一起
    U_test[k,k] -= tmp_norm2
    @turbo for i in (k+1):n ## TODO::修改一下@turbo看看有没有加速
        max_norm2 = -(input[i,k] * tmp_norm2)
        U_test[i,k] += max_norm2
        for j in (k+1):n
            U_test[k,j] += max_norm2 * U_test[i, j]
        end
    end
    
    # @turbo for i in (k+1):n
    #     tmp_norm2 = input[i,k]
    #     input[i,k] -= tmp_norm2
    #     for j in (k+1):n
    #         U_test[i,j] += tmp_norm2 * U_test[k,j]  # 是用+号而不是用-号
    #     end
    # end

    @turbo for i in (k+1):n ## TODO::修改一下@turbo看看有没有加速  22/116
        for j in (k+1):n
            U_test[i,j] += input[i,k] * U_test[k,j]  # 是用+号而不是用-号  ## TODO::修改一下这个loop
        end
    end

    ## 注意: 下面这种@turbo算出来的结果是错误的，可能跟@turbo的用法有关
    # @turbo for i in (k+1):n
    #     for j in (k+1):n
    #         U_test[i,j] += input[i,k] * U_test[k,j]  # 是用+号而不是用-号
    #     end
    #     input[i,k] = 0.0
    # end

end

## not apply pivot
@inline function get_diagonal_and_uppertriagular(input::Matrix{Float64}, D::Vector{Float64}, tmp_norm2::Float64, n::Int)
    tmp_norm2 = input[n,n]
    input[n,n] = sign(tmp_norm2)
    D[n] = abs(tmp_norm2)

    @turbo for j in 1:n-1 
        input[n,j] = 0.0
    end

    for i in n-1:-1:2
        tmp_norm2 = input[i,i]
        input[i,i] = sign(tmp_norm2)
        tmp_norm2 = abs(tmp_norm2)
        D[i] = tmp_norm2
        tmp_norm2 = 1 / tmp_norm2

        @turbo for j in 1:i-1 
            input[i,j] = 0.0
        end

        @turbo for j in i+1:n
            input[i,j] *= tmp_norm2
        end
    end

    tmp_norm2 = input[1,1]
    input[1,1] = sign(tmp_norm2)
    tmp_norm2 = abs(tmp_norm2)
    D[1] = tmp_norm2
    tmp_norm2 = 1 / tmp_norm2

    @turbo for j in 2:n
        input[1,j] *= tmp_norm2
    end
end

# apply pivot
@inline function get_diagonal_and_uppertriagular(input::Matrix{Float64}, D::Vector{Float64}, pivot::Vector{Int}, tmp_norm2::Float64, n::Int)
    tmp_norm2 = input[1,1]
    input[1,1] = sign(tmp_norm2)
    tmp_norm2 = abs(tmp_norm2)
    D[1] = tmp_norm2
    tmp_norm2 = 1 / tmp_norm2

    @turbo for j in 2:n
        input[1,j] *= tmp_norm2
    end

    for i in 2:n-1
        tmp_norm2 = input[i,i]
        input[i,i] = sign(tmp_norm2)
        tmp_norm2 = abs(tmp_norm2)
        D[i] = tmp_norm2
        tmp_norm2 = 1 / tmp_norm2

        @turbo for j in 1:i-1 
            input[i,j] = 0.0
        end

        @turbo for j in i+1:n
            input[i,j] *= tmp_norm2
        end
    end

    tmp_norm2 = input[n,n]
    input[n,n] = sign(tmp_norm2)
    D[n] = abs(tmp_norm2)

    @turbo for j in 1:n-1 
        input[n,j] = 0.0
    end
end

function get_triangular(input::Matrix{Float64}, n::Int)
    @inbounds for k in 1:n-1
        @turbo for i in (k+1):n
            input[i,k] = 0.0
        end
    end
end