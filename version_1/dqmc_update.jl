## CDW channel of Hubbard model
## CDW channel的特定: G↑↑ = G↓↓, G↓↑ = G↑↓ = 0, 即我们只需要计算自旋↑或↓即可
include("dqmc_eqlt_greens.jl")
include("dqmc_matrix_cdw_channel.jl")
include("dqmc_helper.jl")

function thermalization_sweeps(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr, thermal_sweeps::Int64 = 1000)
    @inbounds begin
        for sweep_idx in 1:thermal_sweeps

            sweep_up_local_modified!(hs_field, stack)
            sweep_down_local_modified!(hs_field, stack)
        
        end
    end
end

function measurement_sweeps()

end

@def stack_green_shortcuts begin
    greens = stack.greens
    Ul = stack.Ul
    Dl = stack.Dl
    Tl = stack.Tl
    Ur = stack.Ur
    Dr = stack.Dr
    Tr = stack.Tr
    pivot = stack.pivot_tmp
end

@def stack_bmats_shortcuts begin
    hopping_matrix_exp = stack.hopping_matrix_exp
    hopping_matrix_exp_inv = stack.hopping_matrix_exp_inv
    B_slice = stack.current_B
    B_slice_inv = stack.current_B_inv
end

function sweep_up_local!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    accepted = Int64(0)
    dims = stack.dims
    n_slices = stack.n_slices
    current_range = Int64(1)
    last_slice = Int64(1)
    diag_element = Float64(0)

    greens = stack.greens
    Ul = stack.Ul
    Dl = stack.Dl
    Tl = stack.Tl

    vector_tmp1 = stack.vector_tmp1
    vector_tmp2 = stack.vector_tmp2
    matrix_tmp = stack.matrix_tmp1

    hopping_matrix_exp = stack.hopping_matrix_exp
    hopping_matrix_exp_inv = stack.hopping_matrix_exp_inv
    B_slice = stack.current_B
    B_slice_inv = stack.current_B_inv

    @inbounds begin
        # 单独计算一下第一个range
        last_slice = last(stack.ranges[1])
        for current_slice in 1:last_slice
            get_single_B_slices!(hs_field, current_slice, vector_tmp1, hopping_matrix_exp, B_slice,
                vector_tmp2, hopping_matrix_exp_inv, B_slice_inv)

            # wrap up: G(slice, slice) = B(slice) * G(slice-1, slice-1) * B^{-1}(slice)
            # 这一步我们都用wrap up,不采用numerical stable
            get_eqlt_green_by_wrap_up!(greens, B_slice, B_slice_inv, matrix_tmp)

            # update greens function and B_slice
            for current_site in 1:stack.dims
                propose_local(hs_field, greens, current_site, current_slice)
                if hs_field.ratio > 1 || rand() < hs_field.ratio
                    accept_or_reject(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
                    # 同时更新B矩阵
                    diag_element = hs_field.conf[i, slice_index] < 0 ? field.elements[2] : field.elements[1]
                    update_B_slice!(B_slice, hopping_matrix_exp, diag_element, current_site, dims)
                    accepted += 1
                end
            end

            # 计算Ur = B(l) * Ur
            vmul!(matrix_tmp, B_slice, Ul)
            copyto!(Ul, matrix_tmp)
            # 根据slice_rem是否为零来进行数值稳定与否
            if current_slice == last_slice
                mul_diag_right!(matrix_tmp, Dl, dims)

                current_range += 1
                copyto!(stack.Ur, stack.u_stack[current_range])
                copyto!(stack.Dr, stack.d_stack[current_range])
                copyto!(stack.Tr, stack.t_stack[current_range])

                qr_real_pivot!(Ul, matrix_tmp, Dl, stack.pivot_tmp, dims, Val(true))
                stack.u_stack[current_range] = Ul
                permute_columns!(stack.t_stack[2], matrix_tmp, stack.pivot_tmp, dims)

                copyto!(stack.current_range, current_range)
                copyto!(stack.u_stack[current_range], Ul)
                copyto!(stack.d_stack[current_range], Dl)
                copyto!(stack.t_stack[current_range], Tl)

                ## 更新格林函数
                get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, stack.Ur, stack.Dr, stack.Tr, vector_tmp1, stack.pivot_tmp, dims)
            end

            # TODO::保存文件(不用每个slice都要停下来保存,每隔比如10个slice保存一下)
        end

        if stack.n_stacks == 1
            nothing
        end

        last_slice = last(stack.ranges[2])
        for current_slice in first(stack.ranges[2]):n_slices
            get_single_B_slices!(hs_field, current_slice, vector_tmp1, hopping_matrix_exp, B_slice,
                vector_tmp2, hopping_matrix_exp_inv, B_slice_inv)

            # wrap up: G(slice, slice) = B(slice) * G(slice-1, slice-1) * B^{-1}(slice)
            # 这一步我们都用wrap up,不采用numerical stable
            get_eqlt_green_by_wrap_up!(greens, B_slice, B_slice_inv, matrix_tmp)

            # update greens function and B_slice
            for current_site in 1:stack.dims
                propose_local(hs_field, greens, current_site, current_slice)
                if hs_field.ratio > 1 || rand() < hs_field.ratio
                    accept_or_reject(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
                    # 同时更新B矩阵
                    diag_element = hs_field.conf[i, slice_index] < 0 ? field.elements[2] : field.elements[1]
                    update_B_slice!(B_slice, hopping_matrix_exp, diag_element, current_site, dims)
                    accepted += 1
                end
            end

            # 计算Ur = B(l) * Ur
            vmul!(matrix_tmp, B_slice, Ul)
            copyto!(Ul, matrix_tmp)
            # 根据slice_rem是否为零来进行数值稳定与否
            if current_slice == last_slice
                mul_diag_right!(matrix_tmp, Dl, dims)

                current_range += 1
                copyto!(stack.Ur, stack.u_stack[current_range])
                copyto!(stack.Dr, stack.d_stack[current_range])
                copyto!(stack.Tr, stack.t_stack[current_range])

                qr_real_pivot!(Ul, matrix_tmp, Dl, stack.pivot_tmp, dims)
                stack.u_stack[current_range] = Ul

                permute_rows!(greens, Tl, stack.pivot_tmp, dims)
                lmul!(UpperTriangular(matrix_tmp), greens)
                copyto!(Tl, greens)

                copyto!(stack.current_range, current_range)
                copyto!(stack.u_stack[current_range], Ul)
                copyto!(stack.d_stack[current_range], Dl)
                copyto!(stack.t_stack[current_range], Tl)

                ## 更新格林函数
                get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, stack.Ur, stack.Dr, stack.Tr, vector_tmp1, stack.pivot_tmp, dims)

                last_slice = last(stack.ranges[current_range])
            end

            # TODO::保存文件(不用每个slice都要停下来保存,每隔比如10个slice保存一下)
        end

        accepted_ratio = Float64(accepted) / (2 * dims)
    end
end

function sweep_down_local!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    accepted = Int64(0)
    dims = stack.dims
    n_slices = stack.n_slices
    current_range = stack.n_stacks
    last_slice = Int64(1)
    diag_element = Float64(0)

    greens = stack.greens
    pivot = stack.pivot_tmp
    Ur = stack.Ur
    Dr = stack.Dr
    Tr = stack.Tr
    Ul = stack.Ul
    Dl = stack.Dl
    Tl = stack.Tl

    vector_tmp1 = stack.vector_tmp1
    vector_tmp2 = stack.vector_tmp2
    matrix_tmp = stack.matrix_tmp1

    hopping_matrix_exp = stack.hopping_matrix_exp
    hopping_matrix_exp_inv = stack.hopping_matrix_exp_inv
    B_slice = stack.current_B
    B_slice_inv = stack.current_B_inv

    @inbounds begin
        # 单独计算一下第n_slices的情况, 因为已经在sweep up的最后计算过了

        for current_slice in (n_slices-1):-1:1
            # 根据slice_rem是否为零来进行数值稳定与否
            if stack.slice_rem[current_slice] == 0
                mul_diag_right!(matrix_tmp, Dr, dims)

                current_range -= 1
                copyto!(Ul, stack.u_stack[current_range])
                copyto!(Dl, stack.d_stack[current_range])
                copyto!(Tl, stack.t_stack[current_range])

                qr_real_pivot!(Ur, matrix_tmp, Dr, pivot, dims)

                permute_rows!(greens, Tr, pivot, dims)
                lmul!(UpperTriangular(matrix_tmp), greens)
                copyto!(Tr, greens)

                copyto!(stack.current_range, current_range)
                copyto!(stack.u_stack[current_range], Ur)
                copyto!(stack.d_stack[current_range], Dr)
                copyto!(stack.t_stack[current_range], Tr)

                ## 更新格林函数
                get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, Ul, Dl, Tl,
                    vector_tmp1, vector_tmp2, matrix_tmp,
                    pivot, dims)
            else
                # wrap down: G(slice, slice) = B^{-1}(slice) * G(slice+1, slice+1) * B(slice)
                get_eqlt_green_by_wrap_down!(greens, B_slice_inv, B_slice, matrix_tmp)
            end

            # update greens function and B_slice
            for current_site in 1:stack.dims
                propose_local(hs_field, greens, current_site, current_slice)
                if hs_field.ratio > 1 || rand() < hs_field.ratio
                    accept_or_reject(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
                    # 同时更新B矩阵
                    diag_element = hs_field.conf[i, slice_index] < 0 ? field.elements[2] : field.elements[1]
                    accepted += 1
                end
            end

            # 与sweep up的不同之处:get B slice的计算和Ur的更新放在最后
            get_single_B_slices!(hs_field, current_slice, vector_tmp1, hopping_matrix_exp, B_slice,
                vector_tmp2, hopping_matrix_exp_inv, B_slice_inv)
            # 计算Ur = B(l) * Ur
            mul_adjoint_left!(matrix_tmp, B_slice, Ur)  # 与sweep up的不同之处
            copyto!(Ur, matrix_tmp)

            # TODO::保存文件(不用每个slice都要停下来保存,每隔比如10个slice保存一下)
            # stack.current_slice = current_slice
        end
    end

    accepted_ratio = Float64(accepted) / (2 * dims)
end

function sweep_local!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    # sweep up

    # sweep down

end

@inline function propose_local(hs_field::HSField_CDW, greens_old::Matrix{Float64}, current_site::Int64, current_slice::Int64)
    hs_field.Δ = hs_field.conf[current_site, current_slice] == 1 ? hs_field.Δs[1] : hs_field.Δs[2]
    hs_field.boson_ratio = hs_field.conf[current_site, current_slice] == 1 ? hs_field.boson_ratios[1] : hs_field.boson_ratios[2]

    hs_field.det_ratio = 1 + (1 - greens_old[current_site, current_site]) * hs_field.Δ
    hs_field.ratio = hs_field.boson_ratio * (hs_field.det_ratio)^2
end

@inline function accept_or_reject(hs_field::HSField_CDW, greens_old::Matrix{Float64}, vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64},
    current_site::Int64, current_slice::Int64, n::Int64)
    @inbounds begin
        hs_field.conf[current_site, current_slice] = -hs_field.conf[current_site, current_slice]

        x = hs_field.Δ / hs_field.det_ratio

        @turbo for i = 1:n
            vector_tmp1[i] = greens_old[i, current_site] * x
            vector_tmp2[i] = greens_old[current_site, i]
        end

        @turbo for i = 1:n
            for j = 1:n
                greens_old[i, j] += vector_tmp1[i] * vector_tmp2[j]
            end
            greens_old[i, current_site] -= vector_tmp1[i]
        end
        nothing
    end

end

@inline function update_B_slice!(B_slice::Matrix{Float64}, hopping_matrix_exp::Matrix{Float64}, diag_element::Float64, current_site::Int64, dims::Int64)
    @turbo for i in 1:dims
        B_slice[current_site, i] = hopping_matrix_exp[current_site, i] * diag_element
    end
end

# ----------------------------------------------------------------------------------------------------------------------
function sweep_local_modified!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    # sweep up
    sweep_up_local_modified!(hs_field, stack)
    # println(hs_field.accepted_ratio)
    # sweep down
    hs_field.accepted_ratio += sweep_down_local_modified!(hs_field, stack)
    hs_field.accepted_ratio /= 2
end

function sweep_up_local_modified!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    dims = stack.dims
    n_slices = stack.n_slices

    @stack_green_shortcuts

    @stack_bmats_shortcuts

    vector_tmp1 = stack.vector_tmp1
    vector_tmp2 = stack.vector_tmp2
    matrix_tmp = stack.matrix_tmp1
    
    accepted = Int64(0)

    sweep_up_local_modified!(hs_field, stack, greens,
            Ul, Dl, Tl, Ur, Dr, Tr, 
            vector_tmp1, vector_tmp2, matrix_tmp,
            hopping_matrix_exp, hopping_matrix_exp_inv, B_slice, B_slice_inv,
            pivot, n_slices, accepted, dims)
end

function sweep_up_local_modified!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr, greens::Matrix{Float64},
    Ul::Matrix{Float64}, Dl::Vector{Float64}, Tl::Matrix{Float64},
    Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64},
    vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64}, matrix_tmp::Matrix{Float64},
    hopping_matrix_exp::Matrix{Float64}, hopping_matrix_exp_inv::Matrix{Float64},
    B_slice::Matrix{Float64}, B_slice_inv::Matrix{Float64},
    pivot::Vector{Int64}, n_slices::Int64, accepted::Int64, dims::Int64)

    @inbounds for current_slice in 1:n_slices  # 对每一层的B(l)进行更新
        # flush(stdout) ## 清空一下缓存区,好像对于大体系，一定要加这一句话，否则会崩溃???
        # 更新第current_slice层,不更新B矩阵
        accepted = update_one_slice_modified!(hs_field, greens, vector_tmp1, vector_tmp2, accepted, current_slice, dims)
        # println(accepted)
        # 得到B(current_slice)
        get_single_B_slices_modified!(hs_field, current_slice, hopping_matrix_exp, vector_tmp1, B_slice,
            hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)

        # 计算Ul = B(current_slice) * Ul
        vmul!(matrix_tmp, B_slice, Ul)
        copyto!(Ul, matrix_tmp)

        # 进行 wrap up 还是 数值稳定 ??
        if stack.slice_rem[current_slice] == 0
            stack.current_range = stack.current_range + 1
            sweep_up_qr!(stack, greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, matrix_tmp, pivot, stack.current_range, dims)
            # TODO::保存文件(不用每个slice都要停下来保存,比如每作一次数值稳定性之后保存一下文件)
        else
        #     # wrap up: G(slice, slice) = B(slice) * G(slice-1, slice-1) * B^{-1}(slice)
        #     # 这一步我们都用wrap up,不采用numerical stable
            get_eqlt_green_by_wrap_up!(greens, B_slice, B_slice_inv, matrix_tmp)
        end

    end

    ## 当最后一片slice运行完之后,Ur/Dr/Tr应该被赋值为单位元素，以便接下来sweep down使用
    copyto!(Ur, I)
    Dr .= one(Float64)
    copyto!(Tr, I)

    copyto!(stack.u_stack[end], I)
    stack.d_stack[end] .= one(Float64)
    copyto!(stack.t_stack[end], I)

    hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)
    nothing
end

function sweep_up_qr!(stack::dqmc_matrices_stack_real_qr, greens::Matrix{Float64}, 
    Ul::Matrix{Float64}, Dl::Vector{Float64}, Tl::Matrix{Float64},
    Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64},
    vector_tmp1::Vector{Float64}, matrix_tmp::Matrix{Float64}, pivot::Vector{Int64}, current_range::Int64, dims::Int64)

    mul_diag_right!(matrix_tmp, Dl, dims)
    
    copyto!(Ur, stack.u_stack[current_range])
    copyto!(Dr, stack.d_stack[current_range])
    copyto!(Tr, stack.t_stack[current_range])
    
    qr_real_pivot!(Ul, matrix_tmp, Dl, pivot, dims)
    permute_rows!(stack.t_stack[current_range], Tl, pivot, dims)
    lmul!(UpperTriangular(matrix_tmp), stack.t_stack[current_range])  # TODO::能不能自己写个更快的函数??

    copyto!(Tl, stack.t_stack[current_range])
    copyto!(stack.u_stack[current_range], Ul)
    copyto!(stack.d_stack[current_range], Dl)
    ## 数值稳定地计算格林函数
    get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, pivot, dims)

    copyto!(Ul, stack.u_stack[current_range])
    copyto!(Dl, stack.d_stack[current_range])
    copyto!(Tl, stack.t_stack[current_range])
end

function sweep_down_local_modified!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr)
    dims = stack.dims
    n_slices = stack.n_slices
    diag_element = Float64(0)

    @stack_green_shortcuts

    @stack_bmats_shortcuts

    vector_tmp1 = stack.vector_tmp1
    vector_tmp2 = stack.vector_tmp2
    matrix_tmp = stack.matrix_tmp1

    accepted = Int64(0)
    diag_element = Float64(0)
    sweep_down_local_modified!(hs_field, stack, greens,
            Ul, Dl, Tl, Ur, Dr, Tr, 
            vector_tmp1, vector_tmp2, matrix_tmp,
            hopping_matrix_exp, hopping_matrix_exp_inv, B_slice, B_slice_inv,
            pivot, n_slices, accepted, dims, diag_element)
end

function sweep_down_local_modified!(hs_field::HSField_CDW, stack::dqmc_matrices_stack_real_qr, greens::Matrix{Float64},
    Ul::Matrix{Float64}, Dl::Vector{Float64}, Tl::Matrix{Float64},
    Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64},
    vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64}, matrix_tmp::Matrix{Float64},
    hopping_matrix_exp::Matrix{Float64}, hopping_matrix_exp_inv::Matrix{Float64},
    B_slice::Matrix{Float64}, B_slice_inv::Matrix{Float64},
    pivot::Vector{Int64}, n_slices::Int64, accepted::Int64, dims::Int64, diag_element::Float64)
    @inbounds for current_slice in n_slices:-1:1  # 对每一层的B(l)进行更新
        # flush(stdout) ## 清空一下缓存区,好像对于大体系，一定要加这一句话，否则会崩溃
        # wrap down: G(slice, slice) = B(slice) * G(slice, slice) * B^{-1}(slice)
        # 这一步我们都用wrap up,不采用numerical stable
        get_eqlt_green_by_wrap_down!(greens, B_slice_inv, B_slice, matrix_tmp)

        # 更新第current_slice层,不更新B矩阵
        accepted = update_one_slice_modified!(hs_field, greens, vector_tmp1, vector_tmp2, B_slice, hopping_matrix_exp,
            diag_element, accepted, current_slice, dims)

        # 计算Ur = {B(current_slice)}^{†} * Ul
        mul_adjoint_left!(matrix_tmp, B_slice, Ur)
        copyto!(Ur, matrix_tmp)

        # 是否进行数值稳定
        if stack.slice_rem[current_slice] == 1
            stack.current_range = stack.current_range - 1
            sweep_down_qr!(stack, greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, matrix_tmp, pivot, stack.current_range, dims)
        end

        # 得到old config下的B(current_slice-1)
        get_single_B_slices_modified!(hs_field, current_slice - 1, hopping_matrix_exp, vector_tmp1, B_slice,
            hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)

        # TODO::保存文件(不用每个slice都要停下来保存,比如每作一次数值稳定性之后保存一下文件)
    end
    ## 当最后一片slice运行完之后,Ul/Dl/Tl应该被赋值为单位元素，以便接下来sweep up使用
    copyto!(Ul, I)
    Dl .= one(Float64)
    copyto!(Tl, I)

    copyto!(stack.u_stack[1], I)
    stack.d_stack[1] .= one(Float64)
    copyto!(stack.t_stack[1], I)

    hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)
    nothing
end

function sweep_down_qr!(stack::dqmc_matrices_stack_real_qr, greens::Matrix{Float64}, 
    Ul::Matrix{Float64}, Dl::Vector{Float64}, Tl::Matrix{Float64},
    Ur::Matrix{Float64}, Dr::Vector{Float64}, Tr::Matrix{Float64},
    vector_tmp1::Vector{Float64}, matrix_tmp::Matrix{Float64}, pivot::Vector{Int64}, current_range::Int64, dims::Int64)

    mul_diag_right!(matrix_tmp, Dr, dims)

    # current_range -= 1
    copyto!(Ul, stack.u_stack[current_range])
    copyto!(Dl, stack.d_stack[current_range])
    copyto!(Tl, stack.t_stack[current_range])

    qr_real_pivot!(Ur, matrix_tmp, Dr, pivot, dims)
    permute_rows!(stack.t_stack[current_range], Tr, pivot, dims)
    lmul!(UpperTriangular(matrix_tmp), stack.t_stack[current_range])  # TODO::能不能自己写个更快的函数??

    copyto!(Tr, stack.t_stack[current_range])
    copyto!(stack.u_stack[current_range], Ur)
    copyto!(stack.d_stack[current_range], Dr)

    ## 数值稳定地计算格林函数
    get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, pivot, dims)
    copyto!(Ur, stack.u_stack[current_range])
    copyto!(Dr, stack.d_stack[current_range])
    copyto!(Tr, stack.t_stack[current_range])
    # stack.current_range = current_range
end

@inline function update_one_slice_modified!(hs_field::HSField_CDW, greens::Matrix{Float64}, vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64},
    accepted::Int64, current_slice::Int64, dims::Int64)
    # update greens function and B_slice
    for current_site in 1:dims
        propose_local(hs_field, greens, current_site, current_slice)
        if hs_field.ratio > 1 || rand() < hs_field.ratio
            accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
            # 不更新B矩阵
            accepted += 1
        end
    end
    return accepted
end

@inline function update_one_slice_modified!(hs_field::HSField_CDW, greens::Matrix{Float64}, vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64},
    B_slice::Matrix{Float64}, hopping_matrix_exp::Matrix{Float64}, diag_element::Float64, accepted::Int64, current_slice::Int64, dims::Int64)
    # update greens function and B_slice
    for current_site in 1:dims
        propose_local(hs_field, greens, current_site, current_slice)
        if hs_field.ratio > 1 || rand() < hs_field.ratio
            accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
            accepted += 1
            # println(accepted)
            # 同时更新B矩阵
            diag_element = hs_field.conf[current_site, current_slice] < 0 ? hs_field.elements[2] : hs_field.elements[1]
            @turbo for ii = 1:dims
                B_slice[ii, current_site] = diag_element * hopping_matrix_exp[ii, current_site]
            end
        end
    end
    return accepted
end

@inline function accept_or_reject_modified(hs_field::HSField_CDW, greens_old::Matrix{Float64}, vector_tmp1::Vector{Float64}, vector_tmp2::Vector{Float64},
    current_site::Int64, current_slice::Int64, n::Int64)
    @inbounds begin
        hs_field.conf[current_site, current_slice] = -hs_field.conf[current_site, current_slice]

        x = hs_field.Δ / hs_field.det_ratio

        @turbo for i = 1:n
            vector_tmp1[i] = greens_old[i, current_site]
            vector_tmp2[i] = greens_old[current_site, i] * x
        end

        @turbo for i = 1:n
            for j = 1:n
                greens_old[i, j] += vector_tmp1[i] * vector_tmp2[j]
            end
            greens_old[current_site, i] -= vector_tmp2[i]
        end

        nothing
    end
end