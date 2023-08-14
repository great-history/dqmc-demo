# current_slice = 1
# dims = stack.dims
# accepted = 0
# accepted_modified = Int64(0)

# for i in 1:dims
#     propose_local(hs_field, stack.greens, i, current_slice)
#     detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, i, current_slice)
#     p = exp(-ΔE_boson) * detratio

#     println(p - hs_field.ratio)
#     if real(p) > 1 || rand() < real(p)
#         accept_local!(mc, m, mc.field, i, current_slice, detratio, ΔE_boson, passthrough)
#         global accepted += 1
#         accept_or_reject_modified(hs_field, stack.greens, stack.vector_tmp1, stack.vector_tmp2, i, current_slice, dims)
#     end
# end

@stack_green_shortcuts
@stack_bmats_shortcuts

vector_tmp1 = stack.vector_tmp1
vector_tmp2 = stack.vector_tmp2
matrix_tmp = stack.matrix_tmp1

accepted = Int64(0)
dims = stack.dims
n_slices = stack.n_slices
current_range = Int64(1)

for idx = 1:1000

    for current_slice in 1:n_slices
        for current_site in 1:dims
            propose_local(hs_field, greens, current_site, current_slice)
            detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, current_site, current_slice)
            p = exp(-ΔE_boson) * detratio
            println(p - hs_field.ratio)
            if hs_field.ratio > 1 || rand() < hs_field.ratio
                accept_local!(mc, m, mc.field, current_site, current_slice, detratio, ΔE_boson, passthrough)
                accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
                # 不更新B矩阵
                global accepted += 1
            end
        end

        MonteCarlo.propagate(mc)
        get_single_B_slices_modified!(hs_field, current_slice, hopping_matrix_exp, vector_tmp1, B_slice,
            hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)

        vmul!(matrix_tmp, B_slice, Ul)
        copyto!(Ul, matrix_tmp)

        if stack.slice_rem[current_slice] == 0
            mul_diag_right!(matrix_tmp, Dl, dims)

            global current_range += 1
            copyto!(Ur, stack.u_stack[current_range])
            copyto!(Dr, stack.d_stack[current_range])
            copyto!(Tr, stack.t_stack[current_range])

            qr_real_pivot!(Ul, matrix_tmp, Dl, pivot, dims)
            permute_rows!(stack.t_stack[current_range], Tl, pivot, dims)
            lmul!(UpperTriangular(matrix_tmp), stack.t_stack[current_range])

            copyto!(Tl, stack.t_stack[current_range])
            copyto!(stack.u_stack[current_range], Ul)
            copyto!(stack.d_stack[current_range], Dl)

            get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, pivot, dims)

            stack.current_range = current_range

            copyto!(Ul, stack.u_stack[current_range])
            copyto!(Dl, stack.d_stack[current_range])
            copyto!(Tl, stack.t_stack[current_range])

        else
            get_eqlt_green_by_wrap_up!(greens, B_slice, B_slice_inv, matrix_tmp)
        end
        # println("下面是格林函数差")
        # println(maximum(abs.((greens - mc.stack.greens))))
    end

    println(maximum(abs.((greens - mc.stack.greens))))
    hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)

    diag_element = Float64(0)
    copyto!(Ur, I)
    Dr .= one(eltype(mc.stack.d_stack[end]))
    copyto!(Tr, I)

    for current_slice in n_slices:-1:1
        get_eqlt_green_by_wrap_down!(greens, B_slice_inv, B_slice, matrix_tmp)

        println("下面是sweep down时格林函数差")
        println(maximum(abs.((greens - mc.stack.greens))))

        for current_site in 1:dims
            detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, current_site, current_slice)
            p = exp(-ΔE_boson) * detratio
            propose_local(hs_field, greens, current_site, current_slice)
            println(p - hs_field.ratio)

            if hs_field.ratio > 1 || rand() < hs_field.ratio
                accept_local!(mc, m, mc.field, current_site, current_slice, detratio, ΔE_boson, passthrough)
                accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
                global accepted += 1
    
                # 同时更新B矩阵
                diag_element = hs_field.conf[current_site, current_slice] < 0 ? hs_field.elements[2] : hs_field.elements[1]
                @turbo for ii = 1:dims
                    B_slice[ii, current_site] = diag_element * hopping_matrix_exp[ii, current_site]
                end
            end
        end

        mul_adjoint_left!(matrix_tmp, B_slice, Ur)
        copyto!(Ur, matrix_tmp)

        if stack.slice_rem[current_slice] == 1
            mul_diag_right!(matrix_tmp, Dr, dims)

            global current_range -= 1
            copyto!(Ul, stack.u_stack[current_range])
            copyto!(Dl, stack.d_stack[current_range])
            copyto!(Tl, stack.t_stack[current_range])

            qr_real_pivot!(Ur, matrix_tmp, Dr, pivot, dims)
            permute_rows!(stack.t_stack[current_range], Tr, pivot, dims)
            lmul!(UpperTriangular(matrix_tmp), stack.t_stack[current_range])

            copyto!(Tr, stack.t_stack[current_range])
            copyto!(stack.u_stack[current_range], Ur)
            copyto!(stack.d_stack[current_range], Dr)

            get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, pivot, dims)

            copyto!(Ur, stack.u_stack[current_range])
            copyto!(Dr, stack.d_stack[current_range])
            copyto!(Tr, stack.t_stack[current_range])

            stack.current_range = current_range
        end

        get_single_B_slices_modified!(hs_field, current_slice - 1, hopping_matrix_exp, vector_tmp1, B_slice,
            hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)
        MonteCarlo.propagate(mc)
    end

    println(maximum(abs.((greens - mc.stack.greens))))
    println("\n\n")

    hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)
    copyto!(Ul, I)
    Dl .= one(eltype(mc.stack.d_stack[end]))
    copyto!(Tl, I)
end
# println(current_range)
# @inbounds begin
#     for current_slice in 1:n_slices

#         for current_site in 1:dims
#             propose_local(hs_field, greens, current_site, current_slice)
#             detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, current_site, current_slice)
#             p = exp(-ΔE_boson) * detratio
#             println(p - hs_field.ratio)
#             if hs_field.ratio > 1 || rand() < hs_field.ratio
#                 accept_local!(mc, m, mc.field, current_site, current_slice, detratio, ΔE_boson, passthrough)
#                 accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
#                 # 不更新B矩阵
#                 global accepted += 1
#             end
#         end

#         MonteCarlo.propagate(mc)
#         get_single_B_slices_modified!(hs_field, current_slice, hopping_matrix_exp, vector_tmp1, B_slice,
#             hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)

#         vmul!(matrix_tmp, B_slice, Ul)
#         copyto!(Ul, matrix_tmp)

#         if stack.slice_rem[current_slice] == 0
#             mul_diag_right!(matrix_tmp, Dl, dims)

#             global current_range += 1
#             copyto!(Ur, stack.u_stack[current_range])
#             copyto!(Dr, stack.d_stack[current_range])
#             copyto!(Tr, stack.t_stack[current_range])

#             qr_real_pivot!(Ul, matrix_tmp, Dl, pivot, dims)
#             permute_rows!(stack.t_stack[current_range], Tl, pivot, dims)
#             lmul!(UpperTriangular(matrix_tmp), stack.t_stack[current_range])

#             copyto!(Tl, stack.t_stack[current_range])
#             copyto!(stack.u_stack[current_range], Ul)
#             copyto!(stack.d_stack[current_range], Dl)

#             get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, pivot, dims)

#             stack.current_range = current_range

#             copyto!(Ul, stack.u_stack[current_range])
#             copyto!(Dl, stack.d_stack[current_range])
#             copyto!(Tl, stack.t_stack[current_range])

#         else
#             get_eqlt_green_by_wrap_up!(greens, B_slice, B_slice_inv, matrix_tmp)
#         end
#         # println("下面是格林函数差")
#         # println(maximum(abs.((greens - mc.stack.greens))))
#     end
# end
# println(maximum(abs.((greens - mc.stack.greens))))
# hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)

# diag_element = Float64(0)
# copyto!(Ur, I)
# Dr .= one(eltype(mc.stack.d_stack[end]))
# copyto!(Tr, I)
# println("\n\n")

# @inbounds begin
#     for current_slice in n_slices:-1:1
#         get_eqlt_green_by_wrap_down!(greens, B_slice_inv, B_slice, matrix_tmp)
        
#         # println("下面是sweep down时格林函数差")
#         # println(maximum(abs.((greens - mc.stack.greens))))

#         for current_site in 1:dims
#             detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, current_site, current_slice)
#             p = exp(-ΔE_boson) * detratio
#             propose_local(hs_field, greens, current_site, current_slice)
#             # println(p - hs_field.ratio)

#             if hs_field.ratio > 1 || rand() < hs_field.ratio
#                 accept_local!(mc, m, mc.field, current_site, current_slice, detratio, ΔE_boson, passthrough)
#                 accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
#                 global accepted += 1
    
#                 # 同时更新B矩阵
#                 global diag_element = hs_field.conf[current_site, current_slice] < 0 ? hs_field.elements[2] : hs_field.elements[1]
#                 @turbo for ii = 1:dims
#                     B_slice[ii, current_site] = diag_element * hopping_matrix_exp[ii, current_site]
#                 end
#             end
#         end

#         mul_adjoint_left!(matrix_tmp, B_slice, Ur)
#         copyto!(Ur, matrix_tmp)

#         if stack.slice_rem[current_slice] == 1
#             mul_diag_right!(matrix_tmp, Dr, dims)

#             global current_range -= 1
#             copyto!(Ul, stack.u_stack[current_range])
#             copyto!(Dl, stack.d_stack[current_range])
#             copyto!(Tl, stack.t_stack[current_range])

#             qr_real_pivot!(Ur, matrix_tmp, Dr, pivot, dims)
#             permute_rows!(stack.t_stack[current_range], Tr, pivot, dims)
#             lmul!(UpperTriangular(matrix_tmp), stack.t_stack[current_range])

#             copyto!(Tr, stack.t_stack[current_range])
#             copyto!(stack.u_stack[current_range], Ur)
#             copyto!(stack.d_stack[current_range], Dr)

#             get_eqlt_green_by_qr!(greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, pivot, dims)

#             copyto!(Ur, stack.u_stack[current_range])
#             copyto!(Dr, stack.d_stack[current_range])
#             copyto!(Tr, stack.t_stack[current_range])

#             stack.current_range = current_range
#         end

#         get_single_B_slices_modified!(hs_field, current_slice - 1, hopping_matrix_exp, vector_tmp1, B_slice,
#             hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)
#         MonteCarlo.propagate(mc)
#     end
# end
# flush(stdout)
# println(maximum(abs.((greens - mc.stack.greens))))