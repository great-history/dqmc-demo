using Random
using LinearAlgebra
using MonteCarlo
using LoopVectorization
using Profile
using BenchmarkTools

include("lattice_model.jl")
include("dqmc_parameters.jl")
include("dqmc_fields.jl")
include("dqmc_matrix_cdw_channel.jl")
# include("LinearAlgebra.jl")
include("dqmc_stable.jl")
include("dqmc_eqlt_greens.jl")
include("dqmc_update.jl")
# include("update_test.jl")
include("dqmc_helper.jl")
include("dqmc_ueqlt_greens.jl")

##### -----------------------------------------------------------------------------------------------------------------------分割线, 以下上是对stack初始化的测试
# Lx = 8;
# beta = 5.0

# tri_lat = Triangular_Lattice(Lx,Lx);
# t = 1.0;
# Hubbard_U = 4.0;
# mu = -2.0;
# tri_model = Single_Band_Hubbard_Model(t = t, U = Hubbard_U, mu = mu, lattice = tri_lat);
# parameters = dqmc_parameters(beta = beta);
# hs_field = HSField_CDW(delta_tau = parameters.delta_tau, slice_num = parameters.slices, n_sites = tri_model.lattice.sites, U = tri_model.Hubbard_U);

# stack = dqmc_matrices_stack_real_qr(tri_model, parameters)


# mus = - 2.0
# lattice = TriangularLattice(Lx)
# m = HubbardModel(l = lattice, t = 1.0, U = 4.0, mu = mu)
# dqmc = DQMC(
#     m, beta = beta, delta_tau = 0.125, safe_mult = 8, 
#     thermalization = 1000, sweeps = 1000, measure_rate = 1,
#     recorder = Discarder()
# )

# MonteCarlo.init!(dqmc)
# hs_field.conf = deepcopy(dqmc.field.conf)
# mc = dqmc

### 这段代码主要用来进行数值稳定性的测试比较

# ## 从这里开始是得到上三角矩阵和Householder_matrix,以及得到pivot(列置换初等变换)

# # ### test permutation
# # # @time vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv)
# # # @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv); end)
# # # @profile (for i = 1:10000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv); end)
# # vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1]))
# # udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv)

# vector_β = zeros(n_site)
# # vector_tmp = zeros(16)
# # vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1]))
# # qr_real_pivot!(input, pivot, squared_norms)

# # @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; qr_real_pivot!(input, pivot, squared_norms, vector_β); end)
# # @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv); end)
# copyto!(dqmc.stack.u_stack[end], I)
# dqmc.stack.d_stack[end] .= one(eltype(dqmc.stack.d_stack[end]))
# copyto!(dqmc.stack.t_stack[end], I)
# U = similar(dqmc.stack.curr_U)

# # @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv); end)
# vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1]))
# udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv)
# vmul!(dqmc.stack.t_stack[idx], input, dqmc.stack.t_stack[idx + 1])

# # @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; qr_real_pivot!(U_test, input, pivot, squared_norms, vector_β); end)
# vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1]))

# copyto!(U_test, I)
# qr_real_pivot!(U_test, input, pivot, squared_norms)


# vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1]))
# @time qr_real_pivot!(U_test, input, pivot, squared_norms)
# vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1]))
# @time udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv)
# @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; qr_real_pivot!(U_test, input, pivot, squared_norms); end)
# @time (for i = 1:1000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv); end)

# # @profile (for i = 1:100000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; qr_real_pivot!(U_test, input, pivot, squared_norms, vector_β); end)
# # @profile (for i = 1:100000; vmul!(input, dqmc.stack.curr_U, Diagonal(dqmc.stack.d_stack[idx + 1])) ; udt_AVX_pivot!(dqmc.stack.u_stack[idx], dqmc.stack.d_stack[idx], input, pivot, tempv); end)
# # Profile.clear()


### 下面代码主要用来计算等时/非等时格林函数

# copyto!(stack.u_stack[end], I)
# stack.d_stack[end] .= one(eltype(stack.d_stack[end]))
# copyto!(stack.t_stack[end], I)
# copyto!(stack.Ur, stack.u_stack[end])

# eT = stack.hopping_matrix_exp
# eV = stack.vector_tmp1
# B_slice = stack.current_B
# Ur = stack.Ur   ## TODO::规定Ur存放的是adjoint之后的U矩阵
# matrix_tmp1 = stack.matrix_tmp1

# for slice_index in reverse(stack.ranges[2])
#     get_single_B_slice_modified!(hs_field, slice_index, eT, eV, B_slice)
#     mul_adjoint_left!(matrix_tmp1, B_slice, Ur)
#     copyto!(Ur, matrix_tmp1)
# end

# @time (for i = 1:1000; build_udt_stacks_modified!(hs_field, stack); end)
# @time (for i = 1:1000; reverse_build_stack(dqmc); end)

# copyto!(stack.u_stack[end], I)
# copyto!(stack.Ur, stack.u_stack[end])
# stack.d_stack[end] .= one(Float64)
# copyto!(stack.t_stack[end], I)

# eT = stack.hopping_matrix_exp
# eV = stack.vector_tmp1
# B_slice = stack.current_B
# matrix_tmp1 = stack.matrix_tmp1
# Ur_test = stack.Ur
# pivot = stack.pivot_tmp
# dims = stack.dims;

# copyto!(Ur_test,stack.u_stack[end])
# for slice_index in reverse(stack.ranges[end])
#     get_single_B_slice_modified!(hs_field, slice_index, eT, eV, B_slice)
#     mul_adjoint_left!(matrix_tmp1, B_slice, Ur_test)
#     copyto!(Ur_test, matrix_tmp1)
# end

# @time reverse_build_stack(dqmc);
# @btime reverse_build_stack(dqmc);
# @time build_udt_stacks_modified!(hs_field, stack);
# @btime build_udt_stacks_modified!(hs_field, stack);
##### -----------------------------------------------------------------------------------------------------------------------分割线, 以上是对stack初始化的测试

##### -----------------------------------------------------------------------------------------------------------------------分割线, 以下是对dqmc simulation的测试
# Lx = 4;
# beta = 2.0;
# mu = -2.0;
# t = 1.0;
# Hubbard_U = 4.0;

# lattice = TriangularLattice(Lx)
# m = HubbardModel(l=lattice, t=t, U=Hubbard_U, mu=mu)
# mc = DQMC(
#     m, beta=beta, delta_tau=0.125, safe_mult=8,
#     thermalization=1000, sweeps=1000, measure_rate=1,
#     recorder=Discarder()
# )
# MonteCarlo.init!(mc)
# tri_lat = Triangular_Lattice(Lx, Lx);

# tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat);
# parameters = dqmc_parameters(beta=beta);
# hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U);

# stack = dqmc_matrices_stack_real_qr(tri_model, parameters)

# hs_field.conf = deepcopy(mc.field.conf)
# build_udt_stacks_modified!(hs_field, stack)

# thermalization = mc.parameters.thermalization
# sweeps = mc.parameters.sweeps
# total_sweeps = sweeps + thermalization

# min_update_rate = 0.001
# using Dates
# start_time = now()
# last_checkpoint = now()
# max_sweep_duration = 0.0
# println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

# # fresh stack
# println("Preparing Green's function stack")
# reverse_build_stack(mc)
# MonteCarlo.propagate(mc)

# min_sweeps = round(Int, 1 / min_update_rate)
# _time = time() # for step estimations
# t0 = time() # for analysis.runtime, may need to reset
# println("\n\nThermalization stage - ", thermalization)

# next_print = (div(mc.last_sweep, mc.parameters.print_rate) + 1) * mc.parameters.print_rate
# println("格林函数之差:", maximum(abs.(mc.stack.greens - stack.greens)))
# # Check assumptions for global updates
# # try
# #     copyto!(mc.stack.tmp2, mc.stack.greens)
# #     udt_AVX_pivot!(mc.stack.tmp1, mc.stack.tempvf, mc.stack.tmp2, mc.stack.pivot, mc.stack.tempv)
# #     ud = det(Matrix(mc.stack.tmp1))  # 幺正矩阵的行列式一定为1
# #     td = det(Matrix(mc.stack.tmp2))  # 上三角矩阵(即使置换了列)的行列式一定为对角元的乘积,在这里不是1就是-1
# #     if !(0.9999999 <= abs(td) <= 1.0000001) || !(0.9999999 <= abs(ud) <= 1.0000001)
# #         @error("Assumptions for global updates broken! ($td, $ud should be 1)")
# #     end
# # catch e
# #     @warn "Could not verify global update" exception = e
# # end

# while mc.last_sweep < total_sweeps
#     (mc.last_sweep == thermalization + 1) && println("\n\nMeasurement stage - ", sweeps)

#     # Perform whatever update is scheduled next
#     update(mc.scheduler, mc, mc.model)
# end

# while mc.last_sweep < total_sweeps
#     update(mc.scheduler, mc, mc.model)
# end

# @time detratio, ΔE_boson, passthrough = propose_local(mc, mc.model, mc.field, 1, 1)
# p = exp(- ΔE_boson) * detratio
# accept_local!(mc, m, mc.field, 1, 1, detratio, ΔE_boson, passthrough)


# current_site = 1
# current_slice = 1
# dims = stack.dims
# accepted = Int64(0)
# propose_local(hs_field, stack.greens, current_site, current_slice)
# accept_or_reject_modified(hs_field, stack.greens, stack.vector_tmp1, stack.vector_tmp2, accepted, current_site, current_slice, dims)

# current_slice = 1
# dims = stack.dims
# accepted = 0
# accepted_modified = Int64(0)
# @inbounds begin
#     # for current_slice in 1:stack.n_slices
#         for i in 1:dims
#             detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, i, current_slice)
#             p = exp(- ΔE_boson) * detratio

#             propose_local(hs_field, stack.greens, i, current_slice)
#             println(p - hs_field.ratio)

#             # if mc.parameters.check_sign_problem
#             #     if abs(imag(p)) > 1e-6
#             #         push!(mc.analysis.imaginary_probability, abs(imag(p)))
#             #         mc.parameters.silent || println(
#             #             "Did you expect a sign problem? imag. probability:  %.9e\n", 
#             #             abs(imag(p))
#             #         )
#             #     end
#             #     if real(p) < 0.0
#             #         push!(mc.analysis.negative_probability, real(p))
#             #         mc.parameters.silent || println(
#             #             "Did you expect a sign problem? negative probability %.9e\n",
#             #             real(p)
#             #         )
#             #     end
#             # end

#             if real(p) > 1 || rand() < real(p)
#                 accept_local!(mc, m, mc.field, i, current_slice, detratio, ΔE_boson, passthrough)
#                 accepted += 1
#                 accept_or_reject_modified(hs_field, stack.greens, stack.vector_tmp1, stack.vector_tmp2, i, current_slice, dims)
#             end
#         end
#     # end
# end

# for i in 1:dims
#     detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, i, current_slice)
#     p = exp(-ΔE_boson) * detratio

#     propose_local(hs_field, stack.greens, i, current_slice)
#     println(p - hs_field.ratio)
#     if real(p) > 1 || rand() < real(p)
#         accept_local!(mc, m, mc.field, i, current_slice, detratio, ΔE_boson, passthrough)
#         accepted += 1
#         accept_or_reject_modified(hs_field, stack.greens, stack.vector_tmp1, stack.vector_tmp2, i, current_slice, dims)
#     end
# end
#  --------------------------------------------------------------------------------------------------------
## 测试propagate是不是没问题
# println(maximum(abs.(stack.greens - mc.stack.greens)))
# dims = stack.dims
# Ul = similar(stack.Ul)
# copyto!(Ul, stack.Ul)
# Dl = similar(stack.Dl)
# copyto!(Dl, stack.Dl)
# Tl = similar(stack.Tl)
# copyto!(Tl, stack.Tl)
# Ur = similar(stack.Ur)
# copyto!(Ur, stack.Ur)
# Dr = similar(stack.Dr)
# copyto!(Dr, stack.Dr)
# Tr = similar(stack.Tr)
# copyto!(Tr, stack.Tr)
# vector_tmp1 = stack.vector_tmp1
# vector_tmp2 = stack.vector_tmp2
# matrix_tmp = stack.matrix_tmp1
# copyto!(Ul, I)
# ------------------------------------------------------------------------------------------------------------
# for i = 1:8
#     get_single_B_slices_modified!(hs_field, i, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B,
#         stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv)
#     vmul!(matrix_tmp, stack.current_B, Ul)
#     copyto!(Ul, matrix_tmp)
# end

# mul_diag_right!(matrix_tmp, Dl, dims)
# copyto!(Ur, stack.u_stack[2])
# copyto!(Dr, stack.d_stack[2])
# copyto!(Tr, stack.t_stack[2])

# qr_real_pivot!(Val(true), Ul, matrix_tmp, Dl, stack.pivot_tmp, dims)
# permute_columns!(Tl, matrix_tmp, stack.pivot_tmp, dims)

# copyto!(stack.u_stack[2], Ul)
# copyto!(stack.d_stack[2], Dl)
# copyto!(stack.t_stack[2], Tl)

# idx = mc.stack.current_range
# @debug("Stabilize: decompose into $idx -> $(idx+1)")

# copyto!(mc.stack.Ur, mc.stack.u_stack[idx+1])
# copyto!(mc.stack.Dr, mc.stack.d_stack[idx+1])
# copyto!(mc.stack.Tr, mc.stack.t_stack[idx+1])
# MonteCarlo.add_slice_sequence_left(mc, idx)
# copyto!(mc.stack.Ul, mc.stack.u_stack[idx+1])
# copyto!(mc.stack.Dl, mc.stack.d_stack[idx+1])
# copyto!(mc.stack.Tl, mc.stack.t_stack[idx+1])

# @def compare begin
#     println(maximum(abs.((mc.stack.Ur-Ur)[:, 1:end-1])))
#     println(maximum(abs.(mc.stack.Tr - Tr)))
#     println(maximum(abs.((mc.stack.Dr - Dr) ./ Dr)))
#     println(maximum(abs.((mc.stack.Ul-Ul)[:, 1:end-1])))
#     println(maximum(abs.(mc.stack.Tl - Tl)))
#     println(maximum(abs.((mc.stack.Dl - Dl) ./ Dl)))
# end

# @compare

# calculate_greens_AVX!(
#     mc.stack.Ul, mc.stack.Dl, mc.stack.Tl,
#     mc.stack.Ur, mc.stack.Dr, mc.stack.Tr,
#     output, mc.stack.pivot, mc.stack.tempv
# )

# 下面是sweep up的测试
# ----------------------------------------------------------------------------------------------------------------
# for i in 1:mc.parameters.slices
#     MonteCarlo.propagate(mc)

#     get_single_B_slices_modified!(hs_field, i, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B,
#         stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv)
#     vmul!(matrix_tmp, stack.current_B, Ul)
#     copyto!(Ul, matrix_tmp)

#     if stack.slice_rem[i] == 0
#         stack.current_range += 1
#         mul_diag_right!(matrix_tmp, Dl, dims)
#         copyto!(Ur, stack.u_stack[stack.current_range])
#         copyto!(Dr, stack.d_stack[stack.current_range])
#         copyto!(Tr, stack.t_stack[stack.current_range])

#         qr_real_pivot!(Ul, matrix_tmp, Dl, stack.pivot_tmp, dims)
#         permute_rows!(stack.t_stack[stack.current_range], Tl, stack.pivot_tmp, dims)
#         lmul!(UpperTriangular(matrix_tmp), stack.t_stack[stack.current_range])

#         # permute_columns!(stack.t_stack[stack.current_range], matrix_tmp, stack.pivot_tmp, dims)
#         copyto!(Tl, stack.t_stack[stack.current_range])
#         copyto!(stack.u_stack[stack.current_range], Ul)
#         copyto!(stack.d_stack[stack.current_range], Dl)

#         get_eqlt_green_by_qr!(stack.greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, stack.pivot_tmp, dims)
#         copyto!(Ul, stack.u_stack[stack.current_range])
#         copyto!(Dl, stack.d_stack[stack.current_range])
#         copyto!(Tl, stack.t_stack[stack.current_range])

#         if i == stack.n_slices
#             stack.greens = stack.current_B_inv * stack.greens * stack.current_B
#         end
#     else
#         get_eqlt_green_by_wrap_up!(stack.greens, stack.current_B, stack.current_B_inv, stack.matrix_tmp1)
#     end

#     ## 没有进行数值稳定性,所以只有前几个是对的上的
#     println(maximum(abs.((stack.greens - mc.stack.greens))))
# end

# 下面是sweep down的测试
# ------------------------------------------------------------------------------------------------------
# copyto!(Ur, I)
# Dr .= one(eltype(mc.stack.d_stack[end]))
# copyto!(Tr, I)
# get_single_B_slices_modified!(hs_field, mc.parameters.slices, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B,
#     stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv)
# mul_adjoint_left!(matrix_tmp, stack.current_B, Ur)
# copyto!(Ur, matrix_tmp)

# for i in mc.parameters.slices-1:-1:1
#     MonteCarlo.propagate(mc)

#     get_single_B_slices_modified!(hs_field, i, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B,
#         stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv)
#     mul_adjoint_left!(matrix_tmp, stack.current_B, Ur)
#     copyto!(Ur, matrix_tmp)

#     if stack.slice_rem[i] == 1
#         stack.current_range -= 1
#         mul_diag_right!(matrix_tmp, Dr, dims)
#         copyto!(Ul, stack.u_stack[stack.current_range])
#         copyto!(Dl, stack.d_stack[stack.current_range])
#         copyto!(Tl, stack.t_stack[stack.current_range])

#         qr_real_pivot!(Ur, matrix_tmp, Dr, stack.pivot_tmp, dims)
#         permute_rows!(stack.t_stack[stack.current_range], Tr, stack.pivot_tmp, dims)
#         lmul!(UpperTriangular(matrix_tmp), stack.t_stack[stack.current_range])

#         # permute_columns!(stack.t_stack[stack.current_range], matrix_tmp, stack.pivot_tmp, dims)
#         copyto!(Tr, stack.t_stack[stack.current_range])
#         copyto!(stack.u_stack[stack.current_range], Ur)
#         copyto!(stack.d_stack[stack.current_range], Dr)

#         get_eqlt_green_by_qr!(stack.greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, stack.pivot_tmp, dims)
#         copyto!(Ur, stack.u_stack[stack.current_range])
#         copyto!(Dr, stack.d_stack[stack.current_range])
#         copyto!(Tr, stack.t_stack[stack.current_range])

#     else
#         get_eqlt_green_by_wrap_down!(stack.greens, stack.current_B_inv, stack.current_B, matrix_tmp)
#     end

#     ## 没有进行数值稳定性,所以只有前几个是对的上的
#     println(maximum(abs.((stack.greens - mc.stack.greens))))
# end

# 以下又是sweep up的测试, 经过初步调试，应该是没啥大问题了
# -----------------------------------------------------------------------------------------------------------
# copyto!(stack.u_stack[1], I)
# stack.d_stack[1] .= one(eltype(mc.stack.d_stack[end]))
# copyto!(stack.t_stack[1], I)

# copyto!(Ul, I)
# Dl .= one(eltype(mc.stack.d_stack[end]))
# copyto!(Tl, I)

# mc.stack.direction = 1

# for i in 1:mc.parameters.slices
#     MonteCarlo.propagate(mc)

#     get_single_B_slices_modified!(hs_field, i, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B,
#         stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv)
#     vmul!(matrix_tmp, stack.current_B, Ul)
#     copyto!(Ul, matrix_tmp)

#     if stack.slice_rem[i] == 0
#         stack.current_range += 1
#         mul_diag_right!(matrix_tmp, Dl, dims)
#         copyto!(Ur, stack.u_stack[stack.current_range])
#         copyto!(Dr, stack.d_stack[stack.current_range])
#         copyto!(Tr, stack.t_stack[stack.current_range])

#         qr_real_pivot!(Val(true), Ul, matrix_tmp, Dl, stack.pivot_tmp, dims)
#         permute_rows!(stack.t_stack[stack.current_range], Tl, stack.pivot_tmp, dims)
#         lmul!(UpperTriangular(matrix_tmp), stack.t_stack[stack.current_range])

#         # permute_columns!(stack.t_stack[stack.current_range], matrix_tmp, stack.pivot_tmp, dims)
#         copyto!(Tl, stack.t_stack[stack.current_range])
#         copyto!(stack.u_stack[stack.current_range], Ul)
#         copyto!(stack.d_stack[stack.current_range], Dl)

#         get_eqlt_green_by_qr!(stack.greens, Ul, Dl, Tl, Ur, Dr, Tr, vector_tmp1, stack.pivot_tmp, dims)
#         copyto!(Ul, stack.u_stack[stack.current_range])
#         copyto!(Dl, stack.d_stack[stack.current_range])
#         copyto!(Tl, stack.t_stack[stack.current_range])

#         copyto!(stack.u_stack[stack.current_range], I)
#         stack.d_stack[stack.current_range] .= one(eltype(mc.stack.d_stack[end]))
#         copyto!(stack.t_stack[stack.current_range], I)

#         if i == stack.n_slices
#             stack.greens = stack.current_B_inv * stack.greens * stack.current_B
#         end
#     else
#         get_eqlt_green_by_wrap_up!(stack.greens, stack.current_B, stack.current_B_inv, stack.matrix_tmp1)
#     end

#     ## 没有进行数值稳定性,所以只有前几个是对的上的
#     println(maximum(abs.((stack.greens - mc.stack.greens))))
# end



# copyto!(stack.u_stack[end], I)
# stack.d_stack[end] .= one(Float64)
# copyto!(stack.t_stack[end], I)

# Lx = 4;
# beta = 5.0;
# mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)

# mu = -1.25;
# t = 1.0;
# Hubbard_U = 4.0;

# lattice = TriangularLattice(Lx)
# m = HubbardModel(l=lattice, t=t, U=Hubbard_U, mu=mu)
# mc = DQMC(
#     m, beta=beta, delta_tau=0.125, safe_mult=8,
#     thermalization=1000, sweeps=1000, measure_rate=1,
#     recorder=Discarder()
# )
# MonteCarlo.init!(mc)
# tri_lat = Triangular_Lattice(Lx, Lx);

# tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat);
# parameters = dqmc_parameters(beta=beta);
# hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U);

# stack = dqmc_matrices_stack_real_qr(tri_model, parameters)

# hs_field.conf = deepcopy(mc.field.conf)
# build_udt_stacks_modified!(hs_field, stack)

# thermalization = mc.parameters.thermalization
# sweeps = mc.parameters.sweeps
# total_sweeps = sweeps + thermalization

# min_update_rate = 0.001
# using Dates
# start_time = now()
# last_checkpoint = now()
# max_sweep_duration = 0.0
# println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

# # fresh stack
# println("Preparing Green's function stack")
# reverse_build_stack(mc)
# MonteCarlo.propagate(mc)

# min_sweeps = round(Int, 1 / min_update_rate)
# _time = time() # for step estimations
# t0 = time() # for analysis.runtime, may need to reset
# println("\n\nThermalization stage - ", thermalization)

# next_print = (div(mc.last_sweep, mc.parameters.print_rate) + 1) * mc.parameters.print_rate
# println("格林函数之差:", maximum(abs.(mc.stack.greens - stack.greens)))

# -------------------------------------------------------------------------------------------------------------------------
Lx = 8;
betas = 8.0;
mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)
# mus = 2.0
global counter = 0
dqmc_pc = []
for beta in betas, mu in mus
    t = 1.0;
    Hubbard_U = 4.0;

    # lattice = TriangularLattice(Lx)
    # m = HubbardModel(l=lattice, t=t, U=Hubbard_U, mu=mu)
    # mc = DQMC(
    #     m, beta=beta, delta_tau=0.125, safe_mult=8,
    #     thermalization=1000, sweeps=1000, measure_rate=1,
    #     recorder=Discarder()
    # )
    # MonteCarlo.init!(mc)
    tri_lat = Triangular_Lattice(Lx, Lx);

    tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat);
    parameters = dqmc_parameters(beta=beta);
    hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U);

    stack = dqmc_matrices_stack_real_qr(tri_model, parameters)

    # hs_field.conf = deepcopy(mc.field.conf)

    build_udt_stacks_modified!(hs_field, stack)
    # reverse_build_stack(mc)
    # MonteCarlo.propagate(mc)

    @stack_green_shortcuts
    @stack_bmats_shortcuts

    vector_tmp1 = stack.vector_tmp1
    vector_tmp2 = stack.vector_tmp2
    matrix_tmp = stack.matrix_tmp1

    accepted = Int64(0)
    dims = stack.dims
    n_slices = stack.n_slices
    current_range = Int64(1)

    bins = zeros(Float64, 1000)
    for idx = 1:2000
        println("开始了")
        for current_slice in 1:n_slices
            for current_site in 1:dims
                propose_local(hs_field, greens, current_site, current_slice)
                # detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, current_site, current_slice)
                # p = exp(-ΔE_boson) * detratio
                # println(p - hs_field.ratio)
                if hs_field.ratio > 1 || rand() < hs_field.ratio
                    # accept_local!(mc, m, mc.field, current_site, current_slice, detratio, ΔE_boson, passthrough)
                    accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
                    # 不更新B矩阵
                    # global accepted += 1
                    accepted += 1
                end
            end

            # MonteCarlo.propagate(mc)
            get_single_B_slices_modified!(hs_field, current_slice, hopping_matrix_exp, vector_tmp1, B_slice,
                hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)

            vmul!(matrix_tmp, B_slice, Ul)
            copyto!(Ul, matrix_tmp)

            if stack.slice_rem[current_slice] == 0
                # global current_range += 1
                current_range += 1
                mul_diag_right!(matrix_tmp, Dl, dims)

                # println(current_range)
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
            # println("下面是sweep up时格林函数差")
            # println(maximum(abs.((greens - mc.stack.greens))))

        end

        hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)

        diag_element = Float64(0)
        copyto!(Ur, I)
        Dr .= one(Float64)
        copyto!(Tr, I)

        copyto!(stack.u_stack[end], I)
        stack.d_stack[end] .= one(Float64)
        copyto!(stack.t_stack[end], I)

        for current_slice in n_slices:-1:1
            get_eqlt_green_by_wrap_down!(greens, B_slice_inv, B_slice, matrix_tmp)

            # println("下面是sweep down时格林函数差")
            # println(maximum(abs.((greens - mc.stack.greens))))

            for current_site in 1:dims
                # detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, current_site, current_slice)
                # p = exp(-ΔE_boson) * detratio
                propose_local(hs_field, greens, current_site, current_slice)
                # println(p - hs_field.ratio)

                if hs_field.ratio > 1 || rand() < hs_field.ratio
                    # accept_local!(mc, m, mc.field, current_site, current_slice, detratio, ΔE_boson, passthrough)
                    accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
                    # global accepted += 1
                    accepted += 1
        
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

                # global current_range -= 1
                current_range -= 1
                # println(current_range)
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
            # MonteCarlo.propagate(mc)
        end

        hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)
        copyto!(Ul, I)
        Dl .= one(Float64)
        copyto!(Tl, I)

        copyto!(stack.u_stack[1], I)
        stack.d_stack[1] .= one(Float64)
        copyto!(stack.t_stack[1], I)

        if idx > 1000
            # occ_ = Float64(0)
            # for site = 1:16
            #     occ_ += (1 - greens[site,site])
            # end
            # bins[idx - 1000] = occ_ / 16

            pc_q0 = Float64(0)
            n_sites = 16
            eThalfminus = stack.hopping_matrix_exp_half
            eThalfplus = stack.hopping_matrix_exp_inv_half
            vmul!(stack.matrix_tmp4, stack.greens, eThalfminus)
            vmul!(stack.greens_temp, eThalfplus, stack.matrix_tmp4)
            for site1 = 1:n_sites, site2 = 1:n_sites
                # pc_q0 += bins_pc[sweep_idx][m,n]
                pc_q0 += 2 * (stack.greens_temp[site1, site2])^2
            end

            for site = 1:n_sites
                pc_q0 += (1 - 2 * stack.greens_temp[site, site])
            end
            bins[idx - 1000] = pc_q0
        end
    end

    # counter += 1
    push!(dqmc_pc, sum(bins) /1000 / 16)
    
end

# -----------------------------------------------------------------------------------------------------------
# Lx = 4;
# betas = 2.0;
# mu = -2.0
# t = 1.0;
# Hubbard_U = 4.0;

# lattice = TriangularLattice(Lx)
# m = HubbardModel(l=lattice, t=t, U=Hubbard_U, mu=mu)
# mc = DQMC(
#     m, beta=beta, delta_tau=0.125, safe_mult=8,
#     thermalization=1000, sweeps=1000, measure_rate=1,
#     recorder=Discarder()
# )
# MonteCarlo.init!(mc)
# tri_lat = Triangular_Lattice(Lx, Lx);

# tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat);
# parameters = dqmc_parameters(beta=beta);
# hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U);

# stack = dqmc_matrices_stack_real_qr(tri_model, parameters)

# hs_field.conf = deepcopy(mc.field.conf)

# build_udt_stacks_modified!(hs_field, stack)
# reverse_build_stack(mc)
# MonteCarlo.propagate(mc)

# @stack_green_shortcuts
# @stack_bmats_shortcuts

# vector_tmp1 = stack.vector_tmp1
# vector_tmp2 = stack.vector_tmp2
# matrix_tmp = stack.matrix_tmp1

# global accepted = Int64(0)
# dims = stack.dims
# n_slices = stack.n_slices
# global current_range = Int64(1)

# bins = zeros(Float64, 1000)
# for idx = 1:2000

#     for current_slice in 1:n_slices
#         for current_site in 1:dims
#             propose_local(hs_field, greens, current_site, current_slice)
#             detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, current_site, current_slice)
#             p = exp(-ΔE_boson) * detratio
#             # println(p - hs_field.ratio)
#             if hs_field.ratio > 1 || rand() < hs_field.ratio
#                 accept_local!(mc, m, mc.field, current_site, current_slice, detratio, ΔE_boson, passthrough)
#                 accept_or_reject_modified(hs_field, greens, vector_tmp1, vector_tmp2, current_site, current_slice, dims)
#                 # 不更新B矩阵
#                 global accepted += 1
#                 # accepted += 1
#             end
#         end

#         MonteCarlo.propagate(mc)
#         get_single_B_slices_modified!(hs_field, current_slice, hopping_matrix_exp, vector_tmp1, B_slice,
#             hopping_matrix_exp_inv, vector_tmp2, B_slice_inv)

#         vmul!(matrix_tmp, B_slice, Ul)
#         copyto!(Ul, matrix_tmp)

#         if stack.slice_rem[current_slice] == 0
#             global current_range += 1
#             # current_range += 1
#             mul_diag_right!(matrix_tmp, Dl, dims)

#             # println(current_range)
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
#         # println("下面是sweep up时格林函数差")
#         # println(maximum(abs.((greens - mc.stack.greens))))

#     end

#     hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)

#     diag_element = Float64(0)
#     copyto!(Ur, I)
#     Dr .= one(Float64)
#     copyto!(Tr, I)

#     copyto!(stack.u_stack[end], I)
#     stack.d_stack[end] .= one(Float64)
#     copyto!(stack.t_stack[end], I)

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
#                 # accepted += 1
    
#                 # 同时更新B矩阵
#                 diag_element = hs_field.conf[current_site, current_slice] < 0 ? hs_field.elements[2] : hs_field.elements[1]
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
#             # current_range -= 1
#             # println(current_range)
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

#     hs_field.accepted_ratio = Float64(accepted) / (n_slices * dims)
#     copyto!(Ul, I)
#     Dl .= one(eltype(mc.stack.d_stack[end]))
#     copyto!(Tl, I)

#     copyto!(stack.u_stack[1], I)
#     stack.d_stack[1] .= one(Float64)
#     copyto!(stack.t_stack[1], I)

#     if idx > 1000
#         # occ_ = Float64(0)
#         # for site = 1:16
#         #     occ_ += (1 - greens[site,site])
#         # end
#         # bins[idx - 1000] = occ_ / 16

#         pc_q0 = Float64(0)
#         n_sites = 16
#         eThalfminus = stack.hopping_matrix_exp_half
#         eThalfplus = stack.hopping_matrix_exp_inv_half
#         vmul!(stack.matrix_tmp4, stack.greens, eThalfminus)
#         vmul!(stack.greens_temp, eThalfplus, stack.matrix_tmp4)
#         for site1 = 1:n_sites, site2 = 1:n_sites
#             # pc_q0 += bins_pc[sweep_idx][m,n]
#             pc_q0 += 2 * (stack.greens_temp[site1, site2])^2
#         end

#         for site = 1:n_sites
#             pc_q0 += (1 - 2 * stack.greens_temp[site, site])
#         end
#         bins[idx - 1000] = pc_q0
#     end
# end

# using Plots
# plot(dqmc_pc,linewidth=2,title="Pairing Correlation")

Lx = 20;
beta = 10.0;
mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)
mu = mus[1]
dqmc_pc = []
t = 1.0;
Hubbard_U = 4.0;

lattice = TriangularLattice(Lx)
m = HubbardModel(l=lattice, t=t, U=Hubbard_U, mu=mu)
mc = DQMC(
    m, beta=beta, delta_tau=0.125, safe_mult=8,
    thermalization=1000, sweeps=1000, measure_rate=1,
    recorder=Discarder()
)
mc[:SDZ] = spin_density_susceptibility(mc, model, :z)
MonteCarlo.init!(mc)

groups = MonteCarlo.generate_groups(
        mc, mc.model, 
        [mc.measurements[k] for k in keys(mc.measurements)]
    )
greens_iterator = MonteCarlo.CombinedGreensIterator(mc)
# 下面是我的代码
tri_lat = Triangular_Lattice(Lx, Lx);

tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat);
parameters = dqmc_parameters(beta=beta);
hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U);

stack = dqmc_matrices_stack_real_qr(tri_model, parameters)

hs_field.conf = deepcopy(mc.field.conf)

build_udt_stacks_modified!(hs_field, stack)
# copyto!(stack.u_stack[1], I)
# stack.d_stack[1] .= one(Float64)
# copyto!(stack.t_stack[1], I)

MonteCarlo.reverse_build_stack(mc, mc.stack)
MonteCarlo.propagate(mc)

global counter = 0
# global range = 1
Gll = similar(stack.greens)
copyto!(Gll, stack.greens)
Gl0 = similar(stack.greens)
copyto!(Gl0, stack.greens)
G0l = similar(stack.greens)
G0l = Gll - I

dims = stack.dims
U_tmp = Matrix{Float64}(I, dims, dims)
D_tmp = ones(Float64, dims)
T_tmp = Matrix{Float64}(I, dims, dims)


@time for (G0l_check, Gl0_check, Gll_check) in MonteCarlo.init(mc, greens_iterator)
    counter += 1
    get_uneqlt_greens(stack, hs_field, Gll, Gl0, G0l, U_tmp, D_tmp, T_tmp, counter, dims)
    # get_single_B_slices_modified!(hs_field, counter, stack.hopping_matrix_exp, stack.vector_tmp1, stack.current_B, stack.hopping_matrix_exp_inv, stack.vector_tmp2, stack.current_B_inv)
    # vmul!(stack.matrix_tmp1, stack.current_B, stack.Ul)
    # if stack.slice_rem[counter] == 0 # 做一次数值稳定    
    #     mul_diag_right!(stack.matrix_tmp1, stack.Dl, dims)

    #     range += 1
    #     println("做了一次数值稳定性")
    #     copyto!(stack.Ur, stack.u_stack[range])
    #     copyto!(stack.Dr, stack.d_stack[range])
    #     copyto!(stack.Tr, stack.t_stack[range])

    #     qr_real_pivot!(stack.Ul, stack.matrix_tmp1, stack.Dl, stack.pivot_tmp, dims)
    #     permute_rows!(T_tmp, stack.Tl, stack.pivot_tmp, dims)
    #     lmul!(UpperTriangular(stack.matrix_tmp1), T_tmp)  # TODO::能不能自己写个更快的函数??
    #     copyto!(stack.Tl, T_tmp)
    #     copyto!(U_tmp, stack.Ul)
    #     copyto!(D_tmp, stack.Dl)

    #     get_uneqlt_green_by_qr!(Gll, Gl0, G0l, U_tmp, D_tmp, T_tmp, stack.Ur, stack.Dr, stack.Tr, 
    #         stack.vector_tmp1, stack.pivot_tmp, stack.dims)
        
    # else
    #     copyto!(stack.Ul, stack.matrix_tmp1)
    #     get_uneqlt_green_by_wrap_up!(Gll, Gl0, G0l, stack.matrix_tmp1, stack.Ul, stack.current_B, stack.current_B_inv)
    # end
    
    
    # display(G0l_check - G0l)
    print(maximum(abs.(G0l_check - stack.hopping_matrix_exp_inv_half * G0l * stack.hopping_matrix_exp_half)))
    print("\t")
    print(sum(abs.(G0l_check - stack.hopping_matrix_exp_inv_half * G0l * stack.hopping_matrix_exp_half)))
    print("\t")
    # display(Gl0_check - Gl0)
    print(maximum(abs.(Gl0_check - stack.hopping_matrix_exp_inv_half * Gl0 * stack.hopping_matrix_exp_half)))
    print("\t")
    print(sum(abs.(Gl0_check - stack.hopping_matrix_exp_inv_half * Gl0 * stack.hopping_matrix_exp_half)))
    print("\t")
    # display(Gll_check - Gll)
    print(maximum(abs.(Gll_check - stack.hopping_matrix_exp_inv_half * Gll * stack.hopping_matrix_exp_half)))
    print("\t")
    print(sum(abs.(Gll_check - stack.hopping_matrix_exp_inv_half * Gll * stack.hopping_matrix_exp_half)))
    print("\n\n")
end

# Ul/Dl/Tl/Ur/Dr/Tr重新赋值一遍
copyto!(stack.Ur, stack.u_stack[1])
copyto!(stack.Dr, stack.d_stack[1])
copyto!(stack.Tr, stack.t_stack[1])

copyto!(stack.Ul, I)
stack.Dl = ones(Float64, dims)
copyto!(stack.Tl, I)
