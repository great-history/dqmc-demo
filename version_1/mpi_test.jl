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

import MPI
MPI.Init()
comm = MPI.COMM_WORLD

# N_process = MPI.Comm_size(comm)
# root = 0

# N_betas = 11
# N_mus = 26
# a = N_betas % N_process
# b = N_betas ÷ N_process
# rank = MPI.Comm_rank(comm)

# if rank == root
#     betas = vcat(2.0:0.5:7.0)
#     mus = vcat(-2.0:0.16:2.0)
#     print(" Running on $(MPI.Comm_size(comm)) processes\n")
#     if a > 0
#         betas_rank = zeros(Float64, b + 1) 
#         for i in 1:b+1
#             betas_rank[i] = betas[1+(i-1)*N_process]
#         end
#     else
#         betas_rank = zeros(Float64, b)
#         for i in 1:b
#             betas_rank[i] = betas[1+(i-1)*N_process]
#         end
#     end
# else
#     betas = zeros(Float64, N_betas)
#     mus = zeros(Float64, N_mus)
#     if rank < a
#         betas_rank = zeros(Float64, b + 1)
#     else
#         betas_rank = zeros(Float64, b)
#     end
# end

# MPI.Bcast!(betas, root, comm)
# MPI.Bcast!(mus, root, comm)
# MPI.Barrier(comm)

# if rank != root
#     if rank < a
#         for i in 1:b+1
#             betas_rank[i] = betas[1 + rank + (i-1)*N_process]
#         end
#     else
#         for i in 1:b
#             betas_rank[i] = betas[1 + rank + (i-1)*N_process]
#         end
#     end
# end
# MPI.Barrier(comm)

Lx = 4;
beta = 2.0;
mu = -2.0;
t = 1.0;
Hubbard_U = 4.0;

tri_lat = Triangular_Lattice(Lx, Lx);
# counter = 0
# N = length(mus) * length(betas_rank)
# @time for beta in betas_rank, mu in mus
#     global counter += 1
#     print("\r[", lpad("$counter", 2), "/$N]")
#     tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat)
#     parameters = dqmc_parameters(beta=beta)
#     hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, 
#         n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U)
#     stack = dqmc_matrices_stack_real_qr(tri_model, parameters)

#     thermalization_sweeps(hs_field, stack)
# end
tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat)
parameters = dqmc_parameters(beta=beta)
hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, 
    n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U)
stack = dqmc_matrices_stack_real_qr(tri_model, parameters)

thermalization_sweeps(hs_field, stack)