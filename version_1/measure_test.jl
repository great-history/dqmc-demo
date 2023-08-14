using MonteCarlo
using Dates

global cc_counter = 0

function Base.iterate(iter::MonteCarlo._EachLocalQuadByDistance)
    src = iter.src_mask[1]
    dir12 = iter.srctrg2dir[src, src]
    dir, trg = iter.filtered_src2dirtrg[src][1]
    combined_dir = dir12 + iter.mult[1] * (dir-1) + iter.mult[2] * (dir-1)
    # state = (src1 mask index, src2 mask index, filter1 index, filter2 index)
    return ((combined_dir, src, trg, src, trg), (1, 1, 1, 1))
end

function Base.iterate(iter::MonteCarlo._EachLocalQuadByDistance, state)
    midx1, midx2, fidx1, fidx2 = state
    src1 = iter.src_mask[midx1]
    src2 = iter.src_mask[midx2]

    b1 = fidx2 == length(iter.filtered_src2dirtrg[src2])
    b2 = b1 && (fidx1 == length(iter.filtered_src2dirtrg[src1]))
    b3 = b2 && (midx2 == length(iter.src_mask))
    b4 = b3 && (midx1 == length(iter.src_mask))
    b4 && return nothing
    fidx2 = Int64(b1 || (fidx2 + 1))
    fidx1 = Int64(b2 || (fidx1 + b1))
    midx2 = Int64(b3 || (midx2 + b2))
    midx1 = Int64(midx1 + b3)

    src1 = iter.src_mask[midx1]
    src2 = iter.src_mask[midx2]
    dir12 = iter.srctrg2dir[src1, src2]
    dir1, trg1 = iter.filtered_src2dirtrg[src1][fidx1]
    dir2, trg2 = iter.filtered_src2dirtrg[src2][fidx2]
    combined_dir = dir12 + iter.mult[1] * (dir1-1) + iter.mult[2] * (dir2-1)
    # state = (src1 mask index, src2 mask index, filter1 index, filter2 index)
    return ((combined_dir, src1, trg1, src2, trg2), (midx1, midx2, fidx1, fidx2))
end


prepare!(::MonteCarlo.AbstractLatticeIterator, model, m) = m.temp .= zero(eltype(m.temp))

function apply!(temp::Array, iter::MonteCarlo.DeferredLatticeIterator, measurement, mc::DQMC, model, packed_greens)
    flv = Val(MonteCarlo.nflavors(mc))
    @inbounds for idxs in iter
        # if first(idxs) == 10
        #     println(idxs)
        #     global cc_counter += 1
        #     println(cc_counter)
        # end
        temp[first(idxs)] += measurement.kernel(mc, model, idxs[2:end], packed_greens, flv)

    end
    nothing
end

function apply!(temp::Array, iter::MonteCarlo.DirectLatticeIterator, measurement, mc::DQMC, model, packed_greens)
    flv = Val(MonteCarlo.nflavors(mc))
    for i in iter
        # print(i)
        temp[i] += measurement.kernel(mc, model, i, packed_greens, flv)
    end
    nothing
end

# Call kernel for each pair (src, trg) (Nsties² total)
function apply!(temp::Array, iter::MonteCarlo.EachSitePair, measurement, mc::DQMC, model, packed_greens)
    flv = Val(MonteCarlo.nflavors(mc))
    for (i, j) in iter
        temp[i, j] += measurement.kernel(mc, model, (i, j), packed_greens, flv)
    end
    nothing
end

function apply!(temp::Array, s::MonteCarlo.LatticeIterationWrapper, m, mc, model, pg)
    apply!(temp, s.iter, m, mc, model, pg)
end

# Call kernel for each pair (site, site) (i.e. on-site) 
function apply!(temp::Array, iter::MonteCarlo._OnSite, measurement, mc::DQMC, model, packed_greens)
    flv = Val(MonteCarlo.nflavors(mc))
    for (i, j) in iter
        temp[i] += measurement.kernel(mc, model, (i, j), packed_greens, flv)
    end
    nothing
end

function measure!(lattice_iterator, measurement, mc::DQMC, model, sweep, packed_greens)
    # ignore sweep
    apply!(measurement.temp, lattice_iterator, measurement, mc, model, packed_greens)
    nothing
end

@inline function finalize_temp!(::MonteCarlo.AbstractLatticeIterator, model, m, factor)
    m.temp .*= factor
end

@inline commit!(::MonteCarlo.AbstractLatticeIterator, m) = push!(m.observable, m.temp)

@inline function finish!(li, model, m, factor=1.0)
    finalize_temp!(li, model, m, factor)
    commit!(li, m)
end

function apply!(::Greens, combined::Vector{<:Tuple}, mc::DQMC, model, sweep)
    G = greens!(mc)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, model, measurement)
        measure!(lattice_iterator, measurement, mc, model, sweep, G)
        finish!(lattice_iterator, model, measurement)
    end
    nothing
end

function apply!(iter::CombinedGreensIterator, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, model, measurement)
    end

    G00 = greens!(mc)
    for (G0l, Gl0, Gll) in init(mc, iter)
        # print(G0l)
        for (lattice_iterator, measurement) in combined
            measure!(lattice_iterator, measurement, mc, model, sweep, (G00, G0l, Gl0, Gll))
        end
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, model, measurement, mc.parameters.delta_tau)
    end
    nothing
end

# ------------------------------------------------------------------------------------------------------下面是测试代码
Lx = 4;
beta = 2.0;
mu = -2.0;
# betas = (2.0, 5.0, 7.0)
# mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)
t = 1.0;
Hubbard_U = 4.0;
ignore = tuple()
verbose = true

lattice = TriangularLattice(Lx)
m = HubbardModel(l=lattice, t=t, U=Hubbard_U, mu=mu)
mc = DQMC(
    m, beta=beta, delta_tau=0.125, safe_mult=8,
    thermalization=1000, sweeps=1000, measure_rate=1,
    recorder=Discarder()
)
MonteCarlo.init!(mc)
# tri_lat = Triangular_Lattice(Lx, Lx);
# tri_model = Single_Band_Hubbard_Model(t=t, U=Hubbard_U, mu=mu, lattice=tri_lat);
# parameters = dqmc_parameters(beta=beta);
# hs_field = HSField_CDW(delta_tau=parameters.delta_tau, slice_num=parameters.slices, n_sites=tri_model.lattice.sites, U=tri_model.Hubbard_U);
# stack = dqmc_matrices_stack_real_qr(tri_model, parameters)
# hs_field.conf = deepcopy(mc.field.conf)
# build_udt_stacks_modified!(hs_field, stack)

# mc[:occ] = MonteCarlo.occupation(mc, m)
function my_kernel(::DQMC, ::HubbardModel, ij::NTuple{2}, G::AbstractArray)
    i, j = ij
    4 * (I[j, i] - G[j, i]) * G[i, j]
end

mc[:CDC] = MonteCarlo.Measurement(mc, model, Greens, EachSitePairByDistance, my_kernel)

mc[:PC] = MonteCarlo.pairing_correlation(mc, m, kernel=MonteCarlo.pc_kernel)

thermalization = mc.parameters.thermalization
sweeps = mc.parameters.sweeps
total_sweeps = sweeps + thermalization

th_groups = MonteCarlo.generate_groups(
    mc, mc.model,
    [mc.measurements[k] for k in keys(mc.thermalization_measurements) if !(k in ignore)]
)
groups = MonteCarlo.generate_groups(
    mc, mc.model,
    [mc.measurements[k] for k in keys(mc.measurements) if !(k in ignore)]
)
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
# fresh stack
println("Preparing Green's function stack")
MonteCarlo.reverse_build_stack(mc, mc.stack)
MonteCarlo.propagate(mc)

bins_pc_av = zeros(Float64, 1000)
while mc.last_sweep < total_sweeps
    verbose && (mc.last_sweep == thermalization + 1) && println("\n\nMeasurement stage - ", sweeps)
    MonteCarlo.update(mc.scheduler, mc, mc.model)
    if mc.last_sweep > thermalization
        # 来自FFreyer
        push!(mc.recorder, mc.field, mc.last_sweep)
        if iszero(mc.last_sweep % mc.parameters.measure_rate)
            for (requirement, group) in groups
                # println(typeof(requirement))
                apply!(requirement, group, mc, mc.model, mc.last_sweep)
            end
        end

        # 来自我的代码
        pc_q0 = zero(Float64)
        n_sites = Lx * Lx
        eThalfminus = mc.stack.hopping_matrix_exp
        eThalfplus = mc.stack.hopping_matrix_exp_inv
        A = similar(mc.stack.greens)
        B = similar(mc.stack.greens)
        vmul!(A, mc.stack.greens, eThalfminus)
        vmul!(B, eThalfplus, A)
        for site1 = 1:n_sites, site2 = 1:n_sites
            # pc_q0 += bins_pc[sweep_idx][m,n]
            pc_q0 += (B[site1, site2])^2
        end

        bins_pc_av[mc.last_sweep - 1000] = pc_q0  # 应该是除以格点数的平方
    end
    # println(mc.last_sweep)
end

println(maximum(bins_pc_av))
println(minimum(bins_pc_av))
sum(bins_pc_av) / 1000 / 256
sum(mean(mc[:PC])[:, 1, 1]) / 256

# # measure_test:如何计算pairing correlation
# betas = (2.0, 5.0, 7.0)
# mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)
# lattice = TriangularLattice(4)
# dqmcs = []

# counter = 0
# N = length(mus) * length(betas)
# # ------------------------------------------------------------------------------------------------------------



# # ------------------------------------------------------------------------------------------------------------
# @time for beta in betas, mu in mus
#     global counter += 1
#     print("\r[", lpad("$counter", 2), "/$N]")
#     m = HubbardModel(l = lattice, t = 1.0, U = 4.0, mu = mu)
#     dqmc = DQMC(
#         m, beta = beta, delta_tau = 0.125, safe_mult = 8, 
#         thermalization = 1000, sweeps = 1000, measure_rate = 1,
#         recorder = Discarder()
#     )
#     dqmc[:occ] = occupation(dqmc, m)
#     dqmc[:PC] = pairing_correlation(dqmc, m, kernel = MonteCarlo.pc_kernel)
#     # run!(dqmc, verbose = false)
#     MonteCarlo.init!(dqmc)

#     thermalization = dqmc.parameters.thermalization
#     sweeps = dqmc.parameters.sweeps
#     total_sweeps = sweeps + thermalization
    
#     th_groups = MonteCarlo.generate_groups(
#         dqmc, dqmc.model,
#         [dqmc.measurements[k] for k in keys(dqmc.thermalization_measurements)]
#     )
#     groups = MonteCarlo.generate_groups(
#         dqmc, dqmc.model,
#         [dqmc.measurements[k] for k in keys(dqmc.measurements)]
#     )

#     MonteCarlo.reverse_build_stack(dqmc, dqmc.stack)
#     MonteCarlo.propagate(dqmc)
#     while dqmc.last_sweep < total_sweeps
#         verbose && (dqmc.last_sweep == thermalization + 1) && println("\n\nMeasurement stage - ", sweeps)
#         MonteCarlo.update(dqmc.scheduler, dqmc, dqmc.model)
#         if dqmc.last_sweep > thermalization
#             push!(dqmc.recorder, dqmc.field, dqmc.last_sweep)
#             if iszero(dqmc.last_sweep % dqmc.parameters.measure_rate)
#                 for (requirement, group) in groups
#                     apply!(requirement, group, dqmc, dqmc.model, dqmc.last_sweep)
#                 end
#             end
#         end
#         # println(mc.last_sweep)
#     end

#     # for simplicity we just keep the whole simulation around
#     push!(dqmcs, dqmc)
# end



# equal time cdc 
bins_occ = zeros(Float64, meas_sweeps)
bins_cdc = [Matrix{Float64}(I, Lx, Lx) for _ in 1:meas_sweeps]
if idx > 1000
    n_sites = 16
    eThalfminus = stack.hopping_matrix_exp_half
    eThalfplus = stack.hopping_matrix_exp_inv_half
    vmul!(stack.matrix_tmp4, stack.greens, eThalfminus)
    vmul!(stack.greens_temp, eThalfplus, stack.matrix_tmp4)

    occ_ = Float64(0)
    for site = 1:16
        occ_ += (1 - stack.greens_temp[site,site])
    end
    bins[idx - 1000] = occ_ / 16
    pc_q0 = Float64(0)
    
    # 对称性 i,j位置互换其实是不变的 TODO::修改，防止重复计算
    for site1 = 1:n_sites, site2 = 1:n_sites
        bins_cdc[idx-1000][site1, site2] = 4 * (1 - stack.greens_temp[site1, site1]) * (1 - stack.greens_temp[site2, site2])
            - 2 * stack.greens_temp[site2, site1] * stack.greens_temp[site1, site2]
        if site1 == site2
            bins_cdc[idx-1000][site1, site1] += 2 * stack.greens_temp[site1, site1]
        end
    end
    
end


betas = (2.0, 5.0, 7.0)
mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)

Lx = 4
Ly = 4
t = 1.0
Hubbard_U = 4.0

thermal_sweeps = 1000
meas_sweeps = 1000

beta = betas[1]
mu = mus[1]

