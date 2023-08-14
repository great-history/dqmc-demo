## ffreyer's code test::
## 更新

function update(s::SimpleScheduler, mc::DQMC, model, field=mc.field)
    while true
        s.idx = mod1(s.idx + 1, length(s.sequence))
        update(s.sequence[s.idx], mc, model, field)
        MonteCarlo.is_full_sweep(s.sequence[s.idx]) && break
    end
    mc.last_sweep += 1
    return nothing
end

function update(w::MonteCarlo.AcceptanceStatistics, mc, m, field)
    accepted = update(w.update, mc, m, field)
    w.total += 1
    w.accepted += accepted
    return accepted
end

update(::LocalSweep, mc::DQMC, model, field) = local_sweep(mc, model) / 2length(field)

function local_sweep(mc::DQMC, model)
    accepted = 0
    for _ in 1:2*mc.parameters.slices
        accepted += sweep_spatial(mc, model)
        MonteCarlo.propagate(mc)
    end
    return accepted
end

function sweep_spatial(mc::DQMC, m)
    N = size(mc.field.conf, 1)

    # @inbounds for i in rand(1:N, N)
    accepted = 0
    @inbounds for i in 1:N
        detratio, ΔE_boson, passthrough = propose_local(mc, m, mc.field, i, mc.stack.current_slice)

        p = exp(-ΔE_boson) * detratio

        if mc.parameters.check_sign_problem
            if abs(imag(p)) > 1e-6
                push!(mc.analysis.imaginary_probability, abs(imag(p)))
                mc.parameters.silent || println(
                    "Did you expect a sign problem? imag. probability:  %.9e\n",
                    abs(imag(p))
                )
            end
            if real(p) < 0.0
                push!(mc.analysis.negative_probability, real(p))
                mc.parameters.silent || println(
                    "Did you expect a sign problem? negative probability %.9e\n",
                    real(p)
                )
            end
        end

        # Gibbs/Heat bath
        # p = p / (1.0 + p)
        # Metropolis
        if real(p) > 1 || rand() < real(p)
            accept_local!(mc, m, mc.field, i, mc.stack.current_slice, detratio, ΔE_boson, passthrough)
            accepted += 1
        end
    end

    return accepted
end

@inline propose_local(mc, m, field, i, slice) = propose_local(mc, field, i, slice)

function propose_local(mc, f::MonteCarlo.DensityHirschField, i, slice)
    @inbounds ΔE_boson = -2.0 * f.α * f.conf[i, slice]
    mc.stack.field_cache.Δ = exp(ΔE_boson) - 1
    detratio = MonteCarlo.calculate_detratio!(mc.stack.field_cache, mc.stack.greens, i)
    return detratio, ΔE_boson, nothing
end

@inline function accept_local!(
    mc, m, field, i, slice, detratio, ΔE_boson, passthrough
)
    accept_local!(mc, field, i, slice, detratio, ΔE_boson, passthrough)
end

function accept_local!(mc, f::MonteCarlo.AbstractHirschField, i, slice, args...)
    update_greens!(mc.stack.field_cache, mc.stack.greens, i, size(f.conf, 1))
    @inbounds f.conf[i, slice] *= -1
    nothing
end

function update_greens!(cache::MonteCarlo.StandardFieldCache, G, i, N)
    # calculate Δ R⁻¹
    MonteCarlo.vldiv22!(cache, cache.R, cache.Δ)

    # calculate (I - G)[:, i:N:end]
    MonteCarlo.vsub!(cache.IG, I, G, i, N)

    # calculate {Δ R⁻¹} * G[i:N:end, :]
    MonteCarlo.vmul!(cache.G, cache.invRΔ, G, i, N)

    # update greens function 
    # G[m, n] -= {(I - G)[m, i:N:end]} {{Δ R⁻¹} * G[i:N:end, n]}
    MonteCarlo.vsubkron!(G, cache.IG, cache.G)

    nothing
end

@inline function wrap_greens!(mc::DQMC, gf, curr_slice::Int, direction::Int)
    if direction == -1
        multiply_slice_matrix_inv_left!(mc, mc.model, curr_slice - 1, gf)
        multiply_slice_matrix_right!(mc, mc.model, curr_slice - 1, gf)
    else
        multiply_slice_matrix_left!(mc, mc.model, curr_slice, gf)
        multiply_slice_matrix_inv_right!(mc, mc.model, curr_slice, gf)
    end
    nothing
end

function multiply_slice_matrix_left!(
    mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field=field(mc)
)
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, mc.stack.tmp2, M)
    M .= mc.stack.tmp1
    nothing
end
function multiply_slice_matrix_right!(
    mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field=field(mc)
)
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, M, mc.stack.tmp2)
    M .= mc.stack.tmp1
    nothing
end
function multiply_slice_matrix_inv_right!(
    mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field=field(mc)
)
    slice_matrix!(mc, m, slice, -1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, M, mc.stack.tmp2)
    M .= mc.stack.tmp1
    nothing
end
function multiply_slice_matrix_inv_left!(
    mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field=field(mc)
)
    slice_matrix!(mc, m, slice, -1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, mc.stack.tmp2, M)
    M .= mc.stack.tmp1
    nothing
end

function slice_matrix!(
    mc::DQMC, m::Model, slice::Int, power::Float64=1.0,
    result::AbstractArray=mc.stack.tmp2, field=field(mc)
)
    eT2 = mc.stack.hopping_matrix_exp_squared
    eTinv2 = mc.stack.hopping_matrix_exp_inv_squared
    eV = mc.stack.eV

    interaction_matrix_exp!(mc, m, field, eV, slice, power)

    if power > 0
        # eT * (eT * eV)
        vmul!(result, eT2, eV)
    else
        # ev * (eTinv * eTinv)
        vmul!(result, eV, eTinv2)
    end
    return result
end

function add_slice_sequence_left(mc::DQMC, idx::Int)
    @inbounds begin
        copyto!(mc.stack.curr_U, mc.stack.u_stack[idx])

        # println("Adding slice seq left $idx = ", mc.stack.ranges[idx])
        for slice in mc.stack.ranges[idx]
            multiply_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
        end

        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.d_stack[idx]))
        udt_AVX_pivot!(
            mc.stack.u_stack[idx+1], mc.stack.d_stack[idx+1], mc.stack.tmp1,
            mc.stack.pivot, mc.stack.tempv
        )
        vmul!(mc.stack.t_stack[idx+1], mc.stack.tmp1, mc.stack.t_stack[idx])
    end
end

# Green's function propagation
@inline function wrap_greens!(mc::DQMC, gf, curr_slice::Int, direction::Int)
    if direction == -1
        multiply_slice_matrix_inv_left!(mc, mc.model, curr_slice - 1, gf)
        multiply_slice_matrix_right!(mc, mc.model, curr_slice - 1, gf)
    else
        multiply_slice_matrix_left!(mc, mc.model, curr_slice, gf)
        multiply_slice_matrix_inv_right!(mc, mc.model, curr_slice, gf)
    end
    nothing
end

function propagate(mc::DQMC)
    flush(stdout)
    @debug(
        '[' * lpad(mc.stack.current_slice, 3, ' ') * " -> " *
        rpad(mc.stack.current_slice + mc.stack.direction, 3, ' ') *
        ", " * mc.stack.direction == +1 ? '+' : '-', "] "
    )

    # Advance according to direction
    mc.stack.current_slice += mc.stack.direction

    @inbounds if mc.stack.direction == 1
        if mc.stack.current_slice == 1
            @debug("init direction, clearing 1 to I")
            copyto!(mc.stack.u_stack[1], I)
            mc.stack.d_stack[1] .= one(eltype(mc.stack.d_stack[1]))
            copyto!(mc.stack.t_stack[1], I)

        elseif mc.stack.current_slice - 1 == last(mc.stack.ranges[mc.stack.current_range])
            idx = mc.stack.current_range
            @debug("Stabilize: decompose into $idx -> $(idx+1)")

            copyto!(mc.stack.Ur, mc.stack.u_stack[idx+1])
            copyto!(mc.stack.Dr, mc.stack.d_stack[idx+1])
            copyto!(mc.stack.Tr, mc.stack.t_stack[idx+1])
            add_slice_sequence_left(mc, idx)
            copyto!(mc.stack.Ul, mc.stack.u_stack[idx+1])
            copyto!(mc.stack.Dl, mc.stack.d_stack[idx+1])
            copyto!(mc.stack.Tl, mc.stack.t_stack[idx+1])

            if mc.parameters.check_propagation_error
                copyto!(mc.stack.greens_temp, mc.stack.greens)
            end

            # Should this be mc.stack.greens_temp?
            # If so, shouldn't this only run w/ mc.parameters.all_checks = true?
            wrap_greens!(mc, mc.stack.greens_temp, mc.stack.current_slice - 1, 1)

            calculate_greens(mc) # greens_{slice we are propagating to}

            if mc.parameters.check_propagation_error
                # OPT: could probably be optimized through explicit loop
                greensdiff = maximum(abs.(mc.stack.greens_temp - mc.stack.greens))
                if greensdiff > 1e-7
                    push!(mc.analysis.propagation_error, greensdiff)
                    mc.parameters.silent || println(
                        "->%d \t+1 Propagation instability\t %.1e\n",
                        mc.stack.current_slice, greensdiff
                    )
                end
            end

            if mc.stack.current_range == length(mc.stack.ranges)
                # We are going from M -> M+1. Switch direction.
                @assert mc.stack.current_slice == mc.parameters.slices + 1
                mc.stack.direction = -1
                propagate(mc)
            else
                mc.stack.current_range += 1
            end

        else
            @debug("standard wrap")
            # Wrapping (we already advanced current_slice but wrap according to previous)
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice - 1, 1)
        end

    else # REVERSE
        if mc.stack.current_slice == mc.parameters.slices
            @debug("init direction, clearing end to I")
            copyto!(mc.stack.u_stack[end], I)
            mc.stack.d_stack[end] .= one(eltype(mc.stack.d_stack[end]))
            copyto!(mc.stack.t_stack[end], I)

            # wrap to greens_{mc.parameters.slices}
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice + 1, -1)

        elseif mc.stack.current_slice + 1 == first(mc.stack.ranges[mc.stack.current_range])
            idx = mc.stack.current_range
            @debug("Stabilize: decompose into $(idx+1) -> $idx")

            copyto!(mc.stack.Ul, mc.stack.u_stack[idx])
            copyto!(mc.stack.Dl, mc.stack.d_stack[idx])
            copyto!(mc.stack.Tl, mc.stack.t_stack[idx])
            add_slice_sequence_right(mc, idx)
            copyto!(mc.stack.Ur, mc.stack.u_stack[idx])
            copyto!(mc.stack.Dr, mc.stack.d_stack[idx])
            copyto!(mc.stack.Tr, mc.stack.t_stack[idx])

            if mc.parameters.check_propagation_error
                copyto!(mc.stack.greens_temp, mc.stack.greens)
            end

            calculate_greens(mc)

            if mc.parameters.check_propagation_error
                # OPT: could probably be optimized through explicit loop
                greensdiff = maximum(abs.(mc.stack.greens_temp - mc.stack.greens))
                if greensdiff > 1e-7
                    push!(mc.analysis.propagation_error, greensdiff)
                    mc.parameters.silent || println(
                        "->%d \t-1 Propagation instability\t %.1e\n",
                        mc.stack.current_slice, greensdiff
                    )
                end
            end

            if mc.stack.current_range == 1
                # We are going from 1 -> 0. Switch direction.
                @assert mc.stack.current_slice == 0
                mc.stack.direction = +1
                propagate(mc)
            else
                # We'd be undoing this wrap in forward 1 if we always applied it
                wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice + 1, -1)
                mc.stack.current_range -= 1
            end

        else
            @debug("standard wrap")
            # Wrapping (we already advanced current_slice but wrap according to previous)
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice + 1, -1)
        end
    end

    nothing
end

function calculate_greens(mc::DQMC, output::AbstractMatrix=mc.stack.greens)
    calculate_greens_AVX!(
        mc.stack.Ul, mc.stack.Dl, mc.stack.Tl,
        mc.stack.Ur, mc.stack.Dr, mc.stack.Tr,
        output, mc.stack.pivot, mc.stack.tempv
    )
    output
end

function calculate_greens_AVX!(
    Ul, Dl, Tl, Ur, Dr, Tr, G::AbstractArray{T},
    pivot=Vector{Int64}(undef, length(Dl)),
    temp=Vector{T}(undef, length(Dl))
) where {T}
    # @bm "B1" begin
    # Used: Ul, Dl, Tl, Ur, Dr, Tr
    # TODO: [I + Ul Dl Tl Tr^† Dr Ur^†]^-1
    # Compute: Dl * ((Tl * Tr) * Dr) -> Tr * Dr * G   (UDT)
    vmul!(G, Tl, adjoint(Tr))
    vmul!(Tr, G, Diagonal(Dr))
    vmul!(G, Diagonal(Dl), Tr)
    udt_AVX_pivot!(Tr, Dr, G, pivot, temp, Val(false)) # Dl available
    # end

    # @bm "B2" begin
    # Used: Ul, Ur, G, Tr, Dr  (Ul, Ur, Tr unitary (inv = adjoint))
    # TODO: [I + Ul Tr Dr G Ur^†]^-1
    #     = [(Ul Tr) ((Ul Tr)^-1 (G Ur^†) + Dr) (G Ur)]^-1
    #     = Ur G^-1 [(Ul Tr)^† Ur G^-1 + Dr]^-1 (Ul Tr)^†
    # Compute: Ul Tr -> Tl
    #          (Ur G^-1) -> Ur
    #          ((Ul Tr)^† Ur G^-1) -> Tr
    vmul!(Tl, Ul, Tr)
    rdivp!(Ur, G, Ul, pivot) # requires unpivoted udt decompostion (Val(false))
    vmul!(Tr, adjoint(Tl), Ur)
    # end

    # @bm "B3" begin
    # Used: Tl, Ur, Tr, Dr
    # TODO: Ur [Tr + Dr]^-1 Tl^† -> Ur [Tr]^-1 Tl^†
    rvadd!(Tr, Diagonal(Dr))
    # end

    # @bm "B4" begin
    # Used: Ur, Tr, Tl
    # TODO: Ur [Tr]^-1 Tl^† -> Ur [Ul Dr Tr]^-1 Tl^† 
    #    -> Ur Tr^-1 Dr^-1 Ul^† Tl^† -> Ur Tr^-1 Dr^-1 (Tl Ul)^†
    # Compute: Ur Tr^-1 -> Ur,  Tl Ul -> Tr
    udt_AVX_pivot!(Ul, Dr, Tr, pivot, temp, Val(false)) # Dl available
    rdivp!(Ur, Tr, G, pivot) # requires unpivoted udt decompostion (false)
    vmul!(Tr, Tl, Ul)
    # end

    # @bm "B5" begin
    vinv!(Dl, Dr)
    # end

    # @bm "B6" begin
    # Used: Ur, Tr, Dl, Ul, Tl
    # TODO: (Ur Dl) Tr^† -> G
    vmul!(Ul, Ur, Diagonal(Dl))
    vmul!(G, Ul, adjoint(Tr))
    # end
end