## --------------------------------------------------------------------------------------- build_stack test
function reverse_build_stack(mc::DQMC)
    copyto!(mc.stack.u_stack[end], I)
    mc.stack.d_stack[end] .= one(eltype(mc.stack.d_stack[end]))
    copyto!(mc.stack.t_stack[end], I)

    @inbounds for i in length(mc.stack.ranges):-1:1
        copyto!(mc.stack.curr_U, mc.stack.u_stack[i + 1])
        add_slice_sequence_right(mc, i)
    end

    mc.stack.current_slice = 0
    # strictly this should be 0 but we special based on current_slice anyway
    mc.stack.current_range = 1
    mc.stack.direction = 1

    # Calculate valid greens function
    copyto!(mc.stack.Ul, I)
    mc.stack.Dl .= one(eltype(mc.stack.Dl))
    copyto!(mc.stack.Tl, I)
    copyto!(mc.stack.Ur, mc.stack.u_stack[1])
    copyto!(mc.stack.Dr, mc.stack.d_stack[1])
    copyto!(mc.stack.Tr, mc.stack.t_stack[1])
    calculate_greens(mc)

    nothing
end

function add_slice_sequence_right(mc::DQMC, idx::Int)
    @inbounds begin
        copyto!(mc.stack.curr_U, mc.stack.u_stack[idx + 1])

        for slice in reverse(mc.stack.ranges[idx])
            multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
        end

        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.d_stack[idx + 1]))

        udt_AVX_pivot!(
            mc.stack.u_stack[idx], mc.stack.d_stack[idx], mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv
        )

        vmul!(mc.stack.t_stack[idx], mc.stack.tmp1, mc.stack.t_stack[idx + 1])
    end
end

function multiply_daggered_slice_matrix_left!(
    mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = mc.field
)
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, adjoint(mc.stack.tmp2), M)
    M .= mc.stack.tmp1
    nothing
end

function slice_matrix!(
    mc::DQMC, m::Model, slice::Int, power::Float64 = 1.0, 
    result::AbstractArray = mc.stack.tmp2, field = mc.field
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

@inline function interaction_matrix_exp!(f::DensityHirschField, result::Diagonal, slice, power)
    N = size(f.conf, 1)
    @inbounds for i in eachindex(result.diag)
        result.diag[i] = exp(power * f.α * f.conf[mod1(i, N), slice])
    end
    nothing
end

@inline function interaction_matrix_exp!(
    mc, model, field, result, slice, power = +1.0
)
    interaction_matrix_exp!(field, result, slice, power)
end

#################################################################################################
function udt_AVX_pivot!(
    U::AbstractArray{C, 2}, 
    D::AbstractArray{C, 1}, 
    input::AbstractArray{C, 2},
    pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
    temp::AbstractArray{C, 1} = Vector{C}(undef, length(D)),
    apply_pivot::Val = Val(true)
) where {C<:Real}
# Assumptions:
# - all matrices same size
# - input can be mutated (input becomes T)

# @bm "reset pivot" begin
    n = size(input, 1)
    @inbounds for i in 1:n
        pivot[i] = i
    end
# end

# @bm "QR decomposition" begin
    @inbounds for j = 1:n
        # Find column with maximum norm in trailing submatrix
        # @bm "get jm" begin
            jm, maxval = indmaxcolumn(input, j, n)
        # end

        # @bm "pivot" begin
            if jm != j
                # Flip elements in pivoting vector
                tmpp = pivot[jm]
                pivot[jm] = pivot[j]
                pivot[j] = tmpp

                # Update matrix with
                @turbo for i = 1:n
                    tmp = input[i,jm]
                    input[i,jm] = input[i,j]
                    input[i,j] = tmp
                end
            end
        # end

        # Compute reflector of columns j
        # @bm "Reflector" begin
            τj = reflector!(input, maxval, j, n)
            temp[j] = τj
        # end

        # Update trailing submatrix with reflector
        # @bm "apply" begin
            # TODO optimize?
            x = LinearAlgebra.view(input, j:n, j)
            MonteCarlo.reflectorApply!(x, τj, LinearAlgebra.view(input, j:n, j+1:n))
        # end
    end
# end

# @bm "Calculate Q" begin
    copyto!(U, I)
    @inbounds begin
        U[n, n] -= temp[n]
        for k = n-1:-1:1
            for j = k:n
                vBj = U[k,j]
                @turbo for i = k+1:n
                    vBj += conj(input[i,k]) * U[i,j]
                end
                vBj = temp[k]*vBj
                U[k,j] -= vBj
                @turbo for i = k+1:n
                    U[i,j] -= input[i,k]*vBj
                end
            end
        end
    end
    # U done
# end

# @bm "Calculate D" begin
    @inbounds for i in 1:n
        D[i] = abs(real(input[i, i]))
    end
# end

# @bm "pivoted zeroed T w/ inv(D)" begin
    _apply_pivot!(input, D, temp, pivot, apply_pivot)
# end

nothing
end

function indmaxcolumn(A::Matrix{C}, j=1, n=size(A, 1)) where {C <: Real}
    squared_norm = 0.0
    @turbo for k in j:n
        squared_norm += abs2(A[k, j])
    end
    ii = j
    @inbounds for i in j+1:n
        mi = 0.0
        @turbo for k in j:n
            mi += abs2(A[k, i])
        end
        if abs(mi) > squared_norm
            squared_norm = mi
            ii = i
        end
    end
    return ii, squared_norm
end

@inline function reflector!(x::Matrix{C}, normu, j=1, n=size(x, 1)) where {C <: Real}
    @inbounds begin
        ξ1 = x[j, j]
        if iszero(normu)
            return zero(ξ1) #zero(ξ1/normu)
        end
        normu = sqrt(normu)
        ν = LinearAlgebra.copysign(normu, real(ξ1))
        ξ1 += ν
        x[j, j] = -ν
        @turbo for i = j+1:n
            x[i, j] /= ξ1
        end
    end
    ξ1/ν
end

# Needed for pivoted as well
@inline function reflectorApply!(x::AbstractVector{<: Real}, τ::Real, A::StridedMatrix{<: Real})
    m, n = size(A)
    @inbounds for j = 1:n
        # dot
        vAj = A[1, j]
        @turbo for i = 2:m
            vAj += conj(x[i]) * A[i, j]
        end

        vAj = conj(τ)*vAj

        # ger
        A[1, j] -= vAj
        @turbo for i = 2:m
            A[i, j] -= x[i]*vAj
        end
    end
    return A
end

function _apply_pivot!(input::Matrix{C}, D, temp, pivot, ::Val{true}) where {C <: Real}
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @inbounds for j in 1:i-1
            temp[pivot[j]] = zero(C)
        end
        @turbo for j in i:n  # 时间花费较多
            temp[pivot[j]] = d * input[i, j]
        end
        @turbo for j in 1:n  # 时间花费较多
            input[i, j] = temp[j]
        end
    end
end

function _apply_pivot!(input::Matrix{C}, D, temp, pivot, ::Val{false}) where {C <: Real}
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @turbo for j in i:n
            input[i, j] = d * input[i, j]
        end
    end
end

function calculate_greens(mc::DQMC, output::AbstractMatrix = mc.stack.greens)
    calculate_greens_AVX!(
        mc.stack.Ul, mc.stack.Dl, mc.stack.Tl,
        mc.stack.Ur, mc.stack.Dr, mc.stack.Tr,
        output, mc.stack.pivot, mc.stack.tempv
    )
    output
end

function calculate_greens_AVX!(
    Ul, Dl, Tl, Ur, Dr, Tr, G::AbstractArray{T},
    pivot = Vector{Int64}(undef, length(Dl)),
    temp = Vector{T}(undef, length(Dl))
) where T
# @bm "B1" begin
    # Used: Ul, Dl, Tl, Ur, Dr, Tr
    # TODO: [I + Ul Dl Tl Tr^† Dr Ur^†]^-1
    # Compute: Dl * ((Tl * Tr^†) * Dr) -> Tr * Dr * G   (UDT)
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

function vmul!(C::Matrix{T}, A::Diagonal{T}, B::Matrix{T}) where {T <: Real}
    @turbo for m in 1:size(C, 1), n in 1:size(C, 2)
        C[m,n] = A.diag[m] * B[m,n]
    end
end

function vmul!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = A[m,n] * B.diag[n]
    end
end

function vmul!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}) where {T <: Real}
    B = X.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * conj(B[n, k])
        end
        C[m,n] = Cmn
    end
end

function vmul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

function vmul!(C::Matrix{T}, X::Adjoint{T}, B::Matrix{T}) where {T <: Real}
    A = X.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += conj(A[k,m]) * B[k,n]
        end
        C[m,n] = Cmn
    end
end

function vinv!(v::Vector{T}, w::Vector{T}) where {T<:Real}
    T1 = one(T)
    @turbo for i in eachindex(v)
        v[i] = T1 / w[i]
    end
    v
end

function rdivp!(A::Matrix, T, O, pivot)
    # assume Diagonal is ±1!
    @inbounds begin
        N = size(A, 1)

        # Apply pivot
        for (j, p) in enumerate(pivot)
            @turbo for i in 1:N
                O[i, j] = A[i, p]
            end
        end

        # do the rdiv
        # @turbo will segfault on `k in 1:0`, so pull out first loop 
        @turbo for i in 1:N
            A[i, 1] = O[i, 1] / T[1, 1]
        end
        for j in 2:N
            @turbo for i in 1:N
                x = O[i, j]
                for k in 1:j-1
                    x -= A[i, k] * T[k, j]
                end
                A[i, j] = x / T[j, j]
            end
        end
    end
    A
end

function rvadd!(A::Matrix{T}, D::Diagonal{T}) where {T <: Real}
    @turbo for i in axes(A, 1)
        A[i, i] = A[i, i] + D.diag[i]
    end
end