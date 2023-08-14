# Lattices

abstract type AbstractLattice end

struct Triangular_Lattice{C <: Int} <: AbstractLattice
    Lx::C
    Ly::C
    sites::C
    n_coord::C

    Lattice::Matrix{C}
    nearest_neighbors::Matrix{C}
end

function Triangular_Lattice(Lx::Int, Ly::Int)
    sites = Lx * Ly;
    lattice_array = convert(Array, reshape(1:sites, (Lx, Ly)))
    n_coord = 6;

    nearest_neighbors = Lattice_build_nn(Triangular_Lattice, lattice_array)
    
    return Triangular_Lattice(Lx, Ly, sites, n_coord, lattice_array, nearest_neighbors)
end

function Lattice_build_nn(::Type{Triangular_Lattice}, lattice_array)
    up = circshift(lattice_array, (1,0))
    upleft = circshift(lattice_array, (1,1))
    left = circshift(lattice_array, (0,-1))

    down = circshift(lattice_array, (-1,0))
    downright = circshift(lattice_array, (-1,-1))
    right = circshift(lattice_array, (0,1))

    return vcat(up[:]',upleft[:]',right[:]',down[:]',downright[:]',left[:]')
end

function get_coords(lattice::Triangular_Lattice)  # TODO::多重分派
    vec_a1 = [1 ; 0]
    vec_a2 = [1/2 ; sqrt(3)/2]

    coords = zeros(Float64, 2, lattice.Lx, lattice.Ly)
    for i in 1:lattice.Lx
        for j in 1:lattice.Ly
            coords[:, i, j] = - (j - 1) * vec_a2 + (i - 1) * vec_a1
        end
    end

    return coords
end

function directions_all_build(lattice::Triangular_Lattice)  # TODO::多重分派
    vec_a1 = [1 ; 0]
    vec_a2 = [1/2 ; sqrt(3)/2]

    vectors = zeros(Float64, 2, 9)
    vectors[:, 1] = [0 ; 0]
    vectors[:, 2] = lattice.Lx * vec_a1
    vectors[:, 3] = lattice.Ly * vec_a2
    vectors[:, 4] = - lattice.Lx * vec_a1
    vectors[:, 5] = - lattice.Ly * vec_a2
    
    vectors[:, 6] = lattice.Lx * vec_a1 + lattice.Ly * vec_a2
    vectors[:, 7] = - lattice.Lx * vec_a1 + lattice.Ly * vec_a2
    vectors[:, 8] = - lattice.Lx * vec_a1 - lattice.Ly * vec_a2
    vectors[:, 9] = lattice.Lx * vec_a1 - lattice.Ly * vec_a2

    directions = zeros(Float64, 2, lattice.Lx, lattice.Ly)
    norms = zeros(Float64, lattice.Lx, lattice.Ly)
    coords = get_coords(lattice)
    directions = coords
    for i in 1:lattice.Lx
        for j in 1:lattice.Ly
            norms[i, j] = norm(coords[:, i, j])
        end
    end 

    distance = 0.0
    for i in 1:lattice.Lx
        for j in 1:lattice.Ly
            distance = norms[i, j]
            global idx = 1
            for k in 1:9
                distance_new = norm(coords[:, i, j] - vectors[:, k])
                if distance_new < distance
                    distance = distance_new
                    idx = k
                end
            end    
            norms[i, j] = distance
            directions[:, i, j] = coords[:, i, j] - vectors[:, idx]
        end
    end

    return directions

end

# Models
abstract type Hubbard_Model end

struct Single_Band_Hubbard_Model{LT <: AbstractLattice} <: Hubbard_Model
    t::Float64
    Hubbard_U::Float64
    mu::Float64
    lattice::LT
    n_flavors::Int8
end

function Single_Band_Hubbard_Model(;t::Float64, U::AbstractFloat, mu::Float64, lattice::AbstractLattice)
    # rescale
    if (t != 1.0) && (t != 0.0)
        hop_t = 1.0
        hub_u = U / t
        chem_mu = mu / t
    else
        hop_t = t
        hub_u = U
        chem_mu = mu
    end
    
    if (hub_u < 0.0) && (chem_mu != 0.0)
        @warn("A repulsive Hubbard model(U < 0.0) with chemical potential μ = $mu will have a sign problem")
    end

    Single_Band_Hubbard_Model(hop_t, hub_u, chem_mu, lattice, Int8(1))
end