# # 1D bratu equation from (Kan2022-ko)[@cite]

# ## Necessary packages
using NewtonKrylov, Krylov
using KrylovPreconditioners
using SparseArrays, LinearAlgebra
using OffsetArrays
using MPI

using EnzymeCore
import EnzymeCore.EnzymeRules: forward
using EnzymeCore.EnzymeRules

function forward(::FwdConfig, func::Const{MPI.Request}, ::Type{<:Duplicated})
    # Enzyme: unhandled forward for jl_f_finalizer
    return Duplicated(MPI.Request(), MPI.Request())
end

MPI.Init()

nranks = MPI.Comm_size(MPI.COMM_WORLD)
myid   = MPI.Comm_rank(MPI.COMM_WORLD)

# We are going to split a domain of size N
# into equal chunks.

function localdomain(N, nranks, myrank)
    n = cld(N, nranks)
    first = (n * myrank) + 1
    last = min(n * (myrank + 1), N)
    first:last 
end

# ## Choice of parameters
const N = 10_000
const λ = 3.51382
const dx = 1 / (N + 1) # Grid-spacing

# ### Domain and Inital condition
LI = localdomain(N, 1, 0) # TODO

x = LinRange(0.0, 1.0, N+2) # Global
u₀ = sin.(x .* π)

# ## 1D Bratu equations
# $y′′ + λ * exp(y) = 0$



function bratu!(_res, _y, Δx, λ, N)
    res = OffsetArray(_res, 0:(N+1))
    y  = OffsetArray(_y, 0:(N+1))

    # Set boundary conditions
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    myid   = MPI.Comm_rank(MPI.COMM_WORLD)
   
    ## BVP boundary condition
    if myid == 0
        y[0] = 0
    end
    if myid == (nranks - 1)
        y[N+1] = 0
    end

    ## domain splitting
    # reqs = MPI.MultiRequest(4)  Enzyme strugles with this
    reqs = [MPI.Request() for _ in 1:4]
    if myid != (nranks - 1)
        # Send & data to the right
        MPI.Isend(view(y, N), MPI.COMM_WORLD, reqs[1]; dest = myid + 1)
        MPI.Irecv!(view(y, N+1), MPI.COMM_WORLD, reqs[2]; src = myid + 1)
    end
    if myid != 0
        # Send data to the left
        MPI.Isend(view(y, 1), MPI.COMM_WORLD, reqs[3]; dest = myid - 1)
        MPI.Irecv!(view(y, 0), MPI.COMM_WORLD, reqs[4]; src = myid - 1)
    end
    MPI.Waitall(reqs)

    ## Calculate residual
    for i in 1:N
        y_l = y[i - 1]
        y_r = y[i + 1]
        y′′ = (y_r - 2y[i] + y_l) / Δx^2

        res[i] = y′′ + λ * exp(y[i]) # = 0
    end
    return nothing
end

function bratu(y, dx, λ, N)
    res = similar(y)
    bratu!(res, y, dx, λ, N)
    return res
end

# ## Reference solution
function true_sol_bratu(x)
    ## for λ = 3.51382, 2nd sol θ = 4.8057
    θ = 4.79173
    return -2 * log(cosh(θ * (x - 0.5) / 2) / (cosh(θ / 4)))
end

# ## Solving using inplace variant and CG
uₖ, _ = newton_krylov!(
    (res, u) -> bratu!(res, u, dx, λ, N),
    copy(u₀), similar(u₀);
    Solver = CgSolver,
)

