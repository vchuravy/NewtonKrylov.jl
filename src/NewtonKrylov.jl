module NewtonKrylov

export JacobianOperator, JacobianOperatorInPlace, newton_krylov!, solution, residual

using Krylov
using LinearAlgebra
using Enzyme

"""
    AbstractJacobianOperator{N,M,T}
"""
abstract type AbstractJacobianOperator{N, M, T} end
Base.size(::AbstractJacobianOperator{N,M}) where {N,M} = (N,M)
Base.eltype(::AbstractJacobianOperator{N,M,T}) where {N,M,T} = T

function residual end
function update! end
function solution end

# LinearAlgebra.mul!(out, J, v)
# TODO: LinearAlgebra.mul!(out, transpose(J), v)

include("jacobian_operators.jl")

"""
"""
function newton_krylov!(J::AbstractJacobianOperator{N,M};
                        tol_rel=1e-6,
                        tol_abs=1e-12,
                        max_niter = 50) where {N,M}
	solver = CgSolver(N, M, typeof(solution(J)))

    res₀ = residual(J)
    n_res₀ = norm(res₀)
    tol = tol_rel * n_res₀ + tol_abs

    res = res₀
    n_res = n_res₀

    @info "Jacobian-Free Newton-Krylov" n_res₀ tol tol_rel tol_abs
    n_iter = 0
	while n_res > tol && n_iter <= max_niter
		solve!(solver, J, -res) # Jx = -res
        update!(J, solver.x)
       
		res = residual(J)
        n_res = norm(res)
        n_iter += 1
        @info "Newton" iter=n_iter n_res
	end
	return solution(J)
end

# TODO:
# - Better statistic
# - Allow choice of Krylov solver
#   - maxit_krylov?
# - Inexact Netwon-Krylov
#   - First constant η = 0.1
#   - Eisenstat-Walker 
#   - 3.4.3
# - Preconditoners
#  - See 3.3


end # module NewtonKrylov
