import LinearAlgebra: mul!

function maybe_duplicated(f,df)
    if Enzyme.Compiler.active_reg(typeof(f))
        return DuplicatedNoNeed(f,df)
    else
        return Const(f)
    end
end

struct JacobianOperator{N, M, T, F, A<:AbstractArray{T}} <: AbstractJacobianOperator{N, M, T}
    f::F # res = F(u)
    f_cache::F
    u::A
    function JacobianOperator{N,M}(f::F, u0) where {N,M,F}
        if length(u0) != N
            throw(ArgumentError("$u0 is not of length $N")) 
        end
        f_cache = Enzyme.make_zero(f)
        new{N,M,eltype(u0),F,typeof(u0)}(f, f_cache, u0)
    end
end
JacobianOperator{N}(f, u0) where {N} = JacobianOperator{N,N}(f, u0)

function residual(J::JacobianOperator)
    J.f(J.u)
end

function update!(J::JacobianOperator, δu)
    J.u .+= δu
    nothing
end

solution(J::JacobianOperator) = J.u

function mul!(out, J::JacobianOperator, v)
    Enzyme.make_zero!(J.f_cache)
    out .= only(autodiff(Forward, 
        maybe_duplicated(J.f, J.f_cache), 
        Duplicated, DuplicatedNoNeed(J.u, v)
    ))

    nothing
end

struct JacobianOperatorInPlace{N, M, T, F, A<:AbstractArray{T}} <: AbstractJacobianOperator{N, M, T}
    f::F # F!(res, u)
    f_cache::F
    u::A
    res::A # residual
    function JacobianOperatorInPlace{N,M}(f::F, u0) where {N,M,F}
        if length(u0) != N
            throw(ArgumentError("$u0 is not of length $N")) 
        end
        f_cache = Enzyme.make_zero(f)
        res = similar(u0, M)
        new{N,M,eltype(u0),F,typeof(u0)}(f, f_cache, u0, res)
    end
end
JacobianOperatorInPlace{N}(f, u0) where {N} = JacobianOperatorInPlace{N,N}(f, u0)

function residual(J::JacobianOperatorInPlace)
    J.f(J.res, J.u)
    J.res
end

function update!(J::JacobianOperatorInPlace, δu)
    J.u .+= δu
    nothing
end

solution(J::JacobianOperatorInPlace) = J.u


function mul!(out, J::JacobianOperatorInPlace, v)
    Enzyme.make_zero!(J.f_cache)
    autodiff(Forward, 
        maybe_duplicated(J.f, J.f_cache), Const, 
        DuplicatedNoNeed(J.res, out), DuplicatedNoNeed(J.u, v)
    )
    nothing
end
