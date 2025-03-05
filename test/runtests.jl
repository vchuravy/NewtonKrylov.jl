using Test
using NewtonKrylov

function F!(res, x)
    res[1] = x[1]^2 + x[2]^2 - 2
    return res[2] = exp(x[1] - 1) + x[2]^2 - 2
end

function F(x)
    res = similar(x)
    F!(res, x)
    return res
end

let x₀ = [2.0, 0.5]
    x, stats = newton_krylov!(F!, x₀)
    @test stats.solved
end

let x₀ = [3.0, 5.0]
    x, stats = newton_krylov(F, x₀)
    @test stats.solved
end

import NewtonKrylov: JacobianOperator
using Enzyme, LinearAlgebra

function df(x, a)
    return autodiff(Forward, F, DuplicatedNoNeed(x, a)) |> first
end

function df!(out, x, a)
    res = similar(out)
    autodiff(Forward, F!, DuplicatedNoNeed(res, out), DuplicatedNoNeed(x, a))
    return nothing
end

@testset "Jacobian" begin
    x = [3.0, 5.0]
    v = rand(2)

    J_Enz = jacobian(Forward, F, x) |> only
    J = JacobianOperator(F!, zeros(2), x)
    J_NK = collect(J)

    @test J_NK == J_Enz

    jvp = similar(v)
    mul!(jvp, J, v)

    jvp2 = df(x, v)
    @test jvp == jvp2

    jvp3 = similar(v)
    df!(jvp3, x, v)
    @test jvp == jvp3

    @test jvp ≈ J_Enz * v
end

# Differentiate F with respect to x twice.
function ddf(x, a)
    return autodiff(Forward, df, DuplicatedNoNeed(x, a), Const(a)) |> first
end

function ddf!(out, x, a)
    _out = similar(out)
    autodiff(Forward, df!, DuplicatedNoNeed(_out, out), DuplicatedNoNeed(x, a), Const(a))
    return nothing
end

@testset "2nd order directional derivative" begin
    x = [3.0, 5.0]
    v = rand(2)

    hvvp = similar(x)
    ddf!(hvvp, x, v)

    hvvp2 = ddf(x, v)
    @test hvvp == hvvp2

    J = JacobianOperator(F!, zeros(2), x)
    H = HessianOperator(J)

    hvvp3 = similar(x)
    mul!(hvvp3, H, v)

    @test hvvp == hvvp3
end
