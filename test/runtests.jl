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

@testset "Jacobian" begin
    J_Enz = jacobian(Forward, F, [3.0, 5.0]) |> only
    J = JacobianOperator(F!, zeros(2), [3.0, 5.0])
    J_NK = collect(J)

    @test J_NK == J_Enz

    v = rand(2)
    out = similar(v)
    mul!(out, J, v)

    @test out ≈ J_Enz * v
end
