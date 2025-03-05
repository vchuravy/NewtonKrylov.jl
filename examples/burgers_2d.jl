@kernel function stencil_kernel!(lastu, nextu, lastv, nextv, dx, dy, dt, μ, nx, ny)
    i, j = @index(Global, NTuple)
    nextu[i + 1, j + 1] =
        lastu[i + 1, j + 1] +
        dt * (
        (
            -lastu[i + 1, j + 1] / (2 * dx) * (lastu[i + 2, j + 1] - lastu[i, j + 1]) -
                lastv[i + 1, j + 1] / (2 * dy) * (lastu[i + 1, j + 2] - lastu[i + 1, j])
        ) +
            μ * (
            (lastu[i + 2, j + 1] - 2 * lastu[i + 1, j + 1] + lastu[i, j + 1]) / dx^2 +
                (lastu[i + 1, j + 2] - 2 * lastu[i + 1, j + 1] + lastu[i + 1, j]) / dy^2
        )
    )
    nextv[i + 1, j + 1] =
        lastv[i + 1, j + 1] +
        dt * (
        (
            -lastu[i + 1, j + 1] / (2 * dx) * (lastv[i + 2, j + 1] - lastv[i, j + 1]) -
                lastv[i + 1, j + 1] / (2 * dy) * (lastv[i + 1, j + 2] - lastv[i + 1, j])
        ) +
            μ * (
            (lastv[i + 2, j + 1] - 2 * lastv[i + 1, j + 1] + lastv[i, j + 1]) / dx^2 +
                (lastv[i + 1, j + 2] - 2 * lastv[i + 1, j + 1] + lastv[i + 1, j]) / dy^2
        )
    )
end
