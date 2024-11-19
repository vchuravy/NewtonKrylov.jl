pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add NewtonKrylov to environment stack

using NewtonKrylov
using Documenter
import Documenter.Remotes: GitHub
using Literate
using DocumenterCitations

DocMeta.setdocmeta!(NewtonKrylov, :DocTestSetup, :(using NewtonKrylov); recursive = true)


##
# Generate examples
##

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

examples = [
    "Bratu 1D" => "bratu",
    "Simple" => "simple",
    "BVP" => "bvp",
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
end

examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    modules = [NewtonKrylov],
    authors = "Valentin Churavy",
    repo = GitHub("vchuravy", "NewtonKrylov.jl"),
    sitename = "NewtonKrylov.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://vchuravy.dev/NewtonKrylov.jl",
        assets = [
            asset(
                "https://plausible.io/js/plausible.js",
                class = :js,
                attributes = Dict(Symbol("data-domain") => "vchuravy.dev", :defer => "")
            ),
            "assets/citations.css",
        ],
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
    ],
    doctest = true,
    linkcheck = true,
    plugins = [bib]
)

deploydocs(;
    repo = "github.com/vchuravy/NewtonKrylov.jl.git",
    devbranch = "main",
    push_preview = true,
)
