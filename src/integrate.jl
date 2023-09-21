module Integrate

using Distributions
import Sobol: skip, SobolSeq
import Base.Iterators: take, Repeated, product, repeated
import HCubature: hcubature
import LinearAlgebra: cholesky
import FastGaussQuadrature: gausshermite
import SparseGrids: sparsegrid

abstract type AbstractIntegrator end

(∫::AbstractIntegrator)(f::Function) = sum(w*f(x) for (w,x) in zip(∫.w, ∫.x))

struct FixedNodeIntegrator{Tx,Tw} <: AbstractIntegrator
    x::Tx
    w::Tw
end

MonteCarloIntegrator(distribution::Distribution, ndraw=100)=FixedNodeIntegrator([rand(distribution) for i=1:ndraw], Repeated(1/ndraw))

function QuasiMonteCarloIntegrator(distribution::UnivariateDistribution, ndraws=100)
    ss = skip(SobolSeq(1), ndraw)
    x = [quantile(distribution, x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 

function QuasiMonteCarloIntegrator(distribution::AbstractMvNormal, ndraw=100)
    ss = skip(SobolSeq(length(distribution)), ndraw)
    L = cholesky(distribution.Σ).L
    x = [L*quantile.(Normal(), x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 

struct QuadratureIntegrator{QX, QW, QL, Qμ, QDX} <: AbstractIntegrator
    X::QX
    W::QW
    L::QL
    μ::Qμ
    dx::QDX
end

(∫::QuadratureIntegrator)(f::Function) = sum(f(√2*∫.L*vcat(x...)+∫.μ)*w for (x,w) ∈ zip(∫.X, ∫.W))/(π^(∫.dx/2))

function QuadratureIntegrator(dist::AbstractMvNormal, ndraw = 100)
    n = Int(ceil(ndraw^(1/length(dist))))
    x_1, w_1 = gausshermite(n)
    X = product(repeated(x_1, length(dist))...)
    W = prod.(product(repeated(w_1, length(dist))...))
    L = cholesky(dist.Σ).L
    μ = dist.μ
    dx = length(dist)
    QuadratureIntegrator(X,W,L,μ,dx)
end

function SparseQIntegrator(dist::AbstractMvNormal, order=5)
    X,W = sparsegrid(length(dist), order, gausshermite, sym=true)
    L = cholesky(dist.Σ).L
    μ = dist.μ
    dx = length(dist)
    QuadratureIntegrator(X,W,L,μ,dx)
end

struct AdaptiveIntegrator{FE,FT,FJ,A,L} <: AbstractIntegrator
    eval::FE
    transform::FT
    detJ::FJ
    args::A
    limits::L
end

(∫::AdaptiveIntegrator)(f::Function) = ∫.eval(t->f(∫.transform(t))*∫.detJ(t), ∫.limits...; ∫.args...)[1]

function AdaptiveIntegrator(dist::AbstractMvNormal; eval=hcubature, options=())
    D = length(dist)
    x(t) = t./(1 .- t.^2)
    Dx(t) = prod((1 .+ t.^2)./(1 .- t.^2).^2)*pdf(dist,x(t))
    args = options
    limits = (-ones(D), ones(D))
    AdaptiveIntegrator(hcubature,x,Dx,args, limits)
end

end