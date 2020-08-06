# Momentum-oriented technical indicator functions

"""
```
aroon(hl::Matrix{T}; n::Int64=25)::Array{Float64}
```

Aroon up/down/oscillator

*Output*

- Column 1: Aroon Up
- Column 2: Aroon Down
- Column 3: Aroon Oscillator
"""
function aroon(hl::Matrix{T}; n::Int64=25)::Matrix{Float64} where {T<:Real}
    @assert size(hl,2) == 2 "Argument `hl` must have exactly 2 columns."
    @assert n < size(hl,1) "Argument `n` must be less than the number of rows in argument `hl`."
    out = zeros(T, (size(hl,1),3))
    @inbounds for i in n:size(hl,1)
        out[i,1] = 100.0 * (n - findmax(hl[i-n+1:i,1])[2]) / 25.0
    end
    @inbounds for i in n:size(hl,1)
        out[i,2] = 100.0 * (n - findmin(hl[i-n+1:i,2])[2]) / 25.0
    end
    out[:,3] = out[:,1]-out[:,2]  # aroon oscillator
    out[1:n-1,:] .= NaN
    return out
end

"""
```
donch(hl::Matrix{T}; n::Int64=10, inclusive::Bool=true)::Array{Float64}
```

Donchian channel (if inclusive is set to true, will include current bar in calculations.)

*Output*

- Column 1: Lowest low of last `n` periods
- Column 2: Average of highest high and lowest low of last `n` periods
- Column 3: Highest high of last `n` periods
"""
function donch(hl::Matrix{T}; n::Int64=10, inclusive::Bool=true)::Matrix{Float64} where {T<:Real}
    @assert size(hl,2) == 2 "Argument `hl` must have exactly 2 columns."
    local lower::Array{T} = runmin(hl[:,2], n=n, cumulative=false, inclusive=inclusive)
    local upper::Array{T} = runmax(hl[:,1], n=n, cumulative=false, inclusive=inclusive)
    local middle::Array{T} = (lower .+ upper) ./ 2.0
    return [lower middle upper]
end

"""
```
momentum(x::Array{T}; n::Int64=1)::Array{Float64}
```

Momentum indicator (price now vs price `n` periods back)
"""
function momentum(x::Array{T}; n::Int64=1)::Array{Float64} where {T<:Real}
    @assert n>0 "Argument n must be positive."
    return diffn(x, n=n)
end

"""
```
roc(x::Array{T}; n::Int64=1)::Array{Float64}
```

Rate of change indicator (percent change between i'th observation and (i-n)'th observation)
"""
function roc(x::Array{T}; n::Int64=1)::Array{Float64} where {T<:Real}
    @assert n<size(x,1) && n>0 "Argument n out of bounds."
    out = zeros(size(x)) .* NaN
    @inbounds for i = (n+1):size(x,1)
        out[i] = x[i]/x[i-n] - 1.0
    end
    return out * 100.0
end

"""
```
macd(x::Array{T}; nfast::Int64=12, nslow::Int64=26, nsig::Int64=9)::Array{Float64}
```

Moving average convergence-divergence

*Output*

- Column 1: MACD
- Column 2: MACD Signal Line
- Column 3: MACD Histogram
"""
function macd(x::Array{T}; nfast::Int64=12, nslow::Int64=26, nsig::Int64=9,
    fastMA::Function=ema, slowMA::Function=ema, signalMA::Function=sma)::Matrix{Float64} where {T<:Real}
    out = zeros(T, (length(x),3))
    out[:,1] = fastMA(x, n=nfast) - slowMA(x, n=nslow)
    out[:,2] = signalMA(out[:,1], n=nsig)
    out[:,3] = out[:,1] - out[:,2]
    return out
end

"""
```
rsi(x::Array{T}; n::Int64=14, ma::Function=ema, args...)::Array{Float64}
```

Relative strength index
"""
function rsi(x::Array{T}; n::Int64=14, ma::Function=ema, args...)::Array{Float64} where {T<:Real}
    @assert n<size(x,1) && n>0 "Argument n is out of bounds."
    N = size(x,1)
    ups = zeros(N)
    dns = zeros(N)
    zro = 0.0
    dx = [NaN; ndims(x) > 1 ? diff(x, dims=1) : diff(x)]
    @inbounds for i=2:N
        if dx[i] > zro
            ups[i] = dx[i]
        elseif dx[i] < zro
            dns[i] = -dx[i]
        end
    end
    rs = [NaN; ma(ups[2:end], n=n; args...) ./ ma(dns[2:end], n=n; args...)]
    return 100.0 .- 100.0 ./ (1.0 .+ rs)
end

"""
```
adx(hlc::Array{T}; n::Int64=14, wilder=true)::Array{Float64}
```

Average directional index

*Output*

- Column 1: DI+
- Column 2: DI-
- Column 3: ADX
"""
function adx(hlc::Array{T}; n::Int64=14, ma::Function=ema, args...)::Matrix{Float64} where {T<:Real}
    @assert n<size(hlc,1) && n>0 "Argument n is out of bounds."
    if size(hlc,2) != 3
        error("HLC array must have three columns")
    end
    N = size(hlc,1)
    updm = zeros(N)
    dndm = zeros(N)
    updm[1] = dndm[1] = NaN
    @inbounds for i = 2:N
        upmove = hlc[i,1] - hlc[i-1,1]
        dnmove = hlc[i-1,2] - hlc[i,2]
        if upmove > dnmove && upmove > 0.0
            updm[i] = upmove
        elseif dnmove > upmove && dnmove > 0.0
            dndm[i] = dnmove
        end
    end
    dip = [NaN; ma(updm[2:N], n=n; args...)] ./ atr(hlc, n=n) * 100.0
    dim = [NaN; ma(dndm[2:N], n=n; args...)] ./ atr(hlc, n=n) * 100.0
    dmx = abs.(dip-dim) ./ (dip+dim)
    adx = [fill(NaN,n); ma(dmx[n+1:N], n=n; args...)] * 100.0
    return [dip dim adx]
end

"""
```
psar(hl::Array{T}; af_min::T=0.02, af_max::T=0.2, af_inc::T=af_min)::Array{Float64}
```

Parabolic stop and reverse (SAR)

*Arguments*
- `hl`: 2D array of high and low prices in first and second columns respectively
- `af_min`: starting/initial value for acceleration factor
- `af_max`: maximum acceleration factor (accel factor capped at this value)
- `af_inc`: increment to the acceleration factor (speed of increase in accel factor)
"""
function psar(hl::Array{T}; af_min::T=0.02, af_max::T=0.2, af_inc::T=af_min)::Array{Float64} where {T<:Real}
    @assert af_min<1.0 && af_min>0.0 "Argument af_min must be in [0,1]."
    @assert af_max<1.0 && af_max>0.0 "Argument af_max must be in [0,1]."
    @assert af_inc<1.0 && af_inc>0.0 "Argument af_inc must be in [0,1]."
    @assert size(hl,2) == 2 "Argument hl must have 2 columns."
    ls0 = 1
    ls = 0
    af0 = af_min
    af = 0.0
    ep0 = hl[1,1]
    ep = 0.0
    maxi = 0.0
    mini = 0.0
    sar = zeros(T,size(hl,1))
    sar[1] = hl[1,2] - std(hl[:,1]-hl[:,2])
    @inbounds for i = 2:size(hl,1)
        ls = ls0
        ep = ep0
        af = af0
        mini = min(hl[i-1,2], hl[i,2])
        maxi = max(hl[i-1,1], hl[i,1])
        # Long/short signals and local extrema
        if (ls == 1)
            ls0 = hl[i,2] > sar[i-1] ? 1 : -1
            ep0 = max(maxi, ep)
        else
            ls0 = hl[i,1] < sar[i-1] ? -1 : 1
            ep0 = min(mini, ep)
        end
        # Acceleration vector
        if ls0 == ls  # no signal change
            sar[i] = sar[i-1] + af*(ep-sar[i-1])
            af0 = (af == af_max) ? af_max : (af + af_inc)
            if ls0 == 1  # current long signal
                af0 = (ep0 > ep) ? af0 : af
                sar[i] = min(sar[i], mini)
            else  # current short signal
                af0 = (ep0 < ep) ? af0 : af
                sar[i] = max(sar[i], maxi)
            end
        else  # new signal
            af0 = af_min
            sar[i] = ep0
        end
    end
    return sar
end

"""
```
kst(x::Array{T};
    nroc::Array{Int64}=[10,15,20,30], navg::Array{Int64}=[10,10,10,15],
    wgts::Array{Int64}=collect(1:length(nroc)), ma::Function=sma)::Array{Float64}

```

KST (Know Sure Thing) -- smoothed and summed rates of change
"""
function kst(x::Array{T}; nroc::Array{Int64}=[10,15,20,30], navg::Array{Int64}=[10,10,10,15],
    wgts::Array{Int64}=collect(1:length(nroc)), ma::Function=sma)::Array{Float64} where {T<:Real}
    @assert length(nroc) == length(navg)
    @assert all(nroc.>0) && all(nroc.<size(x,1))
    @assert all(navg.>0) && all(navg.<size(x,1))
    N = length(x)
    k = length(nroc)
    out = zeros(size(x))
    @inbounds for j = 1:k
        out += ma(roc(x, n=nroc[j]), n=navg[j]) * wgts[j]
    end
    return out
end

"""
```
wpr(hlc::Matrix{T}, n::Int64=14)::Array{Float64}
```

Williams %R
"""
function wpr(hlc::Matrix{T}; n::Int64=14)::Array{Float64} where {T<:Real}
    hihi = runmax(hlc[:,1], n=n, cumulative=false)
    lolo = runmin(hlc[:,2], n=n, cumulative=false)
    return -100 * (hihi - hlc[:,3]) ./ (hihi - lolo)
end

"""
```
cci(hlc::Matrix{T}; n::Int64=20, c::T=0.015, ma::Function=sma)::Array{Float64}
```

Commodity channel index
"""
function cci(hlc::Matrix{T}; n::Int64=20, c::T=0.015, ma::Function=sma, args...)::Array{Float64} where {T<:Real}
    tp = (hlc[:,1] + hlc[:,2] + hlc[:,3]) ./ 3.0
    dev = runmad(tp, n=n, cumulative=false, fun=mean)
    avg = ma(tp, n=n; args...)
    return (tp - avg) ./ (c * dev)
end

"""
```
stoch(hlc::Matrix{T}; nK::Int64=14, nD::Int64=3, kind::Symbol=:fast, ma::Function=sma, args...)::Matrix{Float64}
```

Stochastic oscillator (fast or slow)
"""
function stoch(hlc::Matrix{T}; nK::Int64=14, nD::Int64=3,
    kind::Symbol=:fast, ma::Function=sma, args...)::Matrix{Float64} where {T<:Real}
    @assert kind == :fast || kind == :slow "Argument `kind` must be either :fast or :slow"
    @assert nK<size(hlc,1) && nK>0 "Argument `nK` out of bounds."
    @assert nD<size(hlc,1) && nD>0 "Argument `nD` out of bounds."
    hihi = runmax(hlc[:,1], n=nK, cumulative=false)
    lolo = runmin(hlc[:,2], n=nK, cumulative=false)
    out = zeros(T, (size(hlc,1),2))
    out[:,1] = (hlc[:,3]-lolo) ./ (hihi-lolo) * 100.0
    out[:,2] = ma(out[:,1], n=nD; args...)
    if kind == :slow
        out[:,1] = out[:,2]
        out[:,2] = ma(out[:,1], n=nD; args...)
    end
    return out
end

"""
```
smi(hlc::Matrix{T}; n::Int64=13, nFast::Int64=2, nSlow::Int64=25, nSig::Int64=9,
    maFast::Function=ema, maSlow::Function=ema, maSig::Function=sma)::Matrix{Float64}
```

SMI (stochastic momentum oscillator)
"""
function smi(hlc::Matrix{T}; n::Int64=13, nFast::Int64=2, nSlow::Int64=25, nSig::Int64=9,
    maFast::Function=ema, maSlow::Function=ema, maSig::Function=sma)::Matrix{Float64} where {T<:Real}
    hihi = runmax(hlc[:,1], n=n, cumulative=false)
    lolo = runmin(hlc[:,2], n=n, cumulative=false)
    hldif = hihi-lolo
    delta = hlc[:,3] - (hihi+lolo) / 2.0
    numer = maSlow(maFast(delta, n=nFast), n=nSlow)
    denom = maSlow(maFast(hldif, n=nFast), n=nSlow) / 2.0
    out = zeros(T, (size(hlc,1),2))
    out[:,1] = 100.0*(numer./denom)
    out[:,2] = maSig(out[:,1], n=nSig)
    return out
end


"""
MFI (Money Flow Index) gives almost the same results as tradingview.com
"""
function MFI_logic(x::Array{T}; n::Int64=14, args...)::Array{T} where {T<:Real}
    @assert n<size(x,1) && n>0 "Argument n is out of bounds."
    N = size(x,1)
    ups = zeros(N)
    dns = zeros(N)
    zro = 0.0
    dx = [NaN; ndims(x[:,2]) > 1 ? diff(x[:,2], dims=1) : diff(x[:,2])]
    @inbounds for i=2:N
        if dx[i] > zro
            ups[i] = x[i,3]
        elseif dx[i] < zro
            dns[i] = x[i,3]
        end
    end
    rs = sma(ups[1:end], n=n;) ./ sma(dns[1:end], n=n;)
    return 100.0 .- 100.0 ./ (1.0 .+ rs)
end

##### X::TS should contain traditional OHLC+Volume too
function MFI(X::TS, flds::Vector{Symbol}; args...)
    if size(X,2) > 1 && has_close(X)
        typical_price = sum(X[[:high, :low, :close]].values, dims = 2)/3
        close = X[:close].values
        volume = X[:volume].values
        rmf = typical_price.*volume
        return ts(MFI_logic([close typical_price rmf]; args...), X.index, flds)
        #return ts(MFI_logic(X.values; args...), X.index, flds)
    else
        error("Must be univariate or contain Close/Settle/Last.")
    end
end
