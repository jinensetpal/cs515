## ::hide() ::end()
using Statistics, LinearAlgebra, StatsBase, Random, CairoMakie, JSON

## Building blocks for the GPT-2 model ::simple-blocks

"""
    gelu_tanh(x)
Approximate GELU activation function using the tanh approximation from  
Hendrycks and Gimpel (2016), "Gaussian Error Linear Units (GELUs)",
https://arxiv.org/abs/1606.08415. This is what's used in all GPT-2 models.
"""
gelu_tanh(x) = 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))
"""
    softmax!(x)
Normalize the entries of vector `x` in-place using a numerically stable
softmax. The routine subtracts the maximum entry from `x` to avoid
overflow before exponentiating and finally scales the result so the
entries sum to one. Returns the mutated input vector so it can be used in
broadcasted calls such as `softmax!.(eachrow(scores))`.
"""
function softmax!(x) # in-place stable softmax
  x .-= maximum(x)
  @. x = exp(x)
  x ./= sum(x)
end
""" 
    LayerNorm(gain, bias) 
A layer normalization layer with learnable `gain` and `bias` parameters.
Use `layernorm(X, ln)` to apply the layer norm to each row of `X`.
Use `layernorm!(X, ln)` to do the operation in-place on `X`.
"""
struct LayerNorm{T}
  gain::Array{T,1}
  bias::Array{T,1}
end
"""
    layernorm!(X, ln)
Apply layer normalization to each row of `X` in-place using the
parameters in `ln::LayerNorm`. Returns the mutated `X`.
"""
function layernorm!(X, ln::LayerNorm)
  for i in axes(X,1) 
    x = @view X[i,:]
    m = mean(x) 
    v = var(x; corrected=false)
    @. x = ln.gain * ((x - m)/ sqrt(v + 1e-5)) + ln.bias
  end 
  return X
end
"""
    layernorm(X, ln)
Apply layer normalization to each row of `X` using the parameters in
`ln::LayerNorm`. Returns a new array and does not modify `X`.
"""
layernorm(X, ln::LayerNorm) = layernorm!(copy(X), ln)

## The entire GPT-2 model in Julia ::gpt2
""" 
    GPTTransformerLayer(QKV, bqkv, O, bO, W1, b1, W2, b2, lns)
A single transformer layer for GPT-2 with parameters:
- QKV: weights for query, key, value (d x 3*d)
- bqkv: biases for query, key, value (3*d)
- O: output projection weights after attention (d x d)
- bO: output projection bias after attention (d)
- W1: first weight matrix in feedforward (d x d_f)
- b1: first bias in feedforward (d_f)
- W2: second weight matrix in feedforward (d_f x d)
- b2: second bias in feedforward (d)
- lns: tuple of two LayerNorm layers

Use `apply!(X, layer)` to apply the layer to input `X` (sequence length x d_model).
"""
struct GPTTransformerLayer{T}
  QKV::Matrix{T}      # weights for Q, K, V
  bqkv::Vector{T}     # biases for Q, K, V
  O::Matrix{T}        # projection weights after attention
  bO::Vector{T}       # projection bias after attention
  W1::Matrix{T}       # first weight matrix in feedforward
  b1::Vector{T}       # first bias in feedforward
  W2::Matrix{T}       # second weight matrix in feedforward
  b2::Vector{T}       # second bias in feedforward
  lns::Tuple{LayerNorm{T},LayerNorm{T}}  # layer norms
end 
"""
    apply!(X, layer)
Apply the transformer layer `layer` to input `X` (sequence length x d_model).
Returns the output array.
"""
function apply!(X, layer::GPTTransformerLayer) 
  m, d = size(X)  # m = sequence length, d = em
  Xbar = layernorm(X, layer.lns[1])
  QKV = Xbar * layer.QKV .+ layer.bqkv'  # m x 3d
  Q = QKV[:, 1:d]; K = QKV[:, d+1:2d]; V = QKV[:, 2d+1:3d]
  A = similar(X); dh = 64 # multihead attention...
  for head_offset = 1:dh:d 
    head_offsets = head_offset:head_offset+dh-1
    Qh = Q[:, head_offsets]; Kh = K[:, head_offsets]; Vh = V[:, head_offsets]
    scores = (Qh * Kh') ./ sqrt(dh)                          # Attention scores
    scores .= ifelse.(triu(ones(m,m),1) .== 1, -Inf, scores) # Causal mask 
    softmax!.(eachrow(scores))
    Ah = scores * Vh
    A[:, head_offsets] .= Ah
  end 
  # Project back to d-dim
  attn_proj = A * layer.O .+ layer.bO'
  X1 = X  .+ attn_proj  # residual connection
  X1bar = layernorm!(X1, layer.lns[2])
  # Feedforward network
  ff1 = gelu_tanh.(X1bar * layer.W1 .+ layer.b1') 
  ff2 = ff1 * layer.W2 .+ layer.b2'
  X .+= ff2 .+ attn_proj  # in-place update  X = X1 + ff2
  return X
end
"""
    GPT2(E, P, layers, ln_final)
A GPT-2 model with parameters:
- E: token embedding matrix (vocab size x d_model)
- P: positional encoding matrix (max position x d_model)
- layers: vector of GPTTransformerLayer layers
- ln_final: final LayerNorm layer
Use `gpt2(token_sequence, model)` to apply the model to a sequence of token IDs (0 index based).
"""
struct GPT2{T}
  E::Array{T,2}       # token embedding
  P::Array{T,2}       # positional encoding
  layers::Vector{GPTTransformerLayer{T}}  # transformer layers
  ln_final::LayerNorm{T} # final layer norm
end 
function gpt2func(token_sequence::Vector, model::GPT2)
  T = size(token_sequence, 1)  # sequence length
  @assert T <= size(model.P, 1) "token sequence too long for positional encoding"
  X = model.E[token_sequence .+ 1, :] .+ model.P[1:T, :]  # initial embedding + pos enc
  for layer in model.layers
    X = apply!(X, layer)
  end
  X = layernorm!(X, model.ln_final) # Final layer norm
  return X # return before the vocabulary projection
end 
function gpt2(token_sequence::Vector, model::GPT2)
  T = size(token_sequence, 1)  # sequence length
  @assert T <= size(model.P, 1) "token sequence too long for positional encoding"
  X = model.E[token_sequence .+ 1, :] .+ model.P[1:T, :]  # initial embedding + pos enc
  for layer in model.layers
    X = apply!(X, layer)
  end
  X = layernorm!(X, model.ln_final) # Final layer norm
  return X * model.E'  # logit weights 
end

## Extra helpers ::gpt2-extra
_layernorm_param_count(ln::LayerNorm) = length(ln.gain) + length(ln.bias)
function _param_count(layer::GPTTransformerLayer)
  weight_terms = length(layer.QKV) + length(layer.O) + length(layer.W1) + length(layer.W2)
  bias_terms = length(layer.bqkv) + length(layer.bO) + length(layer.b1) + length(layer.b2)
  return weight_terms + bias_terms + sum(_layernorm_param_count, layer.lns)
end
function Base.show(io::IO, ::MIME"text/plain", layer::GPTTransformerLayer{T}) where {T}
  d_model = size(layer.O, 1)
  ff_dim = size(layer.W1, 2)
  params = _param_count(layer)
  print(io, "GPTTransformerLayer{$T}\n")
  print(io, "  model dimension: ", d_model, "\n")
  print(io, "  feedforward dimension: ", ff_dim, "\n")
  print(io, "  parameters: ", params, "\n")
end
function Base.show(io::IO, ::MIME"text/plain", model::GPT2{T}) where {T}
  vocab = size(model.E, 1)
  d_model = size(model.E, 2)
  max_position = size(model.P, 1)
  layer_count = length(model.layers)
  layer_params = sum(_param_count, model.layers)
  total_params = length(model.E) + length(model.P) + layer_params + _layernorm_param_count(model.ln_final)
  print(io, "GPT2{$T} model\n")
  print(io, "  vocabulary size: ", vocab, "\n")
  print(io, "  max positions: ", max_position, "\n")
  print(io, "  model dimension: ", d_model, "\n")
  print(io, "  transformer layers: ", layer_count, "\n")
  print(io, "  parameters: ", total_params, "\n")
end
function attention_logits(X, layer::GPTTransformerLayer) 
  m, d = size(X)  # m = sequence length, d = em
  Xbar = layernorm(X, layer.lns[1])
  QKV = Xbar * layer.QKV .+ layer.bqkv'  # m x 3d
  Q = QKV[:, 1:d]; K = QKV[:, d+1:2d]; V = QKV[:, 2d+1:3d]
  dh = 64 # multihead attention...
  S = Vector{Matrix{eltype(X)}}(undef, div(d, dh))
  for head_offset = 1:dh:d 
    head_offsets = head_offset:head_offset+dh-1
    Qh = Q[:, head_offsets]; Kh = K[:, head_offsets]; Vh = V[:, head_offsets]
    scores = (Qh * Kh') ./ sqrt(dh)                          # Attention scores
    scores .= ifelse.(triu(ones(m,m),1) .== 1, -Inf, scores) # Causal mask 
    softmax!.(eachrow(scores))
    S[div(head_offset-1, dh)+1] = scores
  end 
  return S
end


## Load the GPT-2 small model weights ::hide() ::load
using JSON
if isfile("gpt2-small.safetensors") == false 
    println("Downloading GPT-2 small model weights (548MB)...")
    download("https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors?download=true", "gpt2-small.safetensors")
    println("Downloading gpt2 vocab... (1MB)")
    download("https://huggingface.co/openai-community/gpt2/raw/main/vocab.json", "gpt2-vocab.json")
end
function read_safetensor_header(filename)
  open(filename, "r") do io
    # read a uint64 for the header size
    header_size_bytes = read(io, UInt64)
    # read the header
    header_bytes = read(io, Int(header_size_bytes))
    json = JSON.parse(String(header_bytes))
    json["header_size"] = header_size_bytes
    return json
  end
end 

"""
    _from_c_order(vec, shape; materialize=false)

Given a flat vector `vec` read from a row-major (C-order) file and the
row-major `shape::NTuple`, return an N-d array with Julia (column-major)
semantics and the same logical shape.

# Written by ChatGPT - Sept. 12, 2025
"""
function _from_c_order(vec::AbstractVector{T}, shape::NTuple{N,Int}) where {T,N}
  return PermutedDimsArray(reshape(vec, reverse(shape)), N:-1:1)
end
function _typeof(dtype::String)
  if dtype == "F32"
    return Float32
  elseif dtype == "F16"
    return Float16
  elseif dtype == "BF16"
    @assert false "BFloat16 not supported in this simple Julia code"
    #return BFloat16
  elseif dtype == "I64"
    return Int64
  elseif dtype == "I32"
    return Int32
  elseif dtype == "I16"
    return Int16
  elseif dtype == "I8"
    return Int8
  elseif dtype == "U64"
    return UInt64
  elseif dtype == "U32"
    return UInt32
  elseif dtype == "U16"
    return UInt16
  elseif dtype == "U8"
    return UInt8
  else
    error("unsupported dtype $dtype")
  end
end
function read_safetensor_array(io, hdr, name, ::Type{T}) where T
  @assert haskey(hdr, name) "array $name not found in header"
  meta = hdr[name]
  offsets = meta["data_offsets"]
  @assert all(x -> x >= 0, offsets) "negative offsets not supported"
  @assert offsets[2] > offsets[1] "empty arrays not supported"
  seek(io, offsets[1]+hdr["header_size"] + 8)  # +8 for the initial uint64
  shape = Tuple(meta["shape"])
  @assert shape != () "scalar arrays not supported"
  @assert all(x -> x > 0, shape) "zero-length/nonpositive dimensions not supported"
  dtype = _typeof(meta["dtype"])
  @assert (dtype) == T "dtype $dtype does not match requested type $T"
  # determine the number of elements
  nelements = prod(shape)
  # read the raw data
  data = Vector{T}(undef, nelements)
  read!(io, data)
  return _from_c_order(data, shape)
end
function read_gpt2_safetensors(filename)
  hdr = read_safetensor_header(filename)
  open(filename, "r") do io
    # read the arrays
    E = Matrix(read_safetensor_array(io, hdr, "wte.weight", Float32))
    P = Matrix(read_safetensor_array(io, hdr, "wpe.weight", Float32))
    ln_f_gain = Vector(read_safetensor_array(io, hdr, "ln_f.weight", Float32)[:])
    ln_f_basis = Vector(read_safetensor_array(io, hdr, "ln_f.bias", Float32)[:])
    layers = GPTTransformerLayer{Float32}[]
    N = 0
    while true
      layer_prefix = "h.$N."
      try
        QKV = Matrix(read_safetensor_array(io, hdr, layer_prefix * "attn.c_attn.weight", Float32))
        bqkv = Vector(read_safetensor_array(io, hdr, layer_prefix * "attn.c_attn.bias", Float32)[:])
        O = Matrix(read_safetensor_array(io, hdr, layer_prefix * "attn.c_proj.weight", Float32))
        bO = Vector(read_safetensor_array(io, hdr, layer_prefix * "attn.c_proj.bias", Float32)[:])
        ln1gain = Vector(read_safetensor_array(io, hdr, layer_prefix * "ln_1.weight", Float32)[:])
        ln1basis = Vector(read_safetensor_array(io, hdr, layer_prefix * "ln_1.bias", Float32)[:]) 
        W1 = Matrix(read_safetensor_array(io, hdr, layer_prefix * "mlp.c_fc.weight", Float32))
        b1 = Vector(read_safetensor_array(io, hdr, layer_prefix * "mlp.c_fc.bias", Float32)[:])
        W2 = Matrix(read_safetensor_array(io, hdr, layer_prefix * "mlp.c_proj.weight", Float32))
        b2 = Vector(read_safetensor_array(io, hdr, layer_prefix * "mlp.c_proj.bias", Float32)[:])
        ln2gain = Vector(read_safetensor_array(io, hdr, layer_prefix * "ln_2.weight", Float32)[:])
        ln2basis = Vector(read_safetensor_array(io, hdr, layer_prefix * "ln_2.bias", Float32)[:]) 
        lnx = LayerNorm(ln1gain, ln1basis)
        lnp = LayerNorm(ln2gain, ln2basis)
        lns = (lnx, lnp)
        push!(layers, GPTTransformerLayer(QKV, bqkv, O, bO, W1, b1, W2, b2, lns))
        N += 1
      catch e
        if isa(e, AssertionError)
          break
        else
          rethrow(e)
        end
      end
    end 
    return GPT2(E, P, layers, LayerNorm(ln_f_gain, ln_f_basis))
  end
end

gpt2model = read_gpt2_safetensors("gpt2-small.safetensors")

""" 
    SimpleTokenizer(token_to_id, id_to_token)
A simple tokenizer for GPT-2 based on a provided vocabulary mapping.
This doesn't do exactly BPE tokenization, but a greedy longest-match
strategy using the GPT-2 vocab. It's approximately the same for 
english text, but may fail on some edge cases.
"""
struct SimpleTokenizer
  token_to_id::Dict{String,Int}
  id_to_token::Vector{String}
end
function SimpleTokenizer(vocabfile::String)
  vocab = JSON.parsefile(vocabfile)
  # decode all the bytes in the vocab keys
  bs = vcat(collect(33:126), collect(161:172), collect(174:255))
  cs = copy(bs)
  present = Set(bs)
  n = 0
  for b in 0:255
    if !(b in present)
      push!(bs, b)
      push!(cs, 256 + n)
      n += 1
    end
  end
  d = Dict{Char,UInt8}()
  for (b, c) in zip(bs, cs)
    d[Char(c)] = UInt8(b)
  end
  token_to_id = Dict{String,Int}()
  for (k,v) in vocab
    decoded_chars = String([d[c] for c in k])
    token_to_id[decoded_chars] = v  
  end
  id_to_token = Vector{String}(undef, length(vocab))
  for (k,v) in token_to_id
    id_to_token[v+1] = k # shift to one-based to use vector lookup 
  end
  return SimpleTokenizer(token_to_id, id_to_token)
end

"""
  encode(tokenizer::SimpleTokenizer, text::String) -> Vector{Int}

Tokenizes the input `text` using the provided `tokenizer` (of type `SimpleTokenizer`).

This function implements a greedy tokenization strategy, selecting the longest matching 
token at each position in the input string. It returns a vector of token IDs corresponding 
to the matched tokens.
- `tokenizer::SimpleTokenizer`: The tokenizer object containing a mapping from tokens to their IDs.
- `text::String`: The input string to tokenize.

Return `Vector{Int}`, a vector of token IDs representing the tokenized input text.

`encode("Matrices are my friends and")` returns `[19044, 45977, 389, 616, 2460, 290]`.
"""
function encode(tokenizer::SimpleTokenizer, text::String)
  # A greedy tokenizer that uses the longest next string match
  tokens = String[]
  i = 1
  while i <= lastindex(text)
    match = ""
    for tok in keys(tokenizer.token_to_id)
      if startswith(@view(text[i:end]), tok) && length(tok) > length(match)
        match = tok
      end
    end
    if match == ""
      error("no token match found at position $i in text")
    end
    push!(tokens, match)
    i += lastindex(match)
  end
  return [tokenizer.token_to_id[t] for t in tokens]
end
function decode(tokenizer::SimpleTokenizer, token_ids::Vector{Int})
  return join(tokenizer.id_to_token[id+1] for id in token_ids) # shift to one-based to use vector lookup 
end 

tokenizer = SimpleTokenizer("gpt2-vocab.json")

## The full sequence prediction and gpt text function ::gpt2-predict 
function sequence_predict(model::GPT2, start_sequence; maxlen=20, temperature=1.0)
  token_sequence = copy(start_sequence)
  for i in 1:maxlen
    logits = gpt2(token_sequence, model)
    last_logits = logits[end, :] ./ temperature
    softmax!(last_logits)
    next_token = sample(1:length(last_logits), Weights(last_logits))
    push!(token_sequence, next_token - 1)  # convert to 0-based
  end
  return token_sequence
end
function gpt2(input::String; kwargs...)
  token_sequence = encode(tokenizer, input)
  predicted_sequence = sequence_predict(gpt2model, token_sequence; kwargs...)
  return decode(tokenizer, predicted_sequence)
end

## ::example-output ::with_terminal()
Y = gpt2([19044, 45977, 389, 616, 2460, 290], gpt2model)

## ::example-friends-tokens ::unlimitdisplay()
sequence_predict(gpt2model, 
  [19044, 45977, 389, 616, 2460, 290]; 
  maxlen=11, temperature=0.0001)'

## ::example-friends 
gpt2("Matrices are my friends and"; maxlen=11, temperature=1e-4)
## ::example-numbers ::display() 
gpt2("
1, 2, 3, 4, 5,"; maxlen=50, temperature=0.2)
## ::example-letters ::display()
gpt2("
A, B, C, D, E,"; maxlen=50, temperature=0.2)
## ::example-pattern ::display()
Random.seed!(4)
gpt2("12, 21, 13, 31, 14, 41, 15, 51, 16,"; 
  maxlen=20, temperature=0.2)
## ::example-matrix ::display() 
Random.seed!(7)
gpt2("Let \$A\$ be a matrix. A matrix \$A\$ is symmetric if \$A ="; 
  maxlen=10, temperature=0.5)

## Show the matrices across the layers as heatmaps. 
function gpt2_layer_outputs(token_sequence::Vector{Int}, model::GPT2)
  T = size(token_sequence, 1)  # sequence length
  X = model.E[token_sequence .+ 1, :] .+ model.P[1:T, :]  # initial embedding + pos enc
  layer_outputs = [copy(X)]
  for layer in model.layers
    X = apply!(X, layer)
    push!(layer_outputs, copy(X))
  end
  X = layernorm!(X, model.ln_final) # Final layer norm
  push!(layer_outputs, copy(X))
  return layer_outputs
end
token_sequence = encode(tokenizer, "Matrices are my friends and")
layer_outputs = gpt2_layer_outputs(token_sequence, gpt2model)

##
using Colors
"""
  redwhite_colormap(n=256)

Create a custom colormap that goes from white (for zero) to crimson red (for one).
"""
function redwhite_colormap(n=256)
  white = RGB(1,1,1)
  crimson = RGB(220/255, 20/255, 60/255) # Crimson
  return [RGB(
    white.r + (crimson.r - white.r) * t,
    white.g + (crimson.g - white.g) * t,
    white.b + (crimson.b - white.b) * t
  ) for t in range(0, 1, length=n)]
end

heatmap(softmax!(copy(layer_outputs[1]))', colormap = redwhite_colormap())

## Attention Matrices across layers
function attention_matrices(token_sequence::Vector{Int}, model::GPT2)
  T = size(token_sequence, 1)  # sequence length
  X = model.E[token_sequence .+ 1, :] .+ model.P[1:T, :]  # initial embedding + pos enc
  attentions = Vector{Vector{Matrix{Float32}}}(undef, length(model.layers))
  for (i, layer) in enumerate(model.layers)
    S = attention_logits(X, layer)
    attentions[i] = S
    X = apply!(X, layer)
  end
  return attentions
end

## Display attention heatmaps for each layer and head ::token-attention-friends
fullwidth = 5.5*96
function _add_label(figspot, labeltext, flipchar, char1, char2; charfirst = false, kwopts...)
  if labeltext !== nothing 
    labelchar = char1 
    if flipchar == true 
      labelchar = char2 
    end 
    if charfirst == false 
      return Label(figspot, "$labeltext $labelchar", 
        rotation=0, tellwidth=false, fontsize=11; kwopts...)
    else 
      return Label(figspot, "$labelchar $labeltext", 
        rotation=0, tellwidth=false, fontsize=11; kwopts...)
    end 
  end
  return nothing 
end 
token_sequence = encode(tokenizer, "Matrices are my friends and")
attentions = attention_matrices(token_sequence, gpt2model)
n_layers = length(attentions)
n_heads = length(attentions[1])
f = Figure(size = (fullwidth*96/2, fullwidth*96/2), 
  fontsize = 11, backgroundcolor = :transparent, 
  figure_padding=0)
for layer_idx in 1:n_layers
  if layer_idx == 1
    lblt = _add_label(f[-1,0], "layers / heads →", true, '↑', '↓';  
      halign=:left, valign=:bottom, charfirst = true, alignmode = Outside(),
      padding=(0, 0, 0, 0))
    for head_idx in 1:n_heads
      Label(f[0, head_idx], "$head_idx", 
        tellwidth=false, tellheight=true, alignmode = Outside(),
        padding=(0, 0, 0, 0))
    end
  end
  Label(f[layer_idx, 0], "$layer_idx", tellheight=false, halign=:right,
    padding=(0, 0, 0, 0))
  for head_idx in 1:n_heads
    ax = Axis(f[layer_idx, head_idx], aspect=DataAspect(), 
      backgroundcolor = :transparent, alignmode = Outside())
    # the transpose here is because heatmap transposed data... <shrug>
    heatmap!(ax, attentions[layer_idx][head_idx]'; 
      colormap=:PuBu)
    ax.yreversed = true
    hidedecorations!(ax)
    hidespines!(ax)
  end
end
rowgap!(f.layout, 2)
colgap!(f.layout, 2)
f

## Display attention heatmaps for each layer and head ::token-attention-numbers
token_sequence = encode(tokenizer, "1, 2, 3,")
attentions = attention_matrices(token_sequence, gpt2model)
n_layers = length(attentions)
n_heads = length(attentions[1])
f = Figure(size = (fullwidth*96/2, fullwidth*96/2), 
  fontsize = 11, backgroundcolor = :transparent, 
  figure_padding=0)
for layer_idx in 1:n_layers
  if layer_idx == 1
    lblt = _add_label(f[-1,0], "layers / heads →", true, '↑', '↓';  
      halign=:left, valign=:bottom, charfirst = true, alignmode = Outside(),
      padding=(0, 0, 0, 0))

    for head_idx in 1:n_heads
      Label(f[0, head_idx], "$head_idx", 
        tellwidth=false, tellheight=true, alignmode = Outside(),
        padding=(0, 0, 0, 0))
    end
  end
  Label(f[layer_idx, 0], "$layer_idx", tellheight=false, halign=:right,
    padding=(0, 0, 0, 0))
  for head_idx in 1:n_heads
    ax = Axis(f[layer_idx, head_idx], aspect=DataAspect(), 
      backgroundcolor = :transparent, alignmode = Outside())
    # the transpose here is because heatmap transposed data... <shrug>
    heatmap!(ax, attentions[layer_idx][head_idx]'; 
      colormap=:PuBu)
    ax.yreversed = true
    hidedecorations!(ax)
    hidespines!(ax)
  end
end
rowgap!(f.layout, 2)
colgap!(f.layout, 2)
f

## Number of parameters
function gpt2params(T::Int, L::Int, d::Int, df::Int, N::Int)
    # Embedding + positional encoding
    p_emb = (T + L) * d
    # Transformer layers
    p_layer = (4 * d^2 + 6d + 2d * df + df + 3d)
    #p_layer = 3d^2 + 3d + d^2 + d + 4d + d*df + df + d*df + d
    #p_layer = 4d^2 + 4d + 4d + d*df + df + d*df + d
    #p_layer = 4d^2 + 4d + 4d + d*df + df + d*df + d
    # Final layer norm and projection
    p_final = 2d
    @show p_emb p_layer p_final
    return p_emb + p_layer*N + p_final
end
println("small model:")
@show gpt2params(50257, 1024, 768, 4*768, 12)  # GPT-2 small
println("medium model:")
@show gpt2params(50257, 1024, 1024, 4*1024, 24)  # GPT-2 medium
println("large model:")
@show gpt2params(50257, 1024, 1280, 4*1280, 36)  # GPT-2 large
println("XL model:")
@show gpt2params(50257, 1024, 1600, 4*1600, 48)  # GPT-2 XL


## Testing the model on a known input ::hide()
Y = gpt2([19044, 45977, 389, 616, 2460, 290], gpt2model)
@assert ≈(-31.1058;rtol=1e-5)(Y[1,1])
@assert ≈(-31.0338;rtol=1e-5)(Y[1,end])
@assert ≈(-88.2154;rtol=1e-5)(Y[2,1])
@assert ≈(-103.4866;rtol=1e-5)(Y[end,1])

#
# Output from PyTorch
# for Input 
# tensor([[19044, 45977,   389,   616,  2460,   290]])
# Output after embedding layer:
# torch.Size([1, 6, 768])
# tensor([[[ 0.1652, -0.3270,  0.0742,  ..., -0.1828, -0.0881, -0.0904],
#          [-0.0413,  0.0413,  0.0853,  ...,  0.1023,  0.1343,  0.0346],
#          [ 0.0798, -0.0289,  0.1026,  ...,  0.0793, -0.0357,  0.0632],
#          [ 0.1575,  0.0353,  0.1793,  ...,  0.0607,  0.1234,  0.0210],
#          [-0.0381, -0.1290,  0.2370,  ..., -0.0265,  0.1071, -0.0336],
#          [-0.0432, -0.0350,  0.1615,  ...,  0.0987,  0.0207,  0.0665]]],
#        grad_fn=<AddBackward0>)
# Output after Transformer Layer 1:
# torch.Size([1, 6, 768])
# tensor([[[ 2.1200, -0.5339,  2.8074,  ..., -2.1211, -0.3335, -1.4958],
#          [ 0.3042, -1.1035,  2.1712,  ..., -0.4095,  0.3919, -0.0702],
#          [-0.1200, -0.1891, -0.4527,  ...,  0.4750,  0.7414,  0.9730],
#          [-0.0431,  0.4511,  0.5569,  ...,  1.4691,  1.8839, -0.1017],
#          [ 0.4594,  0.1565,  1.6352,  ..., -0.6582, -0.7287, -0.0591],
#          [ 2.2747,  0.4057, -0.6187,  ...,  0.4224,  0.3128, -0.0726]]])

