# Fourier Feature Network

An n-dimensional Fourier feature network, written in Julia using
[Flux.jl][1], based on [this paper][2].

Passing input points through a simple Fourier feature mapping enables a
multilayer perceptron (MLP) to better approximate higher-frequency
functions in low-dimensional problem domains, compared to an MLP
receiving only the coordinates as input.

[1]: https://github.com/FluxML/Flux.jl
[2]: https://arxiv.org/abs/2006.10739
