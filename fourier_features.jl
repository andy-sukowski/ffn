# See LICENSE file for copyright and license details.

# generate Fourier series upto frequency n, without sin(0)=1
fourier(n::Int, x::Float32) = vcat(1, collect.(sincos.((1:n) * x))...)

# n-dimensional Fourier series in a hypercube (every coordinate in [-π, π])
function fourier_nd(freq::NTuple{N, Int}, coords::NTuple{N, Float32}) where {N}
	cartesian = Iterators.product(fourier.(freq, coords)...)
	return prod.(cartesian) # tensor
end

# n-dimensional Fourier features for every voxel of tensor
function gen_features(freq::NTuple{N, Int}, in_size::NTuple{N, Int}) where {N}
	features = Array{Float32}(undef, prod(1 .+ 2 .* freq), in_size...)
	for coords in Iterators.product(range.(1, in_size)...)
		norm_coords = Float32.(coords ./ in_size .* 2 .* π .- π)
		features[:, coords...] = vec(fourier_nd(freq, norm_coords))
	end
	return features
end
