# See LICENSE file for copyright and license details.

using CUDA
using FileIO
using Flux
using Images
using ProgressMeter
using VideoIO

include("../fourier_features.jl")

# tweak parameters to your liking
in_filename = "new_york.png"
out_filename = "new_york.mkv"
max_freq = (10, 10)
batchsize = 4
epochs = 200
η = 3.0 # learning rate

img = load(in_filename)
img_ = channelview(img)

# define multilayer perceptron model
model = Chain(
	Dense(prod(1 .+ 2 .* max_freq) => 150, σ),
	Dense(150 => 50, σ),
	Dense(50 => 15, σ),
	Dense(15 => 3, σ) # last layer size: number of color channels
) |> gpu
optim = Flux.setup(Descent(η), model)

# generate training data
features = gen_features(max_freq, size(img))
train_x = reshape(features, size(features, 1), :)
train_y = Float32.(reshape(img_, size(img_, 1), :))
loader = Flux.DataLoader((train_x, train_y) |> gpu; batchsize, shuffle=true)

# train network and save video
p = Progress(epochs * length(loader); desc="Training network:", dt=0.1)
losses = Matrix{Float32}(undef, length(loader), epochs)
open_video_out(out_filename, RGB{N0f8}, size(img)) do writer
	epoch_loss = "n/a"
	for epoch in 1:epochs
		for (i, (x, y)) in zip(1:length(loader), loader)
			losses[i, epoch], grads = Flux.withgradient(model) do m
				Flux.mse(m(x), y)
			end
			Flux.update!(optim, model, grads[1])
			next!(p; showvalues = [(:epoch, epoch), (:batch, i), (:loss, epoch_loss)])
		end
		epoch_loss = sum(losses[:, epoch]) / length(loader)
		write(writer, RGB{N0f8}.(colorview(RGB, reshape(model(train_x), size(img_)))))
	end
end
finish!(p)
