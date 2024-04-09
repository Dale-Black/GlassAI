### A Pluto.jl notebook ###
# v0.19.40

#> [frontmatter]
#> title = "Background Remover"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 14618997-e3b2-4ed6-a290-56b1d177ab8e
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()

	using Downloads
	using PythonCall

	# import Python packages
	const np = pyimport("numpy")
	const efficient_sam = pyimport("efficient_sam.build_efficient_sam")
	const torch = pyimport("torch")
	const torchvision = pyimport("torchvision")
	
	const build_efficient_sam_vitt = efficient_sam.build_efficient_sam_vitt
	const efficient_sam_vitt_model = build_efficient_sam_vitt()
	efficient_sam_vitt_model.eval()
	
	using Images
	using GLMakie
end

# ╔═╡ 690a08a6-b365-4962-85fd-02536e7746d3
using PlutoUI

# ╔═╡ 355bf774-d3e8-48d8-8354-96f9cfc6fd98
md"""
# Image Segmentation Dashboard

Welcome to the Image Segmentation Dashboard! This interactive web app allows you to load an image, select a prompt point for segmentation, and view the segmented mask.

## How to Use

1. Load an image by clicking on the "Choose File" button below. [The default image can be found here](https://upload.wikimedia.org/wikipedia/commons/a/a1/Beagle_and_sleeping_black_and_white_kitty-01.jpg)
2. Adjust the sliders `X` and `Y` below to specify the coordinate of the prompt point for segmentation.
3. Check the "Run Segmentation" checkbox to initiate the segmentation process.
4. The segmented mask will be displayed below the image.

Let's get started!
"""

# ╔═╡ 61093d24-9e24-4f55-b675-497b9c7d376c
md"""
---
"""

# ╔═╡ cbdba8b3-06f5-4f0d-ab99-8ddb74ca733b
function default_image()
	url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Beagle_and_sleeping_black_and_white_kitty-01.jpg"

	img = load(Downloads.download(url))
	img = convert(Matrix{RGB{N0f8}}, img)
end

# ╔═╡ 9fb50aec-a21e-495d-8106-86f13c10a9da
function preprocess(
	img::Matrix{<:Colorant},
	point_prompt::NTuple{2,Int},
	)

	# preprocess Julia's image data
	np_img = np.asarray(float32.(channelview(img)))
	# [C, H, W] -> [B, C, H, W]
	img_tensor = torch.from_numpy(np_img).unsqueeze(0)
	
	# preprocess points
	pt = collect(point_prompt) # convert to Vector{Int}
	input_points = hcat([pt]...) # 2x1 Matrix

	num_pts = size(input_points, 2)
	
	pts_sampled = reshape(input_points, (1, 1, num_pts, 2))
	# prepare pts_labels
	pts_labels = ones(Int, num_pts)

	return (
		img_tensor, 
		torch.from_numpy(np.array(pts_sampled, dtype=np.float32)),
		torch.from_numpy(np.array(pts_labels, dtype=np.float32)),
	)
end

# ╔═╡ 25c0631c-734f-4c87-9b6a-d8cafc5698a0
function run_efficient_sam(model, img, point_prompt)
	img_tensor_py, pts_sampled_py, pts_labels_py = preprocess(
		img, point_prompt
	)
    
    # Run the model
    predicted_logits_py, predicted_iou_py = model(
        img_tensor_py,
        pts_sampled_py,
        pts_labels_py,
    )
    
    # Convert PyTorch tensors to NumPy arrays
    predicted_logits_np = pyconvert(
        Array{Float32}, predicted_logits_py.cpu().detach().numpy()
    )
    
    # Postprocess the output data in Julia
    predicted_mask = predicted_logits_np[1, 1, 1, :, :] .< 0
    
    return predicted_mask
end

# ╔═╡ 1f07da64-4618-40e7-bbd3-64869e48c4a2
ui_filepicker = @bind _im FilePicker([MIME("image/*")])

# ╔═╡ e50b6e0a-cd17-44f5-873f-6f048c149a29
md"""
Load an image by clicking on the "Choose File"

$(ui_filepicker)
"""

# ╔═╡ 75f464bc-e098-4ca2-a699-12ac2a7afc56
begin
	img = isnothing(_im) ? default_image() : load(IOBuffer(_im["data"]))
	img = convert(Matrix{RGB{N0f8}}, img)
	H, W = size(img)
	ui_prompt_y = @bind prompt_y PlutoUI.Slider(1:H, default=H÷2)
	ui_prompt_x = @bind prompt_x PlutoUI.Slider(1:W, default=W÷2)
	ui_run_segmentation = @bind run_segmentation CheckBox()
	nothing
end

# ╔═╡ 8da5e2da-3168-4f0b-9bb6-c818e1053c70
md"""
X: $(ui_prompt_x) $(prompt_x)

Y: $(ui_prompt_y) $(prompt_y)

Run Segmentation: $(ui_run_segmentation)
"""

# ╔═╡ 59c465ed-3335-4f33-b7ae-ce48c2cd4d58
function demo(img, prompt_x::Int, prompt_y::Int, run_segmentation::Bool)
	# setup canvas
	makie_img = Makie.image(rotr90(img))
	makie_img.axis.autolimitaspect = 1
	hidedecorations!(makie_img.axis)
	# draw a prompt point
	point_prompt = (prompt_x, prompt_y)
	scatter!(
		makie_img.axis, 
		[prompt_x], [H - prompt_y], 
		marker=:cross, markersize=50, alpha=0.5,
		color=:red,
	)

	if run_segmentation
		predicted_mask = run_efficient_sam(
			efficient_sam_vitt_model, 
			img,
			point_prompt
		)
	
		m = rotr90(map(predicted_mask) do x
			 x ? RGBAf(0, 1, 0, 0.5) : RGBAf(0, 0, 0, 0)
		end)
		image!(makie_img.axis, m; transparency=true)
	end
	makie_img
end

# ╔═╡ fada5bf7-36f6-48fb-b783-934eb062c097
demo(img, prompt_x, prompt_y, run_segmentation)

# ╔═╡ Cell order:
# ╟─355bf774-d3e8-48d8-8354-96f9cfc6fd98
# ╟─e50b6e0a-cd17-44f5-873f-6f048c149a29
# ╟─8da5e2da-3168-4f0b-9bb6-c818e1053c70
# ╟─fada5bf7-36f6-48fb-b783-934eb062c097
# ╟─61093d24-9e24-4f55-b675-497b9c7d376c
# ╠═14618997-e3b2-4ed6-a290-56b1d177ab8e
# ╠═690a08a6-b365-4962-85fd-02536e7746d3
# ╠═cbdba8b3-06f5-4f0d-ab99-8ddb74ca733b
# ╠═9fb50aec-a21e-495d-8106-86f13c10a9da
# ╠═25c0631c-734f-4c87-9b6a-d8cafc5698a0
# ╠═1f07da64-4618-40e7-bbd3-64869e48c4a2
# ╠═75f464bc-e098-4ca2-a699-12ac2a7afc56
# ╠═59c465ed-3335-4f33-b7ae-ce48c2cd4d58
