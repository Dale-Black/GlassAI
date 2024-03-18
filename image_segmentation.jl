### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ e4d881c6-e31c-11ee-3cba-090365ce157a
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".")

# ╔═╡ 3c073890-af5d-4c37-843c-4b03fc7c1f31
using PlutoUI: TableOfContents, FilePicker

# ╔═╡ 58ecc385-02df-43a8-86d7-cf7316e78fb3
# ╠═╡ show_logs = false
using PythonCall

# ╔═╡ d567adda-a6b9-43b2-838b-5dd6edffcc95
using PlutoPlotly

# ╔═╡ b8c82a92-7613-4707-ba5b-29f171f59220
# ╠═╡ show_logs = false
using Images: load, channelview

# ╔═╡ 2f5631e4-baeb-46b3-947d-310b2a99d71a
md"""
# Load Image
"""

# ╔═╡ 2b3c2be9-e179-45bc-ac06-e67acd37f880
@bind im FilePicker()

# ╔═╡ 0f7a88af-4a82-46d0-ac3e-94c9a71736b4
md"""
## Choose Segmentation Points

Click on any two locations in the picture, to initiate the segmentation of that area
"""

# ╔═╡ 089d68e2-3c72-43c6-8f1d-0162e3f058ff
md"""
## View Segmentation Mask
"""

# ╔═╡ 62ed71dd-1598-4f1b-995c-db4483be5497
md"""
#### Appendix
"""

# ╔═╡ d4935235-404f-4cca-a4a1-a39031c12953
import CairoMakie

# ╔═╡ 27f17af6-d8f8-4d15-b1f5-3b21decad981
TableOfContents()

# ╔═╡ dbe6f5c9-b7f6-4ae5-89f3-800fcff7d179
begin
	### Import Python modules
	np = pyimport("numpy")
	efficient_sam = pyimport("efficient_sam.build_efficient_sam")
	zipfile = pyimport("zipfile")
	torch = pyimport("torch")
	torchvision = pyimport("torchvision")
end

# ╔═╡ 0bc72508-f018-441d-9faa-aa59f2c3564a
function preprocess(img::AbstractArray{T, 3}, pts_sampled, pts_labels) where {T}
	# Preprocess the input data
    image_np = Float32.(img)
    img_tensor = permutedims(image_np, (3, 1, 2))
    img_tensor = reshape(img_tensor, (1, size(img_tensor)...))
    
    pts_sampled = reshape(pts_sampled, (1, 1, size(pts_sampled, 1), 2))
    pts_labels = reshape(pts_labels, (1, 1, size(pts_labels, 1)))
    
    # Convert Julia arrays to PyTorch tensors
    img_tensor_py = torch.tensor(np.array(img_tensor, dtype=np.float32))
    pts_sampled_py = torch.tensor(np.array(pts_sampled, dtype=np.float32))
    pts_labels_py = torch.tensor(np.array(pts_labels, dtype=np.float32))

	return img_tensor_py, pts_sampled_py, pts_labels_py
end

# ╔═╡ 14949606-2b9b-4e26-aa4e-b56545768c09
function run_efficient_sam(
	img::AbstractArray{T, 3}, pts_sampled, pts_labels, model) where {T}
	img_tensor_py, pts_sampled_py, pts_labels_py = preprocess(img, pts_sampled, pts_labels) 
    
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

# ╔═╡ 58a53f29-4f85-47d8-8a01-43ce60547cf2
begin
	build_efficient_sam_vitt = efficient_sam.build_efficient_sam_vitt
	efficient_sam_vitt_model = build_efficient_sam_vitt()
	efficient_sam_vitt_model.eval()
end

# ╔═╡ a7d7fb49-904f-4bfd-87d0-9e853e614333
begin
	img = load(IOBuffer(im["data"]))
	img_arr = reverse(permutedims(channelview(img), (2, 3, 1)), dims = 1);
end;

# ╔═╡ c3d9374c-16e5-4712-8014-44767792c54a
@bind clicks let
	p = plot(heatmap(z=img_arr[:, :, 1], colorscale = "Greys"))
	add_plotly_listener!(p, "plotly_click", "
		(function() {
			var clicks = [];
			return function(e) {
				let dt = e.points[0];
				clicks.push([dt.x, dt.y]);
				if (clicks.length === 2) {
					PLOT.value = clicks;
					PLOT.dispatchEvent(new CustomEvent('input'));
					clicks = [];
				}
			};
		})()
	")
	p
end

# ╔═╡ f8cb60a6-7836-4a71-863f-a23756a7292b
if clicks != nothing
	input_point = hcat(clicks...)
end

# ╔═╡ a461f0e5-6690-4602-b3fd-c5c14923b984
input_label = [1, 1]

# ╔═╡ b3a92d8c-768d-4e03-9423-0bd4b4780f90
# ╠═╡ show_logs = false
if clicks != nothing
	mask = run_efficient_sam(img_arr, input_point, input_label, efficient_sam_vitt_model)
	mask_float = Float64.(mask)
end;

# ╔═╡ a3d33cdf-9e43-4df7-bf6e-1053720b1370
if clicks != nothing && mask_float != nothing
	let
		f = CairoMakie.Figure()
		ax = CairoMakie.Axis(f[1, 1])
		CairoMakie.heatmap!(ax, transpose(img_arr[:, :, 1]), colormap = :grays)
		CairoMakie.heatmap!(ax, transpose(mask), colormap = (:jet, 0.5))
		f
	end
end

# ╔═╡ Cell order:
# ╟─2f5631e4-baeb-46b3-947d-310b2a99d71a
# ╟─2b3c2be9-e179-45bc-ac06-e67acd37f880
# ╟─0f7a88af-4a82-46d0-ac3e-94c9a71736b4
# ╟─c3d9374c-16e5-4712-8014-44767792c54a
# ╟─089d68e2-3c72-43c6-8f1d-0162e3f058ff
# ╟─a3d33cdf-9e43-4df7-bf6e-1053720b1370
# ╟─62ed71dd-1598-4f1b-995c-db4483be5497
# ╠═e4d881c6-e31c-11ee-3cba-090365ce157a
# ╠═3c073890-af5d-4c37-843c-4b03fc7c1f31
# ╠═58ecc385-02df-43a8-86d7-cf7316e78fb3
# ╠═d567adda-a6b9-43b2-838b-5dd6edffcc95
# ╠═b8c82a92-7613-4707-ba5b-29f171f59220
# ╠═d4935235-404f-4cca-a4a1-a39031c12953
# ╠═27f17af6-d8f8-4d15-b1f5-3b21decad981
# ╠═dbe6f5c9-b7f6-4ae5-89f3-800fcff7d179
# ╠═0bc72508-f018-441d-9faa-aa59f2c3564a
# ╠═14949606-2b9b-4e26-aa4e-b56545768c09
# ╠═58a53f29-4f85-47d8-8a01-43ce60547cf2
# ╠═a7d7fb49-904f-4bfd-87d0-9e853e614333
# ╠═f8cb60a6-7836-4a71-863f-a23756a7292b
# ╠═a461f0e5-6690-4602-b3fd-c5c14923b984
# ╠═b3a92d8c-768d-4e03-9423-0bd4b4780f90
