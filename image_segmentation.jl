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

# ╔═╡ 211b4260-be19-4127-8da2-527e290f6e2d
# ╠═╡ show_logs = false
begin
	using Pkg; Pkg.activate("."); Pkg.instantiate()
	using PythonCall

	### Import Python modules
	np = pyimport("numpy")
	efficient_sam = pyimport("efficient_sam.build_efficient_sam")
	zipfile = pyimport("zipfile")
	torch = pyimport("torch")
	torchvision = pyimport("torchvision")

	build_efficient_sam_vitt = efficient_sam.build_efficient_sam_vitt
	efficient_sam_vitt_model = build_efficient_sam_vitt()
	efficient_sam_vitt_model.eval()
end

# ╔═╡ 1023bb4d-219f-4843-8dc2-878a0760ee93
using PlutoUI: TableOfContents, FilePicker, CheckBox

# ╔═╡ 115cdd0c-f798-4478-96c0-64339f9d04bd
using PlutoPlotly

# ╔═╡ a6e5ec2f-df65-4853-8936-9254acfce324
using PlotlyBase: add_trace!, attr

# ╔═╡ b8e53ad9-41a7-46a9-a54b-8bdccd9fc31b
using Images: load, channelview

# ╔═╡ d0c22900-22d3-4a88-9c81-5b5e78812edc
md"""
# Load Image
"""

# ╔═╡ 58fbdfd0-bac9-476b-816d-babf166fba84
@bind im FilePicker([MIME("image/*")])

# ╔═╡ 2caa8cf2-d9db-4fd6-99fc-6831e81bc61d
begin
	img = im == nothing ? nothing : load(IOBuffer(im["data"]))
	img_arr = img == nothing ? zeros(100, 100, 3) : reverse(permutedims(channelview(img), (2, 3, 1)), dims = 1)
end;

# ╔═╡ 2838443c-3d45-4cea-aac7-91e32d83606b
md"""
## Choose Segmentation Points

Click on any number of locations in the picture to select points for segmentation. Press "Run Segmentation" to initiate the segmentation.
"""

# ╔═╡ 1b2c42a2-fbc0-4aba-a0d5-99ca1c614fa9
md"""
Run Segmentation: $(@bind run_segmentation CheckBox())
"""

# ╔═╡ 8656006a-a65d-4ea8-ae17-780688b9b203
@bind clicks let
    p = plot(heatmap(z=img_arr[:, :, 1], colorscale = "Greys"))
    if img != nothing
        add_plotly_listener!(p, "plotly_click", "
            (function() {
                var clicks = [];
                return function(e) {
                    let dt = e.points[0];
                    clicks.push([dt.x, dt.y]);
                    
                    // Add a scatter trace for the clicked point
                    let trace = {
                        x: [dt.x],
                        y: [dt.y],
                        mode: 'markers',
                        marker: {
                            size: 10,
                            color: 'red'
                        }
                    };
                    Plotly.addTraces(PLOT, [trace]);
                    
                    PLOT.value = clicks;
                    PLOT.dispatchEvent(new CustomEvent('input'));
                };
            })()
        ")
    end
    p
end

# ╔═╡ c584b2f9-fae5-40a9-b9b5-591a42aedfa0
if run_segmentation && clicks != nothing && img != nothing
	input_points = hcat(clicks...)
	input_labels = ones(Int, size(input_points, 2))
end

# ╔═╡ 9c3b19d4-ef1e-45e0-8d64-53698c0f39c0
md"""
## View Segmentation Mask
"""

# ╔═╡ 47163547-f805-4f57-a98a-6fd045f0440f
md"""
#### Appendix
"""

# ╔═╡ 27a8a934-9b14-4bec-8d0c-daed631a0fa7
import PlotlyJS

# ╔═╡ 5d9b7817-93e0-487b-aecf-fe69891a93fd
TableOfContents()

# ╔═╡ ddae5f2c-38aa-4702-a93d-1c1702f8f09c
function preprocess(img::AbstractArray{T, 3}, pts_sampled, pts_labels) where {T}
    # Preprocess the input data
    image_np = Float32.(img)
    img_tensor = permutedims(image_np, (3, 1, 2))
    img_tensor = reshape(img_tensor, (1, size(img_tensor)...))
    
    pts_sampled = reshape(pts_sampled, (1, 1, size(pts_sampled, 2), 2))
    pts_labels = reshape(pts_labels, (1, length(pts_labels)))
    
    # Convert Julia arrays to PyTorch tensors
    img_tensor_py = torch.tensor(np.array(img_tensor, dtype=np.float32))
    pts_sampled_py = torch.tensor(np.array(pts_sampled, dtype=np.float32))
    pts_labels_py = torch.tensor(np.array(pts_labels, dtype=np.float32))

    return img_tensor_py, pts_sampled_py, pts_labels_py
end

# ╔═╡ 95fd162b-33f1-4801-9725-53e73b1edc6b
function run_efficient_sam(
	img::AbstractArray{T, 3}, pts_sampled, pts_labels, model
	) where {T}
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

# ╔═╡ 8339a4c6-d69e-4798-a0f5-ad90bf331a35
if run_segmentation && clicks != nothing && img != nothing
	mask = run_efficient_sam(img_arr, input_points, input_labels, efficient_sam_vitt_model)
	mask_float = Float64.(mask)
else
	mask_float = nothing
end;

# ╔═╡ 7652852c-fb47-43f0-bed8-ba29700f95a5
let
	# Create a heatmap of the original image
	p = plot(heatmap(z=img_arr[:, :, 1], colorscale = "Greys"))

	if mask_float != nothing
		# Create a new trace for the segmentation mask
		mask_trace = PlotlyJS.heatmap(z=mask_float, colorscale = "Jet", opacity=0.5)
		
		# Add the segmentation mask trace to the existing plot
		addtraces!(p, mask_trace)
	end
	
	# Update the layout to set the title
	relayout!(p, title_text = "Segmentation Mask")
	
	p
end

# ╔═╡ Cell order:
# ╟─d0c22900-22d3-4a88-9c81-5b5e78812edc
# ╟─58fbdfd0-bac9-476b-816d-babf166fba84
# ╠═2caa8cf2-d9db-4fd6-99fc-6831e81bc61d
# ╟─2838443c-3d45-4cea-aac7-91e32d83606b
# ╟─1b2c42a2-fbc0-4aba-a0d5-99ca1c614fa9
# ╟─8656006a-a65d-4ea8-ae17-780688b9b203
# ╠═c584b2f9-fae5-40a9-b9b5-591a42aedfa0
# ╠═8339a4c6-d69e-4798-a0f5-ad90bf331a35
# ╟─9c3b19d4-ef1e-45e0-8d64-53698c0f39c0
# ╟─7652852c-fb47-43f0-bed8-ba29700f95a5
# ╟─47163547-f805-4f57-a98a-6fd045f0440f
# ╠═211b4260-be19-4127-8da2-527e290f6e2d
# ╠═1023bb4d-219f-4843-8dc2-878a0760ee93
# ╠═115cdd0c-f798-4478-96c0-64339f9d04bd
# ╠═a6e5ec2f-df65-4853-8936-9254acfce324
# ╠═27a8a934-9b14-4bec-8d0c-daed631a0fa7
# ╠═b8e53ad9-41a7-46a9-a54b-8bdccd9fc31b
# ╠═5d9b7817-93e0-487b-aecf-fe69891a93fd
# ╠═ddae5f2c-38aa-4702-a93d-1c1702f8f09c
# ╠═95fd162b-33f1-4801-9725-53e73b1edc6b
