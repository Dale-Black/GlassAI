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

# ╔═╡ a0a39283-d21f-43ef-b52e-5d644a70d4c0
md"""
# Image Segmentation Dashboard

Welcome to the Image Segmentation Dashboard! This interactive web app allows you to load an image, select points for segmentation, and view the segmented mask.

## How to Use

1. Load an image by clicking on the "Choose File" button below.
2. Click on the image to select points for segmentation. You can select multiple points.
3. Check the "Run Segmentation" checkbox to initiate the segmentation process.
4. The segmented mask will be displayed below the image.
5. To clear the selected points and start over, click the "Clear Points" button.

Let's get started!
"""

# ╔═╡ 62a5940d-b893-4158-bf8a-94b2333d9761
md"""
## Load Image

Click the "Choose File" button to load an image:
"""

# ╔═╡ 2c2a2742-e455-4c56-9f68-d523d517dbca
@bind im FilePicker([MIME("image/*")])

# ╔═╡ a8b8d896-06ff-4702-a827-4beda9204466
begin
    img = im == nothing ? nothing : load(IOBuffer(im["data"]))
    img_arr = img == nothing ? zeros(100, 100, 3) : reverse(permutedims(channelview(img), (2, 3, 1)), dims = 1)
end;

# ╔═╡ 27ac9ce3-28cf-42ef-9e18-e7124b5f3808
md"""
## Select Segmentation Points

Click on the image to select points for segmentation. You can select multiple points.
"""

# ╔═╡ cd88a1ea-52c5-4596-9bd3-7dd7702bc1a5
md"""
Run Segmentation: $(@bind run_segmentation CheckBox())

Clear Points: $(@bind clear_points CheckBox())
"""

# ╔═╡ d09ab4f8-3d93-43d7-ab37-cb48d1acb066
@bind clicks let
    if clear_points
        clicks = []
    end
    
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

# ╔═╡ 78d30dba-5789-474f-bffd-c433e85f9478
if run_segmentation && clicks != nothing && img != nothing
    input_points = hcat(clicks...)
    input_labels = ones(Int, size(input_points, 2))
end

# ╔═╡ 52c5e6e7-a49f-495a-9a3c-4c6411bdee94
md"""
## Segmentation Result

The segmented mask will be displayed below:
"""

# ╔═╡ d9fd8e8e-8188-40d9-b932-44d4a5ac6fb5
md"""
## Appendix

This section contains the necessary dependencies and utility functions used in the dashboard.
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

# ╔═╡ 05ae620f-ac06-4d17-b8b4-f485d4be74be
if run_segmentation && clicks != nothing && img != nothing
    mask = run_efficient_sam(img_arr, input_points, input_labels, efficient_sam_vitt_model)
    mask_float = Float64.(mask)
else
    mask_float = nothing
end;

# ╔═╡ 4104d784-6b91-4246-b1a8-d2ad576cd46d
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
# ╟─a0a39283-d21f-43ef-b52e-5d644a70d4c0
# ╟─62a5940d-b893-4158-bf8a-94b2333d9761
# ╟─2c2a2742-e455-4c56-9f68-d523d517dbca
# ╠═a8b8d896-06ff-4702-a827-4beda9204466
# ╟─27ac9ce3-28cf-42ef-9e18-e7124b5f3808
# ╟─cd88a1ea-52c5-4596-9bd3-7dd7702bc1a5
# ╟─d09ab4f8-3d93-43d7-ab37-cb48d1acb066
# ╠═78d30dba-5789-474f-bffd-c433e85f9478
# ╠═05ae620f-ac06-4d17-b8b4-f485d4be74be
# ╟─52c5e6e7-a49f-495a-9a3c-4c6411bdee94
# ╟─4104d784-6b91-4246-b1a8-d2ad576cd46d
# ╟─d9fd8e8e-8188-40d9-b932-44d4a5ac6fb5
# ╠═211b4260-be19-4127-8da2-527e290f6e2d
# ╠═1023bb4d-219f-4843-8dc2-878a0760ee93
# ╠═115cdd0c-f798-4478-96c0-64339f9d04bd
# ╠═a6e5ec2f-df65-4853-8936-9254acfce324
# ╠═27a8a934-9b14-4bec-8d0c-daed631a0fa7
# ╠═b8e53ad9-41a7-46a9-a54b-8bdccd9fc31b
# ╠═5d9b7817-93e0-487b-aecf-fe69891a93fd
# ╠═ddae5f2c-38aa-4702-a93d-1c1702f8f09c
# ╠═95fd162b-33f1-4801-9725-53e73b1edc6b
