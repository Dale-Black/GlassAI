### A Pluto.jl notebook ###
# v0.19.40

#> [frontmatter]
#> title = "Image Segmentation"

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
using Images: load, channelview, RGB

# ╔═╡ 5f0b4424-b7c7-44e4-9e69-5b6f30f3793a
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

# ╔═╡ 2f104042-6e1f-4741-9f74-78d133676601
md"""
## Load Image

Click the "Choose File" button to load an image:
"""

# ╔═╡ ea15595f-080d-4011-96fb-f47c9a8a9ff5
@bind im FilePicker([MIME("image/*")])

# ╔═╡ 5d69009c-3101-4803-9cab-45dc469794e8
begin
    img = im == nothing ? nothing : load(IOBuffer(im["data"]))
    img_arr = if img == nothing
        zeros(100, 100, 3)
    else
        if size(channelview(img), 3) == 1
            # Grayscale image
			reshape(reverse(Float32.(img), dims = 1), (size(Float32.(img))...), 1)
        else
            # RGB image
            reverse(permutedims(channelview(img), (2, 3, 1)), dims = 1)
        end
    end
end;

# ╔═╡ 639d0fbd-dc9a-43ce-acc3-33c020bfd107
md"""
## Select Segmentation Points

Click on the image to select points for segmentation. You can select up to 5 points.
"""

# ╔═╡ 27b75d9d-ad20-4421-b5b4-533bc1736b08
md"""
Run Segmentation: $(@bind run_segmentation CheckBox())

Clear Points: $(@bind clear_points CheckBox())
"""

# ╔═╡ 0678942a-15f5-401b-8e9f-14de851a2e58
@bind clicks let
    if clear_points
        clicks = []
    end
    
    if img == nothing
        p = plot(heatmap(z=zeros(100, 100), colorscale = "Greys"))
    else
        if size(img_arr, 3) == 1
            # Grayscale image
            p = plot(heatmap(z=img_arr[:, :, 1], colorscale = "Greys"))
        else
            # RGB image
            z = map(img) do i
                [i.r, i.g, i.b] .* 255
            end
            z = collect(eachrow(z))
            im = PlutoPlotly.image(;z)
            p = PlutoPlotly.plot(im)
        end
    end
    
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

# ╔═╡ de21f60d-85fa-4b00-b6bb-d1c506f4a370
if run_segmentation && clicks != nothing && img != nothing
    input_points = hcat(clicks...)
    input_labels = ones(Int, size(input_points, 2))
end

# ╔═╡ 3ac9a240-80d9-4b01-a135-eb0254efa0d6
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

# ╔═╡ 98017e13-3d05-4b7c-992a-f76dcf0d6e39
function preprocess(img::AbstractArray{T, 3}, pts_sampled, pts_labels) where {T}
    # Preprocess the input data
    image = Float32.(img)
    img_tensor = permutedims(image, (3, 1, 2))
    img_tensor = reshape(img_tensor, (1, size(img_tensor)...))
    
    pts_sampled = reshape(pts_sampled, (1, 1, size(pts_sampled, 2), 2))
    pts_labels = reshape(pts_labels, (1, length(pts_labels)))
    
    # Convert Julia arrays to PyTorch tensors
    img_tensor_py = torch.tensor(np.array(img_tensor, dtype=np.float32))
    pts_sampled_py = torch.tensor(np.array(pts_sampled, dtype=np.float32))
    pts_labels_py = torch.tensor(np.array(pts_labels, dtype=np.float32))

    return img_tensor_py, pts_sampled_py, pts_labels_py
end

# ╔═╡ c43f7331-9323-4a72-b317-4bc3463c0a34
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

# ╔═╡ 359266b3-355c-494b-95dd-c4ab6943342c
if run_segmentation && clicks != nothing && img != nothing
    mask = run_efficient_sam(img_arr, input_points, input_labels, efficient_sam_vitt_model)
    mask_float = Float64.(mask)
else
    mask_float = nothing
end;

# ╔═╡ d466087e-dc8c-4ec4-bc30-3a22df6be834
let
    p = plot(heatmap(z=nothing, colorscale = "Greys"))
    
    if mask_float != nothing
		# Grayscale image
		p = plot(heatmap(z=img_arr[:, :, 1], colorscale = "Greys"))
        
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
# ╟─5f0b4424-b7c7-44e4-9e69-5b6f30f3793a
# ╟─2f104042-6e1f-4741-9f74-78d133676601
# ╟─ea15595f-080d-4011-96fb-f47c9a8a9ff5
# ╠═5d69009c-3101-4803-9cab-45dc469794e8
# ╟─639d0fbd-dc9a-43ce-acc3-33c020bfd107
# ╟─27b75d9d-ad20-4421-b5b4-533bc1736b08
# ╟─0678942a-15f5-401b-8e9f-14de851a2e58
# ╠═de21f60d-85fa-4b00-b6bb-d1c506f4a370
# ╠═359266b3-355c-494b-95dd-c4ab6943342c
# ╟─3ac9a240-80d9-4b01-a135-eb0254efa0d6
# ╟─d466087e-dc8c-4ec4-bc30-3a22df6be834
# ╟─d9fd8e8e-8188-40d9-b932-44d4a5ac6fb5
# ╠═211b4260-be19-4127-8da2-527e290f6e2d
# ╠═1023bb4d-219f-4843-8dc2-878a0760ee93
# ╠═115cdd0c-f798-4478-96c0-64339f9d04bd
# ╠═a6e5ec2f-df65-4853-8936-9254acfce324
# ╠═27a8a934-9b14-4bec-8d0c-daed631a0fa7
# ╠═b8e53ad9-41a7-46a9-a54b-8bdccd9fc31b
# ╠═5d9b7817-93e0-487b-aecf-fe69891a93fd
# ╠═98017e13-3d05-4b7c-992a-f76dcf0d6e39
# ╠═c43f7331-9323-4a72-b317-4bc3463c0a34
