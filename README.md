# ai-rt

`ai-rt` is a Rust-based CPU ray tracing project. It will be developed using
agentic AI workflows, with small, reviewable iterations that are verified by
the compiler and automated tests as the renderer grows.

## Current Status

The binary ray traces scenes described in TOML. The default barycentric shading
mode visualizes triangle coordinates as RGB colors; Phong shading remains
available with `--shading-mode phong`, PBR shading is available with
`--shading-mode pbr`, and a Burn MLP approximation of the PBR shader is
available with `--shading-mode mlp`.
The background is black, pixels are ray
traced in parallel with Rayon in 8x8 tiles, and the checked-in `scene.toml`
template contains three spheres. By default, the program loads `scene.toml`
and writes a `64x64` image to `output.png`.

## Development Direction

Planned milestones include:

- Establish a simple, testable ray tracing data model.
- Implement rays, intersections, materials, and a camera.
- Add a CPU rendering pipeline and image output.
- Build scenes incrementally while measuring correctness and performance.
- Use agentic workflows for research, implementation, review, and validation.

Each change should remain focused, explain its intent, and include appropriate
verification before it is merged.

## Getting Started

Run the program with its default settings:

```sh
cargo run
```

Specify the image dimensions and output filename with CLI arguments:

```sh
cargo run -- --width 128 --height 96 --output render.png
```

Select the shading mode explicitly when rendering. Material properties can be
constants or texture maps, sampled from UVs generated for each sphere triangle.
Set a material's `uv_scale = [u, v]` to tile maps across the sphere:

```sh
cargo run -- --shading-mode barycentrics
cargo run -- --shading-mode phong
cargo run -- --shading-mode pbr
cargo run -- --shading-mode mlp --shader-model models/pbr_mlp_v1/model
```

Use a different scene description with `--scene`:

```sh
cargo run -- --scene examples/scene.toml --output render.png
```

Scene files contain `[camera]`, `[light]`, `[geometry]`, and `[materials.*]`
sections, along with any number of `[[objects]]` sphere entries. Materials own
an `albedo`, optional `normal_map`, `roughness`, and `metalness` property. These
accept a direct constant or a texture table. Texture maps use RGB images;
scalar maps use their red channel. Normal maps use tangent-space RGB values
encoded from `[-1, 1]` to `[0, 1]`. The explicit `constant` table form is also
available:

```toml
[materials.example]
albedo = { constant = [0.8, 0.4, 0.2] }
normal_map = { texture = "textures/default_normal.png" }
roughness = { texture = "textures/default_roughness.png" }
metalness = 0.0
# MLP scenes also require a floating-point latent texture:
# latent = { texture = "latents/example.latent" }
```

Roughness and metalness default to `0.5` and `0.0`, respectively. PBR uses the
material normal map when present, then applies Burley diffuse and Cook-Torrance
GGX specular. Each object specifies a material by name with `material = "name"`.
If an object omits that field, the material named `default` is selected, or the
only material is selected when the scene defines exactly one material.
Positions and colors are XYZ/RGB arrays; camera and light `yaw`, `pitch`, and
`roll` values are degrees. Relative texture paths are resolved from the scene
file's directory. The checked-in template uses
`textures/default_albedo.png`, `textures/default_roughness.png`,
`textures/default_metalness.png`, and `textures/default_normal.png` on its first
sphere, while its other spheres exercise constant material properties.

`geometry.latitude_segments` and `geometry.longitude_segments` control the
triangle density of the one unit sphere mesh that is shared by all objects.
See `scene.toml` for a complete example.

Show all available options with:

```sh
cargo run -- --help
```

Format and check the project with:

```sh
cargo fmt --check
cargo check
```

## Training the PBR MLP

The optional training binary distills the shared PBR reference function into a
small Burn MLP while jointly learning a latent vector at every texel of each
source material's latent texture. The default grid is 8x8; use `--latent-width
1 --latent-height 1` for one global vector per material. MLP scenes must
reference the exported textures from each material.

```sh
cargo run --release --features train --bin train_shader -- \
  --scene scene.toml \
  --samples 8192 --epochs 2 --batch-size 256 \
  --latent-size 8 --latent-width 8 --latent-height 8 \
  --output models/pbr_mlp_v1/model
```

The command writes a Burn MessagePack checkpoint and JSON manifest beside it,
plus `latents/manifest.json` and one multi-channel `.latent` file per material. Runtime
inference uses the CPU Flex backend and evaluates visible MLP hits in batches
per render tile rather than creating a tensor per pixel. The model input
contract is normal, light direction, view direction, and the material latent
vector, in that order. The old `phong_mlp_v1` checkpoint is not compatible with
this PBR contract.

## License

License terms will be added as the project is prepared for publication.
