# ai-rt

`ai-rt` is a Rust-based CPU ray tracing project. It will be developed using
agentic AI workflows, with small, reviewable iterations that are verified by
the compiler and automated tests as the renderer grows.

## Current Status

The binary ray traces scenes described in TOML. The default barycentric shading
mode visualizes triangle coordinates as RGB colors; Phong shading remains
available with `--shading-mode phong`, and Burn MLP shading is available with
`--shading-mode mlp`. The background is black, pixels are ray
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

Select the shading mode explicitly when rendering:

```sh
cargo run -- --shading-mode barycentrics
cargo run -- --shading-mode phong
cargo run -- --shading-mode mlp --shader-model models/phong_mlp_v1/model
```

Use a different scene description with `--scene`:

```sh
cargo run -- --scene examples/scene.toml --output render.png
```

Scene files contain `[camera]`, `[light]`, and `[geometry]` sections, along
with any number of `[[objects]]` sphere entries. Positions and colors are
XYZ/RGB arrays; camera and light `yaw`, `pitch`, and `roll` values are degrees.
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

## Training the MLP

The optional training binary distills the shared Phong reference function into
a small Burn MLP. The default training configuration is intentionally short so
the initial checkpoint is only a starting point for later refinement:

```sh
cargo run --release --features train --bin train_shader -- \
  --samples 8192 --epochs 2 --batch-size 256 \
  --output models/phong_mlp_v1/model
```

The command writes a Burn MessagePack checkpoint and a JSON manifest beside
it. Runtime inference uses the CPU Flex backend and evaluates visible MLP hits
in batches per render tile rather than creating a tensor per pixel. The model
input contract is normal, light direction, view direction, and material color,
in that order.

## License

License terms will be added as the project is prepared for publication.
