# ai-rt

`ai-rt` is a Rust-based CPU ray tracing project. It will be developed using
agentic AI workflows, with small, reviewable iterations that are verified by
the compiler and automated tests as the renderer grows.

## Current Status

The binary ray traces scenes described in TOML. The default barycentric shading
mode visualizes triangle coordinates as RGB colors; Phong shading remains
available with `--shading-mode phong`. The background is black, pixels are ray
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

## License

License terms will be added as the project is prepared for publication.
