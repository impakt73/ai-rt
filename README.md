# ai-rt

`ai-rt` is a Rust-based CPU ray tracing project. It will be developed using
agentic AI workflows, with small, reviewable iterations that are verified by
the compiler and automated tests as the renderer grows.

## Current Status

The binary generates a solid red PNG image. By default, it writes a `64x64`
image to `output.png`.

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
