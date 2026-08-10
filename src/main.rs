mod cli;
mod geometry;
mod image;
mod mlp;
mod render;
mod scene;
mod shader;

use std::{error::Error, io, path::PathBuf};

use clap::Parser;

use cli::Args;
use image::write_png;
use render::render;
use scene::{DEFAULT_SCENE_PATH, load_scene};

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.width == 0 || args.height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "image dimensions must be greater than zero",
        )
        .into());
    }

    let scene_path = args
        .scene
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SCENE_PATH));
    let scene = load_scene(&scene_path, args.shading_mode, args.shader_model.as_deref())?;
    (args.width as usize)
        .checked_mul(args.height as usize)
        .and_then(|pixel_count| pixel_count.checked_mul(3))
        .ok_or_else(|| io::Error::other("image dimensions are too large"))?;
    let pixels = render(args.width, args.height, &scene);
    write_png(&args.output, args.width, args.height, &pixels)?;

    println!(
        "Wrote {}x{} ray-traced PNG to {} using scene {}",
        args.width,
        args.height,
        args.output.display(),
        scene_path.display()
    );

    Ok(())
}
