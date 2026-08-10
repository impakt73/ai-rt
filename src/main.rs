use std::{error::Error, fs::File, io, path::PathBuf};

use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about = "Generate a solid red PNG image")]
struct Args {
    /// Image width in pixels.
    #[arg(long, default_value_t = 64)]
    width: u32,

    /// Image height in pixels.
    #[arg(long, default_value_t = 64)]
    height: u32,

    /// Output PNG filename.
    #[arg(short, long, default_value = "output.png")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let pixel_count = (args.width as usize)
        .checked_mul(args.height as usize)
        .ok_or_else(|| io::Error::other("image dimensions are too large"))?;
    let pixels = vec![255, 0, 0].repeat(pixel_count);

    let file = File::create(&args.output)?;
    let mut encoder = png::Encoder::new(file, args.width, args.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;

    println!(
        "Wrote {}x{} red PNG to {}",
        args.width,
        args.height,
        args.output.display()
    );

    Ok(())
}
