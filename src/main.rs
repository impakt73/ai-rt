use std::{error::Error, fs::File, io, path::PathBuf};

use clap::Parser;
use glam::Vec3;
use rayon::prelude::*;

const CAMERA_POSITION: Vec3 = Vec3::ZERO;
const SPHERE_CENTER: Vec3 = Vec3::new(0.0, 0.0, -3.0);
const SPHERE_RADIUS: f32 = 1.0;
const SPHERE_COLOR: Vec3 = Vec3::new(1.0, 0.0, 0.0);
// This vector points from a surface toward the directional light.
const LIGHT_DIRECTION: Vec3 = Vec3::new(-0.5, 0.8, 1.0);
const AMBIENT_STRENGTH: f32 = 0.08;
const SPECULAR_STRENGTH: f32 = 0.35;
const SPECULAR_SHININESS: f32 = 32.0;
const HALF_FOV_TANGENT: f32 = 0.41421357;

#[derive(Debug, Parser)]
#[command(author, version, about = "Render a Phong-shaded red sphere")]
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
    if args.width == 0 || args.height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "image dimensions must be greater than zero",
        )
        .into());
    }

    let pixel_count = (args.width as usize)
        .checked_mul(args.height as usize)
        .ok_or_else(|| io::Error::other("image dimensions are too large"))?;
    let byte_count = pixel_count
        .checked_mul(3)
        .ok_or_else(|| io::Error::other("image dimensions are too large"))?;
    let pixels = render(args.width, args.height, byte_count);

    let file = File::create(&args.output)?;
    let mut encoder = png::Encoder::new(file, args.width, args.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;

    println!(
        "Wrote {}x{} ray-traced PNG to {}",
        args.width,
        args.height,
        args.output.display()
    );

    Ok(())
}

fn render(width: u32, height: u32, byte_count: usize) -> Vec<u8> {
    let aspect_ratio = width as f32 / height as f32;
    let light_direction = LIGHT_DIRECTION.normalize();
    let pixel_count = byte_count / 3;

    (0..pixel_count)
        .into_par_iter()
        .flat_map_iter(|index| {
            let x = index % width as usize;
            let y = index / width as usize;
            let screen_x =
                ((x as f32 + 0.5) / width as f32 * 2.0 - 1.0) * aspect_ratio * HALF_FOV_TANGENT;
            let screen_y = (1.0 - (y as f32 + 0.5) / height as f32 * 2.0) * HALF_FOV_TANGENT;
            let ray_direction = Vec3::new(screen_x, screen_y, -1.0).normalize();
            let color = shade(ray_direction, light_direction);

            [
                (color.x.clamp(0.0, 1.0) * 255.0) as u8,
                (color.y.clamp(0.0, 1.0) * 255.0) as u8,
                (color.z.clamp(0.0, 1.0) * 255.0) as u8,
            ]
        })
        .collect()
}

fn shade(ray_direction: Vec3, light_direction: Vec3) -> Vec3 {
    let Some(distance) = ray_sphere_intersection(CAMERA_POSITION, ray_direction) else {
        return Vec3::ZERO;
    };

    let hit_point = CAMERA_POSITION + ray_direction * distance;
    let normal = (hit_point - SPHERE_CENTER).normalize();
    let diffuse = normal.dot(light_direction).max(0.0);
    let view_direction = (CAMERA_POSITION - hit_point).normalize();
    let reflected_direction = (-light_direction).reflect(normal);
    let specular = reflected_direction
        .dot(view_direction)
        .max(0.0)
        .powf(SPECULAR_SHININESS)
        * SPECULAR_STRENGTH;

    SPHERE_COLOR * (AMBIENT_STRENGTH + diffuse) + Vec3::splat(specular)
}

fn ray_sphere_intersection(origin: Vec3, direction: Vec3) -> Option<f32> {
    let sphere_to_ray = origin - SPHERE_CENTER;
    let half_b = sphere_to_ray.dot(direction);
    let c = sphere_to_ray.length_squared() - SPHERE_RADIUS * SPHERE_RADIUS;
    let discriminant = half_b * half_b - c;

    if discriminant < 0.0 {
        return None;
    }

    let square_root = discriminant.sqrt();
    let near = -half_b - square_root;
    let far = -half_b + square_root;
    [near, far].into_iter().find(|distance| *distance >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_ray_hits_sphere() {
        let distance = ray_sphere_intersection(CAMERA_POSITION, Vec3::NEG_Z).unwrap();

        assert!((distance - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ray_misses_sphere() {
        assert!(ray_sphere_intersection(CAMERA_POSITION, Vec3::X).is_none());
    }
}
