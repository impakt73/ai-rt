use std::{error::Error, fs::File, io, path::PathBuf};

use clap::Parser;
use glam::{EulerRot, Quat, Vec3};
use rayon::prelude::*;
use serde::Deserialize;

const DEFAULT_SCENE_PATH: &str = "scene.toml";
const AMBIENT_STRENGTH: f32 = 0.08;
const SPECULAR_STRENGTH: f32 = 0.35;
const SPECULAR_SHININESS: f32 = 32.0;
const HALF_FOV_TANGENT: f32 = 0.41421357;
const TILE_SIZE: usize = 8;

#[derive(Debug, Parser)]
#[command(author, version, about = "Render a Phong-shaded sphere scene")]
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

    /// TOML scene description. Defaults to scene.toml.
    #[arg(short, long)]
    scene: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Scene {
    camera: CameraDescription,
    light: LightDescription,
    #[serde(default)]
    objects: Vec<SphereDescription>,
}

#[derive(Debug, Deserialize)]
struct CameraDescription {
    position: [f32; 3],
    #[serde(default)]
    yaw: f32,
    #[serde(default)]
    pitch: f32,
    #[serde(default)]
    roll: f32,
}

#[derive(Debug, Deserialize)]
struct LightDescription {
    #[serde(default)]
    yaw: f32,
    #[serde(default)]
    pitch: f32,
    #[serde(default)]
    roll: f32,
}

#[derive(Debug, Deserialize)]
struct SphereDescription {
    position: [f32; 3],
    radius: f32,
    color: [f32; 3],
}

#[derive(Debug)]
struct Camera {
    position: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
}

#[derive(Debug)]
struct Sphere {
    position: Vec3,
    radius: f32,
    color: Vec3,
}

#[derive(Debug)]
struct RenderScene {
    camera: Camera,
    light_direction: Vec3,
    spheres: Vec<Sphere>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PixelData {
    data: [u8; 3],
}

impl PixelData {
    fn new(red: u8, green: u8, blue: u8) -> Self {
        Self {
            data: [red, green, blue],
        }
    }

    fn from_color(color: Vec3) -> Self {
        Self::new(
            (color.x.clamp(0.0, 1.0) * 255.0) as u8,
            (color.y.clamp(0.0, 1.0) * 255.0) as u8,
            (color.z.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    fn write_rgb(&self, destination: &mut [u8]) {
        destination[..self.data.len()].copy_from_slice(&self.data);
    }
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

    let scene_path = args
        .scene
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SCENE_PATH));
    let scene = load_scene(&scene_path)?;
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

fn load_scene(path: &PathBuf) -> Result<RenderScene, Box<dyn Error>> {
    let contents = std::fs::read_to_string(path)?;
    let description: Scene = toml::from_str(&contents)?;
    let camera_rotation = rotation_from_degrees(
        description.camera.yaw,
        description.camera.pitch,
        description.camera.roll,
    );
    let light_rotation = rotation_from_degrees(
        description.light.yaw,
        description.light.pitch,
        description.light.roll,
    );

    let camera = Camera {
        position: Vec3::from_array(description.camera.position),
        right: camera_rotation * Vec3::X,
        up: camera_rotation * Vec3::Y,
        forward: camera_rotation * Vec3::NEG_Z,
    };
    let spheres = description
        .objects
        .into_iter()
        .map(|sphere| Sphere {
            position: Vec3::from_array(sphere.position),
            radius: sphere.radius,
            color: Vec3::from_array(sphere.color),
        })
        .collect();

    Ok(RenderScene {
        camera,
        light_direction: (light_rotation * Vec3::Z).normalize(),
        spheres,
    })
}

fn rotation_from_degrees(yaw: f32, pitch: f32, roll: f32) -> Quat {
    Quat::from_euler(
        EulerRot::YXZ,
        yaw.to_radians(),
        pitch.to_radians(),
        roll.to_radians(),
    )
}

fn render(width: u32, height: u32, scene: &RenderScene) -> Vec<PixelData> {
    let aspect_ratio = width as f32 / height as f32;
    let width = width as usize;
    let height = height as usize;
    let tiles_x = width.div_ceil(TILE_SIZE);
    let tiles_y = height.div_ceil(TILE_SIZE);
    let pixels_per_tile = TILE_SIZE * TILE_SIZE;

    let mut pixels = vec![PixelData::default(); tiles_x * tiles_y * pixels_per_tile];
    pixels
        .par_chunks_mut(pixels_per_tile)
        .enumerate()
        .for_each(|(tile_index, tile_pixels)| {
            let tile_x = tile_index % tiles_x;
            let tile_y = tile_index / tiles_x;
            let tile_start_x = tile_x * TILE_SIZE;
            let tile_start_y = tile_y * TILE_SIZE;

            for (morton_index, tile_pixel) in tile_pixels.iter_mut().enumerate() {
                let (local_x, local_y) = morton_coordinates(morton_index);
                let x = tile_start_x + local_x;
                let y = tile_start_y + local_y;
                if x >= width || y >= height {
                    continue;
                }

                *tile_pixel = render_pixel(x, y, width, height, aspect_ratio, scene);
            }
        });

    pixels
}

fn render_pixel(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    aspect_ratio: f32,
    scene: &RenderScene,
) -> PixelData {
    let screen_x = ((x as f32 + 0.5) / width as f32 * 2.0 - 1.0) * aspect_ratio * HALF_FOV_TANGENT;
    let screen_y = (1.0 - (y as f32 + 0.5) / height as f32 * 2.0) * HALF_FOV_TANGENT;
    let ray_direction =
        (scene.camera.forward + scene.camera.right * screen_x + scene.camera.up * screen_y)
            .normalize();
    let color = shade(scene.camera.position, ray_direction, scene);

    PixelData::from_color(color)
}

fn write_png(
    path: &PathBuf,
    width: u32,
    height: u32,
    tile_pixels: &[PixelData],
) -> Result<(), Box<dyn Error>> {
    let image_pixels = row_major_pixels(width, height, tile_pixels);

    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&image_pixels)?;

    Ok(())
}

fn row_major_pixels(width: u32, height: u32, tile_pixels: &[PixelData]) -> Vec<u8> {
    let width_usize = width as usize;
    let height_usize = height as usize;
    let tiles_x = width_usize.div_ceil(TILE_SIZE);
    let pixels_per_tile = TILE_SIZE * TILE_SIZE;
    let mut image_pixels = vec![0; width_usize * height_usize * 3];

    for (tile_index, tile) in tile_pixels.chunks(pixels_per_tile).enumerate() {
        let tile_x = tile_index % tiles_x;
        let tile_y = tile_index / tiles_x;
        let tile_start_x = tile_x * TILE_SIZE;
        let tile_start_y = tile_y * TILE_SIZE;

        for (morton_index, color) in tile.iter().enumerate() {
            let (local_x, local_y) = morton_coordinates(morton_index);
            let x = tile_start_x + local_x;
            let y = tile_start_y + local_y;
            if x >= width_usize || y >= height_usize {
                continue;
            }

            let offset = (y * width_usize + x) * 3;
            color.write_rgb(&mut image_pixels[offset..offset + 3]);
        }
    }

    image_pixels
}

fn morton_coordinates(index: usize) -> (usize, usize) {
    let mut x = 0;
    let mut y = 0;

    for bit in 0..TILE_SIZE.ilog2() as usize {
        x |= ((index >> (bit * 2)) & 1) << bit;
        y |= ((index >> (bit * 2 + 1)) & 1) << bit;
    }

    (x, y)
}

fn shade(origin: Vec3, ray_direction: Vec3, scene: &RenderScene) -> Vec3 {
    let Some((distance, sphere)) = scene
        .spheres
        .iter()
        .filter_map(|sphere| {
            ray_sphere_intersection(origin, ray_direction, sphere)
                .map(|distance| (distance, sphere))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
    else {
        return Vec3::ZERO;
    };

    let hit_point = origin + ray_direction * distance;
    let normal = (hit_point - sphere.position).normalize();
    let diffuse = normal.dot(scene.light_direction).max(0.0);
    let view_direction = (origin - hit_point).normalize();
    let reflected_direction = (-scene.light_direction).reflect(normal);
    let specular = reflected_direction
        .dot(view_direction)
        .max(0.0)
        .powf(SPECULAR_SHININESS)
        * SPECULAR_STRENGTH;

    sphere.color * (AMBIENT_STRENGTH + diffuse) + Vec3::splat(specular)
}

fn ray_sphere_intersection(origin: Vec3, direction: Vec3, sphere: &Sphere) -> Option<f32> {
    let sphere_to_ray = origin - sphere.position;
    let half_b = sphere_to_ray.dot(direction);
    let c = sphere_to_ray.length_squared() - sphere.radius * sphere.radius;
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

    fn test_sphere() -> Sphere {
        Sphere {
            position: Vec3::new(0.0, 0.0, -3.0),
            radius: 1.0,
            color: Vec3::ONE,
        }
    }

    #[test]
    fn center_ray_hits_sphere() {
        let distance = ray_sphere_intersection(Vec3::ZERO, Vec3::NEG_Z, &test_sphere()).unwrap();

        assert!((distance - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ray_misses_sphere() {
        assert!(ray_sphere_intersection(Vec3::ZERO, Vec3::X, &test_sphere()).is_none());
    }

    #[test]
    fn morton_coordinates_follow_z_order() {
        let coordinates: Vec<_> = (0..8).map(morton_coordinates).collect();

        assert_eq!(
            coordinates,
            vec![
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (3, 0),
                (2, 1),
                (3, 1)
            ]
        );
    }

    #[test]
    fn render_writes_partial_tiles() {
        let scene = RenderScene {
            camera: Camera {
                position: Vec3::ZERO,
                right: Vec3::X,
                up: Vec3::Y,
                forward: Vec3::NEG_Z,
            },
            light_direction: Vec3::Z,
            spheres: vec![test_sphere()],
        };
        let pixels = render(9, 10, &scene);

        assert_eq!(pixels.len(), 2 * 2 * TILE_SIZE * TILE_SIZE);
        assert!(pixels.iter().any(|pixel| *pixel != PixelData::default()));
    }

    #[test]
    fn row_major_pixels_unshuffles_morton_tiles() {
        let tile_pixels: Vec<_> = (0..4)
            .flat_map(|tile_index| {
                (0..TILE_SIZE * TILE_SIZE).map(move |morton_index| {
                    PixelData::new((tile_index * 64 + morton_index) as u8, 0, 0)
                })
            })
            .collect();
        let pixels = row_major_pixels(9, 10, &tile_pixels);

        assert_eq!(&pixels[0..3], &[0, 0, 0]);
        assert_eq!(&pixels[3..6], &[1, 0, 0]);
        assert_eq!(&pixels[(8 * 3)..(9 * 3)], &[64, 0, 0]);
        assert_eq!(&pixels[(9 * 8 * 3)..(9 * 8 * 3 + 3)], &[128, 0, 0]);
        assert_eq!(&pixels[((9 * 9 + 8) * 3)..((9 * 9 + 9) * 3)], &[194, 0, 0]);
    }

    #[test]
    fn scene_file_parses_multiple_spheres() {
        let scene: Scene = toml::from_str(
            r#"
            [camera]
            position = [0.0, 0.0, 0.0]
            yaw = 0.0
            pitch = 0.0
            roll = 0.0

            [light]
            yaw = -25.0
            pitch = -35.0
            roll = 0.0

            [[objects]]
            position = [0.0, 0.0, -3.0]
            radius = 1.0
            color = [1.0, 0.0, 0.0]

            [[objects]]
            position = [2.0, 0.0, -4.0]
            radius = 0.5
            color = [0.0, 1.0, 0.0]
            "#,
        )
        .unwrap();

        assert_eq!(scene.objects.len(), 2);
    }
}
