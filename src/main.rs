use std::{error::Error, fs::File, io, path::PathBuf, sync::Arc};

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
    bvh::Bvh,
    ray::Ray,
};
use clap::{Parser, ValueEnum};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use rayon::prelude::*;
use serde::Deserialize;

const DEFAULT_SCENE_PATH: &str = "scene.toml";
const AMBIENT_STRENGTH: f32 = 0.08;
const SPECULAR_STRENGTH: f32 = 0.35;
const SPECULAR_SHININESS: f32 = 32.0;
const HALF_FOV_TANGENT: f32 = 0.41421357;
const TILE_SIZE: usize = 8;

#[derive(Debug, Parser)]
#[command(author, version, about = "Render a sphere scene")]
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

    /// Shading mode used for visible triangle hits.
    #[arg(long, value_enum, default_value_t = ShadingMode::Barycentrics)]
    shading_mode: ShadingMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ShadingMode {
    Barycentrics,
    Phong,
}

#[derive(Debug, Deserialize)]
struct Scene {
    camera: CameraDescription,
    light: LightDescription,
    #[serde(default)]
    geometry: GeometryDescription,
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
struct GeometryDescription {
    #[serde(default = "default_latitude_segments")]
    latitude_segments: usize,
    #[serde(default = "default_longitude_segments")]
    longitude_segments: usize,
}

impl Default for GeometryDescription {
    fn default() -> Self {
        Self {
            latitude_segments: default_latitude_segments(),
            longitude_segments: default_longitude_segments(),
        }
    }
}

fn default_latitude_segments() -> usize {
    32
}

fn default_longitude_segments() -> usize {
    64
}

#[derive(Debug, Deserialize)]
struct SphereDescription {
    position: [f32; 3],
    radius: f32,
    color: [f32; 3],
}

#[derive(Debug)]
struct Triangle {
    vertices: [Point3<f32>; 3],
    normal: Vector3<f32>,
}

#[derive(Debug)]
struct SphereGeometry {
    triangles: Vec<Triangle>,
}

#[derive(Debug)]
struct Camera {
    position: Point3<f32>,
    right: Vector3<f32>,
    up: Vector3<f32>,
    forward: Vector3<f32>,
}

#[derive(Debug)]
struct Sphere {
    position: Point3<f32>,
    radius: f32,
    color: Vector3<f32>,
    node_index: usize,
}

impl Bounded<f32, 3> for Sphere {
    fn aabb(&self) -> Aabb<f32, 3> {
        let radius = Vector3::repeat(self.radius);
        Aabb::with_bounds(self.position - radius, self.position + radius)
    }
}

impl BHShape<f32, 3> for Sphere {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[derive(Debug)]
struct RenderScene {
    camera: Camera,
    light_direction: Vector3<f32>,
    geometry: Arc<SphereGeometry>,
    spheres: Vec<Sphere>,
    bvh: Bvh<f32, 3>,
    shading_mode: ShadingMode,
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

    fn from_color(color: Vector3<f32>) -> Self {
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
    let scene = load_scene(&scene_path, args.shading_mode)?;
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

fn load_scene(path: &PathBuf, shading_mode: ShadingMode) -> Result<RenderScene, Box<dyn Error>> {
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
        position: point3_from_array(description.camera.position),
        right: camera_rotation * Vector3::new(1.0, 0.0, 0.0),
        up: camera_rotation * Vector3::new(0.0, 1.0, 0.0),
        forward: camera_rotation * Vector3::new(0.0, 0.0, -1.0),
    };
    let geometry = Arc::new(generate_sphere(
        description.geometry.latitude_segments,
        description.geometry.longitude_segments,
    )?);
    let mut spheres = Vec::with_capacity(description.objects.len());
    for sphere in description.objects {
        if !sphere.radius.is_finite() || sphere.radius <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sphere radius must be finite and greater than zero",
            )
            .into());
        }
        spheres.push(Sphere {
            position: point3_from_array(sphere.position),
            radius: sphere.radius,
            color: vector3_from_array(sphere.color),
            node_index: 0,
        });
    }
    let bvh = Bvh::build(&mut spheres);

    Ok(RenderScene {
        camera,
        light_direction: (light_rotation * Vector3::new(0.0, 0.0, 1.0)).normalize(),
        geometry,
        spheres,
        bvh,
        shading_mode,
    })
}

fn generate_sphere(
    latitude_segments: usize,
    longitude_segments: usize,
) -> Result<SphereGeometry, io::Error> {
    if latitude_segments < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sphere latitude segments must be at least 2",
        ));
    }
    if longitude_segments < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sphere longitude segments must be at least 3",
        ));
    }

    let point_on_sphere = |latitude: usize, longitude: usize| {
        let theta = std::f32::consts::PI * latitude as f32 / latitude_segments as f32;
        let phi = 2.0 * std::f32::consts::PI * longitude as f32 / longitude_segments as f32;
        Point3::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        )
    };
    let triangle = |a: Point3<f32>, b: Point3<f32>, c: Point3<f32>| {
        let mut vertices = [a, b, c];
        let mut normal = (vertices[1] - vertices[0])
            .cross(&(vertices[2] - vertices[0]))
            .normalize();
        if normal.dot(&(vertices[0].coords + vertices[1].coords + vertices[2].coords)) < 0.0 {
            vertices.swap(1, 2);
            normal = -normal;
        }
        Triangle { vertices, normal }
    };

    let north_pole = Point3::new(0.0, 1.0, 0.0);
    let south_pole = Point3::new(0.0, -1.0, 0.0);
    let triangle_count = longitude_segments
        .checked_mul(latitude_segments - 1)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "sphere segment counts are too large",
            )
        })?;
    let mut triangles = Vec::with_capacity(triangle_count);

    for longitude in 0..longitude_segments {
        let next_longitude = (longitude + 1) % longitude_segments;
        triangles.push(triangle(
            north_pole,
            point_on_sphere(1, longitude),
            point_on_sphere(1, next_longitude),
        ));

        for latitude in 1..latitude_segments - 1 {
            let current = point_on_sphere(latitude, longitude);
            let next = point_on_sphere(latitude, next_longitude);
            let below = point_on_sphere(latitude + 1, longitude);
            let below_next = point_on_sphere(latitude + 1, next_longitude);
            triangles.push(triangle(current, below, next));
            triangles.push(triangle(next, below, below_next));
        }

        triangles.push(triangle(
            point_on_sphere(latitude_segments - 1, longitude),
            south_pole,
            point_on_sphere(latitude_segments - 1, next_longitude),
        ));
    }

    Ok(SphereGeometry { triangles })
}

fn rotation_from_degrees(yaw: f32, pitch: f32, roll: f32) -> UnitQuaternion<f32> {
    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw.to_radians())
        * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch.to_radians())
        * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), roll.to_radians())
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

fn shade(origin: Point3<f32>, ray_direction: Vector3<f32>, scene: &RenderScene) -> Vector3<f32> {
    let ray = Ray::new(origin, ray_direction);
    let Some((distance, normal, barycentrics, sphere)) = scene
        .bvh
        .nearest_traverse_iterator(&ray, &scene.spheres)
        .filter_map(|sphere| {
            ray_mesh_intersection(origin, ray_direction, sphere, &scene.geometry)
                .map(|(distance, normal, barycentrics)| (distance, normal, barycentrics, sphere))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
    else {
        return Vector3::zeros();
    };

    if scene.shading_mode == ShadingMode::Barycentrics {
        return barycentrics;
    }

    let hit_point = origin + ray_direction * distance;
    let diffuse = normal.dot(&scene.light_direction).max(0.0);
    let view_direction = (origin - hit_point).normalize();
    let reflected_direction = reflect(-scene.light_direction, normal);
    let specular = reflected_direction
        .dot(&view_direction)
        .max(0.0)
        .powf(SPECULAR_SHININESS)
        * SPECULAR_STRENGTH;

    sphere.color * (AMBIENT_STRENGTH + diffuse) + Vector3::repeat(specular)
}

fn ray_mesh_intersection(
    origin: Point3<f32>,
    direction: Vector3<f32>,
    sphere: &Sphere,
    geometry: &SphereGeometry,
) -> Option<(f32, Vector3<f32>, Vector3<f32>)> {
    let local_origin = Point3::from((origin - sphere.position) / sphere.radius);
    let local_direction = direction / sphere.radius;

    geometry
        .triangles
        .iter()
        .filter_map(|triangle| {
            ray_triangle_intersection(local_origin, local_direction, triangle)
                .map(|(distance, barycentrics)| (distance, triangle.normal, barycentrics))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
}

fn ray_triangle_intersection(
    origin: Point3<f32>,
    direction: Vector3<f32>,
    triangle: &Triangle,
) -> Option<(f32, Vector3<f32>)> {
    const PARALLEL_EPSILON: f32 = 1.0e-7;

    let edge1 = triangle.vertices[1] - triangle.vertices[0];
    let edge2 = triangle.vertices[2] - triangle.vertices[0];
    let pvec = direction.cross(&edge2);
    let determinant = edge1.dot(&pvec);
    if determinant.abs() < PARALLEL_EPSILON {
        return None;
    }

    let inverse_determinant = determinant.recip();
    let tvec = origin - triangle.vertices[0];
    let u = tvec.dot(&pvec) * inverse_determinant;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let qvec = tvec.cross(&edge1);
    let v = direction.dot(&qvec) * inverse_determinant;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let distance = edge2.dot(&qvec) * inverse_determinant;
    (distance >= 0.0).then_some((distance, Vector3::new(1.0 - u - v, u, v)))
}

fn point3_from_array(value: [f32; 3]) -> Point3<f32> {
    Point3::from(value)
}

fn vector3_from_array(value: [f32; 3]) -> Vector3<f32> {
    Vector3::new(value[0], value[1], value[2])
}

fn reflect(vector: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    vector - normal * (2.0 * vector.dot(&normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sphere() -> Sphere {
        Sphere {
            position: Point3::new(0.0, 0.0, -3.0),
            radius: 1.0,
            color: Vector3::new(1.0, 1.0, 1.0),
            node_index: 0,
        }
    }

    fn render_scene(spheres: Vec<Sphere>) -> RenderScene {
        let mut spheres = spheres;
        let bvh = Bvh::build(&mut spheres);
        RenderScene {
            camera: Camera {
                position: Point3::origin(),
                right: Vector3::new(1.0, 0.0, 0.0),
                up: Vector3::new(0.0, 1.0, 0.0),
                forward: Vector3::new(0.0, 0.0, -1.0),
            },
            light_direction: Vector3::new(0.0, 0.0, 1.0),
            geometry: Arc::new(generate_sphere(16, 32).unwrap()),
            spheres,
            bvh,
            shading_mode: ShadingMode::Barycentrics,
        }
    }

    #[test]
    fn center_ray_hits_triangle_sphere() {
        let geometry = generate_sphere(16, 32).unwrap();
        let distance = ray_mesh_intersection(
            Point3::origin(),
            Vector3::new(0.0, 0.0, -1.0),
            &test_sphere(),
            &geometry,
        )
        .unwrap();

        assert!((distance.0 - 2.0).abs() < 0.02);
    }

    #[test]
    fn ray_misses_triangle_sphere() {
        let geometry = generate_sphere(16, 32).unwrap();
        assert!(
            ray_mesh_intersection(
                Point3::origin(),
                Vector3::new(1.0, 0.0, 0.0),
                &test_sphere(),
                &geometry,
            )
            .is_none()
        );
    }

    #[test]
    fn sphere_generation_has_configurable_triangle_density() {
        let coarse = generate_sphere(4, 8).unwrap();
        let detailed = generate_sphere(8, 16).unwrap();

        assert_eq!(coarse.triangles.len(), 2 * 8 * (4 - 1));
        assert_eq!(detailed.triangles.len(), 2 * 16 * (8 - 1));
    }

    #[test]
    fn ray_triangle_intersection_hits_front_face() {
        let triangle = Triangle {
            vertices: [
                Point3::new(-1.0, -1.0, -2.0),
                Point3::new(1.0, -1.0, -2.0),
                Point3::new(0.0, 1.0, -2.0),
            ],
            normal: Vector3::new(0.0, 0.0, 1.0),
        };

        assert_eq!(
            ray_triangle_intersection(Point3::origin(), Vector3::new(0.0, 0.0, -1.0), &triangle,)
                .map(|(distance, _)| distance),
            Some(2.0)
        );
    }

    #[test]
    fn ray_triangle_intersection_returns_barycentrics() {
        let triangle = Triangle {
            vertices: [
                Point3::new(-1.0, -1.0, -2.0),
                Point3::new(1.0, -1.0, -2.0),
                Point3::new(0.0, 1.0, -2.0),
            ],
            normal: Vector3::new(0.0, 0.0, 1.0),
        };

        let (_, barycentrics) = ray_triangle_intersection(
            Point3::new(0.0, -1.0 / 3.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            &triangle,
        )
        .unwrap();

        assert!((barycentrics.x - 1.0 / 3.0).abs() < f32::EPSILON);
        assert!((barycentrics.y - 1.0 / 3.0).abs() < f32::EPSILON);
        assert!((barycentrics.z - 1.0 / 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn bvh_traversal_filters_spheres_by_ray_bounds() {
        let scene = render_scene(vec![
            test_sphere(),
            Sphere {
                position: Point3::new(100.0, 0.0, -3.0),
                radius: 1.0,
                color: Vector3::new(1.0, 1.0, 1.0),
                node_index: 0,
            },
        ]);
        let ray = Ray::new(Point3::origin(), Vector3::new(0.0, 0.0, -1.0));

        let candidates: Vec<_> = scene
            .bvh
            .nearest_traverse_iterator(&ray, &scene.spheres)
            .collect();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].position, Point3::new(0.0, 0.0, -3.0));
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
        let scene = render_scene(vec![test_sphere()]);
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

            [geometry]
            latitude_segments = 4
            longitude_segments = 8

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
        assert_eq!(scene.geometry.latitude_segments, 4);
        assert_eq!(scene.geometry.longitude_segments, 8);
    }
}
