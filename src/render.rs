use bvh::ray::Ray;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::{
    cli::ShadingMode,
    geometry::ray_mesh_intersection,
    image::{PixelData, TILE_SIZE, morton_coordinates},
    scene::RenderScene,
};

const AMBIENT_STRENGTH: f32 = 0.08;
const SPECULAR_STRENGTH: f32 = 0.35;
const SPECULAR_SHININESS: f32 = 32.0;
const HALF_FOV_TANGENT: f32 = 0.41421357;

pub(crate) fn render(width: u32, height: u32, scene: &RenderScene) -> Vec<PixelData> {
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

fn shade(origin: Point3<f32>, ray_direction: Vector3<f32>, scene: &RenderScene) -> Vector3<f32> {
    let ray = Ray::new(origin, ray_direction);
    let Some((distance, normal, barycentrics, sphere)) = scene
        .bvh
        .nearest_traverse_iterator(&ray, &scene.spheres)
        .filter_map(|sphere| {
            ray_mesh_intersection(
                origin,
                ray_direction,
                sphere.position,
                sphere.radius,
                &scene.geometry,
            )
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

fn reflect(vector: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    vector - normal * (2.0 * vector.dot(&normal))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        geometry::generate_sphere,
        image::row_major_pixels,
        scene::{Camera, DEFAULT_SCENE_PATH, Sphere, load_scene},
    };
    use bvh::bvh::Bvh;

    fn test_sphere() -> Sphere {
        Sphere::new(
            Point3::new(0.0, 0.0, -3.0),
            1.0,
            Vector3::new(1.0, 1.0, 1.0),
        )
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
    fn bvh_traversal_filters_spheres_by_ray_bounds() {
        let scene = render_scene(vec![
            test_sphere(),
            Sphere::new(
                Point3::new(100.0, 0.0, -3.0),
                1.0,
                Vector3::new(1.0, 1.0, 1.0),
            ),
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
    fn render_writes_partial_tiles() {
        let scene = render_scene(vec![test_sphere()]);
        let pixels = render(9, 10, &scene);

        assert_eq!(pixels.len(), 2 * 2 * TILE_SIZE * TILE_SIZE);
        assert!(pixels.iter().any(|pixel| *pixel != PixelData::default()));
    }

    fn assert_render_matches_gold(
        name: &str,
        width: u32,
        height: u32,
        pixels: &[PixelData],
        gold: &[u8],
    ) {
        const MINIMUM_SIMILARITY: f64 = 0.999;

        let rendered =
            image::RgbImage::from_raw(width, height, row_major_pixels(width, height, pixels))
                .expect("rendered image dimensions should match its pixel buffer");
        let expected = image::load_from_memory(gold)
            .expect("gold image should be a valid image")
            .into_rgb8();
        let similarity = image_compare::rgb_hybrid_compare(&rendered, &expected)
            .expect("rendered and gold images should have identical dimensions");

        assert!(
            similarity.score >= MINIMUM_SIMILARITY,
            "rendered image {name} differs from its gold image (similarity: {:.6}, expected at least {MINIMUM_SIMILARITY:.6})",
            similarity.score,
        );
    }

    #[test]
    fn render_matches_gold_image() {
        let scene_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_SCENE_PATH);
        let scene = load_scene(&scene_path, ShadingMode::Barycentrics).unwrap();
        let pixels = render(32, 32, &scene);

        assert_render_matches_gold(
            "default_scene_barycentrics",
            32,
            32,
            &pixels,
            include_bytes!("../tests/gold/default_scene_barycentrics.png"),
        );
    }
}
