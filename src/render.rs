use bvh::ray::Ray;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::{
    cli::ShadingMode,
    geometry::ray_mesh_intersection,
    image::{PixelData, TILE_SIZE, morton_coordinates},
    scene::{Material, RenderScene},
    shader::{PbrInput, ShaderInput, pbr_color, phong_color},
};

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

                if scene.shading_mode == ShadingMode::Mlp {
                    continue;
                }
                *tile_pixel = render_pixel(x, y, width, height, aspect_ratio, scene);
            }

            if scene.shading_mode == ShadingMode::Mlp {
                render_mlp_tile(
                    tile_start_x,
                    tile_start_y,
                    width,
                    height,
                    aspect_ratio,
                    tile_pixels,
                    scene,
                );
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
    let ray_direction = ray_direction(x, y, width, height, aspect_ratio, scene);
    let color = shade(scene.camera.position, ray_direction, scene);

    PixelData::from_color(color)
}

fn shade(origin: Point3<f32>, ray_direction: Vector3<f32>, scene: &RenderScene) -> Vector3<f32> {
    let Some(hit) = trace_hit(origin, ray_direction, scene) else {
        return Vector3::zeros();
    };

    shade_hit(origin, ray_direction, hit, scene)
}

#[derive(Clone)]
struct Hit {
    distance: f32,
    normal: Vector3<f32>,
    barycentrics: Vector3<f32>,
    uv: nalgebra::Vector2<f32>,
    material: std::sync::Arc<Material>,
}

fn trace_hit(origin: Point3<f32>, ray_direction: Vector3<f32>, scene: &RenderScene) -> Option<Hit> {
    let ray = Ray::new(origin, ray_direction);
    scene
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
            .map(|(distance, normal, barycentrics, uv)| Hit {
                distance,
                normal,
                barycentrics,
                uv,
                material: sphere.material.clone(),
            })
        })
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

fn shade_hit(
    origin: Point3<f32>,
    ray_direction: Vector3<f32>,
    hit: Hit,
    scene: &RenderScene,
) -> Vector3<f32> {
    if scene.shading_mode == ShadingMode::Barycentrics {
        return hit.barycentrics;
    }

    let hit_point = origin + ray_direction * hit.distance;
    let view_direction = (origin - hit_point).normalize();
    let albedo = hit
        .material
        .texture
        .as_ref()
        .map_or(hit.material.albedo, |texture| {
            texture.sample(hit.uv.component_mul(&hit.material.uv_scale))
        });
    match scene.shading_mode {
        ShadingMode::Phong => phong_color(ShaderInput {
            normal: hit.normal,
            light_direction: scene.light_direction,
            view_direction,
            albedo,
        }),
        ShadingMode::Pbr => pbr_color(PbrInput {
            normal: hit.normal,
            light_direction: scene.light_direction,
            view_direction,
            albedo,
            roughness: hit.material.roughness,
            metalness: hit.material.metalness,
        }),
        ShadingMode::Barycentrics | ShadingMode::Mlp => {
            unreachable!("non-direct shading mode reached the direct shader")
        }
    }
}

fn ray_direction(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    aspect_ratio: f32,
    scene: &RenderScene,
) -> Vector3<f32> {
    let screen_x = ((x as f32 + 0.5) / width as f32 * 2.0 - 1.0) * aspect_ratio * HALF_FOV_TANGENT;
    let screen_y = (1.0 - (y as f32 + 0.5) / height as f32 * 2.0) * HALF_FOV_TANGENT;
    (scene.camera.forward + scene.camera.right * screen_x + scene.camera.up * screen_y).normalize()
}

fn render_mlp_tile(
    tile_start_x: usize,
    tile_start_y: usize,
    width: usize,
    height: usize,
    aspect_ratio: f32,
    tile_pixels: &mut [PixelData],
    scene: &RenderScene,
) {
    let mlp = scene
        .mlp
        .as_ref()
        .expect("MLP shading requires a loaded model");
    let mut features = Vec::new();
    let mut destinations = Vec::new();
    let origin = scene.camera.position;

    for (morton_index, tile_pixel) in tile_pixels.iter_mut().enumerate() {
        let (local_x, local_y) = morton_coordinates(morton_index);
        let x = tile_start_x + local_x;
        let y = tile_start_y + local_y;
        if x >= width || y >= height {
            continue;
        }

        let direction = ray_direction(x, y, width, height, aspect_ratio, scene);
        let Some(hit) = trace_hit(origin, direction, scene) else {
            *tile_pixel = PixelData::default();
            continue;
        };
        let hit_point = origin + direction * hit.distance;
        let input = ShaderInput {
            normal: hit.normal,
            light_direction: scene.light_direction,
            view_direction: (origin - hit_point).normalize(),
            albedo: hit.material.albedo,
        };
        features.extend(input.feature_row());
        destinations.push(morton_index);
    }

    if destinations.is_empty() {
        return;
    }

    let predictions = mlp.infer(&features, destinations.len());
    for (row, destination) in destinations.into_iter().enumerate() {
        let offset = row * 3;
        tile_pixels[destination] = PixelData::from_color(Vector3::new(
            predictions[offset],
            predictions[offset + 1],
            predictions[offset + 2],
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        geometry::generate_sphere,
        image::{Texture, row_major_pixels},
        scene::{Camera, DEFAULT_SCENE_PATH, Material, Sphere, load_scene},
        shader::{AMBIENT_STRENGTH, SPECULAR_STRENGTH},
    };
    use bvh::bvh::Bvh;

    fn test_sphere() -> Sphere {
        Sphere::new(
            Point3::new(0.0, 0.0, -3.0),
            1.0,
            Arc::new(Material {
                albedo: Vector3::new(1.0, 1.0, 1.0),
                texture: None,
                uv_scale: nalgebra::Vector2::repeat(1.0),
                roughness: 0.5,
                metalness: 0.0,
            }),
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
            mlp: None,
        }
    }

    #[test]
    fn bvh_traversal_filters_spheres_by_ray_bounds() {
        let scene = render_scene(vec![
            test_sphere(),
            Sphere::new(
                Point3::new(100.0, 0.0, -3.0),
                1.0,
                Arc::new(Material {
                    albedo: Vector3::new(1.0, 1.0, 1.0),
                    texture: None,
                    uv_scale: nalgebra::Vector2::repeat(1.0),
                    roughness: 0.5,
                    metalness: 0.0,
                }),
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

    #[test]
    fn phong_uses_the_interpolated_texture_color() {
        let mut scene = render_scene(vec![test_sphere()]);
        scene.shading_mode = ShadingMode::Phong;

        let color = shade_hit(
            Point3::origin(),
            Vector3::new(0.0, 0.0, -1.0),
            Hit {
                distance: 2.0,
                normal: Vector3::new(0.0, 0.0, 1.0),
                barycentrics: Vector3::repeat(1.0 / 3.0),
                uv: nalgebra::Vector2::new(0.5, 0.5),
                material: Arc::new(Material {
                    albedo: Vector3::new(1.0, 0.0, 0.0),
                    texture: Some(Arc::new(Texture::from_pixels(
                        1,
                        1,
                        vec![Vector3::new(0.0, 1.0, 0.0)],
                    ))),
                    uv_scale: nalgebra::Vector2::repeat(1.0),
                    roughness: 0.5,
                    metalness: 0.0,
                }),
            },
            &scene,
        );

        assert!((color.x - SPECULAR_STRENGTH).abs() < f32::EPSILON);
        assert!((color.y - (AMBIENT_STRENGTH + 1.0 + SPECULAR_STRENGTH)).abs() < f32::EPSILON);
        assert!((color.z - SPECULAR_STRENGTH).abs() < f32::EPSILON);
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
        let scene = load_scene(&scene_path, ShadingMode::Barycentrics, None).unwrap();
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
