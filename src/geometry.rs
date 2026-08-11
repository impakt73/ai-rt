use std::io;

use nalgebra::{Point3, Vector2, Vector3};

#[derive(Debug)]
pub(crate) struct Triangle {
    pub(crate) vertices: [Point3<f32>; 3],
    pub(crate) vertex_normals: [Vector3<f32>; 3],
    pub(crate) uvs: [Vector2<f32>; 3],
}

#[derive(Debug)]
pub(crate) struct SphereGeometry {
    pub(crate) triangles: Vec<Triangle>,
}

type MeshIntersection = (f32, Vector3<f32>, Vector3<f32>, Vector2<f32>);

pub(crate) fn generate_sphere(
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
        (
            Point3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ),
            Vector2::new(
                longitude as f32 / longitude_segments as f32,
                latitude as f32 / latitude_segments as f32,
            ),
        )
    };
    let triangle = |a: (Point3<f32>, Vector2<f32>),
                    b: (Point3<f32>, Vector2<f32>),
                    c: (Point3<f32>, Vector2<f32>)| {
        let mut vertices = [a.0, b.0, c.0];
        let mut uvs = [a.1, b.1, c.1];
        let mut vertex_normals = vertices.map(|vertex| vertex.coords.normalize());
        let normal = (vertices[1] - vertices[0])
            .cross(&(vertices[2] - vertices[0]))
            .normalize();
        if normal.dot(&(vertices[0].coords + vertices[1].coords + vertices[2].coords)) < 0.0 {
            vertices.swap(1, 2);
            vertex_normals.swap(1, 2);
            uvs.swap(1, 2);
        }
        Triangle {
            vertices,
            vertex_normals,
            uvs,
        }
    };

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
        // Keep the final sector's next longitude at one full turn so its UVs
        // interpolate toward u=1 instead of wrapping back to u=0.
        let next_longitude = longitude + 1;
        triangles.push(triangle(
            point_on_sphere(0, longitude),
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
            point_on_sphere(latitude_segments, longitude),
            point_on_sphere(latitude_segments - 1, next_longitude),
        ));
    }

    Ok(SphereGeometry { triangles })
}

pub(crate) fn ray_mesh_intersection(
    origin: Point3<f32>,
    direction: Vector3<f32>,
    sphere_position: Point3<f32>,
    sphere_radius: f32,
    geometry: &SphereGeometry,
) -> Option<MeshIntersection> {
    let local_origin = Point3::from((origin - sphere_position) / sphere_radius);
    let local_direction = direction / sphere_radius;

    geometry
        .triangles
        .iter()
        .filter_map(|triangle| {
            ray_triangle_intersection(local_origin, local_direction, triangle).map(
                |(distance, barycentrics)| {
                    let normal = (triangle.vertex_normals[0] * barycentrics.x
                        + triangle.vertex_normals[1] * barycentrics.y
                        + triangle.vertex_normals[2] * barycentrics.z)
                        .normalize();
                    let uv = triangle.uvs[0] * barycentrics.x
                        + triangle.uvs[1] * barycentrics.y
                        + triangle.uvs[2] * barycentrics.z;
                    (distance, normal, barycentrics, uv)
                },
            )
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
}

pub(crate) fn ray_triangle_intersection(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_ray_hits_triangle_sphere() {
        let geometry = generate_sphere(16, 32).unwrap();
        let distance = ray_mesh_intersection(
            Point3::origin(),
            Vector3::new(0.0, 0.0, -1.0),
            Point3::new(0.0, 0.0, -3.0),
            1.0,
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
                Point3::new(0.0, 0.0, -3.0),
                1.0,
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
    fn sphere_triangles_have_smooth_vertex_normals() {
        let geometry = generate_sphere(4, 8).unwrap();
        let triangle = &geometry.triangles[1];

        assert_ne!(triangle.vertex_normals[0], triangle.vertex_normals[1]);
        for (vertex, normal) in triangle.vertices.iter().zip(triangle.vertex_normals) {
            assert!((vertex.coords.normalize() - normal).norm() < f32::EPSILON);
        }
    }

    #[test]
    fn ray_triangle_intersection_hits_front_face() {
        let triangle = Triangle {
            vertices: [
                Point3::new(-1.0, -1.0, -2.0),
                Point3::new(1.0, -1.0, -2.0),
                Point3::new(0.0, 1.0, -2.0),
            ],
            vertex_normals: [Vector3::new(0.0, 0.0, 1.0); 3],
            uvs: [Vector2::zeros(); 3],
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
            vertex_normals: [Vector3::new(0.0, 0.0, 1.0); 3],
            uvs: [
                Vector2::new(0.0, 0.0),
                Vector2::new(1.0, 0.0),
                Vector2::new(0.0, 1.0),
            ],
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
    fn mesh_intersection_interpolates_uvs_with_barycentrics() {
        let triangle = Triangle {
            vertices: [
                Point3::new(-1.0, -1.0, -2.0),
                Point3::new(1.0, -1.0, -2.0),
                Point3::new(0.0, 1.0, -2.0),
            ],
            vertex_normals: [Vector3::new(0.0, 0.0, 1.0); 3],
            uvs: [
                Vector2::new(0.0, 0.0),
                Vector2::new(1.0, 0.0),
                Vector2::new(0.0, 1.0),
            ],
        };
        let geometry = SphereGeometry {
            triangles: vec![triangle],
        };

        let (_, _, barycentrics, uv) = ray_mesh_intersection(
            Point3::origin(),
            Vector3::new(0.0, 0.0, -1.0),
            Point3::origin(),
            1.0,
            &geometry,
        )
        .unwrap();

        assert!((barycentrics - Vector3::new(0.25, 0.25, 0.5)).norm() < f32::EPSILON);
        assert!((uv - Vector2::new(0.25, 0.5)).norm() < f32::EPSILON);
    }

    #[test]
    fn sphere_uvs_cover_the_longitude_seam_without_wrapping() {
        let geometry = generate_sphere(2, 4).unwrap();
        let seam_triangle = &geometry.triangles[6];

        assert!(
            seam_triangle
                .uvs
                .iter()
                .all(|uv| (0.0..=1.0).contains(&uv.x))
        );
        assert!(
            seam_triangle
                .uvs
                .iter()
                .any(|uv| (uv.x - 1.0).abs() < f32::EPSILON)
        );
    }
}
