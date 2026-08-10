use std::{error::Error, io, path::Path, sync::Arc};

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
    bvh::Bvh,
};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use serde::Deserialize;

use crate::{
    cli::ShadingMode,
    geometry::{SphereGeometry, generate_sphere},
    mlp::LoadedMlpShader,
};

pub(crate) const DEFAULT_SCENE_PATH: &str = "scene.toml";

#[derive(Debug, Deserialize)]
struct SceneDescription {
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
pub(crate) struct Camera {
    pub(crate) position: Point3<f32>,
    pub(crate) right: Vector3<f32>,
    pub(crate) up: Vector3<f32>,
    pub(crate) forward: Vector3<f32>,
}

#[derive(Debug)]
pub(crate) struct Sphere {
    pub(crate) position: Point3<f32>,
    pub(crate) radius: f32,
    pub(crate) color: Vector3<f32>,
    node_index: usize,
}

impl Sphere {
    pub(crate) fn new(position: Point3<f32>, radius: f32, color: Vector3<f32>) -> Self {
        Self {
            position,
            radius,
            color,
            node_index: 0,
        }
    }
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

pub(crate) struct RenderScene {
    pub(crate) camera: Camera,
    pub(crate) light_direction: Vector3<f32>,
    pub(crate) geometry: Arc<SphereGeometry>,
    pub(crate) spheres: Vec<Sphere>,
    pub(crate) bvh: Bvh<f32, 3>,
    pub(crate) shading_mode: ShadingMode,
    pub(crate) mlp: Option<Arc<LoadedMlpShader>>,
}

pub(crate) fn load_scene(
    path: &Path,
    shading_mode: ShadingMode,
    shader_model: Option<&Path>,
) -> Result<RenderScene, Box<dyn Error>> {
    let contents = std::fs::read_to_string(path)?;
    let description: SceneDescription = toml::from_str(&contents)?;
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
        spheres.push(Sphere::new(
            point3_from_array(sphere.position),
            sphere.radius,
            vector3_from_array(sphere.color),
        ));
    }
    let bvh = Bvh::build(&mut spheres);
    let mlp = match shading_mode {
        ShadingMode::Mlp => Some(Arc::new(LoadedMlpShader::load(shader_model.ok_or_else(
            || {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "MLP shading requires --shader-model",
                )
            },
        )?)?)),
        _ => None,
    };

    Ok(RenderScene {
        camera,
        light_direction: (light_rotation * Vector3::new(0.0, 0.0, 1.0)).normalize(),
        geometry,
        spheres,
        bvh,
        shading_mode,
        mlp,
    })
}

fn rotation_from_degrees(yaw: f32, pitch: f32, roll: f32) -> UnitQuaternion<f32> {
    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw.to_radians())
        * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch.to_radians())
        * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), roll.to_radians())
}

fn point3_from_array(value: [f32; 3]) -> Point3<f32> {
    Point3::from(value)
}

fn vector3_from_array(value: [f32; 3]) -> Vector3<f32> {
    Vector3::new(value[0], value[1], value[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_file_parses_multiple_spheres() {
        let scene: SceneDescription = toml::from_str(
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
