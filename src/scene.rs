use std::{collections::HashMap, error::Error, io, path::Path, sync::Arc};

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
    bvh::Bvh,
};
use nalgebra::{Point3, UnitQuaternion, Vector2, Vector3};
use serde::Deserialize;

use crate::{
    cli::ShadingMode,
    geometry::{SphereGeometry, generate_sphere},
    image::Texture,
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
    materials: HashMap<String, MaterialDescription>,
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
struct MaterialDescription {
    albedo: MaterialPropertyDescription<[f32; 3]>,
    #[serde(default = "default_uv_scale")]
    uv_scale: [f32; 2],
    #[serde(default = "default_roughness")]
    roughness: MaterialPropertyDescription<f32>,
    #[serde(default = "default_metalness")]
    metalness: MaterialPropertyDescription<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MaterialPropertyDescription<T> {
    Constant(T),
    Texture { texture: std::path::PathBuf },
    NamedConstant { constant: T },
}

fn default_uv_scale() -> [f32; 2] {
    [1.0, 1.0]
}

fn default_roughness() -> MaterialPropertyDescription<f32> {
    MaterialPropertyDescription::Constant(0.5)
}

fn default_metalness() -> MaterialPropertyDescription<f32> {
    MaterialPropertyDescription::Constant(0.0)
}

#[derive(Debug, Deserialize)]
struct SphereDescription {
    position: [f32; 3],
    radius: f32,
    #[serde(default)]
    material: Option<String>,
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
    pub(crate) material: Arc<Material>,
    node_index: usize,
}

impl Sphere {
    pub(crate) fn new(position: Point3<f32>, radius: f32, material: Arc<Material>) -> Self {
        Self {
            position,
            radius,
            material,
            node_index: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Material {
    pub(crate) albedo: MaterialProperty<Vector3<f32>>,
    pub(crate) uv_scale: Vector2<f32>,
    pub(crate) roughness: MaterialProperty<f32>,
    pub(crate) metalness: MaterialProperty<f32>,
}

#[derive(Debug)]
pub(crate) enum MaterialProperty<T> {
    Constant(T),
    Texture(Arc<Texture>),
}

pub(crate) trait TextureSample: Copy {
    fn from_texture_sample(sample: Vector3<f32>) -> Self;
}

impl TextureSample for Vector3<f32> {
    fn from_texture_sample(sample: Vector3<f32>) -> Self {
        sample
    }
}

impl TextureSample for f32 {
    fn from_texture_sample(sample: Vector3<f32>) -> Self {
        sample.x
    }
}

impl<T: TextureSample> MaterialProperty<T> {
    pub(crate) fn sample(&self, uv: Vector2<f32>) -> T {
        match self {
            Self::Constant(value) => *value,
            Self::Texture(texture) => T::from_texture_sample(texture.sample(uv)),
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
    let materials = load_materials(description.materials, path)?;
    let mut spheres = Vec::with_capacity(description.objects.len());
    for sphere in description.objects {
        if !sphere.radius.is_finite() || sphere.radius <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sphere radius must be finite and greater than zero",
            )
            .into());
        }
        let material = select_material(sphere.material.as_deref(), &materials)?;
        spheres.push(Sphere::new(
            point3_from_array(sphere.position),
            sphere.radius,
            material,
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

fn load_materials(
    descriptions: HashMap<String, MaterialDescription>,
    scene_path: &Path,
) -> Result<HashMap<String, Arc<Material>>, Box<dyn Error>> {
    descriptions
        .into_iter()
        .map(|(name, material)| {
            let albedo = match material.albedo {
                MaterialPropertyDescription::Constant(value)
                | MaterialPropertyDescription::NamedConstant { constant: value } => value,
                MaterialPropertyDescription::Texture { .. } => [0.0; 3],
            };
            if albedo.iter().any(|component| !component.is_finite()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("material {name:?} albedo values must be finite"),
                )
                .into());
            }
            if material
                .uv_scale
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "material {name:?} uv scale values must be finite and greater than zero"
                    ),
                )
                .into());
            }
            validate_scalar_property(&material.roughness, &name, "roughness")?;
            validate_scalar_property(&material.metalness, &name, "metalness")?;

            Ok((
                name,
                Arc::new(Material {
                    albedo: load_property(material.albedo, scene_path).map(|property| {
                        match property {
                            MaterialProperty::Constant(value) => {
                                MaterialProperty::Constant(vector3_from_array(value))
                            }
                            MaterialProperty::Texture(texture) => {
                                MaterialProperty::Texture(texture)
                            }
                        }
                    })?,
                    uv_scale: Vector2::from(material.uv_scale),
                    roughness: load_property(material.roughness, scene_path)?,
                    metalness: load_property(material.metalness, scene_path)?,
                }),
            ))
        })
        .collect()
}

fn validate_scalar_property(
    property: &MaterialPropertyDescription<f32>,
    name: &str,
    property_name: &str,
) -> Result<(), Box<dyn Error>> {
    let value = match property {
        MaterialPropertyDescription::Constant(value)
        | MaterialPropertyDescription::NamedConstant { constant: value } => Some(*value),
        MaterialPropertyDescription::Texture { .. } => None,
    };
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("material {name:?} {property_name} must be finite and between zero and one"),
        )
        .into());
    }
    Ok(())
}

fn load_property<T>(
    property: MaterialPropertyDescription<T>,
    scene_path: &Path,
) -> Result<MaterialProperty<T>, Box<dyn Error>> {
    match property {
        MaterialPropertyDescription::Constant(value)
        | MaterialPropertyDescription::NamedConstant { constant: value } => {
            Ok(MaterialProperty::Constant(value))
        }
        MaterialPropertyDescription::Texture { texture } => {
            let resolved_path = if texture.is_absolute() {
                texture
            } else {
                scene_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(texture)
            };
            Ok(MaterialProperty::Texture(Arc::new(Texture::load(
                &resolved_path,
            )?)))
        }
    }
}

fn select_material(
    requested_name: Option<&str>,
    materials: &HashMap<String, Arc<Material>>,
) -> Result<Arc<Material>, Box<dyn Error>> {
    let name = if let Some(name) = requested_name {
        name
    } else if materials.contains_key("default") {
        "default"
    } else if materials.len() == 1 {
        materials.keys().next().unwrap().as_str()
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "object must specify a material when the scene has no unique default material",
        )
        .into());
    };
    materials.get(name).cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("object references unknown material {name:?}"),
        )
        .into()
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

            [materials.red]
            albedo = [1.0, 0.0, 0.0]

            [materials.green]
            albedo = [0.0, 1.0, 0.0]

            [[objects]]
            position = [0.0, 0.0, -3.0]
            radius = 1.0
            material = "red"

            [[objects]]
            position = [2.0, 0.0, -4.0]
            radius = 0.5
            material = "green"
            "#,
        )
        .unwrap();

        assert_eq!(scene.objects.len(), 2);
        assert_eq!(scene.geometry.latitude_segments, 4);
        assert_eq!(scene.geometry.longitude_segments, 8);
    }

    #[test]
    fn materials_parse_constant_and_texture_properties() {
        let scene: SceneDescription = toml::from_str(
            r#"
            [camera]
            position = [0.0, 0.0, 0.0]

            [light]

            [materials.first]
            albedo = { constant = [1.0, 1.0, 1.0] }
            uv_scale = [2.0, 3.0]
            roughness = { texture = "textures/roughness.png" }

            [materials.second]
            albedo = [0.5, 0.5, 0.5]
            metalness = { texture = "textures/metalness.png" }

            [[objects]]
            position = [0.0, 0.0, -3.0]
            radius = 1.0
            material = "first"

            [[objects]]
            position = [2.0, 0.0, -3.0]
            radius = 1.0
            material = "second"
            "#,
        )
        .unwrap();

        assert!(matches!(
            &scene.materials["first"].albedo,
            MaterialPropertyDescription::NamedConstant { .. }
        ));
        assert!(matches!(
            &scene.materials["first"].roughness,
            MaterialPropertyDescription::Texture { texture }
                if texture == Path::new("textures/roughness.png")
        ));
        assert!(matches!(
            &scene.materials["second"].metalness,
            MaterialPropertyDescription::Texture { texture }
                if texture == Path::new("textures/metalness.png")
        ));
        assert_eq!(scene.materials["first"].uv_scale, [2.0, 3.0]);
        assert_eq!(scene.objects[0].material.as_deref(), Some("first"));
    }

    #[test]
    fn materials_parse_pbr_properties_and_default_them() {
        let scene: SceneDescription = toml::from_str(
            r#"
            [camera]
            position = [0.0, 0.0, 0.0]

            [light]

            [materials.metal]
            albedo = [0.8, 0.6, 0.2]
            roughness = 0.25
            metalness = 0.75

            [materials.plastic]
            albedo = [0.2, 0.4, 0.8]

            [[objects]]
            position = [0.0, 0.0, -3.0]
            radius = 1.0
            material = "metal"
            "#,
        )
        .unwrap();

        assert!(matches!(
            &scene.materials["metal"].roughness,
            MaterialPropertyDescription::Constant(0.25)
        ));
        assert!(matches!(
            &scene.materials["metal"].metalness,
            MaterialPropertyDescription::Constant(0.75)
        ));
        assert!(matches!(
            &scene.materials["plastic"].roughness,
            MaterialPropertyDescription::Constant(0.5)
        ));
        assert!(matches!(
            &scene.materials["plastic"].metalness,
            MaterialPropertyDescription::Constant(0.0)
        ));
    }

    #[test]
    fn scene_loads_texture_and_constant_properties() {
        let scene_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_SCENE_PATH);
        let scene = load_scene(&scene_path, ShadingMode::Barycentrics, None).unwrap();

        assert!(matches!(
            &scene.spheres[0].material.albedo,
            MaterialProperty::Texture(_)
        ));
        assert!(matches!(
            &scene.spheres[1].material.albedo,
            MaterialProperty::Constant(_)
        ));
        assert!(matches!(
            &scene.spheres[2].material.albedo,
            MaterialProperty::Constant(_)
        ));
        assert!(matches!(
            &scene.spheres[0].material.roughness,
            MaterialProperty::Texture(_)
        ));
        assert!(matches!(
            &scene.spheres[0].material.metalness,
            MaterialProperty::Texture(_)
        ));
        assert!(matches!(
            &scene.spheres[1].material.roughness,
            MaterialProperty::Constant(_)
        ));
        assert!(matches!(
            &scene.spheres[1].material.metalness,
            MaterialProperty::Constant(_)
        ));

        let uv = Vector2::zeros();
        assert_eq!(
            scene.spheres[0].material.albedo.sample(uv),
            Vector3::new(220.0, 40.0, 40.0) / 255.0
        );
        assert_eq!(scene.spheres[0].material.roughness.sample(uv), 64.0 / 255.0);
        assert_eq!(scene.spheres[0].material.metalness.sample(uv), 0.0);
        assert_eq!(
            scene.spheres[1].material.albedo.sample(uv),
            Vector3::new(0.0, 0.7, 1.0)
        );
        assert_eq!(scene.spheres[1].material.roughness.sample(uv), 0.5);
        assert_eq!(scene.spheres[1].material.metalness.sample(uv), 0.0);
    }

    #[test]
    fn material_property_samples_rgb_and_scalar_texture_values() {
        let texture = Arc::new(Texture::from_pixels(
            1,
            1,
            vec![Vector3::new(0.25, 0.5, 0.75)],
        ));

        let albedo: MaterialProperty<Vector3<f32>> = MaterialProperty::Texture(texture.clone());
        let roughness: MaterialProperty<f32> = MaterialProperty::Texture(texture.clone());
        let constant = MaterialProperty::Constant(0.75);

        assert_eq!(
            albedo.sample(Vector2::zeros()),
            Vector3::new(0.25, 0.5, 0.75)
        );
        assert_eq!(roughness.sample(Vector2::zeros()), 0.25);
        assert_eq!(constant.sample(Vector2::zeros()), 0.75);
    }

    #[test]
    fn objects_without_material_use_the_default_material() {
        let scene: SceneDescription = toml::from_str(
            r#"
            [camera]
            position = [0.0, 0.0, 0.0]

            [light]

            [materials.default]
            albedo = [1.0, 0.0, 0.0]

            [[objects]]
            position = [0.0, 0.0, -3.0]
            radius = 1.0
            "#,
        )
        .unwrap();

        let materials = load_materials(scene.materials, Path::new("scene.toml")).unwrap();
        let selected = select_material(scene.objects[0].material.as_deref(), &materials).unwrap();
        assert_eq!(
            selected.albedo.sample(Vector2::zeros()),
            Vector3::new(1.0, 0.0, 0.0)
        );
    }
}
