use nalgebra::Vector3;

pub(crate) const AMBIENT_STRENGTH: f32 = 0.08;
pub(crate) const SPECULAR_STRENGTH: f32 = 0.35;
pub(crate) const SPECULAR_SHININESS: f32 = 32.0;
pub(crate) const DIELECTRIC_F0: f32 = 0.04;
const PI: f32 = std::f32::consts::PI;
pub(crate) const INPUT_SIZE: usize = 12;
pub(crate) const OUTPUT_SIZE: usize = 3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShaderInput {
    pub(crate) normal: Vector3<f32>,
    pub(crate) light_direction: Vector3<f32>,
    pub(crate) view_direction: Vector3<f32>,
    pub(crate) albedo: Vector3<f32>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PbrInput {
    pub(crate) normal: Vector3<f32>,
    pub(crate) light_direction: Vector3<f32>,
    pub(crate) view_direction: Vector3<f32>,
    pub(crate) albedo: Vector3<f32>,
    pub(crate) roughness: f32,
    pub(crate) metalness: f32,
}

impl ShaderInput {
    pub(crate) fn feature_row(self) -> [f32; INPUT_SIZE] {
        [
            self.normal.x,
            self.normal.y,
            self.normal.z,
            self.light_direction.x,
            self.light_direction.y,
            self.light_direction.z,
            self.view_direction.x,
            self.view_direction.y,
            self.view_direction.z,
            self.albedo.x,
            self.albedo.y,
            self.albedo.z,
        ]
    }
}

pub(crate) fn phong_color(input: ShaderInput) -> Vector3<f32> {
    let diffuse = input.normal.dot(&input.light_direction).max(0.0);
    let reflected_direction = reflect(-input.light_direction, input.normal);
    let specular = reflected_direction
        .dot(&input.view_direction)
        .max(0.0)
        .powf(SPECULAR_SHININESS)
        * SPECULAR_STRENGTH;

    input.albedo * (AMBIENT_STRENGTH + diffuse) + Vector3::repeat(specular)
}

pub(crate) fn pbr_color(input: PbrInput) -> Vector3<f32> {
    let normal = input.normal.normalize();
    let light_direction = input.light_direction.normalize();
    let view_direction = input.view_direction.normalize();
    let normal_dot_light = normal.dot(&light_direction).max(0.0);
    let normal_dot_view = normal.dot(&view_direction).max(0.0);
    if normal_dot_light == 0.0 || normal_dot_view == 0.0 {
        return Vector3::zeros();
    }

    let halfway = (light_direction + view_direction).normalize();
    let normal_dot_halfway = normal.dot(&halfway).max(0.0);
    let view_dot_halfway = view_direction.dot(&halfway).max(0.0);
    let roughness = input.roughness.clamp(0.0, 1.0);
    let metalness = input.metalness.clamp(0.0, 1.0);

    let diffuse = burley_diffuse(
        input.albedo,
        roughness,
        normal_dot_light,
        normal_dot_view,
        view_dot_halfway,
    ) * (1.0 - metalness);
    let specular = cook_torrance_ggx_specular(
        input.albedo,
        roughness,
        metalness,
        normal_dot_light,
        normal_dot_view,
        normal_dot_halfway,
        view_dot_halfway,
    );

    (diffuse + specular) * normal_dot_light
}

fn burley_diffuse(
    albedo: Vector3<f32>,
    roughness: f32,
    normal_dot_light: f32,
    normal_dot_view: f32,
    light_dot_halfway: f32,
) -> Vector3<f32> {
    let fd90 = 0.5 + 2.0 * roughness * light_dot_halfway.powi(2);
    let light_scatter = 1.0 + (fd90 - 1.0) * schlick_weight(normal_dot_light);
    let view_scatter = 1.0 + (fd90 - 1.0) * schlick_weight(normal_dot_view);
    albedo * (light_scatter * view_scatter / PI)
}

fn cook_torrance_ggx_specular(
    albedo: Vector3<f32>,
    roughness: f32,
    metalness: f32,
    normal_dot_light: f32,
    normal_dot_view: f32,
    normal_dot_halfway: f32,
    view_dot_halfway: f32,
) -> Vector3<f32> {
    let alpha = roughness.max(1.0e-4).powi(2);
    let alpha_squared = alpha.powi(2);
    let denominator = normal_dot_halfway.powi(2) * (alpha_squared - 1.0) + 1.0;
    let distribution = alpha_squared / (PI * denominator.powi(2));

    let f0 = Vector3::repeat(DIELECTRIC_F0).lerp(&albedo, metalness);
    let fresnel = f0 + (Vector3::repeat(1.0) - f0) * schlick_weight(view_dot_halfway);
    let masking_roughness = (roughness + 1.0).powi(2) / 8.0;
    let light_mask =
        normal_dot_light / (normal_dot_light * (1.0 - masking_roughness) + masking_roughness);
    let view_mask =
        normal_dot_view / (normal_dot_view * (1.0 - masking_roughness) + masking_roughness);
    fresnel * (distribution * light_mask * view_mask / (4.0 * normal_dot_light * normal_dot_view))
}

fn schlick_weight(cosine: f32) -> f32 {
    (1.0 - cosine.clamp(0.0, 1.0)).powi(5)
}

pub(crate) fn reflect(vector: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    vector - normal * (2.0 * vector.dot(&normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phong_reference_has_ambient_only_for_back_facing_light() {
        let color = phong_color(ShaderInput {
            normal: Vector3::new(0.0, 0.0, 1.0),
            light_direction: Vector3::new(0.0, 0.0, -1.0),
            view_direction: Vector3::new(0.0, 0.0, 1.0),
            albedo: Vector3::new(0.5, 0.25, 0.75),
        });

        assert_eq!(color, Vector3::new(0.04, 0.02, 0.06));
    }

    #[test]
    fn pbr_reference_uses_burley_diffuse_and_ggx_specular() {
        let color = pbr_color(PbrInput {
            normal: Vector3::new(0.0, 0.0, 1.0),
            light_direction: Vector3::new(0.0, 0.0, 1.0),
            view_direction: Vector3::new(0.0, 0.0, 1.0),
            albedo: Vector3::new(0.8, 0.4, 0.2),
            roughness: 0.5,
            metalness: 0.0,
        });

        let expected_diffuse = Vector3::new(0.8, 0.4, 0.2) / PI;
        let expected_specular = Vector3::repeat(DIELECTRIC_F0 * 16.0 / (4.0 * PI));
        let expected = expected_diffuse + expected_specular;
        for (actual, expected) in color.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn pbr_has_no_direct_contribution_for_back_facing_light() {
        let color = pbr_color(PbrInput {
            normal: Vector3::new(0.0, 0.0, 1.0),
            light_direction: Vector3::new(0.0, 0.0, -1.0),
            view_direction: Vector3::new(0.0, 0.0, 1.0),
            albedo: Vector3::repeat(1.0),
            roughness: 0.5,
            metalness: 0.0,
        });

        assert_eq!(color, Vector3::zeros());
    }

    #[test]
    fn feature_row_has_stable_order() {
        let row = ShaderInput {
            normal: Vector3::new(1.0, 2.0, 3.0),
            light_direction: Vector3::new(4.0, 5.0, 6.0),
            view_direction: Vector3::new(7.0, 8.0, 9.0),
            albedo: Vector3::new(10.0, 11.0, 12.0),
        }
        .feature_row();

        assert_eq!(
            row,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }
}
