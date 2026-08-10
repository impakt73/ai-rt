use nalgebra::Vector3;

pub(crate) const AMBIENT_STRENGTH: f32 = 0.08;
pub(crate) const SPECULAR_STRENGTH: f32 = 0.35;
pub(crate) const SPECULAR_SHININESS: f32 = 32.0;
pub(crate) const INPUT_SIZE: usize = 12;
pub(crate) const OUTPUT_SIZE: usize = 3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShaderInput {
    pub(crate) normal: Vector3<f32>,
    pub(crate) light_direction: Vector3<f32>,
    pub(crate) view_direction: Vector3<f32>,
    pub(crate) material_color: Vector3<f32>,
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
            self.material_color.x,
            self.material_color.y,
            self.material_color.z,
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

    input.material_color * (AMBIENT_STRENGTH + diffuse) + Vector3::repeat(specular)
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
            material_color: Vector3::new(0.5, 0.25, 0.75),
        });

        assert_eq!(color, Vector3::new(0.04, 0.02, 0.06));
    }

    #[test]
    fn feature_row_has_stable_order() {
        let row = ShaderInput {
            normal: Vector3::new(1.0, 2.0, 3.0),
            light_direction: Vector3::new(4.0, 5.0, 6.0),
            view_direction: Vector3::new(7.0, 8.0, 9.0),
            material_color: Vector3::new(10.0, 11.0, 12.0),
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
