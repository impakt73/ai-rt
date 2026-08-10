use std::{error::Error, path::Path};

use burn::{
    backend::Flex,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Shape, Tensor, TensorData,
        backend::{Backend, BackendTypes},
    },
};
use serde::Deserialize;

use crate::shader::{
    AMBIENT_STRENGTH, INPUT_SIZE, OUTPUT_SIZE, SPECULAR_SHININESS, SPECULAR_STRENGTH,
};

const HIDDEN_SIZE: usize = 64;

#[derive(Module, Debug)]
pub(crate) struct MlpShader<B: Backend> {
    input: Linear<B>,
    input_activation: Relu,
    hidden: Linear<B>,
    hidden_activation: Relu,
    output: Linear<B>,
}

impl<B: Backend> MlpShader<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(INPUT_SIZE, HIDDEN_SIZE).init(device),
            input_activation: Relu::new(),
            hidden: LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE).init(device),
            hidden_activation: Relu::new(),
            output: LinearConfig::new(HIDDEN_SIZE, OUTPUT_SIZE).init(device),
        }
    }

    pub(crate) fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let input = self.input_activation.forward(self.input.forward(input));
        let hidden = self.hidden_activation.forward(self.hidden.forward(input));
        self.output.forward(hidden)
    }
}

#[allow(dead_code)]
pub(crate) type InferenceBackend = Flex;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelManifest {
    schema_version: u32,
    input_size: usize,
    output_size: usize,
    hidden_sizes: [usize; 2],
    feature_order: Vec<String>,
    ambient_strength: f32,
    specular_strength: f32,
    specular_shininess: f32,
}

#[allow(dead_code)]
pub(crate) struct LoadedMlpShader {
    model: MlpShader<InferenceBackend>,
    device: <InferenceBackend as BackendTypes>::Device,
}

#[allow(dead_code)]
impl LoadedMlpShader {
    pub(crate) fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let manifest_path = path.with_extension("json");
        let manifest: ModelManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        let expected_features = [
            "normal.x",
            "normal.y",
            "normal.z",
            "light.x",
            "light.y",
            "light.z",
            "view.x",
            "view.y",
            "view.z",
            "material.r",
            "material.g",
            "material.b",
        ];
        if manifest.schema_version != 1
            || manifest.input_size != INPUT_SIZE
            || manifest.output_size != OUTPUT_SIZE
            || manifest.hidden_sizes != [64, 64]
            || manifest.feature_order != expected_features
            || manifest.ambient_strength != AMBIENT_STRENGTH
            || manifest.specular_strength != SPECULAR_STRENGTH
            || manifest.specular_shininess != SPECULAR_SHININESS
        {
            return Err("MLP model manifest does not match the runtime feature contract".into());
        }

        let device = Default::default();
        let model = MlpShader::new(&device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = model.load_file(path, &recorder, &device)?;
        Ok(Self { model, device })
    }

    pub(crate) fn infer(&self, features: &[f32], batch_size: usize) -> Vec<f32> {
        assert_eq!(features.len(), batch_size * INPUT_SIZE);
        let tensor = Tensor::<InferenceBackend, 2>::from_data(
            TensorData::new(features.to_vec(), Shape::new([batch_size, INPUT_SIZE])),
            &self.device,
        );
        self.model
            .forward(tensor)
            .into_data()
            .to_vec()
            .expect("MLP output must be f32")
    }
}
