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

use crate::shader::{DIRECTION_INPUT_SIZE, OUTPUT_SIZE};

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
    pub(crate) fn new(device: &B::Device, input_size: usize) -> Self {
        Self {
            input: LinearConfig::new(input_size, HIDDEN_SIZE).init(device),
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
    #[serde(default)]
    latent_size: usize,
    output_size: usize,
    hidden_sizes: [usize; 2],
    feature_order: Vec<String>,
    #[serde(default)]
    target_shader: String,
    #[serde(default)]
    pbr_shader_version: u32,
}

#[allow(dead_code)]
pub(crate) struct LoadedMlpShader {
    model: MlpShader<InferenceBackend>,
    device: <InferenceBackend as BackendTypes>::Device,
    latent_size: usize,
}

#[allow(dead_code)]
impl LoadedMlpShader {
    pub(crate) fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let manifest_path = path.with_extension("json");
        let manifest: ModelManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        let mut expected_features = vec![
            "normal.x", "normal.y", "normal.z", "light.x", "light.y", "light.z", "view.x",
            "view.y", "view.z",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        expected_features.extend((0..manifest.latent_size).map(|index| format!("latent.{index}")));
        if manifest.schema_version != 2
            || manifest.input_size != DIRECTION_INPUT_SIZE + manifest.latent_size
            || manifest.latent_size == 0
            || manifest.output_size != OUTPUT_SIZE
            || manifest.hidden_sizes != [64, 64]
            || manifest.feature_order != expected_features
            || manifest.target_shader != "pbr"
            || manifest.pbr_shader_version != 1
        {
            return Err("MLP model manifest does not match the runtime feature contract".into());
        }

        let device = Default::default();
        let model = MlpShader::new(&device, manifest.input_size);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = model.load_file(path, &recorder, &device)?;
        Ok(Self {
            model,
            device,
            latent_size: manifest.latent_size,
        })
    }

    pub(crate) fn infer(&self, features: &[f32], batch_size: usize) -> Vec<f32> {
        assert_eq!(
            features.len(),
            batch_size * (DIRECTION_INPUT_SIZE + self.latent_size)
        );
        let tensor = Tensor::<InferenceBackend, 2>::from_data(
            TensorData::new(
                features.to_vec(),
                Shape::new([batch_size, DIRECTION_INPUT_SIZE + self.latent_size]),
            ),
            &self.device,
        );
        self.model
            .forward(tensor)
            .into_data()
            .to_vec()
            .expect("MLP output must be f32")
    }

    pub(crate) fn latent_size(&self) -> usize {
        self.latent_size
    }
}
