#[path = "../mlp.rs"]
mod mlp;
#[path = "../shader.rs"]
mod shader;

use std::{error::Error, fs, path::PathBuf};

use burn::{
    backend::{Autodiff, Flex},
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Shape, Tensor, TensorData},
};
use clap::Parser;
use nalgebra::Vector3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use mlp::MlpShader;
use shader::{
    AMBIENT_STRENGTH, INPUT_SIZE, OUTPUT_SIZE, SPECULAR_SHININESS, SPECULAR_STRENGTH, ShaderInput,
    phong_color,
};

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Parser)]
#[command(about = "Train a Burn MLP to approximate the Phong shader")]
struct Args {
    #[arg(long, default_value_t = 8_192)]
    samples: usize,

    #[arg(long, default_value_t = 2)]
    epochs: usize,

    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value = "models/phong_mlp_v1/model")]
    output: PathBuf,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    input_size: usize,
    output_size: usize,
    hidden_sizes: [usize; 2],
    seed: u64,
    samples: usize,
    epochs: usize,
    batch_size: usize,
    feature_order: [&'static str; 12],
    ambient_strength: f32,
    specular_strength: f32,
    specular_shininess: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.samples == 0 || args.epochs == 0 || args.batch_size == 0 {
        return Err("samples, epochs, and batch size must be greater than zero".into());
    }

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    let device = Default::default();
    let mut model = MlpShader::<TrainingBackend>::new(&device);
    let mut optimizer = AdamConfig::new().init();
    let loss_function = MseLoss::new();
    let steps_per_epoch = args.samples.div_ceil(args.batch_size);

    for epoch in 0..args.epochs {
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed + epoch as u64);
        let mut epoch_loss = 0.0;
        for _ in 0..steps_per_epoch {
            let (features, targets) = make_batch(&mut rng, args.batch_size);
            let inputs = Tensor::<TrainingBackend, 2>::from_data(
                TensorData::new(features, Shape::new([args.batch_size, INPUT_SIZE])),
                &device,
            );
            let targets = Tensor::<TrainingBackend, 2>::from_data(
                TensorData::new(targets, Shape::new([args.batch_size, OUTPUT_SIZE])),
                &device,
            );
            let predictions = model.forward(inputs);
            let loss = loss_function.forward(predictions, targets, Reduction::Mean);
            epoch_loss += loss.clone().into_data().to_vec::<f32>()?[0] as f64;
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(1e-3, model, gradients);
        }
        println!(
            "epoch {}/{} loss {:.6}",
            epoch + 1,
            args.epochs,
            epoch_loss / steps_per_epoch as f64
        );
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.save_file(args.output.clone(), &recorder)?;
    let manifest = Manifest {
        schema_version: 1,
        input_size: INPUT_SIZE,
        output_size: OUTPUT_SIZE,
        hidden_sizes: [64, 64],
        seed: args.seed,
        samples: args.samples,
        epochs: args.epochs,
        batch_size: args.batch_size,
        feature_order: [
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
        ],
        ambient_strength: AMBIENT_STRENGTH,
        specular_strength: SPECULAR_STRENGTH,
        specular_shininess: SPECULAR_SHININESS,
    };
    fs::write(
        args.output.with_extension("json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

fn make_batch(rng: &mut ChaCha8Rng, batch_size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut features = Vec::with_capacity(batch_size * INPUT_SIZE);
    let mut targets = Vec::with_capacity(batch_size * OUTPUT_SIZE);
    for _ in 0..batch_size {
        let input = ShaderInput {
            normal: random_unit_vector(rng),
            light_direction: random_unit_vector(rng),
            view_direction: random_unit_vector(rng),
            material_color: Vector3::new(rng.random(), rng.random(), rng.random()),
        };
        features.extend(input.feature_row());
        let target = phong_color(input);
        targets.extend([target.x, target.y, target.z]);
    }
    (features, targets)
}

fn random_unit_vector(rng: &mut ChaCha8Rng) -> Vector3<f32> {
    loop {
        let vector: Vector3<f32> = Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let length_squared = vector.norm_squared();
        if length_squared > 1.0e-6 && length_squared <= 1.0 {
            return vector / length_squared.sqrt();
        }
    }
}
