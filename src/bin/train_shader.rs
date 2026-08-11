#![allow(dead_code)]

#[path = "../cli.rs"]
mod cli;
#[path = "../geometry.rs"]
mod geometry;
#[path = "../image.rs"]
mod image;
#[path = "../latent.rs"]
mod latent;
#[path = "../mlp.rs"]
mod mlp;
#[path = "../scene.rs"]
mod scene;
#[allow(dead_code)]
#[path = "../shader.rs"]
mod shader;

use std::{error::Error, fs, path::PathBuf, sync::Arc};

use burn::{
    backend::{Autodiff, Flex},
    module::{Module, Param},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Int, Shape, Tensor, TensorData},
};
use clap::Parser;
use nalgebra::{Vector2, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use latent::LatentTexture;
use mlp::MlpShader;
use scene::{Material, RenderScene};
use shader::{
    DEFAULT_LATENT_SIZE, DEFAULT_LATENT_TEXTURE_SIZE, DIRECTION_INPUT_SIZE, MlpInput, OUTPUT_SIZE,
    PbrInput, pbr_color,
};

type TrainingBackend = Autodiff<Flex>;

#[derive(Module, Debug)]
struct TrainingModel<B: burn::tensor::backend::Backend> {
    shader: MlpShader<B>,
    latents: Param<Tensor<B, 2>>,
}

impl<B: burn::tensor::backend::Backend> TrainingModel<B> {
    fn forward(&self, directions: Tensor<B, 2>, material_ids: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let latents = self.latents.val().select(0, material_ids);
        self.shader
            .forward(Tensor::cat(vec![directions, latents], 1))
    }
}

#[derive(Debug, Parser)]
#[command(
    about = "Train a Burn MLP and per-material latent textures to approximate the PBR shader"
)]
struct Args {
    #[arg(long, default_value = "scene.toml")]
    scene: PathBuf,

    #[arg(long, default_value_t = 8_192)]
    samples: usize,

    #[arg(long, default_value_t = 2)]
    epochs: usize,

    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = DEFAULT_LATENT_SIZE)]
    latent_size: usize,

    #[arg(long, default_value_t = DEFAULT_LATENT_TEXTURE_SIZE)]
    latent_width: usize,

    #[arg(long, default_value_t = DEFAULT_LATENT_TEXTURE_SIZE)]
    latent_height: usize,

    #[arg(long, default_value = "models/pbr_mlp_v1/model")]
    output: PathBuf,

    #[arg(long)]
    latent_output: Option<PathBuf>,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    input_size: usize,
    latent_size: usize,
    output_size: usize,
    hidden_sizes: [usize; 2],
    seed: u64,
    samples: usize,
    epochs: usize,
    batch_size: usize,
    feature_order: Vec<String>,
    target_shader: &'static str,
    pbr_shader_version: u32,
}

#[derive(Serialize)]
struct LatentManifest {
    schema_version: u32,
    latent_size: usize,
    width: usize,
    height: usize,
    materials: Vec<LatentMaterial>,
}

#[derive(Serialize)]
struct LatentMaterial {
    name: String,
    texture: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.samples == 0
        || args.epochs == 0
        || args.batch_size == 0
        || args.latent_size == 0
        || args.latent_width == 0
        || args.latent_height == 0
    {
        return Err(
            "samples, epochs, batch size, latent size, and latent dimensions must be greater than zero"
                .into(),
        );
    }

    let source = scene::load_scene(&args.scene, cli::ShadingMode::Pbr, None)?;
    let materials = unique_materials(&source);
    if materials.is_empty() {
        return Err("the training scene must contain at least one object".into());
    }

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let latent_output = args.latent_output.unwrap_or_else(|| {
        args.output
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("latents")
    });
    fs::create_dir_all(&latent_output)?;

    let device = Default::default();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let latent_texels = args
        .latent_width
        .checked_mul(args.latent_height)
        .ok_or("latent texture dimensions are too large")?;
    let latent_rows = materials
        .len()
        .checked_mul(latent_texels)
        .ok_or("latent texture count is too large")?;
    let latent_values = latent_rows
        .checked_mul(args.latent_size)
        .ok_or("latent parameter count is too large")?;
    let initial_latents: Vec<f32> = (0..latent_values)
        .map(|_| rng.random_range(-0.05..0.05))
        .collect();
    let mut model = TrainingModel {
        shader: MlpShader::new(&device, DIRECTION_INPUT_SIZE + args.latent_size),
        latents: Param::from_data(
            TensorData::new(initial_latents, Shape::new([latent_rows, args.latent_size])),
            &device,
        ),
    };
    let mut optimizer = AdamConfig::new().init();
    let loss_function = MseLoss::new();
    let steps_per_epoch = args.samples.div_ceil(args.batch_size);

    for epoch in 0..args.epochs {
        let mut epoch_loss = 0.0;
        for _ in 0..steps_per_epoch {
            let batch = make_batch(
                &mut rng,
                &materials,
                args.batch_size,
                args.latent_width,
                args.latent_height,
            );
            let directions = Tensor::<TrainingBackend, 2>::from_data(
                TensorData::new(
                    batch.directions,
                    Shape::new([args.batch_size, DIRECTION_INPUT_SIZE]),
                ),
                &device,
            );
            let material_ids = Tensor::<TrainingBackend, 1, Int>::from_data(
                TensorData::new(batch.material_ids, Shape::new([args.batch_size])),
                &device,
            );
            let targets = Tensor::<TrainingBackend, 2>::from_data(
                TensorData::new(batch.targets, Shape::new([args.batch_size, OUTPUT_SIZE])),
                &device,
            );
            let predictions = model.forward(directions, material_ids);
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
    write_latent_assets(
        &latent_output,
        &materials,
        args.latent_size,
        args.latent_width,
        args.latent_height,
        &model,
    )?;
    model.shader.save_file(args.output.clone(), &recorder)?;

    let manifest = Manifest {
        schema_version: 2,
        input_size: DIRECTION_INPUT_SIZE + args.latent_size,
        latent_size: args.latent_size,
        output_size: OUTPUT_SIZE,
        hidden_sizes: [64, 64],
        seed: args.seed,
        samples: args.samples,
        epochs: args.epochs,
        batch_size: args.batch_size,
        feature_order: feature_order(args.latent_size),
        target_shader: "pbr",
        pbr_shader_version: 1,
    };
    fs::write(
        args.output.with_extension("json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

struct Batch {
    directions: Vec<f32>,
    material_ids: Vec<i64>,
    targets: Vec<f32>,
}

fn make_batch(
    rng: &mut ChaCha8Rng,
    materials: &[Arc<Material>],
    batch_size: usize,
    latent_width: usize,
    latent_height: usize,
) -> Batch {
    let mut directions = Vec::with_capacity(batch_size * DIRECTION_INPUT_SIZE);
    let mut material_ids = Vec::with_capacity(batch_size);
    let mut targets = Vec::with_capacity(batch_size * OUTPUT_SIZE);
    for _ in 0..batch_size {
        let material_id = rng.random_range(0..materials.len());
        let material = &materials[material_id];
        let latent_x = rng.random_range(0..latent_width);
        let latent_y = rng.random_range(0..latent_height);
        let uv = Vector2::new(
            latent_x as f32 / latent_width as f32,
            latent_y as f32 / latent_height as f32,
        )
        .component_mul(&material.uv_scale);
        let normal = random_unit_vector(rng);
        let light_direction = random_unit_vector(rng);
        let view_direction = random_unit_vector(rng);
        directions.extend(
            MlpInput {
                normal,
                light_direction,
                view_direction,
            }
            .feature_row(&[]),
        );
        let material_texel = latent_y * latent_width + latent_x;
        material_ids.push((material_id * latent_width * latent_height + material_texel) as i64);
        let target = pbr_color(PbrInput {
            normal,
            light_direction,
            view_direction,
            albedo: material.albedo.sample(uv),
            roughness: material.roughness.sample(uv),
            metalness: material.metalness.sample(uv),
        });
        targets.extend([target.x, target.y, target.z]);
    }
    Batch {
        directions,
        material_ids,
        targets,
    }
}

fn write_latent_assets<B: burn::tensor::backend::Backend>(
    output: &std::path::Path,
    materials: &[Arc<Material>],
    latent_size: usize,
    latent_width: usize,
    latent_height: usize,
    model: &TrainingModel<B>,
) -> Result<(), Box<dyn Error>> {
    let values = model.latents.val().into_data().to_vec::<f32>()?;
    let mut manifest = LatentManifest {
        schema_version: 1,
        latent_size,
        width: latent_width,
        height: latent_height,
        materials: Vec::with_capacity(materials.len()),
    };
    for (index, material) in materials.iter().enumerate() {
        let file_name = format!("material_{index}.latent");
        let path = output.join(&file_name);
        let start = index * latent_width * latent_height * latent_size;
        let texture = LatentTexture::from_values(
            latent_width as u32,
            latent_height as u32,
            latent_size,
            values[start..start + latent_width * latent_height * latent_size].to_vec(),
        );
        texture.write(&path)?;
        manifest.materials.push(LatentMaterial {
            name: material.name.clone(),
            texture: file_name,
        });
    }
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

fn feature_order(latent_size: usize) -> Vec<String> {
    let mut order = [
        "normal.x", "normal.y", "normal.z", "light.x", "light.y", "light.z", "view.x", "view.y",
        "view.z",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();
    order.extend((0..latent_size).map(|index| format!("latent.{index}")));
    order
}

fn unique_materials(scene: &RenderScene) -> Vec<Arc<Material>> {
    let mut materials = Vec::new();
    for sphere in &scene.spheres {
        if !materials
            .iter()
            .any(|material: &Arc<Material>| Arc::ptr_eq(material, &sphere.material))
        {
            materials.push(sphere.material.clone());
        }
    }
    materials
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
