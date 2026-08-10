use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(author, version, about = "Render a sphere scene")]
pub(crate) struct Args {
    /// Image width in pixels.
    #[arg(long, default_value_t = 64)]
    pub(crate) width: u32,

    /// Image height in pixels.
    #[arg(long, default_value_t = 64)]
    pub(crate) height: u32,

    /// Output PNG filename.
    #[arg(short, long, default_value = "output.png")]
    pub(crate) output: PathBuf,

    /// TOML scene description. Defaults to scene.toml.
    #[arg(short, long)]
    pub(crate) scene: Option<PathBuf>,

    /// Shading mode used for visible triangle hits.
    #[arg(long, value_enum, default_value_t = ShadingMode::Barycentrics)]
    pub(crate) shading_mode: ShadingMode,

    /// Burn MLP checkpoint base path, required by MLP shading.
    #[arg(long)]
    pub(crate) shader_model: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ShadingMode {
    Barycentrics,
    Phong,
    Pbr,
    Mlp,
}
