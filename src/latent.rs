use std::{error::Error, io, path::Path};

use nalgebra::Vector2;

const MAGIC: &[u8; 8] = b"AIRTLAT1";
const HEADER_SIZE: usize = 20;

#[derive(Debug)]
pub(crate) struct LatentTexture {
    width: u32,
    height: u32,
    channels: usize,
    values: Vec<f32>,
}

impl LatentTexture {
    pub(crate) fn from_values(width: u32, height: u32, channels: usize, values: Vec<f32>) -> Self {
        assert!(width > 0 && height > 0 && channels > 0);
        assert_eq!(values.len(), width as usize * height as usize * channels);
        Self {
            width,
            height,
            channels,
            values,
        }
    }

    pub(crate) fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < HEADER_SIZE || &bytes[..MAGIC.len()] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "latent texture has an invalid header",
            )
            .into());
        }

        let width = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let channels = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        if width == 0 || height == 0 || channels == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "latent texture dimensions and channel count must be greater than zero",
            )
            .into());
        }
        let value_count = (width as usize)
            .checked_mul(height as usize)
            .and_then(|count| count.checked_mul(channels))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "latent texture is too large")
            })?;
        let expected_size = HEADER_SIZE
            .checked_add(
                value_count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "latent texture is too large")
                    })?,
            )
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "latent texture is too large")
            })?;
        if bytes.len() != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "latent texture payload length does not match its header",
            )
            .into());
        }

        let values: Vec<f32> = bytes[HEADER_SIZE..]
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "latent texture values must be finite",
            )
            .into());
        }
        Ok(Self::from_values(width, height, channels, values))
    }

    #[allow(dead_code)]
    pub(crate) fn write(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut bytes = Vec::with_capacity(HEADER_SIZE + self.values.len() * 4);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.width.to_le_bytes());
        bytes.extend_from_slice(&self.height.to_le_bytes());
        bytes.extend_from_slice(&(self.channels as u32).to_le_bytes());
        for value in &self.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub(crate) fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn sample(&self, uv: Vector2<f32>) -> Vec<f32> {
        let u = uv.x.rem_euclid(1.0) * self.width as f32;
        let v = uv.y.rem_euclid(1.0) * self.height as f32;
        let x0 = u.floor() as u32 % self.width;
        let x1 = (x0 + 1) % self.width;
        let y0 = v.floor() as u32 % self.height;
        let y1 = (y0 + 1) % self.height;
        let x_fraction = u.fract();
        let y_fraction = v.fract();
        let mut result = vec![0.0; self.channels];

        for (channel, result) in result.iter_mut().enumerate() {
            let top = self.value(x0, y0, channel) * (1.0 - x_fraction)
                + self.value(x1, y0, channel) * x_fraction;
            let bottom = self.value(x0, y1, channel) * (1.0 - x_fraction)
                + self.value(x1, y1, channel) * x_fraction;
            *result = top * (1.0 - y_fraction) + bottom * y_fraction;
        }
        result
    }

    fn value(&self, x: u32, y: u32, channel: usize) -> f32 {
        self.values[((y * self.width + x) as usize) * self.channels + channel]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_texture_samples_all_channels_with_wrapping() {
        let texture = LatentTexture::from_values(2, 1, 2, vec![1.0, 2.0, 3.0, 4.0]);

        assert_eq!(texture.sample(Vector2::new(0.0, 0.0)), vec![1.0, 2.0]);
        assert_eq!(texture.sample(Vector2::new(1.0, 0.0)), vec![1.0, 2.0]);
        assert_eq!(texture.sample(Vector2::new(0.25, 0.0)), vec![2.0, 3.0]);
    }

    #[test]
    fn latent_texture_round_trips_binary_format() {
        let path = std::env::temp_dir().join(format!("ai-rt-latent-{}.bin", std::process::id()));
        let texture = LatentTexture::from_values(2, 1, 2, vec![1.0, 2.0, 3.0, 4.0]);
        texture.write(&path).unwrap();

        let loaded = LatentTexture::load(&path).unwrap();
        assert_eq!(loaded.channels(), 2);
        assert_eq!(loaded.sample(Vector2::new(0.25, 0.0)), vec![2.0, 3.0]);
        std::fs::remove_file(path).unwrap();
    }
}
