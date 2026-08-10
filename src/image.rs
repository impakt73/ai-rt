use std::{error::Error, fs::File, path::Path};

use nalgebra::{Vector2, Vector3};

pub(crate) const TILE_SIZE: usize = 8;

#[derive(Debug)]
pub(crate) struct Texture {
    width: u32,
    height: u32,
    pixels: Vec<Vector3<f32>>,
}

impl Texture {
    pub(crate) fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let image = ::image::ImageReader::open(path)?.decode()?.into_rgb8();
        let (width, height) = image.dimensions();
        let pixels = image
            .pixels()
            .map(|pixel| {
                Vector3::new(
                    f32::from(pixel[0]) / 255.0,
                    f32::from(pixel[1]) / 255.0,
                    f32::from(pixel[2]) / 255.0,
                )
            })
            .collect();

        Ok(Self::from_pixels(width, height, pixels))
    }

    pub(crate) fn from_pixels(width: u32, height: u32, pixels: Vec<Vector3<f32>>) -> Self {
        assert!(width > 0 && height > 0);
        assert_eq!(pixels.len(), (width * height) as usize);
        Self {
            width,
            height,
            pixels,
        }
    }

    pub(crate) fn sample(&self, uv: Vector2<f32>) -> Vector3<f32> {
        let u = uv.x.rem_euclid(1.0) * self.width as f32;
        let v = uv.y.rem_euclid(1.0) * self.height as f32;
        let x0 = u.floor() as u32 % self.width;
        let x1 = (x0 + 1) % self.width;
        let y0 = v.floor() as u32 % self.height;
        let y1 = (y0 + 1) % self.height;
        let x_fraction = u.fract();
        let y_fraction = v.fract();

        let top = self.texel(x0, y0) * (1.0 - x_fraction) + self.texel(x1, y0) * x_fraction;
        let bottom = self.texel(x0, y1) * (1.0 - x_fraction) + self.texel(x1, y1) * x_fraction;
        top * (1.0 - y_fraction) + bottom * y_fraction
    }

    fn texel(&self, x: u32, y: u32) -> Vector3<f32> {
        self.pixels[(y * self.width + x) as usize]
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PixelData {
    data: [u8; 3],
}

impl PixelData {
    pub(crate) fn new(red: u8, green: u8, blue: u8) -> Self {
        Self {
            data: [red, green, blue],
        }
    }

    pub(crate) fn from_color(color: nalgebra::Vector3<f32>) -> Self {
        Self::new(
            (color.x.clamp(0.0, 1.0) * 255.0) as u8,
            (color.y.clamp(0.0, 1.0) * 255.0) as u8,
            (color.z.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    fn write_rgb(&self, destination: &mut [u8]) {
        destination[..self.data.len()].copy_from_slice(&self.data);
    }
}

pub(crate) fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    tile_pixels: &[PixelData],
) -> Result<(), Box<dyn Error>> {
    let image_pixels = row_major_pixels(width, height, tile_pixels);

    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&image_pixels)?;

    Ok(())
}

pub(crate) fn row_major_pixels(width: u32, height: u32, tile_pixels: &[PixelData]) -> Vec<u8> {
    let width_usize = width as usize;
    let height_usize = height as usize;
    let tiles_x = width_usize.div_ceil(TILE_SIZE);
    let pixels_per_tile = TILE_SIZE * TILE_SIZE;
    let mut image_pixels = vec![0; width_usize * height_usize * 3];

    for (tile_index, tile) in tile_pixels.chunks(pixels_per_tile).enumerate() {
        let tile_x = tile_index % tiles_x;
        let tile_y = tile_index / tiles_x;
        let tile_start_x = tile_x * TILE_SIZE;
        let tile_start_y = tile_y * TILE_SIZE;

        for (morton_index, color) in tile.iter().enumerate() {
            let (local_x, local_y) = morton_coordinates(morton_index);
            let x = tile_start_x + local_x;
            let y = tile_start_y + local_y;
            if x >= width_usize || y >= height_usize {
                continue;
            }

            let offset = (y * width_usize + x) * 3;
            color.write_rgb(&mut image_pixels[offset..offset + 3]);
        }
    }

    image_pixels
}

pub(crate) fn morton_coordinates(index: usize) -> (usize, usize) {
    let mut x = 0;
    let mut y = 0;

    for bit in 0..TILE_SIZE.ilog2() as usize {
        x |= ((index >> (bit * 2)) & 1) << bit;
        y |= ((index >> (bit * 2 + 1)) & 1) << bit;
    }

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_texture() -> Texture {
        Texture {
            width: 2,
            height: 2,
            pixels: vec![
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
            ],
        }
    }

    #[test]
    fn morton_coordinates_follow_z_order() {
        let coordinates: Vec<_> = (0..8).map(morton_coordinates).collect();

        assert_eq!(
            coordinates,
            vec![
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (3, 0),
                (2, 1),
                (3, 1)
            ]
        );
    }

    #[test]
    fn row_major_pixels_unshuffles_morton_tiles() {
        let tile_pixels: Vec<_> = (0..4)
            .flat_map(|tile_index| {
                (0..TILE_SIZE * TILE_SIZE).map(move |morton_index| {
                    PixelData::new((tile_index * 64 + morton_index) as u8, 0, 0)
                })
            })
            .collect();
        let pixels = row_major_pixels(9, 10, &tile_pixels);

        assert_eq!(&pixels[0..3], &[0, 0, 0]);
        assert_eq!(&pixels[3..6], &[1, 0, 0]);
        assert_eq!(&pixels[(8 * 3)..(9 * 3)], &[64, 0, 0]);
        assert_eq!(&pixels[(9 * 8 * 3)..(9 * 8 * 3 + 3)], &[128, 0, 0]);
        assert_eq!(&pixels[((9 * 9 + 8) * 3)..((9 * 9 + 9) * 3)], &[194, 0, 0]);
    }

    #[test]
    fn texture_sampling_wraps_both_axes() {
        let texture = test_texture();

        assert_eq!(
            texture.sample(Vector2::new(1.0, 0.0)),
            Vector3::new(1.0, 0.0, 0.0)
        );
        assert_eq!(
            texture.sample(Vector2::new(0.0, 1.0)),
            Vector3::new(1.0, 0.0, 0.0)
        );
    }
}
