use std::{error::Error, fs::File, path::Path};

pub(crate) const TILE_SIZE: usize = 8;

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
}
