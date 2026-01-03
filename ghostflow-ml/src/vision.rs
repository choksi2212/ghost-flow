//! Computer Vision - Image Processing and Augmentation
//!
//! This module provides image processing utilities for computer vision tasks.

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Image data augmentation
pub struct ImageAugmentation {
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub rotation_range: f32,
    pub width_shift_range: f32,
    pub height_shift_range: f32,
    pub zoom_range: (f32, f32),
    pub brightness_range: (f32, f32),
    pub random_seed: Option<u64>,
}

impl ImageAugmentation {
    pub fn new() -> Self {
        ImageAugmentation {
            horizontal_flip: false,
            vertical_flip: false,
            rotation_range: 0.0,
            width_shift_range: 0.0,
            height_shift_range: 0.0,
            zoom_range: (1.0, 1.0),
            brightness_range: (1.0, 1.0),
            random_seed: None,
        }
    }

    pub fn horizontal_flip(mut self, flip: bool) -> Self {
        self.horizontal_flip = flip;
        self
    }

    pub fn vertical_flip(mut self, flip: bool) -> Self {
        self.vertical_flip = flip;
        self
    }

    pub fn rotation_range(mut self, degrees: f32) -> Self {
        self.rotation_range = degrees;
        self
    }

    pub fn shift_range(mut self, width: f32, height: f32) -> Self {
        self.width_shift_range = width;
        self.height_shift_range = height;
        self
    }

    pub fn zoom_range(mut self, min: f32, max: f32) -> Self {
        self.zoom_range = (min, max);
        self
    }

    pub fn brightness_range(mut self, min: f32, max: f32) -> Self {
        self.brightness_range = (min, max);
        self
    }

    pub fn augment(&self, image: &Tensor) -> Tensor {
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let dims = image.dims();
        let mut data = image.data_f32().to_vec();

        // Horizontal flip
        if self.horizontal_flip && rng.gen::<f32>() > 0.5 {
            data = self.flip_horizontal(&data, dims);
        }

        // Vertical flip
        if self.vertical_flip && rng.gen::<f32>() > 0.5 {
            data = self.flip_vertical(&data, dims);
        }

        // Brightness adjustment
        if self.brightness_range.0 != 1.0 || self.brightness_range.1 != 1.0 {
            let factor = rng.gen::<f32>() * (self.brightness_range.1 - self.brightness_range.0) 
                + self.brightness_range.0;
            for pixel in &mut data {
                *pixel *= factor;
                *pixel = pixel.clamp(0.0, 1.0);
            }
        }

        Tensor::from_slice(&data, dims).unwrap()
    }

    fn flip_horizontal(&self, data: &[f32], dims: &[usize]) -> Vec<f32> {
        let (channels, height, width) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            (1, dims[0], dims[1])
        };

        let mut flipped = vec![0.0f32; data.len()];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_idx = c * height * width + h * width + w;
                    let dst_idx = c * height * width + h * width + (width - 1 - w);
                    flipped[dst_idx] = data[src_idx];
                }
            }
        }

        flipped
    }

    fn flip_vertical(&self, data: &[f32], dims: &[usize]) -> Vec<f32> {
        let (channels, height, width) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            (1, dims[0], dims[1])
        };

        let mut flipped = vec![0.0f32; data.len()];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_idx = c * height * width + h * width + w;
                    let dst_idx = c * height * width + (height - 1 - h) * width + w;
                    flipped[dst_idx] = data[src_idx];
                }
            }
        }

        flipped
    }
}

impl Default for ImageAugmentation {
    fn default() -> Self { Self::new() }
}

/// Image normalization
pub struct ImageNormalization {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl ImageNormalization {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        ImageNormalization { mean, std }
    }

    /// ImageNet normalization
    pub fn imagenet() -> Self {
        ImageNormalization {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }

    pub fn normalize(&self, image: &Tensor) -> Tensor {
        let dims = image.dims();
        let data = image.data_f32();
        let _channels = if dims.len() == 3 { dims[0] } else { 1 };

        let normalized: Vec<f32> = data.iter()
            .enumerate()
            .map(|(i, &pixel)| {
                let c = if dims.len() == 3 {
                    i / (dims[1] * dims[2])
                } else {
                    0
                };
                (pixel - self.mean[c % self.mean.len()]) / self.std[c % self.std.len()]
            })
            .collect();

        Tensor::from_slice(&normalized, dims).unwrap()
    }

    pub fn denormalize(&self, image: &Tensor) -> Tensor {
        let dims = image.dims();
        let data = image.data_f32();
        let _channels = if dims.len() == 3 { dims[0] } else { 1 };

        let denormalized: Vec<f32> = data.iter()
            .enumerate()
            .map(|(i, &pixel)| {
                let c = if dims.len() == 3 {
                    i / (dims[1] * dims[2])
                } else {
                    0
                };
                pixel * self.std[c % self.std.len()] + self.mean[c % self.mean.len()]
            })
            .collect();

        Tensor::from_slice(&denormalized, dims).unwrap()
    }
}

/// Image resizing
pub struct ImageResize {
    pub target_size: (usize, usize),
    pub interpolation: Interpolation,
}

#[derive(Clone, Copy)]
pub enum Interpolation {
    Nearest,
    Bilinear,
}

impl ImageResize {
    pub fn new(width: usize, height: usize) -> Self {
        ImageResize {
            target_size: (width, height),
            interpolation: Interpolation::Bilinear,
        }
    }

    pub fn interpolation(mut self, interp: Interpolation) -> Self {
        self.interpolation = interp;
        self
    }

    pub fn resize(&self, image: &Tensor) -> Tensor {
        let dims = image.dims();
        let data = image.data_f32();

        let (channels, src_height, src_width) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            (1, dims[0], dims[1])
        };

        let (dst_width, dst_height) = self.target_size;

        match self.interpolation {
            Interpolation::Nearest => {
                self.resize_nearest(&data, channels, src_height, src_width, dst_height, dst_width)
            }
            Interpolation::Bilinear => {
                self.resize_bilinear(&data, channels, src_height, src_width, dst_height, dst_width)
            }
        }
    }

    fn resize_nearest(
        &self,
        data: &[f32],
        channels: usize,
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Tensor {
        let mut resized = vec![0.0f32; channels * dst_h * dst_w];

        let scale_h = src_h as f32 / dst_h as f32;
        let scale_w = src_w as f32 / dst_w as f32;

        for c in 0..channels {
            for h in 0..dst_h {
                for w in 0..dst_w {
                    let src_h_idx = (h as f32 * scale_h) as usize;
                    let src_w_idx = (w as f32 * scale_w) as usize;

                    let src_idx = c * src_h * src_w + src_h_idx * src_w + src_w_idx;
                    let dst_idx = c * dst_h * dst_w + h * dst_w + w;

                    resized[dst_idx] = data[src_idx];
                }
            }
        }

        let dims = if channels == 1 {
            vec![dst_h, dst_w]
        } else {
            vec![channels, dst_h, dst_w]
        };

        Tensor::from_slice(&resized, &dims).unwrap()
    }

    fn resize_bilinear(
        &self,
        data: &[f32],
        channels: usize,
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Tensor {
        let mut resized = vec![0.0f32; channels * dst_h * dst_w];

        let scale_h = src_h as f32 / dst_h as f32;
        let scale_w = src_w as f32 / dst_w as f32;

        for c in 0..channels {
            for h in 0..dst_h {
                for w in 0..dst_w {
                    let src_h_f = h as f32 * scale_h;
                    let src_w_f = w as f32 * scale_w;

                    let h0 = src_h_f.floor() as usize;
                    let w0 = src_w_f.floor() as usize;
                    let h1 = (h0 + 1).min(src_h - 1);
                    let w1 = (w0 + 1).min(src_w - 1);

                    let dh = src_h_f - h0 as f32;
                    let dw = src_w_f - w0 as f32;

                    let idx00 = c * src_h * src_w + h0 * src_w + w0;
                    let idx01 = c * src_h * src_w + h0 * src_w + w1;
                    let idx10 = c * src_h * src_w + h1 * src_w + w0;
                    let idx11 = c * src_h * src_w + h1 * src_w + w1;

                    let val = (1.0 - dh) * (1.0 - dw) * data[idx00]
                        + (1.0 - dh) * dw * data[idx01]
                        + dh * (1.0 - dw) * data[idx10]
                        + dh * dw * data[idx11];

                    let dst_idx = c * dst_h * dst_w + h * dst_w + w;
                    resized[dst_idx] = val;
                }
            }
        }

        let dims = if channels == 1 {
            vec![dst_h, dst_w]
        } else {
            vec![channels, dst_h, dst_w]
        };

        Tensor::from_slice(&resized, &dims).unwrap()
    }
}

/// Image cropping
pub struct ImageCrop {
    pub top: usize,
    pub left: usize,
    pub height: usize,
    pub width: usize,
}

impl ImageCrop {
    pub fn new(top: usize, left: usize, height: usize, width: usize) -> Self {
        ImageCrop { top, left, height, width }
    }

    pub fn center_crop(image_height: usize, image_width: usize, crop_size: usize) -> Self {
        let top = (image_height - crop_size) / 2;
        let left = (image_width - crop_size) / 2;
        ImageCrop {
            top,
            left,
            height: crop_size,
            width: crop_size,
        }
    }

    pub fn crop(&self, image: &Tensor) -> Tensor {
        let dims = image.dims();
        let data = image.data_f32();

        let (channels, src_height, src_width) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            (1, dims[0], dims[1])
        };

        let mut cropped = vec![0.0f32; channels * self.height * self.width];

        for c in 0..channels {
            for h in 0..self.height {
                for w in 0..self.width {
                    let src_h = self.top + h;
                    let src_w = self.left + w;

                    if src_h < src_height && src_w < src_width {
                        let src_idx = c * src_height * src_width + src_h * src_width + src_w;
                        let dst_idx = c * self.height * self.width + h * self.width + w;
                        cropped[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        let dims = if channels == 1 {
            vec![self.height, self.width]
        } else {
            vec![channels, self.height, self.width]
        };

        Tensor::from_slice(&cropped, &dims).unwrap()
    }
}

/// Random crop
pub struct RandomCrop {
    pub height: usize,
    pub width: usize,
    pub random_seed: Option<u64>,
}

impl RandomCrop {
    pub fn new(height: usize, width: usize) -> Self {
        RandomCrop {
            height,
            width,
            random_seed: None,
        }
    }

    pub fn crop(&self, image: &Tensor) -> Tensor {
        let dims = image.dims();
        let (_, src_height, src_width) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            (1, dims[0], dims[1])
        };

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let max_top = src_height.saturating_sub(self.height);
        let max_left = src_width.saturating_sub(self.width);

        let top = if max_top > 0 { rng.gen_range(0..=max_top) } else { 0 };
        let left = if max_left > 0 { rng.gen_range(0..=max_left) } else { 0 };

        let crop = ImageCrop::new(top, left, self.height, self.width);
        crop.crop(image)
    }
}

/// Color jitter
pub struct ColorJitter {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
}

impl ColorJitter {
    pub fn new() -> Self {
        ColorJitter {
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            hue: 0.0,
        }
    }

    pub fn brightness(mut self, factor: f32) -> Self {
        self.brightness = factor;
        self
    }

    pub fn contrast(mut self, factor: f32) -> Self {
        self.contrast = factor;
        self
    }

    pub fn apply(&self, image: &Tensor) -> Tensor {
        let mut rng = thread_rng();
        let data = image.data_f32();
        let mut jittered = data.to_vec();

        // Brightness
        if self.brightness > 0.0 {
            let factor = 1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * self.brightness;
            for pixel in &mut jittered {
                *pixel = (*pixel * factor).clamp(0.0, 1.0);
            }
        }

        // Contrast
        if self.contrast > 0.0 {
            let mean = jittered.iter().sum::<f32>() / jittered.len() as f32;
            let factor = 1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * self.contrast;
            for pixel in &mut jittered {
                *pixel = ((*pixel - mean) * factor + mean).clamp(0.0, 1.0);
            }
        }

        Tensor::from_slice(&jittered, image.dims()).unwrap()
    }
}

impl Default for ColorJitter {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_augmentation() {
        let image = Tensor::from_slice(&vec![0.5f32; 3 * 32 * 32], &[3, 32, 32]).unwrap();
        
        let aug = ImageAugmentation::new()
            .horizontal_flip(true)
            .brightness_range(0.8, 1.2);

        let augmented = aug.augment(&image);
        assert_eq!(augmented.dims(), image.dims());
    }

    #[test]
    fn test_image_normalization() {
        let image = Tensor::from_slice(&vec![0.5f32; 3 * 32 * 32], &[3, 32, 32]).unwrap();
        let norm = ImageNormalization::imagenet();

        let normalized = norm.normalize(&image);
        let denormalized = norm.denormalize(&normalized);

        assert_eq!(normalized.dims(), image.dims());
        assert_eq!(denormalized.dims(), image.dims());
    }

    #[test]
    fn test_image_resize() {
        let image = Tensor::from_slice(&vec![0.5f32; 3 * 64 * 64], &[3, 64, 64]).unwrap();
        let resize = ImageResize::new(32, 32);

        let resized = resize.resize(&image);
        assert_eq!(resized.dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_image_crop() {
        let image = Tensor::from_slice(&vec![0.5f32; 3 * 64 * 64], &[3, 64, 64]).unwrap();
        let crop = ImageCrop::center_crop(64, 64, 32);

        let cropped = crop.crop(&image);
        assert_eq!(cropped.dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_color_jitter() {
        let image = Tensor::from_slice(&vec![0.5f32; 3 * 32 * 32], &[3, 32, 32]).unwrap();
        let jitter = ColorJitter::new().brightness(0.2).contrast(0.2);

        let jittered = jitter.apply(&image);
        assert_eq!(jittered.dims(), image.dims());
    }
}
