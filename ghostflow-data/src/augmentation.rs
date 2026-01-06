//! Data Augmentation
//!
//! Common data augmentation techniques for training.

use ghostflow_core::tensor::Tensor;
use rand::Rng;

/// Random horizontal flip
pub struct RandomHorizontalFlip {
    pub probability: f32,
}

impl RandomHorizontalFlip {
    pub fn new(probability: f32) -> Self {
        Self { probability }
    }

    pub fn apply(&self, image: &Tensor) -> Tensor {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < self.probability {
            self.flip_horizontal(image)
        } else {
            image.clone()
        }
    }

    fn flip_horizontal(&self, image: &Tensor) -> Tensor {
        let shape = image.shape().dims();
        let data = image.storage().as_slice::<f32>();
        
        // Assume image is [C, H, W] or [H, W]
        let (channels, height, width) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            (1, shape[0], shape[1])
        };

        let mut flipped = Vec::with_capacity(data.len());

        for c in 0..channels {
            for h in 0..height {
                for w in (0..width).rev() {
                    let idx = c * height * width + h * width + w;
                    flipped.push(data[idx]);
                }
            }
        }

        Tensor::from_slice(&flipped, shape).unwrap()
    }
}

/// Random vertical flip
pub struct RandomVerticalFlip {
    pub probability: f32,
}

impl RandomVerticalFlip {
    pub fn new(probability: f32) -> Self {
        Self { probability }
    }

    pub fn apply(&self, image: &Tensor) -> Tensor {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < self.probability {
            self.flip_vertical(image)
        } else {
            image.clone()
        }
    }

    fn flip_vertical(&self, image: &Tensor) -> Tensor {
        let shape = image.shape().dims();
        let data = image.storage().as_slice::<f32>();
        
        let (channels, height, width) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            (1, shape[0], shape[1])
        };

        let mut flipped = Vec::with_capacity(data.len());

        for c in 0..channels {
            for h in (0..height).rev() {
                for w in 0..width {
                    let idx = c * height * width + h * width + w;
                    flipped.push(data[idx]);
                }
            }
        }

        Tensor::from_slice(&flipped, shape).unwrap()
    }
}

/// Random rotation
pub struct RandomRotation {
    pub max_degrees: f32,
}

impl RandomRotation {
    pub fn new(max_degrees: f32) -> Self {
        Self { max_degrees }
    }

    pub fn apply(&self, image: &Tensor) -> Tensor {
        let mut rng = rand::thread_rng();
        let degrees = rng.gen_range(-self.max_degrees..=self.max_degrees);
        self.rotate(image, degrees)
    }

    fn rotate(&self, image: &Tensor, _degrees: f32) -> Tensor {
        // Simplified rotation - in practice would use proper image rotation
        // For now, just return the image
        image.clone()
    }
}

/// Random crop
pub struct RandomCrop {
    pub size: (usize, usize),
}

impl RandomCrop {
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }

    pub fn apply(&self, image: &Tensor) -> Tensor {
        let shape = image.shape().dims();
        let (channels, height, width) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            (1, shape[0], shape[1])
        };

        let (crop_h, crop_w) = self.size;
        
        if crop_h > height || crop_w > width {
            return image.clone();
        }

        let mut rng = rand::thread_rng();
        let top = rng.gen_range(0..=(height - crop_h));
        let left = rng.gen_range(0..=(width - crop_w));

        self.crop(image, top, left, crop_h, crop_w)
    }

    fn crop(&self, image: &Tensor, top: usize, left: usize, height: usize, width: usize) -> Tensor {
        let shape = image.shape().dims();
        let data = image.storage().as_slice::<f32>();
        
        let (channels, img_height, img_width) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            (1, shape[0], shape[1])
        };

        let mut cropped = Vec::new();

        for c in 0..channels {
            for h in top..(top + height) {
                for w in left..(left + width) {
                    let idx = c * img_height * img_width + h * img_width + w;
                    cropped.push(data[idx]);
                }
            }
        }

        let new_shape = if shape.len() == 3 {
            vec![channels, height, width]
        } else {
            vec![height, width]
        };

        Tensor::from_slice(&cropped, &new_shape).unwrap()
    }
}

/// Normalize
pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Normalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    /// ImageNet normalization
    pub fn imagenet() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }

    pub fn apply(&self, image: &Tensor) -> Tensor {
        let shape = image.shape().dims();
        let data = image.storage().as_slice::<f32>();
        
        let channels = if shape.len() == 3 { shape[0] } else { 1 };
        let pixels_per_channel = data.len() / channels;

        let mut normalized = Vec::with_capacity(data.len());

        for c in 0..channels {
            let mean = self.mean.get(c).copied().unwrap_or(0.0);
            let std = self.std.get(c).copied().unwrap_or(1.0);
            
            for i in 0..pixels_per_channel {
                let idx = c * pixels_per_channel + i;
                normalized.push((data[idx] - mean) / std);
            }
        }

        Tensor::from_slice(&normalized, shape).unwrap()
    }
}

/// Compose multiple augmentations
pub struct Compose {
    transforms: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
}

impl Compose {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn add<F>(mut self, transform: F) -> Self
    where
        F: Fn(&Tensor) -> Tensor + 'static,
    {
        self.transforms.push(Box::new(transform));
        self
    }

    pub fn apply(&self, image: &Tensor) -> Tensor {
        let mut result = image.clone();
        for transform in &self.transforms {
            result = transform(&result);
        }
        result
    }
}

impl Default for Compose {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_horizontal_flip() {
        let image = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let flip = RandomHorizontalFlip::new(1.0); // Always flip
        
        let flipped = flip.apply(&image);
        assert_eq!(flipped.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_random_crop() {
        let image = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();
        let crop = RandomCrop::new((2, 2));
        
        let cropped = crop.apply(&image);
        assert_eq!(cropped.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_normalize() {
        let image = Tensor::from_slice(&[0.5f32, 0.6, 0.7], &[3]).unwrap();
        let normalize = Normalize::new(vec![0.5], vec![0.1]);
        
        let normalized = normalize.apply(&image);
        assert_eq!(normalized.shape().dims(), &[3]);
        
        let data = normalized.storage().as_slice::<f32>();
        assert!((data[0] - 0.0).abs() < 0.01); // (0.5 - 0.5) / 0.1 = 0
    }

    #[test]
    fn test_compose() {
        let image = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        
        let flip = RandomHorizontalFlip::new(1.0);
        let normalize = Normalize::new(vec![0.5], vec![0.1]);
        
        let compose = Compose::new()
            .add(move |img| flip.apply(img))
            .add(move |img| normalize.apply(img));
        
        let result = compose.apply(&image);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }
}
