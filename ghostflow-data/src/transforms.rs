//! Data transforms and augmentations

use ghostflow_core::Tensor;
use rand::Rng;

/// Trait for data transforms
pub trait Transform: Send + Sync {
    fn apply(&self, tensor: &Tensor) -> Tensor;
}

/// Normalize tensor with mean and std
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Normalize { mean, std }
    }

    /// ImageNet normalization
    pub fn imagenet() -> Self {
        Normalize {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }
}

impl Transform for Normalize {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let dims = tensor.dims();
        let data = tensor.data_f32();
        
        // Assume tensor is [C, H, W] or [N, C, H, W]
        let channels = if dims.len() == 3 { dims[0] } else { dims[1] };
        let spatial_size: usize = dims[dims.len()-2..].iter().product();
        
        let mut result = data.clone();
        
        if dims.len() == 3 {
            // [C, H, W]
            for c in 0..channels {
                let start = c * spatial_size;
                let end = start + spatial_size;
                for item in result.iter_mut().take(end).skip(start) {
                    *item = (*item - self.mean[c]) / self.std[c];
                }
            }
        } else {
            // [N, C, H, W]
            let batch_size = dims[0];
            let batch_stride = channels * spatial_size;
            
            for b in 0..batch_size {
                for c in 0..channels {
                    let start = b * batch_stride + c * spatial_size;
                    let end = start + spatial_size;
                    for item in result.iter_mut().take(end).skip(start) {
                        *item = (*item - self.mean[c]) / self.std[c];
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, dims).unwrap()
    }
}

/// Random horizontal flip
pub struct RandomHorizontalFlip {
    p: f32,
}

impl RandomHorizontalFlip {
    pub fn new(p: f32) -> Self {
        RandomHorizontalFlip { p }
    }
}

impl Default for RandomHorizontalFlip {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        if rand::thread_rng().gen::<f32>() > self.p {
            return tensor.clone();
        }

        let dims = tensor.dims();
        let data = tensor.data_f32();
        
        // Assume [C, H, W] or [N, C, H, W]
        let (height, width) = if dims.len() == 3 {
            (dims[1], dims[2])
        } else {
            (dims[2], dims[3])
        };
        
        let mut result = data.clone();
        
        // Flip along width dimension
        if dims.len() == 3 {
            let channels = dims[0];
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width / 2 {
                        let idx1 = c * height * width + h * width + w;
                        let idx2 = c * height * width + h * width + (width - 1 - w);
                        result.swap(idx1, idx2);
                    }
                }
            }
        }
        
        Tensor::from_slice(&result, dims).unwrap()
    }
}

/// Random crop
pub struct RandomCrop {
    size: (usize, usize),
}

impl RandomCrop {
    pub fn new(height: usize, width: usize) -> Self {
        RandomCrop { size: (height, width) }
    }
}

impl Transform for RandomCrop {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let dims = tensor.dims();
        let data = tensor.data_f32();
        
        let (channels, in_h, in_w) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            panic!("RandomCrop expects 3D tensor [C, H, W]");
        };
        
        let (out_h, out_w) = self.size;
        
        if in_h < out_h || in_w < out_w {
            panic!("Crop size larger than input");
        }
        
        let mut rng = rand::thread_rng();
        let top = rng.gen_range(0..=in_h - out_h);
        let left = rng.gen_range(0..=in_w - out_w);
        
        let mut result = Vec::with_capacity(channels * out_h * out_w);
        
        for c in 0..channels {
            for h in 0..out_h {
                for w in 0..out_w {
                    let src_idx = c * in_h * in_w + (top + h) * in_w + (left + w);
                    result.push(data[src_idx]);
                }
            }
        }
        
        Tensor::from_slice(&result, &[channels, out_h, out_w]).unwrap()
    }
}

/// Compose multiple transforms
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Compose { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut result = tensor.clone();
        for t in &self.transforms {
            result = t.apply(&result);
        }
        result
    }
}

/// Convert to tensor (identity for already-tensor data)
pub struct ToTensor;

impl Transform for ToTensor {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        tensor.clone()
    }
}

/// Random erasing augmentation
pub struct RandomErasing {
    p: f32,
    scale: (f32, f32),
    ratio: (f32, f32),
    value: f32,
}

impl RandomErasing {
    pub fn new(p: f32) -> Self {
        RandomErasing {
            p,
            scale: (0.02, 0.33),
            ratio: (0.3, 3.3),
            value: 0.0,
        }
    }
}

impl Default for RandomErasing {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Transform for RandomErasing {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        if rand::thread_rng().gen::<f32>() > self.p {
            return tensor.clone();
        }

        let dims = tensor.dims();
        let mut data = tensor.data_f32();
        
        let (channels, height, width) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            return tensor.clone();
        };
        
        let area = (height * width) as f32;
        let mut rng = rand::thread_rng();
        
        for _ in 0..10 {
            let target_area = rng.gen_range(self.scale.0..self.scale.1) * area;
            let aspect_ratio = rng.gen_range(self.ratio.0..self.ratio.1);
            
            let h = (target_area * aspect_ratio).sqrt() as usize;
            let w = (target_area / aspect_ratio).sqrt() as usize;
            
            if h < height && w < width {
                let top = rng.gen_range(0..height - h);
                let left = rng.gen_range(0..width - w);
                
                for c in 0..channels {
                    for y in top..top + h {
                        for x in left..left + w {
                            let idx = c * height * width + y * width + x;
                            data[idx] = self.value;
                        }
                    }
                }
                break;
            }
        }
        
        Tensor::from_slice(&data, dims).unwrap()
    }
}
