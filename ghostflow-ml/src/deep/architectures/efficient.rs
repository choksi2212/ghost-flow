//! Efficient Model Architectures - GhostNet, MnasNet, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, Dense, BatchNorm2d};
use crate::deep::activations::ReLU;

/// GhostNet
pub struct GhostNet {
    stem: Conv2d,
    blocks: Vec<GhostBottleneck>,
    fc: Dense,
}

struct GhostBottleneck {
    ghost1: GhostModule,
    ghost2: GhostModule,
    shortcut: Option<Conv2d>,
}

struct GhostModule {
    primary_conv: Conv2d,
    cheap_operation: Conv2d,
    bn1: BatchNorm2d,
    bn2: BatchNorm2d,
}

impl GhostModule {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        let primary_channels = out_channels / 2;
        GhostModule {
            primary_conv: Conv2d::new(in_channels, primary_channels, (1, 1)),
            cheap_operation: Conv2d::new(primary_channels, out_channels - primary_channels, (3, 3)).padding((1, 1)),
            bn1: BatchNorm2d::new(primary_channels),
            bn2: BatchNorm2d::new(out_channels - primary_channels),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut primary = self.primary_conv.forward(x, training);
        primary = self.bn1.forward(&primary, training);
        primary = ReLU::new().forward(&primary);
        
        let mut cheap = self.cheap_operation.forward(&primary, training);
        cheap = self.bn2.forward(&cheap, training);
        cheap = ReLU::new().forward(&cheap);
        
        self.concatenate(&primary, &cheap)
    }

    fn concatenate(&self, x1: &Tensor, x2: &Tensor) -> Tensor {
        let dims1 = x1.dims();
        let dims2 = x2.dims();
        let batch = dims1[0];
        let channels1 = dims1[1];
        let channels2 = dims2[1];
        let height = dims1[2];
        let width = dims1[3];

        let total_channels = channels1 + channels2;
        let mut result = Vec::new();

        for b in 0..batch {
            for c in 0..channels1 {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels1 + c) * height + h) * width + w;
                        result.push(x1.data_f32()[idx]);
                    }
                }
            }
            for c in 0..channels2 {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels2 + c) * height + h) * width + w;
                        result.push(x2.data_f32()[idx]);
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, total_channels, height, width]).unwrap()
    }
}

impl GhostBottleneck {
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let shortcut = if stride != 1 || in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, (1, 1)).stride((stride, stride)))
        } else {
            None
        };

        GhostBottleneck {
            ghost1: GhostModule::new(in_channels, out_channels),
            ghost2: GhostModule::new(out_channels, out_channels),
            shortcut,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = if let Some(ref mut shortcut) = self.shortcut {
            shortcut.forward(x, training)
        } else {
            x.clone()
        };

        let mut out = self.ghost1.forward(x, training);
        out = self.ghost2.forward(&out, training);
        
        self.add_tensors(&out, &identity)
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl GhostNet {
    pub fn new(num_classes: usize) -> Self {
        GhostNet {
            stem: Conv2d::new(3, 16, (3, 3)).stride((2, 2)).padding((1, 1)),
            blocks: vec![
                GhostBottleneck::new(16, 16, 1),
                GhostBottleneck::new(16, 24, 2),
                GhostBottleneck::new(24, 40, 2),
                GhostBottleneck::new(40, 80, 2),
                GhostBottleneck::new(80, 112, 1),
                GhostBottleneck::new(112, 160, 2),
            ],
            fc: Dense::new(160, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        out = ReLU::new().forward(&out);
        
        for block in &mut self.blocks {
            out = block.forward(&out, training);
        }
        
        self.fc.forward(&out, training)
    }
}

/// MnasNet
pub struct MnasNet {
    stem: Conv2d,
    blocks: Vec<MnasBlock>,
    fc: Dense,
}

struct MnasBlock {
    expand: Conv2d,
    depthwise: Conv2d,
    project: Conv2d,
    bn1: BatchNorm2d,
    bn2: BatchNorm2d,
    bn3: BatchNorm2d,
}

impl MnasBlock {
    fn new(in_channels: usize, out_channels: usize, expand_ratio: usize) -> Self {
        let expanded = in_channels * expand_ratio;
        MnasBlock {
            expand: Conv2d::new(in_channels, expanded, (1, 1)),
            depthwise: Conv2d::new(expanded, expanded, (3, 3)).padding((1, 1)),
            project: Conv2d::new(expanded, out_channels, (1, 1)),
            bn1: BatchNorm2d::new(expanded),
            bn2: BatchNorm2d::new(expanded),
            bn3: BatchNorm2d::new(out_channels),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.expand.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.depthwise.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.project.forward(&out, training);
        self.bn3.forward(&out, training)
    }
}

impl MnasNet {
    pub fn new(num_classes: usize) -> Self {
        MnasNet {
            stem: Conv2d::new(3, 32, (3, 3)).stride((2, 2)).padding((1, 1)),
            blocks: vec![
                MnasBlock::new(32, 16, 1),
                MnasBlock::new(16, 24, 6),
                MnasBlock::new(24, 40, 6),
                MnasBlock::new(40, 80, 6),
                MnasBlock::new(80, 96, 6),
                MnasBlock::new(96, 192, 6),
                MnasBlock::new(192, 320, 6),
            ],
            fc: Dense::new(320, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        out = ReLU::new().forward(&out);
        
        for block in &mut self.blocks {
            out = block.forward(&out, training);
        }
        
        self.fc.forward(&out, training)
    }
}

/// MobileNet v3
pub struct MobileNetV3 {
    stem: Conv2d,
    blocks: Vec<MobileNetV3Block>,
    fc: Dense,
}

struct MobileNetV3Block {
    expand: Conv2d,
    depthwise: Conv2d,
    se: SEModule,
    project: Conv2d,
    use_residual: bool,
}

struct SEModule {
    fc1: Dense,
    fc2: Dense,
}

impl SEModule {
    fn new(channels: usize, reduction: usize) -> Self {
        SEModule {
            fc1: Dense::new(channels, channels / reduction),
            fc2: Dense::new(channels / reduction, channels),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let squeezed = self.global_avg_pool(x);
        
        let mut out = self.fc1.forward(&squeezed, training);
        out = ReLU::new().forward(&out);
        
        out = self.fc2.forward(&out, training);
        out = self.sigmoid(&out);
        
        self.scale(x, &out)
    }

    fn global_avg_pool(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();
        
        let mut result = vec![0.0f32; batch * channels];
        
        for b in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        sum += data[idx];
                    }
                }
                result[b * channels + c] = sum / (height * width) as f32;
            }
        }
        
        Tensor::from_slice(&result, &[batch, channels]).unwrap()
    }

    fn sigmoid(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn scale(&self, x: &Tensor, scale: &Tensor) -> Tensor {
        x.clone() // Simplified
    }
}

impl MobileNetV3Block {
    fn new(in_channels: usize, out_channels: usize, expand_ratio: usize, use_residual: bool) -> Self {
        let expanded = in_channels * expand_ratio;
        MobileNetV3Block {
            expand: Conv2d::new(in_channels, expanded, (1, 1)),
            depthwise: Conv2d::new(expanded, expanded, (3, 3)).padding((1, 1)),
            se: SEModule::new(expanded, 4),
            project: Conv2d::new(expanded, out_channels, (1, 1)),
            use_residual,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        
        let mut out = self.expand.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.depthwise.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.se.forward(&out, training);
        
        out = self.project.forward(&out, training);
        
        if self.use_residual {
            self.add_tensors(&out, &identity)
        } else {
            out
        }
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl MobileNetV3 {
    pub fn new(num_classes: usize) -> Self {
        MobileNetV3 {
            stem: Conv2d::new(3, 16, (3, 3)).stride((2, 2)).padding((1, 1)),
            blocks: vec![
                MobileNetV3Block::new(16, 16, 1, true),
                MobileNetV3Block::new(16, 24, 4, false),
                MobileNetV3Block::new(24, 24, 3, true),
                MobileNetV3Block::new(24, 40, 3, false),
                MobileNetV3Block::new(40, 40, 3, true),
                MobileNetV3Block::new(40, 80, 6, false),
            ],
            fc: Dense::new(80, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        out = ReLU::new().forward(&out);
        
        for block in &mut self.blocks {
            out = block.forward(&out, training);
        }
        
        self.fc.forward(&out, training)
    }
}

/// RegNet
pub struct RegNet {
    stem: Conv2d,
    stages: Vec<RegNetStage>,
    fc: Dense,
}

struct RegNetStage {
    blocks: Vec<RegNetBlock>,
}

struct RegNetBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    bn1: BatchNorm2d,
    bn2: BatchNorm2d,
    bn3: BatchNorm2d,
}

impl RegNetBlock {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        let bottleneck_channels = out_channels / 4;
        RegNetBlock {
            conv1: Conv2d::new(in_channels, bottleneck_channels, (1, 1)),
            conv2: Conv2d::new(bottleneck_channels, bottleneck_channels, (3, 3)).padding((1, 1)),
            conv3: Conv2d::new(bottleneck_channels, out_channels, (1, 1)),
            bn1: BatchNorm2d::new(bottleneck_channels),
            bn2: BatchNorm2d::new(bottleneck_channels),
            bn3: BatchNorm2d::new(out_channels),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv3.forward(&out, training);
        self.bn3.forward(&out, training)
    }
}

impl RegNetStage {
    fn new(in_channels: usize, out_channels: usize, num_blocks: usize) -> Self {
        RegNetStage {
            blocks: (0..num_blocks).map(|_| RegNetBlock::new(in_channels, out_channels)).collect(),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        for block in &mut self.blocks {
            out = block.forward(&out, training);
        }
        out
    }
}

impl RegNet {
    pub fn new(num_classes: usize) -> Self {
        RegNet {
            stem: Conv2d::new(3, 32, (3, 3)).stride((2, 2)).padding((1, 1)),
            stages: vec![
                RegNetStage::new(32, 48, 2),
                RegNetStage::new(48, 120, 6),
                RegNetStage::new(120, 336, 17),
                RegNetStage::new(336, 888, 2),
            ],
            fc: Dense::new(888, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        out = ReLU::new().forward(&out);
        
        for stage in &mut self.stages {
            out = stage.forward(&out, training);
        }
        
        self.fc.forward(&out, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghostnet() {
        let mut model = GhostNet::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_mnasnet() {
        let mut model = MnasNet::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}


