//! CNN Architectures - ResNet, VGG, Inception, DenseNet, MobileNet, EfficientNet, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, BatchNorm2d, Dense, MaxPool2d, AvgPool2d, GlobalAvgPool2d};
use crate::deep::activations::ReLU;

/// ResNet Block (Residual Block)
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
    stride: usize,
}

impl ResidualBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some((
                Conv2d::new(in_channels, out_channels, (1, 1)).stride((stride, stride)),
                BatchNorm2d::new(out_channels),
            ))
        } else {
            None
        };

        ResidualBlock {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3))
                .stride((stride, stride))
                .padding((1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3)).padding((1, 1)),
            bn2: BatchNorm2d::new(out_channels),
            downsample,
            stride,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();

        // First conv block
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);

        // Second conv block
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);

        // Downsample identity if needed
        let identity = if let Some((ref mut conv, ref mut bn)) = self.downsample {
            let mut id = conv.forward(&identity, training);
            id = bn.forward(&id, training);
            id
        } else {
            identity
        };

        // Add residual connection
        let out_data = out.data_f32();
        let id_data = identity.data_f32();
        let result: Vec<f32> = out_data.iter()
            .zip(id_data.iter())
            .map(|(&o, &i)| o + i)
            .collect();

        let result_tensor = Tensor::from_slice(&result, out.dims()).unwrap();
        ReLU::new().forward(&result_tensor)
    }
}

/// Bottleneck Block for ResNet-50/101/152
pub struct BottleneckBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
    expansion: usize,
}

impl BottleneckBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let expansion = 4;
        let downsample = if stride != 1 || in_channels != out_channels * expansion {
            Some((
                Conv2d::new(in_channels, out_channels * expansion, (1, 1))
                    .stride((stride, stride)),
                BatchNorm2d::new(out_channels * expansion),
            ))
        } else {
            None
        };

        BottleneckBlock {
            conv1: Conv2d::new(in_channels, out_channels, (1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3))
                .stride((stride, stride))
                .padding((1, 1)),
            bn2: BatchNorm2d::new(out_channels),
            conv3: Conv2d::new(out_channels, out_channels * expansion, (1, 1)),
            bn3: BatchNorm2d::new(out_channels * expansion),
            downsample,
            expansion,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();

        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.conv3.forward(&out, training);
        out = self.bn3.forward(&out, training);

        let identity = if let Some((ref mut conv, ref mut bn)) = self.downsample {
            let mut id = conv.forward(&identity, training);
            bn.forward(&id, training)
        } else {
            identity
        };

        let out_data = out.data_f32();
        let id_data = identity.data_f32();
        let result: Vec<f32> = out_data.iter()
            .zip(id_data.iter())
            .map(|(&o, &i)| o + i)
            .collect();

        let result_tensor = Tensor::from_slice(&result, out.dims()).unwrap();
        ReLU::new().forward(&result_tensor)
    }
}

/// ResNet-18
pub struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    layer1: Vec<ResidualBlock>,
    layer2: Vec<ResidualBlock>,
    layer3: Vec<ResidualBlock>,
    layer4: Vec<ResidualBlock>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl ResNet18 {
    pub fn new(num_classes: usize) -> Self {
        ResNet18 {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(64),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layer1: vec![
                ResidualBlock::new(64, 64, 1),
                ResidualBlock::new(64, 64, 1),
            ],
            layer2: vec![
                ResidualBlock::new(64, 128, 2),
                ResidualBlock::new(128, 128, 1),
            ],
            layer3: vec![
                ResidualBlock::new(128, 256, 2),
                ResidualBlock::new(256, 256, 1),
            ],
            layer4: vec![
                ResidualBlock::new(256, 512, 2),
                ResidualBlock::new(512, 512, 1),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(512, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        for block in &mut self.layer1 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer2 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer3 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer4 {
            out = block.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// ResNet-34
pub struct ResNet34 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    layer1: Vec<ResidualBlock>,
    layer2: Vec<ResidualBlock>,
    layer3: Vec<ResidualBlock>,
    layer4: Vec<ResidualBlock>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl ResNet34 {
    pub fn new(num_classes: usize) -> Self {
        ResNet34 {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(64),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layer1: vec![
                ResidualBlock::new(64, 64, 1),
                ResidualBlock::new(64, 64, 1),
                ResidualBlock::new(64, 64, 1),
            ],
            layer2: vec![
                ResidualBlock::new(64, 128, 2),
                ResidualBlock::new(128, 128, 1),
                ResidualBlock::new(128, 128, 1),
                ResidualBlock::new(128, 128, 1),
            ],
            layer3: vec![
                ResidualBlock::new(128, 256, 2),
                ResidualBlock::new(256, 256, 1),
                ResidualBlock::new(256, 256, 1),
                ResidualBlock::new(256, 256, 1),
                ResidualBlock::new(256, 256, 1),
                ResidualBlock::new(256, 256, 1),
            ],
            layer4: vec![
                ResidualBlock::new(256, 512, 2),
                ResidualBlock::new(512, 512, 1),
                ResidualBlock::new(512, 512, 1),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(512, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        for block in &mut self.layer1 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer2 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer3 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer4 {
            out = block.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// ResNet-50
pub struct ResNet50 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    layer1: Vec<BottleneckBlock>,
    layer2: Vec<BottleneckBlock>,
    layer3: Vec<BottleneckBlock>,
    layer4: Vec<BottleneckBlock>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl ResNet50 {
    pub fn new(num_classes: usize) -> Self {
        ResNet50 {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(64),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layer1: vec![
                BottleneckBlock::new(64, 64, 1),
                BottleneckBlock::new(256, 64, 1),
                BottleneckBlock::new(256, 64, 1),
            ],
            layer2: vec![
                BottleneckBlock::new(256, 128, 2),
                BottleneckBlock::new(512, 128, 1),
                BottleneckBlock::new(512, 128, 1),
                BottleneckBlock::new(512, 128, 1),
            ],
            layer3: vec![
                BottleneckBlock::new(512, 256, 2),
                BottleneckBlock::new(1024, 256, 1),
                BottleneckBlock::new(1024, 256, 1),
                BottleneckBlock::new(1024, 256, 1),
                BottleneckBlock::new(1024, 256, 1),
                BottleneckBlock::new(1024, 256, 1),
            ],
            layer4: vec![
                BottleneckBlock::new(1024, 512, 2),
                BottleneckBlock::new(2048, 512, 1),
                BottleneckBlock::new(2048, 512, 1),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(2048, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        for block in &mut self.layer1 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer2 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer3 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer4 {
            out = block.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// VGG-16
pub struct VGG16 {
    features: Vec<Box<dyn Fn(&Tensor, bool) -> Tensor>>,
    classifier: Vec<Dense>,
}

impl VGG16 {
    pub fn new(num_classes: usize) -> Self {
        // VGG16 architecture will be implemented with proper layer structure
        // This is a placeholder showing the structure
        VGG16 {
            features: Vec::new(),
            classifier: vec![
                Dense::new(512 * 7 * 7, 4096),
                Dense::new(4096, 4096),
                Dense::new(4096, num_classes),
            ],
        }
    }
}

/// VGG-19
pub struct VGG19 {
    conv_blocks: Vec<Vec<Conv2d>>,
    bn_blocks: Vec<Vec<BatchNorm2d>>,
    pools: Vec<MaxPool2d>,
    classifier: Vec<Dense>,
}

impl VGG19 {
    pub fn new(num_classes: usize) -> Self {
        VGG19 {
            conv_blocks: vec![
                vec![Conv2d::new(3, 64, (3, 3)).padding((1, 1)), Conv2d::new(64, 64, (3, 3)).padding((1, 1))],
                vec![Conv2d::new(64, 128, (3, 3)).padding((1, 1)), Conv2d::new(128, 128, (3, 3)).padding((1, 1))],
                vec![
                    Conv2d::new(128, 256, (3, 3)).padding((1, 1)),
                    Conv2d::new(256, 256, (3, 3)).padding((1, 1)),
                    Conv2d::new(256, 256, (3, 3)).padding((1, 1)),
                    Conv2d::new(256, 256, (3, 3)).padding((1, 1)),
                ],
                vec![
                    Conv2d::new(256, 512, (3, 3)).padding((1, 1)),
                    Conv2d::new(512, 512, (3, 3)).padding((1, 1)),
                    Conv2d::new(512, 512, (3, 3)).padding((1, 1)),
                    Conv2d::new(512, 512, (3, 3)).padding((1, 1)),
                ],
                vec![
                    Conv2d::new(512, 512, (3, 3)).padding((1, 1)),
                    Conv2d::new(512, 512, (3, 3)).padding((1, 1)),
                    Conv2d::new(512, 512, (3, 3)).padding((1, 1)),
                    Conv2d::new(512, 512, (3, 3)).padding((1, 1)),
                ],
            ],
            bn_blocks: vec![
                vec![BatchNorm2d::new(64), BatchNorm2d::new(64)],
                vec![BatchNorm2d::new(128), BatchNorm2d::new(128)],
                vec![BatchNorm2d::new(256), BatchNorm2d::new(256), BatchNorm2d::new(256), BatchNorm2d::new(256)],
                vec![BatchNorm2d::new(512), BatchNorm2d::new(512), BatchNorm2d::new(512), BatchNorm2d::new(512)],
                vec![BatchNorm2d::new(512), BatchNorm2d::new(512), BatchNorm2d::new(512), BatchNorm2d::new(512)],
            ],
            pools: vec![
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
            ],
            classifier: vec![
                Dense::new(512 * 7 * 7, 4096),
                Dense::new(4096, 4096),
                Dense::new(4096, num_classes),
            ],
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();

        for (i, conv_block) in self.conv_blocks.iter_mut().enumerate() {
            for (j, conv) in conv_block.iter_mut().enumerate() {
                out = conv.forward(&out, training);
                out = self.bn_blocks[i][j].forward(&out, training);
                out = ReLU::new().forward(&out);
            }
            out = self.pools[i].forward(&out);
        }

        // Flatten
        let batch_size = out.dims()[0];
        let flat_size = out.data_f32().len() / batch_size;
        out = Tensor::from_slice(out.data_f32(), &[batch_size, flat_size]).unwrap();

        // Classifier
        for (i, fc) in self.classifier.iter_mut().enumerate() {
            out = fc.forward(&out, training);
            if i < self.classifier.len() - 1 {
                out = ReLU::new().forward(&out);
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_block() {
        let mut block = ResidualBlock::new(64, 64, 1);
        let input = Tensor::from_slice(&vec![0.5f32; 64 * 32 * 32], &[1, 64, 32, 32]).unwrap();
        let output = block.forward(&input, true);
        assert_eq!(output.dims()[1], 64);
    }

    #[test]
    fn test_resnet18() {
        let mut model = ResNet18::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}


/// Inception Module (Inception v1 / GoogLeNet)
pub struct InceptionModule {
    branch1x1: Conv2d,
    branch3x3_reduce: Conv2d,
    branch3x3: Conv2d,
    branch5x5_reduce: Conv2d,
    branch5x5: Conv2d,
    branch_pool: MaxPool2d,
    branch_pool_proj: Conv2d,
}

impl InceptionModule {
    pub fn new(in_channels: usize, ch1x1: usize, ch3x3_reduce: usize, ch3x3: usize,
               ch5x5_reduce: usize, ch5x5: usize, pool_proj: usize) -> Self {
        InceptionModule {
            branch1x1: Conv2d::new(in_channels, ch1x1, (1, 1)),
            branch3x3_reduce: Conv2d::new(in_channels, ch3x3_reduce, (1, 1)),
            branch3x3: Conv2d::new(ch3x3_reduce, ch3x3, (3, 3)).padding((1, 1)),
            branch5x5_reduce: Conv2d::new(in_channels, ch5x5_reduce, (1, 1)),
            branch5x5: Conv2d::new(ch5x5_reduce, ch5x5, (5, 5)).padding((2, 2)),
            branch_pool: MaxPool2d::new((3, 3), (1, 1), (1, 1)),
            branch_pool_proj: Conv2d::new(in_channels, pool_proj, (1, 1)),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // Branch 1: 1x1 conv
        let branch1 = self.branch1x1.forward(x, training);
        let branch1 = ReLU::new().forward(&branch1);

        // Branch 2: 1x1 conv -> 3x3 conv
        let mut branch2 = self.branch3x3_reduce.forward(x, training);
        branch2 = ReLU::new().forward(&branch2);
        branch2 = self.branch3x3.forward(&branch2, training);
        let branch2 = ReLU::new().forward(&branch2);

        // Branch 3: 1x1 conv -> 5x5 conv
        let mut branch3 = self.branch5x5_reduce.forward(x, training);
        branch3 = ReLU::new().forward(&branch3);
        branch3 = self.branch5x5.forward(&branch3, training);
        let branch3 = ReLU::new().forward(&branch3);

        // Branch 4: 3x3 pool -> 1x1 conv
        let mut branch4 = self.branch_pool.forward(x);
        branch4 = self.branch_pool_proj.forward(&branch4, training);
        let branch4 = ReLU::new().forward(&branch4);

        // Concatenate along channel dimension
        self.concatenate_channels(&[branch1, branch2, branch3, branch4])
    }

    fn concatenate_channels(&self, tensors: &[Tensor]) -> Tensor {
        let batch_size = tensors[0].dims()[0];
        let height = tensors[0].dims()[2];
        let width = tensors[0].dims()[3];
        let total_channels: usize = tensors.iter().map(|t| t.dims()[1]).sum();

        let mut result = Vec::new();
        for b in 0..batch_size {
            for tensor in tensors {
                let channels = tensor.dims()[1];
                let data = tensor.data_f32();
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            result.push(data[idx]);
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch_size, total_channels, height, width]).unwrap()
    }
}

/// GoogLeNet (Inception v1)
pub struct GoogLeNet {
    conv1: Conv2d,
    maxpool1: MaxPool2d,
    conv2: Conv2d,
    conv3: Conv2d,
    maxpool2: MaxPool2d,
    inception3a: InceptionModule,
    inception3b: InceptionModule,
    maxpool3: MaxPool2d,
    inception4a: InceptionModule,
    inception4b: InceptionModule,
    inception4c: InceptionModule,
    inception4d: InceptionModule,
    inception4e: InceptionModule,
    maxpool4: MaxPool2d,
    inception5a: InceptionModule,
    inception5b: InceptionModule,
    avgpool: GlobalAvgPool2d,
    dropout: f32,
    fc: Dense,
}

impl GoogLeNet {
    pub fn new(num_classes: usize) -> Self {
        GoogLeNet {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            maxpool1: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            conv2: Conv2d::new(64, 64, (1, 1)),
            conv3: Conv2d::new(64, 192, (3, 3)).padding((1, 1)),
            maxpool2: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            inception3a: InceptionModule::new(192, 64, 96, 128, 16, 32, 32),
            inception3b: InceptionModule::new(256, 128, 128, 192, 32, 96, 64),
            maxpool3: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            inception4a: InceptionModule::new(480, 192, 96, 208, 16, 48, 64),
            inception4b: InceptionModule::new(512, 160, 112, 224, 24, 64, 64),
            inception4c: InceptionModule::new(512, 128, 128, 256, 24, 64, 64),
            inception4d: InceptionModule::new(512, 112, 144, 288, 32, 64, 64),
            inception4e: InceptionModule::new(528, 256, 160, 320, 32, 128, 128),
            maxpool4: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            inception5a: InceptionModule::new(832, 256, 160, 320, 32, 128, 128),
            inception5b: InceptionModule::new(832, 384, 192, 384, 48, 128, 128),
            avgpool: GlobalAvgPool2d::new(),
            dropout: 0.4,
            fc: Dense::new(1024, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool1.forward(&out);

        out = self.conv2.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.conv3.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool2.forward(&out);

        out = self.inception3a.forward(&out, training);
        out = self.inception3b.forward(&out, training);
        out = self.maxpool3.forward(&out);

        out = self.inception4a.forward(&out, training);
        out = self.inception4b.forward(&out, training);
        out = self.inception4c.forward(&out, training);
        out = self.inception4d.forward(&out, training);
        out = self.inception4e.forward(&out, training);
        out = self.maxpool4.forward(&out);

        out = self.inception5a.forward(&out, training);
        out = self.inception5b.forward(&out, training);

        out = self.avgpool.forward(&out);
        
        // Apply dropout if training
        if training {
            out = self.apply_dropout(&out, self.dropout);
        }

        self.fc.forward(&out, training)
    }

    fn apply_dropout(&self, x: &Tensor, p: f32) -> Tensor {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Dense Block for DenseNet
pub struct DenseBlock {
    layers: Vec<(BatchNorm2d, Conv2d, BatchNorm2d, Conv2d)>,
    growth_rate: usize,
}

impl DenseBlock {
    pub fn new(num_layers: usize, in_channels: usize, growth_rate: usize) -> Self {
        let mut layers = Vec::new();
        let mut current_channels = in_channels;

        for _ in 0..num_layers {
            layers.push((
                BatchNorm2d::new(current_channels),
                Conv2d::new(current_channels, growth_rate * 4, (1, 1)),
                BatchNorm2d::new(growth_rate * 4),
                Conv2d::new(growth_rate * 4, growth_rate, (3, 3)).padding((1, 1)),
            ));
            current_channels += growth_rate;
        }

        DenseBlock { layers, growth_rate }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut features = vec![x.clone()];

        for (bn1, conv1, bn2, conv2) in &mut self.layers {
            // Concatenate all previous features
            let concat = self.concatenate_features(&features);

            // Bottleneck layer
            let mut out = bn1.forward(&concat, training);
            out = ReLU::new().forward(&out);
            out = conv1.forward(&out, training);

            // 3x3 conv
            out = bn2.forward(&out, training);
            out = ReLU::new().forward(&out);
            out = conv2.forward(&out, training);

            features.push(out);
        }

        self.concatenate_features(&features)
    }

    fn concatenate_features(&self, features: &[Tensor]) -> Tensor {
        if features.len() == 1 {
            return features[0].clone();
        }

        let batch_size = features[0].dims()[0];
        let height = features[0].dims()[2];
        let width = features[0].dims()[3];
        let total_channels: usize = features.iter().map(|f| f.dims()[1]).sum();

        let mut result = Vec::new();
        for b in 0..batch_size {
            for feature in features {
                let channels = feature.dims()[1];
                let data = feature.data_f32();
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            result.push(data[idx]);
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch_size, total_channels, height, width]).unwrap()
    }
}

/// Transition Layer for DenseNet
pub struct TransitionLayer {
    bn: BatchNorm2d,
    conv: Conv2d,
    pool: AvgPool2d,
}

impl TransitionLayer {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        TransitionLayer {
            bn: BatchNorm2d::new(in_channels),
            conv: Conv2d::new(in_channels, out_channels, (1, 1)),
            pool: AvgPool2d::new((2, 2), (2, 2)),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.bn.forward(x, training);
        out = ReLU::new().forward(&out);
        out = self.conv.forward(&out, training);
        self.pool.forward(&out)
    }
}

/// DenseNet-121
pub struct DenseNet121 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    dense1: DenseBlock,
    trans1: TransitionLayer,
    dense2: DenseBlock,
    trans2: TransitionLayer,
    dense3: DenseBlock,
    trans3: TransitionLayer,
    dense4: DenseBlock,
    bn_final: BatchNorm2d,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl DenseNet121 {
    pub fn new(num_classes: usize) -> Self {
        let growth_rate = 32;
        let num_init_features = 64;

        DenseNet121 {
            conv1: Conv2d::new(3, num_init_features, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(num_init_features),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            dense1: DenseBlock::new(6, num_init_features, growth_rate),
            trans1: TransitionLayer::new(num_init_features + 6 * growth_rate, 
                                        (num_init_features + 6 * growth_rate) / 2),
            dense2: DenseBlock::new(12, (num_init_features + 6 * growth_rate) / 2, growth_rate),
            trans2: TransitionLayer::new((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate,
                                        ((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate) / 2),
            dense3: DenseBlock::new(24, ((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate) / 2, growth_rate),
            trans3: TransitionLayer::new(((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate) / 2 + 24 * growth_rate,
                                        (((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate) / 2 + 24 * growth_rate) / 2),
            dense4: DenseBlock::new(16, (((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate) / 2 + 24 * growth_rate) / 2, growth_rate),
            bn_final: BatchNorm2d::new((((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate) / 2 + 24 * growth_rate) / 2 + 16 * growth_rate),
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new((((num_init_features + 6 * growth_rate) / 2 + 12 * growth_rate) / 2 + 24 * growth_rate) / 2 + 16 * growth_rate, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        out = self.dense1.forward(&out, training);
        out = self.trans1.forward(&out, training);

        out = self.dense2.forward(&out, training);
        out = self.trans2.forward(&out, training);

        out = self.dense3.forward(&out, training);
        out = self.trans3.forward(&out, training);

        out = self.dense4.forward(&out, training);

        out = self.bn_final.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.avgpool.forward(&out);

        self.fc.forward(&out, training)
    }
}

/// Depthwise Separable Convolution for MobileNet
pub struct DepthwiseSeparableConv {
    depthwise: Conv2d,
    pointwise: Conv2d,
    bn1: BatchNorm2d,
    bn2: BatchNorm2d,
}

impl DepthwiseSeparableConv {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        DepthwiseSeparableConv {
            depthwise: Conv2d::new(in_channels, in_channels, (3, 3))
                .stride((stride, stride))
                .padding((1, 1)),
            pointwise: Conv2d::new(in_channels, out_channels, (1, 1)),
            bn1: BatchNorm2d::new(in_channels),
            bn2: BatchNorm2d::new(out_channels),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.depthwise.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.pointwise.forward(&out, training);
        out = self.bn2.forward(&out, training);
        ReLU::new().forward(&out)
    }
}

/// MobileNet v1
pub struct MobileNetV1 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layers: Vec<DepthwiseSeparableConv>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl MobileNetV1 {
    pub fn new(num_classes: usize) -> Self {
        MobileNetV1 {
            conv1: Conv2d::new(3, 32, (3, 3)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(32),
            layers: vec![
                DepthwiseSeparableConv::new(32, 64, 1),
                DepthwiseSeparableConv::new(64, 128, 2),
                DepthwiseSeparableConv::new(128, 128, 1),
                DepthwiseSeparableConv::new(128, 256, 2),
                DepthwiseSeparableConv::new(256, 256, 1),
                DepthwiseSeparableConv::new(256, 512, 2),
                DepthwiseSeparableConv::new(512, 512, 1),
                DepthwiseSeparableConv::new(512, 512, 1),
                DepthwiseSeparableConv::new(512, 512, 1),
                DepthwiseSeparableConv::new(512, 512, 1),
                DepthwiseSeparableConv::new(512, 512, 1),
                DepthwiseSeparableConv::new(512, 1024, 2),
                DepthwiseSeparableConv::new(1024, 1024, 1),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(1024, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);

        for layer in &mut self.layers {
            out = layer.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// Inverted Residual Block for MobileNet v2
pub struct InvertedResidual {
    use_res_connect: bool,
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    stride: usize,
}

impl InvertedResidual {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, expand_ratio: usize) -> Self {
        let hidden_dim = in_channels * expand_ratio;
        let use_res_connect = stride == 1 && in_channels == out_channels;

        InvertedResidual {
            use_res_connect,
            conv1: Conv2d::new(in_channels, hidden_dim, (1, 1)),
            bn1: BatchNorm2d::new(hidden_dim),
            conv2: Conv2d::new(hidden_dim, hidden_dim, (3, 3))
                .stride((stride, stride))
                .padding((1, 1)),
            bn2: BatchNorm2d::new(hidden_dim),
            conv3: Conv2d::new(hidden_dim, out_channels, (1, 1)),
            bn3: BatchNorm2d::new(out_channels),
            stride,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();

        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.conv3.forward(&out, training);
        out = self.bn3.forward(&out, training);

        if self.use_res_connect {
            let out_data = out.data_f32();
            let id_data = identity.data_f32();
            let result: Vec<f32> = out_data.iter()
                .zip(id_data.iter())
                .map(|(&o, &i)| o + i)
                .collect();
            Tensor::from_slice(&result, out.dims()).unwrap()
        } else {
            out
        }
    }
}

/// MobileNet v2
pub struct MobileNetV2 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layers: Vec<InvertedResidual>,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl MobileNetV2 {
    pub fn new(num_classes: usize) -> Self {
        MobileNetV2 {
            conv1: Conv2d::new(3, 32, (3, 3)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(32),
            layers: vec![
                InvertedResidual::new(32, 16, 1, 1),
                InvertedResidual::new(16, 24, 2, 6),
                InvertedResidual::new(24, 24, 1, 6),
                InvertedResidual::new(24, 32, 2, 6),
                InvertedResidual::new(32, 32, 1, 6),
                InvertedResidual::new(32, 32, 1, 6),
                InvertedResidual::new(32, 64, 2, 6),
                InvertedResidual::new(64, 64, 1, 6),
                InvertedResidual::new(64, 64, 1, 6),
                InvertedResidual::new(64, 64, 1, 6),
                InvertedResidual::new(64, 96, 1, 6),
                InvertedResidual::new(96, 96, 1, 6),
                InvertedResidual::new(96, 96, 1, 6),
                InvertedResidual::new(96, 160, 2, 6),
                InvertedResidual::new(160, 160, 1, 6),
                InvertedResidual::new(160, 160, 1, 6),
                InvertedResidual::new(160, 320, 1, 6),
            ],
            conv2: Conv2d::new(320, 1280, (1, 1)),
            bn2: BatchNorm2d::new(1280),
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(1280, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);

        for layer in &mut self.layers {
            out = layer.forward(&out, training);
        }

        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

#[cfg(test)]
mod tests_extended {
    use super::*;

    #[test]
    fn test_inception_module() {
        let mut module = InceptionModule::new(192, 64, 96, 128, 16, 32, 32);
        let input = Tensor::from_slice(&vec![0.5f32; 192 * 28 * 28], &[1, 192, 28, 28]).unwrap();
        let output = module.forward(&input, true);
        assert_eq!(output.dims()[0], 1);
    }

    #[test]
    fn test_googlenet() {
        let mut model = GoogLeNet::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_densenet121() {
        let mut model = DenseNet121::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_mobilenetv1() {
        let mut model = MobileNetV1::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_mobilenetv2() {
        let mut model = MobileNetV2::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}


/// Squeeze-and-Excitation Block
pub struct SEBlock {
    fc1: Dense,
    fc2: Dense,
    reduction: usize,
}

impl SEBlock {
    pub fn new(channels: usize, reduction: usize) -> Self {
        SEBlock {
            fc1: Dense::new(channels, channels / reduction),
            fc2: Dense::new(channels / reduction, channels),
            reduction,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let batch_size = x.dims()[0];
        let channels = x.dims()[1];
        let height = x.dims()[2];
        let width = x.dims()[3];

        // Global average pooling
        let mut pooled = vec![0.0f32; batch_size * channels];
        let data = x.data_f32();
        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        sum += data[idx];
                    }
                }
                pooled[b * channels + c] = sum / (height * width) as f32;
            }
        }

        let pooled_tensor = Tensor::from_slice(&pooled, &[batch_size, channels]).unwrap();

        // Excitation
        let mut out = self.fc1.forward(&pooled_tensor, training);
        out = ReLU::new().forward(&out);
        out = self.fc2.forward(&out, training);
        
        // Sigmoid activation
        let out_data = out.data_f32();
        let sigmoid: Vec<f32> = out_data.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        // Scale original input
        let mut result = vec![0.0f32; data.len()];
        for b in 0..batch_size {
            for c in 0..channels {
                let scale = sigmoid[b * channels + c];
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        result[idx] = data[idx] * scale;
                    }
                }
            }
        }

        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// MBConv Block for EfficientNet
pub struct MBConvBlock {
    expand_conv: Option<Conv2d>,
    expand_bn: Option<BatchNorm2d>,
    depthwise_conv: Conv2d,
    depthwise_bn: BatchNorm2d,
    se: Option<SEBlock>,
    project_conv: Conv2d,
    project_bn: BatchNorm2d,
    use_res_connect: bool,
}

impl MBConvBlock {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize,
               stride: usize, expand_ratio: usize, se_ratio: f32) -> Self {
        let expanded_channels = in_channels * expand_ratio;
        let use_res_connect = stride == 1 && in_channels == out_channels;

        let (expand_conv, expand_bn) = if expand_ratio != 1 {
            (
                Some(Conv2d::new(in_channels, expanded_channels, (1, 1))),
                Some(BatchNorm2d::new(expanded_channels)),
            )
        } else {
            (None, None)
        };

        let se = if se_ratio > 0.0 {
            let se_channels = (in_channels as f32 * se_ratio).max(1.0) as usize;
            Some(SEBlock::new(expanded_channels, expanded_channels / se_channels))
        } else {
            None
        };

        MBConvBlock {
            expand_conv,
            expand_bn,
            depthwise_conv: Conv2d::new(expanded_channels, expanded_channels, (kernel_size, kernel_size))
                .stride((stride, stride))
                .padding((kernel_size / 2, kernel_size / 2)),
            depthwise_bn: BatchNorm2d::new(expanded_channels),
            se,
            project_conv: Conv2d::new(expanded_channels, out_channels, (1, 1)),
            project_bn: BatchNorm2d::new(out_channels),
            use_res_connect,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        let mut out = x.clone();

        // Expansion phase
        if let (Some(ref mut conv), Some(ref mut bn)) = (&mut self.expand_conv, &mut self.expand_bn) {
            out = conv.forward(&out, training);
            out = bn.forward(&out, training);
            out = ReLU::new().forward(&out);
        }

        // Depthwise convolution
        out = self.depthwise_conv.forward(&out, training);
        out = self.depthwise_bn.forward(&out, training);
        out = ReLU::new().forward(&out);

        // Squeeze and Excitation
        if let Some(ref mut se) = self.se {
            out = se.forward(&out, training);
        }

        // Projection phase
        out = self.project_conv.forward(&out, training);
        out = self.project_bn.forward(&out, training);

        // Residual connection
        if self.use_res_connect {
            let out_data = out.data_f32();
            let id_data = identity.data_f32();
            let result: Vec<f32> = out_data.iter()
                .zip(id_data.iter())
                .map(|(&o, &i)| o + i)
                .collect();
            Tensor::from_slice(&result, out.dims()).unwrap()
        } else {
            out
        }
    }
}

/// EfficientNet-B0
pub struct EfficientNetB0 {
    stem: Conv2d,
    stem_bn: BatchNorm2d,
    blocks: Vec<MBConvBlock>,
    head: Conv2d,
    head_bn: BatchNorm2d,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl EfficientNetB0 {
    pub fn new(num_classes: usize) -> Self {
        EfficientNetB0 {
            stem: Conv2d::new(3, 32, (3, 3)).stride((2, 2)).padding((1, 1)),
            stem_bn: BatchNorm2d::new(32),
            blocks: vec![
                // Stage 1
                MBConvBlock::new(32, 16, 3, 1, 1, 0.25),
                // Stage 2
                MBConvBlock::new(16, 24, 3, 2, 6, 0.25),
                MBConvBlock::new(24, 24, 3, 1, 6, 0.25),
                // Stage 3
                MBConvBlock::new(24, 40, 5, 2, 6, 0.25),
                MBConvBlock::new(40, 40, 5, 1, 6, 0.25),
                // Stage 4
                MBConvBlock::new(40, 80, 3, 2, 6, 0.25),
                MBConvBlock::new(80, 80, 3, 1, 6, 0.25),
                MBConvBlock::new(80, 80, 3, 1, 6, 0.25),
                // Stage 5
                MBConvBlock::new(80, 112, 5, 1, 6, 0.25),
                MBConvBlock::new(112, 112, 5, 1, 6, 0.25),
                MBConvBlock::new(112, 112, 5, 1, 6, 0.25),
                // Stage 6
                MBConvBlock::new(112, 192, 5, 2, 6, 0.25),
                MBConvBlock::new(192, 192, 5, 1, 6, 0.25),
                MBConvBlock::new(192, 192, 5, 1, 6, 0.25),
                MBConvBlock::new(192, 192, 5, 1, 6, 0.25),
                // Stage 7
                MBConvBlock::new(192, 320, 3, 1, 6, 0.25),
            ],
            head: Conv2d::new(320, 1280, (1, 1)),
            head_bn: BatchNorm2d::new(1280),
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(1280, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        out = self.stem_bn.forward(&out, training);
        out = ReLU::new().forward(&out);

        for block in &mut self.blocks {
            out = block.forward(&out, training);
        }

        out = self.head.forward(&out, training);
        out = self.head_bn.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// Fire Module for SqueezeNet
pub struct FireModule {
    squeeze: Conv2d,
    squeeze_bn: BatchNorm2d,
    expand1x1: Conv2d,
    expand1x1_bn: BatchNorm2d,
    expand3x3: Conv2d,
    expand3x3_bn: BatchNorm2d,
}

impl FireModule {
    pub fn new(in_channels: usize, squeeze_channels: usize, expand_channels: usize) -> Self {
        FireModule {
            squeeze: Conv2d::new(in_channels, squeeze_channels, (1, 1)),
            squeeze_bn: BatchNorm2d::new(squeeze_channels),
            expand1x1: Conv2d::new(squeeze_channels, expand_channels, (1, 1)),
            expand1x1_bn: BatchNorm2d::new(expand_channels),
            expand3x3: Conv2d::new(squeeze_channels, expand_channels, (3, 3)).padding((1, 1)),
            expand3x3_bn: BatchNorm2d::new(expand_channels),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // Squeeze
        let mut out = self.squeeze.forward(x, training);
        out = self.squeeze_bn.forward(&out, training);
        out = ReLU::new().forward(&out);

        // Expand 1x1
        let mut expand1 = self.expand1x1.forward(&out, training);
        expand1 = self.expand1x1_bn.forward(&expand1, training);
        expand1 = ReLU::new().forward(&expand1);

        // Expand 3x3
        let mut expand3 = self.expand3x3.forward(&out, training);
        expand3 = self.expand3x3_bn.forward(&expand3, training);
        expand3 = ReLU::new().forward(&expand3);

        // Concatenate
        self.concatenate_channels(&[expand1, expand3])
    }

    fn concatenate_channels(&self, tensors: &[Tensor]) -> Tensor {
        let batch_size = tensors[0].dims()[0];
        let height = tensors[0].dims()[2];
        let width = tensors[0].dims()[3];
        let total_channels: usize = tensors.iter().map(|t| t.dims()[1]).sum();

        let mut result = Vec::new();
        for b in 0..batch_size {
            for tensor in tensors {
                let channels = tensor.dims()[1];
                let data = tensor.data_f32();
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            result.push(data[idx]);
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch_size, total_channels, height, width]).unwrap()
    }
}

/// SqueezeNet v1.1
pub struct SqueezeNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool1: MaxPool2d,
    fire2: FireModule,
    fire3: FireModule,
    maxpool2: MaxPool2d,
    fire4: FireModule,
    fire5: FireModule,
    maxpool3: MaxPool2d,
    fire6: FireModule,
    fire7: FireModule,
    fire8: FireModule,
    fire9: FireModule,
    conv10: Conv2d,
    avgpool: GlobalAvgPool2d,
}

impl SqueezeNet {
    pub fn new(num_classes: usize) -> Self {
        SqueezeNet {
            conv1: Conv2d::new(3, 64, (3, 3)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(64),
            maxpool1: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            fire2: FireModule::new(64, 16, 64),
            fire3: FireModule::new(128, 16, 64),
            maxpool2: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            fire4: FireModule::new(128, 32, 128),
            fire5: FireModule::new(256, 32, 128),
            maxpool3: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            fire6: FireModule::new(256, 48, 192),
            fire7: FireModule::new(384, 48, 192),
            fire8: FireModule::new(384, 64, 256),
            fire9: FireModule::new(512, 64, 256),
            conv10: Conv2d::new(512, num_classes, (1, 1)),
            avgpool: GlobalAvgPool2d::new(),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool1.forward(&out);

        out = self.fire2.forward(&out, training);
        out = self.fire3.forward(&out, training);
        out = self.maxpool2.forward(&out);

        out = self.fire4.forward(&out, training);
        out = self.fire5.forward(&out, training);
        out = self.maxpool3.forward(&out);

        out = self.fire6.forward(&out, training);
        out = self.fire7.forward(&out, training);
        out = self.fire8.forward(&out, training);
        out = self.fire9.forward(&out, training);

        out = self.conv10.forward(&out, training);
        out = ReLU::new().forward(&out);
        self.avgpool.forward(&out)
    }
}

#[cfg(test)]
mod tests_efficient {
    use super::*;

    #[test]
    fn test_se_block() {
        let mut se = SEBlock::new(64, 16);
        let input = Tensor::from_slice(&vec![0.5f32; 64 * 32 * 32], &[1, 64, 32, 32]).unwrap();
        let output = se.forward(&input, true);
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn test_efficientnet_b0() {
        let mut model = EfficientNetB0::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_squeezenet() {
        let mut model = SqueezeNet::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}


/// ResNet-101
pub struct ResNet101 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    layer1: Vec<BottleneckBlock>,
    layer2: Vec<BottleneckBlock>,
    layer3: Vec<BottleneckBlock>,
    layer4: Vec<BottleneckBlock>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl ResNet101 {
    pub fn new(num_classes: usize) -> Self {
        ResNet101 {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(64),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layer1: vec![
                BottleneckBlock::new(64, 64, 1),
                BottleneckBlock::new(256, 64, 1),
                BottleneckBlock::new(256, 64, 1),
            ],
            layer2: vec![
                BottleneckBlock::new(256, 128, 2),
                BottleneckBlock::new(512, 128, 1),
                BottleneckBlock::new(512, 128, 1),
                BottleneckBlock::new(512, 128, 1),
            ],
            layer3: {
                let mut blocks = vec![BottleneckBlock::new(512, 256, 2)];
                for _ in 0..22 {
                    blocks.push(BottleneckBlock::new(1024, 256, 1));
                }
                blocks
            },
            layer4: vec![
                BottleneckBlock::new(1024, 512, 2),
                BottleneckBlock::new(2048, 512, 1),
                BottleneckBlock::new(2048, 512, 1),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(2048, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        for block in &mut self.layer1 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer2 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer3 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer4 {
            out = block.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// ResNet-152
pub struct ResNet152 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    layer1: Vec<BottleneckBlock>,
    layer2: Vec<BottleneckBlock>,
    layer3: Vec<BottleneckBlock>,
    layer4: Vec<BottleneckBlock>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl ResNet152 {
    pub fn new(num_classes: usize) -> Self {
        ResNet152 {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(64),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layer1: vec![
                BottleneckBlock::new(64, 64, 1),
                BottleneckBlock::new(256, 64, 1),
                BottleneckBlock::new(256, 64, 1),
            ],
            layer2: {
                let mut blocks = vec![BottleneckBlock::new(256, 128, 2)];
                for _ in 0..7 {
                    blocks.push(BottleneckBlock::new(512, 128, 1));
                }
                blocks
            },
            layer3: {
                let mut blocks = vec![BottleneckBlock::new(512, 256, 2)];
                for _ in 0..35 {
                    blocks.push(BottleneckBlock::new(1024, 256, 1));
                }
                blocks
            },
            layer4: vec![
                BottleneckBlock::new(1024, 512, 2),
                BottleneckBlock::new(2048, 512, 1),
                BottleneckBlock::new(2048, 512, 1),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(2048, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        for block in &mut self.layer1 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer2 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer3 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer4 {
            out = block.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// ResNeXt Block (Aggregated Residual Transformations)
pub struct ResNeXtBlock {
    convs: Vec<(Conv2d, BatchNorm2d, Conv2d, BatchNorm2d, Conv2d, BatchNorm2d)>,
    downsample: Option<(Conv2d, BatchNorm2d)>,
    cardinality: usize,
}

impl ResNeXtBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, cardinality: usize, base_width: usize) -> Self {
        let width = (out_channels as f32 * (base_width as f32 / 64.0)) as usize;
        let downsample = if stride != 1 || in_channels != out_channels * 4 {
            Some((
                Conv2d::new(in_channels, out_channels * 4, (1, 1)).stride((stride, stride)),
                BatchNorm2d::new(out_channels * 4),
            ))
        } else {
            None
        };

        let mut convs = Vec::new();
        for _ in 0..cardinality {
            convs.push((
                Conv2d::new(in_channels, width, (1, 1)),
                BatchNorm2d::new(width),
                Conv2d::new(width, width, (3, 3)).stride((stride, stride)).padding((1, 1)),
                BatchNorm2d::new(width),
                Conv2d::new(width, out_channels * 4, (1, 1)),
                BatchNorm2d::new(out_channels * 4),
            ));
        }

        ResNeXtBlock {
            convs,
            downsample,
            cardinality,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();

        // Process each path
        let mut outputs = Vec::new();
        for (conv1, bn1, conv2, bn2, conv3, bn3) in &mut self.convs {
            let mut out = conv1.forward(x, training);
            out = bn1.forward(&out, training);
            out = ReLU::new().forward(&out);

            out = conv2.forward(&out, training);
            out = bn2.forward(&out, training);
            out = ReLU::new().forward(&out);

            out = conv3.forward(&out, training);
            out = bn3.forward(&out, training);

            outputs.push(out);
        }

        // Sum all paths
        let mut result = outputs[0].clone();
        for output in &outputs[1..] {
            let result_data = result.data_f32();
            let output_data = output.data_f32();
            let summed: Vec<f32> = result_data.iter()
                .zip(output_data.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            result = Tensor::from_slice(&summed, result.dims()).unwrap();
        }

        // Downsample identity if needed
        let identity = if let Some((ref mut conv, ref mut bn)) = self.downsample {
            let mut id = conv.forward(&identity, training);
            bn.forward(&id, training)
        } else {
            identity
        };

        // Add residual
        let result_data = result.data_f32();
        let id_data = identity.data_f32();
        let final_result: Vec<f32> = result_data.iter()
            .zip(id_data.iter())
            .map(|(&r, &i)| r + i)
            .collect();

        let final_tensor = Tensor::from_slice(&final_result, result.dims()).unwrap();
        ReLU::new().forward(&final_tensor)
    }
}

/// ResNeXt-50
pub struct ResNeXt50 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    layer1: Vec<ResNeXtBlock>,
    layer2: Vec<ResNeXtBlock>,
    layer3: Vec<ResNeXtBlock>,
    layer4: Vec<ResNeXtBlock>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl ResNeXt50 {
    pub fn new(num_classes: usize, cardinality: usize) -> Self {
        ResNeXt50 {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(64),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layer1: vec![
                ResNeXtBlock::new(64, 64, 1, cardinality, 4),
                ResNeXtBlock::new(256, 64, 1, cardinality, 4),
                ResNeXtBlock::new(256, 64, 1, cardinality, 4),
            ],
            layer2: vec![
                ResNeXtBlock::new(256, 128, 2, cardinality, 4),
                ResNeXtBlock::new(512, 128, 1, cardinality, 4),
                ResNeXtBlock::new(512, 128, 1, cardinality, 4),
                ResNeXtBlock::new(512, 128, 1, cardinality, 4),
            ],
            layer3: vec![
                ResNeXtBlock::new(512, 256, 2, cardinality, 4),
                ResNeXtBlock::new(1024, 256, 1, cardinality, 4),
                ResNeXtBlock::new(1024, 256, 1, cardinality, 4),
                ResNeXtBlock::new(1024, 256, 1, cardinality, 4),
                ResNeXtBlock::new(1024, 256, 1, cardinality, 4),
                ResNeXtBlock::new(1024, 256, 1, cardinality, 4),
            ],
            layer4: vec![
                ResNeXtBlock::new(1024, 512, 2, cardinality, 4),
                ResNeXtBlock::new(2048, 512, 1, cardinality, 4),
                ResNeXtBlock::new(2048, 512, 1, cardinality, 4),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(2048, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        for block in &mut self.layer1 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer2 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer3 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer4 {
            out = block.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// Wide ResNet Block
pub struct WideResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Conv2d>,
    dropout: f32,
}

impl WideResidualBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, dropout: f32) -> Self {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, (1, 1)).stride((stride, stride)))
        } else {
            None
        };

        WideResidualBlock {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3))
                .stride((stride, stride))
                .padding((1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3)).padding((1, 1)),
            bn2: BatchNorm2d::new(out_channels),
            downsample,
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();

        let mut out = self.bn1.forward(x, training);
        out = ReLU::new().forward(&out);
        out = self.conv1.forward(&out, training);

        out = self.bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        if training && self.dropout > 0.0 {
            out = self.apply_dropout(&out, self.dropout);
        }
        
        out = self.conv2.forward(&out, training);

        let identity = if let Some(ref mut conv) = self.downsample {
            conv.forward(&identity, training)
        } else {
            identity
        };

        let out_data = out.data_f32();
        let id_data = identity.data_f32();
        let result: Vec<f32> = out_data.iter()
            .zip(id_data.iter())
            .map(|(&o, &i)| o + i)
            .collect();

        Tensor::from_slice(&result, out.dims()).unwrap()
    }

    fn apply_dropout(&self, x: &Tensor, p: f32) -> Tensor {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&val| if rng.gen::<f32>() < p { 0.0 } else { val / (1.0 - p) })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }
}

/// Wide ResNet-28-10 (28 layers, width factor 10)
pub struct WideResNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layer1: Vec<WideResidualBlock>,
    layer2: Vec<WideResidualBlock>,
    layer3: Vec<WideResidualBlock>,
    bn_final: BatchNorm2d,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl WideResNet {
    pub fn new(num_classes: usize, depth: usize, width_factor: usize, dropout: f32) -> Self {
        let n = (depth - 4) / 6;
        let base_width = 16;

        WideResNet {
            conv1: Conv2d::new(3, base_width, (3, 3)).padding((1, 1)),
            bn1: BatchNorm2d::new(base_width),
            layer1: {
                let mut blocks = vec![WideResidualBlock::new(base_width, base_width * width_factor, 1, dropout)];
                for _ in 1..n {
                    blocks.push(WideResidualBlock::new(base_width * width_factor, base_width * width_factor, 1, dropout));
                }
                blocks
            },
            layer2: {
                let mut blocks = vec![WideResidualBlock::new(base_width * width_factor, base_width * 2 * width_factor, 2, dropout)];
                for _ in 1..n {
                    blocks.push(WideResidualBlock::new(base_width * 2 * width_factor, base_width * 2 * width_factor, 1, dropout));
                }
                blocks
            },
            layer3: {
                let mut blocks = vec![WideResidualBlock::new(base_width * 2 * width_factor, base_width * 4 * width_factor, 2, dropout)];
                for _ in 1..n {
                    blocks.push(WideResidualBlock::new(base_width * 4 * width_factor, base_width * 4 * width_factor, 1, dropout));
                }
                blocks
            },
            bn_final: BatchNorm2d::new(base_width * 4 * width_factor),
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(base_width * 4 * width_factor, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);

        for block in &mut self.layer1 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer2 {
            out = block.forward(&out, training);
        }
        for block in &mut self.layer3 {
            out = block.forward(&out, training);
        }

        out = self.bn_final.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.avgpool.forward(&out);

        self.fc.forward(&out, training)
    }
}

#[cfg(test)]
mod tests_resnet_extended {
    use super::*;

    #[test]
    fn test_resnet101() {
        let mut model = ResNet101::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_resnet152() {
        let mut model = ResNet152::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_resnext50() {
        let mut model = ResNeXt50::new(1000, 32);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_wide_resnet() {
        let mut model = WideResNet::new(10, 28, 10, 0.3);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 32 * 32], &[1, 3, 32, 32]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 10);
    }
}


/// Inception v3 Module
pub struct InceptionV3Module {
    branch1x1: Conv2d,
    branch5x5_1: Conv2d,
    branch5x5_2: Conv2d,
    branch3x3dbl_1: Conv2d,
    branch3x3dbl_2: Conv2d,
    branch3x3dbl_3: Conv2d,
    branch_pool: AvgPool2d,
    branch_pool_proj: Conv2d,
}

impl InceptionV3Module {
    pub fn new(in_channels: usize, ch1x1: usize, ch5x5_reduce: usize, ch5x5: usize,
               ch3x3_reduce: usize, ch3x3: usize, pool_proj: usize) -> Self {
        InceptionV3Module {
            branch1x1: Conv2d::new(in_channels, ch1x1, (1, 1)),
            branch5x5_1: Conv2d::new(in_channels, ch5x5_reduce, (1, 1)),
            branch5x5_2: Conv2d::new(ch5x5_reduce, ch5x5, (3, 3)).padding((1, 1)),
            branch3x3dbl_1: Conv2d::new(in_channels, ch3x3_reduce, (1, 1)),
            branch3x3dbl_2: Conv2d::new(ch3x3_reduce, ch3x3, (3, 3)).padding((1, 1)),
            branch3x3dbl_3: Conv2d::new(ch3x3, ch3x3, (3, 3)).padding((1, 1)),
            branch_pool: AvgPool2d::new((3, 3), (1, 1)),
            branch_pool_proj: Conv2d::new(in_channels, pool_proj, (1, 1)),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let branch1 = self.branch1x1.forward(x, training);
        let branch1 = ReLU::new().forward(&branch1);

        let mut branch2 = self.branch5x5_1.forward(x, training);
        branch2 = ReLU::new().forward(&branch2);
        branch2 = self.branch5x5_2.forward(&branch2, training);
        let branch2 = ReLU::new().forward(&branch2);

        let mut branch3 = self.branch3x3dbl_1.forward(x, training);
        branch3 = ReLU::new().forward(&branch3);
        branch3 = self.branch3x3dbl_2.forward(&branch3, training);
        branch3 = ReLU::new().forward(&branch3);
        branch3 = self.branch3x3dbl_3.forward(&branch3, training);
        let branch3 = ReLU::new().forward(&branch3);

        let mut branch4 = self.branch_pool.forward(x);
        branch4 = self.branch_pool_proj.forward(&branch4, training);
        let branch4 = ReLU::new().forward(&branch4);

        self.concatenate_channels(&[branch1, branch2, branch3, branch4])
    }

    fn concatenate_channels(&self, tensors: &[Tensor]) -> Tensor {
        let batch_size = tensors[0].dims()[0];
        let height = tensors[0].dims()[2];
        let width = tensors[0].dims()[3];
        let total_channels: usize = tensors.iter().map(|t| t.dims()[1]).sum();

        let mut result = Vec::new();
        for b in 0..batch_size {
            for tensor in tensors {
                let channels = tensor.dims()[1];
                let data = tensor.data_f32();
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            result.push(data[idx]);
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch_size, total_channels, height, width]).unwrap()
    }
}

/// Inception v3
pub struct InceptionV3 {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    maxpool1: MaxPool2d,
    inception_blocks: Vec<InceptionV3Module>,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl InceptionV3 {
    pub fn new(num_classes: usize) -> Self {
        InceptionV3 {
            conv1: Conv2d::new(3, 32, (3, 3)).stride((2, 2)),
            conv2: Conv2d::new(32, 64, (3, 3)),
            conv3: Conv2d::new(64, 192, (3, 3)).padding((1, 1)),
            maxpool1: MaxPool2d::new((3, 3), (2, 2), (0, 0)),
            inception_blocks: vec![
                InceptionV3Module::new(192, 64, 48, 64, 64, 96, 32),
                InceptionV3Module::new(256, 64, 48, 64, 64, 96, 64),
                InceptionV3Module::new(288, 64, 48, 64, 64, 96, 64),
            ],
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(288, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = ReLU::new().forward(&out);
        out = self.conv2.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.conv3.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool1.forward(&out);

        for block in &mut self.inception_blocks {
            out = block.forward(&out, training);
        }

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// ShuffleNet Unit
pub struct ShuffleNetUnit {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    depthwise: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    shortcut: Option<(AvgPool2d, Conv2d, BatchNorm2d)>,
    groups: usize,
}

impl ShuffleNetUnit {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, groups: usize) -> Self {
        let mid_channels = out_channels / 4;
        
        let shortcut = if stride == 2 {
            Some((
                AvgPool2d::new((3, 3), (stride, stride)),
                Conv2d::new(in_channels, out_channels - in_channels, (1, 1)),
                BatchNorm2d::new(out_channels - in_channels),
            ))
        } else {
            None
        };

        ShuffleNetUnit {
            conv1: Conv2d::new(in_channels, mid_channels, (1, 1)),
            bn1: BatchNorm2d::new(mid_channels),
            depthwise: Conv2d::new(mid_channels, mid_channels, (3, 3))
                .stride((stride, stride))
                .padding((1, 1)),
            bn2: BatchNorm2d::new(mid_channels),
            conv3: Conv2d::new(mid_channels, out_channels, (1, 1)),
            bn3: BatchNorm2d::new(out_channels),
            shortcut,
            groups,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();

        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);

        // Channel shuffle (simplified)
        out = self.channel_shuffle(&out);

        out = self.depthwise.forward(&out, training);
        out = self.bn2.forward(&out, training);

        out = self.conv3.forward(&out, training);
        out = self.bn3.forward(&out, training);

        // Shortcut connection
        let identity = if let Some((ref mut pool, ref mut conv, ref mut bn)) = self.shortcut {
            let mut id = pool.forward(&identity);
            id = conv.forward(&id, training);
            id = bn.forward(&id, training);
            
            // Concatenate with identity
            self.concatenate_channels(&[identity, id])
        } else {
            identity
        };

        let out_data = out.data_f32();
        let id_data = identity.data_f32();
        let result: Vec<f32> = out_data.iter()
            .zip(id_data.iter())
            .map(|(&o, &i)| o + i)
            .collect();

        let result_tensor = Tensor::from_slice(&result, out.dims()).unwrap();
        ReLU::new().forward(&result_tensor)
    }

    fn channel_shuffle(&self, x: &Tensor) -> Tensor {
        // Simplified channel shuffle
        x.clone()
    }

    fn concatenate_channels(&self, tensors: &[Tensor]) -> Tensor {
        let batch_size = tensors[0].dims()[0];
        let height = tensors[0].dims()[2];
        let width = tensors[0].dims()[3];
        let total_channels: usize = tensors.iter().map(|t| t.dims()[1]).sum();

        let mut result = Vec::new();
        for b in 0..batch_size {
            for tensor in tensors {
                let channels = tensor.dims()[1];
                let data = tensor.data_f32();
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            result.push(data[idx]);
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch_size, total_channels, height, width]).unwrap()
    }
}

/// ShuffleNet v2
pub struct ShuffleNetV2 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    stage2: Vec<ShuffleNetUnit>,
    stage3: Vec<ShuffleNetUnit>,
    stage4: Vec<ShuffleNetUnit>,
    conv5: Conv2d,
    bn5: BatchNorm2d,
    avgpool: GlobalAvgPool2d,
    fc: Dense,
}

impl ShuffleNetV2 {
    pub fn new(num_classes: usize) -> Self {
        ShuffleNetV2 {
            conv1: Conv2d::new(3, 24, (3, 3)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(24),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            stage2: vec![
                ShuffleNetUnit::new(24, 116, 2, 2),
                ShuffleNetUnit::new(116, 116, 1, 2),
                ShuffleNetUnit::new(116, 116, 1, 2),
                ShuffleNetUnit::new(116, 116, 1, 2),
            ],
            stage3: vec![
                ShuffleNetUnit::new(116, 232, 2, 2),
                ShuffleNetUnit::new(232, 232, 1, 2),
                ShuffleNetUnit::new(232, 232, 1, 2),
                ShuffleNetUnit::new(232, 232, 1, 2),
            ],
            stage4: vec![
                ShuffleNetUnit::new(232, 464, 2, 2),
                ShuffleNetUnit::new(464, 464, 1, 2),
                ShuffleNetUnit::new(464, 464, 1, 2),
                ShuffleNetUnit::new(464, 464, 1, 2),
            ],
            conv5: Conv2d::new(464, 1024, (1, 1)),
            bn5: BatchNorm2d::new(1024),
            avgpool: GlobalAvgPool2d::new(),
            fc: Dense::new(1024, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);

        for unit in &mut self.stage2 {
            out = unit.forward(&out, training);
        }
        for unit in &mut self.stage3 {
            out = unit.forward(&out, training);
        }
        for unit in &mut self.stage4 {
            out = unit.forward(&out, training);
        }

        out = self.conv5.forward(&out, training);
        out = self.bn5.forward(&out, training);
        out = ReLU::new().forward(&out);

        out = self.avgpool.forward(&out);
        self.fc.forward(&out, training)
    }
}

#[cfg(test)]
mod tests_inception_shuffle {
    use super::*;

    #[test]
    fn test_inception_v3() {
        let mut model = InceptionV3::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 299 * 299], &[1, 3, 299, 299]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_shufflenet_v2() {
        let mut model = ShuffleNetV2::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}


/// ConvNeXt
pub struct ConvNeXt {
    stem: Conv2d,
    stages: Vec<ConvNeXtStage>,
    norm: LayerNorm,
    head: Dense,
}

struct ConvNeXtStage {
    blocks: Vec<ConvNeXtBlock>,
    downsample: Option<Conv2d>,
}

struct ConvNeXtBlock {
    dwconv: Conv2d,
    norm: LayerNorm,
    pwconv1: Dense,
    pwconv2: Dense,
    gamma: f32,
}

impl ConvNeXtBlock {
    fn new(dim: usize) -> Self {
        ConvNeXtBlock {
            dwconv: Conv2d::new(dim, dim, (7, 7)).padding((3, 3)),
            norm: LayerNorm::new(vec![dim]),
            pwconv1: Dense::new(dim, dim * 4),
            pwconv2: Dense::new(dim * 4, dim),
            gamma: 1.0,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        
        let mut out = self.dwconv.forward(x, training);
        out = self.norm.forward(&out);
        out = self.pwconv1.forward(&out, training);
        out = self.gelu(&out);
        out = self.pwconv2.forward(&out, training);
        
        self.add_tensors(&identity, &out)
    }

    fn gelu(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| 0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh()))
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let a_data = a.data_f32();
        let b_data = b.data_f32();
        let result: Vec<f32> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + self.gamma * y)
            .collect();
        Tensor::from_slice(&result, a.dims()).unwrap()
    }
}

impl ConvNeXtStage {
    fn new(dim: usize, num_blocks: usize, downsample: bool) -> Self {
        ConvNeXtStage {
            blocks: (0..num_blocks).map(|_| ConvNeXtBlock::new(dim)).collect(),
            downsample: if downsample {
                Some(Conv2d::new(dim, dim * 2, (2, 2)).stride((2, 2)))
            } else {
                None
            },
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for block in &mut self.blocks {
            out = block.forward(&out, training);
        }
        
        if let Some(ref mut ds) = self.downsample {
            out = ds.forward(&out, training);
        }
        
        out
    }
}

impl ConvNeXt {
    pub fn new(num_classes: usize) -> Self {
        ConvNeXt {
            stem: Conv2d::new(3, 96, (4, 4)).stride((4, 4)),
            stages: vec![
                ConvNeXtStage::new(96, 3, true),
                ConvNeXtStage::new(192, 3, true),
                ConvNeXtStage::new(384, 9, true),
                ConvNeXtStage::new(768, 3, false),
            ],
            norm: LayerNorm::new(vec![768]),
            head: Dense::new(768, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        
        for stage in &mut self.stages {
            out = stage.forward(&out, training);
        }
        
        out = self.norm.forward(&out);
        self.head.forward(&out, training)
    }
}

/// NFNet (Normalizer-Free Network)
pub struct NFNet {
    stem: Conv2d,
    stages: Vec<NFNetStage>,
    final_conv: Conv2d,
    pool: GlobalAvgPool2d,
    fc: Dense,
}

struct NFNetStage {
    blocks: Vec<NFNetBlock>,
}

struct NFNetBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    alpha: f32,
    beta: f32,
}

impl NFNetBlock {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        NFNetBlock {
            conv1: Conv2d::new(in_channels, out_channels / 4, (1, 1)),
            conv2: Conv2d::new(out_channels / 4, out_channels / 4, (3, 3)).padding((1, 1)),
            conv3: Conv2d::new(out_channels / 4, out_channels, (1, 1)),
            alpha: 0.2,
            beta: 1.0,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        
        let mut out = self.conv1.forward(x, training);
        out = self.scaled_activation(&out);
        
        out = self.conv2.forward(&out, training);
        out = self.scaled_activation(&out);
        
        out = self.conv3.forward(&out, training);
        
        self.skip_connection(&identity, &out)
    }

    fn scaled_activation(&self, x: &Tensor) -> Tensor {
        let data = x.data_f32();
        let result: Vec<f32> = data.iter()
            .map(|&v| if v > 0.0 { v } else { 0.0 })
            .collect();
        Tensor::from_slice(&result, x.dims()).unwrap()
    }

    fn skip_connection(&self, identity: &Tensor, residual: &Tensor) -> Tensor {
        let id_data = identity.data_f32();
        let res_data = residual.data_f32();
        let result: Vec<f32> = id_data.iter()
            .zip(res_data.iter())
            .map(|(&i, &r)| i * self.beta + r * self.alpha)
            .collect();
        Tensor::from_slice(&result, identity.dims()).unwrap()
    }
}

impl NFNetStage {
    fn new(in_channels: usize, out_channels: usize, num_blocks: usize) -> Self {
        NFNetStage {
            blocks: (0..num_blocks).map(|_| NFNetBlock::new(in_channels, out_channels)).collect(),
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

impl NFNet {
    pub fn new(num_classes: usize) -> Self {
        NFNet {
            stem: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            stages: vec![
                NFNetStage::new(64, 256, 3),
                NFNetStage::new(256, 512, 4),
                NFNetStage::new(512, 1024, 6),
                NFNetStage::new(1024, 2048, 3),
            ],
            final_conv: Conv2d::new(2048, 2048, (1, 1)),
            pool: GlobalAvgPool2d::new(),
            fc: Dense::new(2048, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        
        for stage in &mut self.stages {
            out = stage.forward(&out, training);
        }
        
        out = self.final_conv.forward(&out, training);
        out = self.pool.forward(&out);
        self.fc.forward(&out, training)
    }
}

/// CoAtNet (Convolution and Attention Network)
pub struct CoAtNet {
    stem: Conv2d,
    conv_stages: Vec<CoAtNetConvStage>,
    attn_stages: Vec<CoAtNetAttnStage>,
    head: Dense,
}

struct CoAtNetConvStage {
    blocks: Vec<MBConvBlock>,
}

impl CoAtNetConvStage {
    fn new(in_channels: usize, out_channels: usize, num_blocks: usize) -> Self {
        CoAtNetConvStage {
            blocks: (0..num_blocks).map(|_| MBConvBlock::new(in_channels, out_channels, 4)).collect(),
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

struct CoAtNetAttnStage {
    blocks: Vec<TransformerBlock>,
}

struct TransformerBlock {
    attention: Dense,
    ffn: Dense,
}

impl TransformerBlock {
    fn new(dim: usize) -> Self {
        TransformerBlock {
            attention: Dense::new(dim, dim),
            ffn: Dense::new(dim, dim * 4),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let attn_out = self.attention.forward(x, training);
        let ffn_out = self.ffn.forward(&attn_out, training);
        
        self.add_tensors(x, &ffn_out)
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

impl CoAtNetAttnStage {
    fn new(dim: usize, num_blocks: usize) -> Self {
        CoAtNetAttnStage {
            blocks: (0..num_blocks).map(|_| TransformerBlock::new(dim)).collect(),
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

impl CoAtNet {
    pub fn new(num_classes: usize) -> Self {
        CoAtNet {
            stem: Conv2d::new(3, 64, (3, 3)).stride((2, 2)).padding((1, 1)),
            conv_stages: vec![
                CoAtNetConvStage::new(64, 96, 2),
                CoAtNetConvStage::new(96, 192, 2),
            ],
            attn_stages: vec![
                CoAtNetAttnStage::new(192, 3),
                CoAtNetAttnStage::new(192, 2),
            ],
            head: Dense::new(192, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.stem.forward(x, training);
        
        for stage in &mut self.conv_stages {
            out = stage.forward(&out, training);
        }
        
        for stage in &mut self.attn_stages {
            out = stage.forward(&out, training);
        }
        
        self.head.forward(&out, training)
    }
}

#[cfg(test)]
mod tests_modern_cnns {
    use super::*;

    #[test]
    fn test_convnext() {
        let mut model = ConvNeXt::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }

    #[test]
    fn test_nfnet() {
        let mut model = NFNet::new(1000);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 1000);
    }
}


