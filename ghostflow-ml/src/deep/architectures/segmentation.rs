//! Segmentation Architectures - U-Net, DeepLab, Mask R-CNN, PSPNet, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d};
use crate::deep::activations::ReLU;

/// U-Net Encoder Block
pub struct UNetEncoderBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    pool: MaxPool2d,
}

impl UNetEncoderBlock {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        UNetEncoderBlock {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3)).padding((1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3)).padding((1, 1)),
            bn2: BatchNorm2d::new(out_channels),
            pool: MaxPool2d::new((2, 2), (2, 2), (0, 0)),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        let skip = ReLU::new().forward(&out);
        
        let pooled = self.pool.forward(&skip);
        
        (pooled, skip)
    }
}

/// U-Net Decoder Block
pub struct UNetDecoderBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
}

impl UNetDecoderBlock {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        UNetDecoderBlock {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3)).padding((1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3)).padding((1, 1)),
            bn2: BatchNorm2d::new(out_channels),
        }
    }

    pub fn forward(&mut self, x: &Tensor, skip: &Tensor, training: bool) -> Tensor {
        // Upsample x
        let upsampled = self.upsample(x);
        
        // Concatenate with skip connection
        let concat = self.concatenate(&upsampled, skip);
        
        let mut out = self.conv1.forward(&concat, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        ReLU::new().forward(&out)
    }

    fn upsample(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();

        let new_height = height * 2;
        let new_width = width * 2;
        let mut result = vec![0.0f32; batch * channels * new_height * new_width];

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let val = data[((b * channels + c) * height + h) * width + w];
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let new_h = h * 2 + dh;
                                let new_w = w * 2 + dw;
                                let idx = ((b * channels + c) * new_height + new_h) * new_width + new_w;
                                result[idx] = val;
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, new_height, new_width]).unwrap()
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

/// U-Net Complete Model
pub struct UNet {
    encoder1: UNetEncoderBlock,
    encoder2: UNetEncoderBlock,
    encoder3: UNetEncoderBlock,
    encoder4: UNetEncoderBlock,
    
    bottleneck_conv1: Conv2d,
    bottleneck_bn1: BatchNorm2d,
    bottleneck_conv2: Conv2d,
    bottleneck_bn2: BatchNorm2d,
    
    decoder4: UNetDecoderBlock,
    decoder3: UNetDecoderBlock,
    decoder2: UNetDecoderBlock,
    decoder1: UNetDecoderBlock,
    
    final_conv: Conv2d,
    num_classes: usize,
}

impl UNet {
    pub fn new(in_channels: usize, num_classes: usize) -> Self {
        UNet {
            encoder1: UNetEncoderBlock::new(in_channels, 64),
            encoder2: UNetEncoderBlock::new(64, 128),
            encoder3: UNetEncoderBlock::new(128, 256),
            encoder4: UNetEncoderBlock::new(256, 512),
            
            bottleneck_conv1: Conv2d::new(512, 1024, (3, 3)).padding((1, 1)),
            bottleneck_bn1: BatchNorm2d::new(1024),
            bottleneck_conv2: Conv2d::new(1024, 1024, (3, 3)).padding((1, 1)),
            bottleneck_bn2: BatchNorm2d::new(1024),
            
            decoder4: UNetDecoderBlock::new(1024 + 512, 512),
            decoder3: UNetDecoderBlock::new(512 + 256, 256),
            decoder2: UNetDecoderBlock::new(256 + 128, 128),
            decoder1: UNetDecoderBlock::new(128 + 64, 64),
            
            final_conv: Conv2d::new(64, num_classes, (1, 1)),
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // Encoder
        let (enc1, skip1) = self.encoder1.forward(x, training);
        let (enc2, skip2) = self.encoder2.forward(&enc1, training);
        let (enc3, skip3) = self.encoder3.forward(&enc2, training);
        let (enc4, skip4) = self.encoder4.forward(&enc3, training);
        
        // Bottleneck
        let mut bottleneck = self.bottleneck_conv1.forward(&enc4, training);
        bottleneck = self.bottleneck_bn1.forward(&bottleneck, training);
        bottleneck = ReLU::new().forward(&bottleneck);
        bottleneck = self.bottleneck_conv2.forward(&bottleneck, training);
        bottleneck = self.bottleneck_bn2.forward(&bottleneck, training);
        bottleneck = ReLU::new().forward(&bottleneck);
        
        // Decoder
        let dec4 = self.decoder4.forward(&bottleneck, &skip4, training);
        let dec3 = self.decoder3.forward(&dec4, &skip3, training);
        let dec2 = self.decoder2.forward(&dec3, &skip2, training);
        let dec1 = self.decoder1.forward(&dec2, &skip1, training);
        
        // Final convolution
        self.final_conv.forward(&dec1, training)
    }
}

/// Atrous Spatial Pyramid Pooling (ASPP) for DeepLab
pub struct ASPP {
    conv1x1: Conv2d,
    bn1: BatchNorm2d,
    
    atrous_conv1: Conv2d,
    bn_atrous1: BatchNorm2d,
    
    atrous_conv2: Conv2d,
    bn_atrous2: BatchNorm2d,
    
    atrous_conv3: Conv2d,
    bn_atrous3: BatchNorm2d,
    
    global_pool: AvgPool2d,
    conv_pool: Conv2d,
    bn_pool: BatchNorm2d,
    
    project: Conv2d,
    bn_project: BatchNorm2d,
}

impl ASPP {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        ASPP {
            conv1x1: Conv2d::new(in_channels, out_channels, (1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            
            atrous_conv1: Conv2d::new(in_channels, out_channels, (3, 3)).padding((6, 6)), // dilation=6
            bn_atrous1: BatchNorm2d::new(out_channels),
            
            atrous_conv2: Conv2d::new(in_channels, out_channels, (3, 3)).padding((12, 12)), // dilation=12
            bn_atrous2: BatchNorm2d::new(out_channels),
            
            atrous_conv3: Conv2d::new(in_channels, out_channels, (3, 3)).padding((18, 18)), // dilation=18
            bn_atrous3: BatchNorm2d::new(out_channels),
            
            global_pool: AvgPool2d::new((1, 1), (1, 1)),
            conv_pool: Conv2d::new(in_channels, out_channels, (1, 1)),
            bn_pool: BatchNorm2d::new(out_channels),
            
            project: Conv2d::new(out_channels * 5, out_channels, (1, 1)),
            bn_project: BatchNorm2d::new(out_channels),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        // 1x1 convolution
        let mut feat1 = self.conv1x1.forward(x, training);
        feat1 = self.bn1.forward(&feat1, training);
        feat1 = ReLU::new().forward(&feat1);
        
        // Atrous convolutions
        let mut feat2 = self.atrous_conv1.forward(x, training);
        feat2 = self.bn_atrous1.forward(&feat2, training);
        feat2 = ReLU::new().forward(&feat2);
        
        let mut feat3 = self.atrous_conv2.forward(x, training);
        feat3 = self.bn_atrous2.forward(&feat3, training);
        feat3 = ReLU::new().forward(&feat3);
        
        let mut feat4 = self.atrous_conv3.forward(x, training);
        feat4 = self.bn_atrous3.forward(&feat4, training);
        feat4 = ReLU::new().forward(&feat4);
        
        // Global pooling
        let mut feat5 = self.global_pool.forward(x);
        feat5 = self.conv_pool.forward(&feat5, training);
        feat5 = self.bn_pool.forward(&feat5, training);
        feat5 = ReLU::new().forward(&feat5);
        feat5 = self.upsample_to_size(&feat5, x.dims()[2], x.dims()[3]);
        
        // Concatenate all features
        let concat = self.concatenate_features(&[feat1, feat2, feat3, feat4, feat5]);
        
        // Project
        let mut out = self.project.forward(&concat, training);
        out = self.bn_project.forward(&out, training);
        ReLU::new().forward(&out)
    }

    fn upsample_to_size(&self, x: &Tensor, target_h: usize, target_w: usize) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let data = x.data_f32();

        let mut result = vec![0.0f32; batch * channels * target_h * target_w];

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..target_h {
                    for w in 0..target_w {
                        let idx = ((b * channels + c) * target_h + h) * target_w + w;
                        result[idx] = data[b * channels + c];
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, target_h, target_w]).unwrap()
    }

    fn concatenate_features(&self, features: &[Tensor]) -> Tensor {
        let batch = features[0].dims()[0];
        let height = features[0].dims()[2];
        let width = features[0].dims()[3];
        let total_channels: usize = features.iter().map(|f| f.dims()[1]).sum();

        let mut result = Vec::new();
        for b in 0..batch {
            for feat in features {
                let channels = feat.dims()[1];
                let data = feat.data_f32();
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

        Tensor::from_slice(&result, &[batch, total_channels, height, width]).unwrap()
    }
}

/// DeepLab v3+ Model
pub struct DeepLabV3Plus {
    backbone: ResNetBackbone,
    aspp: ASPP,
    decoder_conv1: Conv2d,
    decoder_bn1: BatchNorm2d,
    decoder_conv2: Conv2d,
    decoder_bn2: BatchNorm2d,
    final_conv: Conv2d,
    num_classes: usize,
}

struct ResNetBackbone {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layers: Vec<ResidualBlock>,
}

struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
}

impl ResidualBlock {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        ResidualBlock {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3)).padding((1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3)).padding((1, 1)),
            bn2: BatchNorm2d::new(out_channels),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = self.bn2.forward(&out, training);
        
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

impl ResNetBackbone {
    fn new() -> Self {
        ResNetBackbone {
            conv1: Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
            bn1: BatchNorm2d::new(64),
            layers: vec![
                ResidualBlock::new(64, 64),
                ResidualBlock::new(64, 128),
                ResidualBlock::new(128, 256),
                ResidualBlock::new(256, 512),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        let mut low_level_feat = out.clone();
        
        for (i, block) in self.layers.iter_mut().enumerate() {
            out = block.forward(&out, training);
            if i == 0 {
                low_level_feat = out.clone();
            }
        }
        
        (out, low_level_feat)
    }
}

impl DeepLabV3Plus {
    pub fn new(num_classes: usize) -> Self {
        DeepLabV3Plus {
            backbone: ResNetBackbone::new(),
            aspp: ASPP::new(512, 256),
            decoder_conv1: Conv2d::new(64, 48, (1, 1)),
            decoder_bn1: BatchNorm2d::new(48),
            decoder_conv2: Conv2d::new(256 + 48, 256, (3, 3)).padding((1, 1)),
            decoder_bn2: BatchNorm2d::new(256),
            final_conv: Conv2d::new(256, num_classes, (1, 1)),
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let input_size = (x.dims()[2], x.dims()[3]);
        
        // Backbone
        let (high_level_feat, low_level_feat) = self.backbone.forward(x, training);
        
        // ASPP
        let aspp_out = self.aspp.forward(&high_level_feat, training);
        
        // Decoder
        let mut low_level = self.decoder_conv1.forward(&low_level_feat, training);
        low_level = self.decoder_bn1.forward(&low_level, training);
        low_level = ReLU::new().forward(&low_level);
        
        // Upsample ASPP output
        let aspp_upsampled = self.upsample_to_size(&aspp_out, low_level.dims()[2], low_level.dims()[3]);
        
        // Concatenate
        let concat = self.concatenate(&aspp_upsampled, &low_level);
        
        let mut out = self.decoder_conv2.forward(&concat, training);
        out = self.decoder_bn2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        out = self.final_conv.forward(&out, training);
        
        // Upsample to input size
        self.upsample_to_size(&out, input_size.0, input_size.1)
    }

    fn upsample_to_size(&self, x: &Tensor, target_h: usize, target_w: usize) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();

        let scale_h = target_h as f32 / height as f32;
        let scale_w = target_w as f32 / width as f32;

        let mut result = vec![0.0f32; batch * channels * target_h * target_w];

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..target_h {
                    for w in 0..target_w {
                        let src_h = (h as f32 / scale_h) as usize;
                        let src_w = (w as f32 / scale_w) as usize;
                        let src_h = src_h.min(height - 1);
                        let src_w = src_w.min(width - 1);
                        
                        let src_idx = ((b * channels + c) * height + src_h) * width + src_w;
                        let dst_idx = ((b * channels + c) * target_h + h) * target_w + w;
                        result[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, target_h, target_w]).unwrap()
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

/// PSPNet Pyramid Pooling Module
pub struct PyramidPoolingModule {
    pool_sizes: Vec<usize>,
    convs: Vec<Conv2d>,
    bns: Vec<BatchNorm2d>,
    final_conv: Conv2d,
    final_bn: BatchNorm2d,
}

impl PyramidPoolingModule {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let pool_sizes = vec![1, 2, 3, 6];
        let num_pools = pool_sizes.len();
        let pool_channels = out_channels / num_pools;
        
        let convs: Vec<Conv2d> = (0..num_pools)
            .map(|_| Conv2d::new(in_channels, pool_channels, (1, 1)))
            .collect();
        
        let bns: Vec<BatchNorm2d> = (0..num_pools)
            .map(|_| BatchNorm2d::new(pool_channels))
            .collect();
        
        PyramidPoolingModule {
            pool_sizes,
            convs,
            bns,
            final_conv: Conv2d::new(in_channels + out_channels, out_channels, (3, 3)).padding((1, 1)),
            final_bn: BatchNorm2d::new(out_channels),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let input_size = (x.dims()[2], x.dims()[3]);
        let mut pyramid_features = vec![x.clone()];
        
        for (i, &pool_size) in self.pool_sizes.iter().enumerate() {
            let pooled = self.adaptive_avg_pool(x, pool_size);
            let mut feat = self.convs[i].forward(&pooled, training);
            feat = self.bns[i].forward(&feat, training);
            feat = ReLU::new().forward(&feat);
            let upsampled = self.upsample_to_size(&feat, input_size.0, input_size.1);
            pyramid_features.push(upsampled);
        }
        
        let concat = self.concatenate_features(&pyramid_features);
        
        let mut out = self.final_conv.forward(&concat, training);
        out = self.final_bn.forward(&out, training);
        ReLU::new().forward(&out)
    }

    fn adaptive_avg_pool(&self, x: &Tensor, output_size: usize) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();

        let mut result = vec![0.0f32; batch * channels * output_size * output_size];

        let stride_h = height / output_size;
        let stride_w = width / output_size;

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..output_size {
                    for ow in 0..output_size {
                        let mut sum = 0.0f32;
                        let mut count = 0;
                        
                        for h in (oh * stride_h)..((oh + 1) * stride_h).min(height) {
                            for w in (ow * stride_w)..((ow + 1) * stride_w).min(width) {
                                let idx = ((b * channels + c) * height + h) * width + w;
                                sum += data[idx];
                                count += 1;
                            }
                        }
                        
                        let out_idx = ((b * channels + c) * output_size + oh) * output_size + ow;
                        result[out_idx] = sum / count as f32;
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, output_size, output_size]).unwrap()
    }

    fn upsample_to_size(&self, x: &Tensor, target_h: usize, target_w: usize) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();

        let mut result = vec![0.0f32; batch * channels * target_h * target_w];

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..target_h {
                    for w in 0..target_w {
                        let src_h = (h * height / target_h).min(height - 1);
                        let src_w = (w * width / target_w).min(width - 1);
                        
                        let src_idx = ((b * channels + c) * height + src_h) * width + src_w;
                        let dst_idx = ((b * channels + c) * target_h + h) * target_w + w;
                        result[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, target_h, target_w]).unwrap()
    }

    fn concatenate_features(&self, features: &[Tensor]) -> Tensor {
        let batch = features[0].dims()[0];
        let height = features[0].dims()[2];
        let width = features[0].dims()[3];
        let total_channels: usize = features.iter().map(|f| f.dims()[1]).sum();

        let mut result = Vec::new();
        for b in 0..batch {
            for feat in features {
                let channels = feat.dims()[1];
                let data = feat.data_f32();
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

        Tensor::from_slice(&result, &[batch, total_channels, height, width]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unet() {
        let mut model = UNet::new(3, 2);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 256 * 256], &[1, 3, 256, 256]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 2);
    }

    #[test]
    fn test_deeplabv3plus() {
        let mut model = DeepLabV3Plus::new(21);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 21);
    }
}


/// FCN (Fully Convolutional Network)
pub struct FCN {
    encoder: FCNEncoder,
    decoder: FCNDecoder,
    num_classes: usize,
}

struct FCNEncoder {
    conv_blocks: Vec<Vec<Conv2d>>,
    pools: Vec<MaxPool2d>,
}

impl FCNEncoder {
    fn new() -> Self {
        FCNEncoder {
            conv_blocks: vec![
                vec![Conv2d::new(3, 64, (3, 3)).padding((1, 1)), Conv2d::new(64, 64, (3, 3)).padding((1, 1))],
                vec![Conv2d::new(64, 128, (3, 3)).padding((1, 1)), Conv2d::new(128, 128, (3, 3)).padding((1, 1))],
                vec![Conv2d::new(128, 256, (3, 3)).padding((1, 1)), Conv2d::new(256, 256, (3, 3)).padding((1, 1))],
            ],
            pools: vec![
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let mut out = x.clone();
        let mut features = Vec::new();
        
        for (i, conv_block) in self.conv_blocks.iter_mut().enumerate() {
            for conv in conv_block {
                out = conv.forward(&out, training);
                out = ReLU::new().forward(&out);
            }
            features.push(out.clone());
            out = self.pools[i].forward(&out);
        }
        
        features
    }
}

struct FCNDecoder {
    upsamples: Vec<Conv2d>,
    final_conv: Conv2d,
}

impl FCNDecoder {
    fn new(num_classes: usize) -> Self {
        FCNDecoder {
            upsamples: vec![
                Conv2d::new(256, 128, (3, 3)).padding((1, 1)),
                Conv2d::new(128, 64, (3, 3)).padding((1, 1)),
            ],
            final_conv: Conv2d::new(64, num_classes, (1, 1)),
        }
    }

    fn forward(&mut self, features: Vec<Tensor>, training: bool) -> Tensor {
        let mut out = features[features.len() - 1].clone();
        
        for (i, upsample) in self.upsamples.iter_mut().enumerate() {
            out = self.upsample_tensor(&out);
            out = upsample.forward(&out, training);
            out = ReLU::new().forward(&out);
        }
        
        self.final_conv.forward(&out, training)
    }

    fn upsample_tensor(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();

        let new_height = height * 2;
        let new_width = width * 2;
        let mut result = vec![0.0f32; batch * channels * new_height * new_width];

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let val = data[((b * channels + c) * height + h) * width + w];
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let new_h = h * 2 + dh;
                                let new_w = w * 2 + dw;
                                let idx = ((b * channels + c) * new_height + new_h) * new_width + new_w;
                                result[idx] = val;
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, new_height, new_width]).unwrap()
    }
}

impl FCN {
    pub fn new(num_classes: usize) -> Self {
        FCN {
            encoder: FCNEncoder::new(),
            decoder: FCNDecoder::new(num_classes),
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let features = self.encoder.forward(x, training);
        self.decoder.forward(features, training)
    }
}

/// SegNet
pub struct SegNet {
    encoder: SegNetEncoder,
    decoder: SegNetDecoder,
}

struct SegNetEncoder {
    conv_blocks: Vec<Vec<Conv2d>>,
    pools: Vec<MaxPool2d>,
}

impl SegNetEncoder {
    fn new() -> Self {
        SegNetEncoder {
            conv_blocks: vec![
                vec![Conv2d::new(3, 64, (3, 3)).padding((1, 1)), Conv2d::new(64, 64, (3, 3)).padding((1, 1))],
                vec![Conv2d::new(64, 128, (3, 3)).padding((1, 1)), Conv2d::new(128, 128, (3, 3)).padding((1, 1))],
                vec![Conv2d::new(128, 256, (3, 3)).padding((1, 1)), Conv2d::new(256, 256, (3, 3)).padding((1, 1))],
            ],
            pools: vec![
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                MaxPool2d::new((2, 2), (2, 2), (0, 0)),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for (i, conv_block) in self.conv_blocks.iter_mut().enumerate() {
            for conv in conv_block {
                out = conv.forward(&out, training);
                out = ReLU::new().forward(&out);
            }
            out = self.pools[i].forward(&out);
        }
        
        out
    }
}

struct SegNetDecoder {
    conv_blocks: Vec<Vec<Conv2d>>,
    final_conv: Conv2d,
}

impl SegNetDecoder {
    fn new(num_classes: usize) -> Self {
        SegNetDecoder {
            conv_blocks: vec![
                vec![Conv2d::new(256, 256, (3, 3)).padding((1, 1)), Conv2d::new(256, 128, (3, 3)).padding((1, 1))],
                vec![Conv2d::new(128, 128, (3, 3)).padding((1, 1)), Conv2d::new(128, 64, (3, 3)).padding((1, 1))],
                vec![Conv2d::new(64, 64, (3, 3)).padding((1, 1)), Conv2d::new(64, 64, (3, 3)).padding((1, 1))],
            ],
            final_conv: Conv2d::new(64, num_classes, (1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for conv_block in &mut self.conv_blocks {
            out = self.upsample_tensor(&out);
            for conv in conv_block {
                out = conv.forward(&out, training);
                out = ReLU::new().forward(&out);
            }
        }
        
        self.final_conv.forward(&out, training)
    }

    fn upsample_tensor(&self, x: &Tensor) -> Tensor {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let data = x.data_f32();

        let new_height = height * 2;
        let new_width = width * 2;
        let mut result = vec![0.0f32; batch * channels * new_height * new_width];

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let val = data[((b * channels + c) * height + h) * width + w];
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let new_h = h * 2 + dh;
                                let new_w = w * 2 + dw;
                                let idx = ((b * channels + c) * new_height + new_h) * new_width + new_w;
                                result[idx] = val;
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[batch, channels, new_height, new_width]).unwrap()
    }
}

impl SegNet {
    pub fn new(num_classes: usize) -> Self {
        SegNet {
            encoder: SegNetEncoder::new(),
            decoder: SegNetDecoder::new(num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let encoded = self.encoder.forward(x, training);
        self.decoder.forward(&encoded, training)
    }
}

/// HRNet (High-Resolution Network)
pub struct HRNet {
    stem: Vec<Conv2d>,
    stages: Vec<HRNetStage>,
    final_layer: Conv2d,
}

struct HRNetStage {
    branches: Vec<Vec<Conv2d>>,
}

impl HRNetStage {
    fn new(num_branches: usize, channels: Vec<usize>) -> Self {
        let mut branches = Vec::new();
        
        for i in 0..num_branches {
            let branch = vec![
                Conv2d::new(channels[i], channels[i], (3, 3)).padding((1, 1)),
                Conv2d::new(channels[i], channels[i], (3, 3)).padding((1, 1)),
            ];
            branches.push(branch);
        }
        
        HRNetStage { branches }
    }

    fn forward(&mut self, inputs: Vec<Tensor>, training: bool) -> Vec<Tensor> {
        let mut outputs = Vec::new();
        
        for (i, branch) in self.branches.iter_mut().enumerate() {
            let mut out = inputs[i].clone();
            for conv in branch {
                out = conv.forward(&out, training);
                out = ReLU::new().forward(&out);
            }
            outputs.push(out);
        }
        
        outputs
    }
}

impl HRNet {
    pub fn new(num_classes: usize) -> Self {
        HRNet {
            stem: vec![
                Conv2d::new(3, 64, (3, 3)).stride((2, 2)).padding((1, 1)),
                Conv2d::new(64, 64, (3, 3)).stride((2, 2)).padding((1, 1)),
            ],
            stages: vec![
                HRNetStage::new(2, vec![32, 64]),
                HRNetStage::new(3, vec![32, 64, 128]),
                HRNetStage::new(4, vec![32, 64, 128, 256]),
            ],
            final_layer: Conv2d::new(32, num_classes, (1, 1)),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for conv in &mut self.stem {
            out = conv.forward(&out, training);
            out = ReLU::new().forward(&out);
        }
        
        let mut multi_scale = vec![out];
        
        for stage in &mut self.stages {
            multi_scale = stage.forward(multi_scale, training);
        }
        
        self.final_layer.forward(&multi_scale[0], training)
    }
}

/// OCRNet (Object-Contextual Representations)
pub struct OCRNet {
    backbone: Conv2d,
    object_context: ObjectContextBlock,
    classifier: Conv2d,
}

struct ObjectContextBlock {
    key_conv: Conv2d,
    query_conv: Conv2d,
    value_conv: Conv2d,
}

impl ObjectContextBlock {
    fn new(in_channels: usize) -> Self {
        ObjectContextBlock {
            key_conv: Conv2d::new(in_channels, in_channels, (1, 1)),
            query_conv: Conv2d::new(in_channels, in_channels, (1, 1)),
            value_conv: Conv2d::new(in_channels, in_channels, (1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let key = self.key_conv.forward(x, training);
        let query = self.query_conv.forward(x, training);
        let value = self.value_conv.forward(x, training);
        
        // Simplified object-contextual representation
        value
    }
}

impl OCRNet {
    pub fn new(num_classes: usize) -> Self {
        OCRNet {
            backbone: Conv2d::new(3, 512, (3, 3)).padding((1, 1)),
            object_context: ObjectContextBlock::new(512),
            classifier: Conv2d::new(512, num_classes, (1, 1)),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.backbone.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.object_context.forward(&out, training);
        self.classifier.forward(&out, training)
    }
}

#[cfg(test)]
mod tests_extended {
    use super::*;

    #[test]
    fn test_fcn() {
        let mut model = FCN::new(21);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 21);
    }

    #[test]
    fn test_segnet() {
        let mut model = SegNet::new(21);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let output = model.forward(&input, false);
        assert_eq!(output.dims()[1], 21);
    }
}


