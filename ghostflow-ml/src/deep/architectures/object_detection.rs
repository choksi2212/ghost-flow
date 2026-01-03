//! Object Detection Architectures - YOLO, Faster R-CNN, RetinaNet, SSD, etc.

use ghostflow_core::Tensor;
use crate::deep::layers::{Conv2d, BatchNorm2d, Dense, MaxPool2d};
use crate::deep::activations::{ReLU, LeakyReLU, Sigmoid};

/// YOLO v3 Detection Layer
pub struct YOLOv3DetectionLayer {
    conv: Conv2d,
    num_classes: usize,
    anchors: Vec<(f32, f32)>,
}

impl YOLOv3DetectionLayer {
    pub fn new(in_channels: usize, num_classes: usize, anchors: Vec<(f32, f32)>) -> Self {
        let num_anchors = anchors.len();
        let out_channels = num_anchors * (5 + num_classes); // (x, y, w, h, conf) + classes
        
        YOLOv3DetectionLayer {
            conv: Conv2d::new(in_channels, out_channels, (1, 1)),
            num_classes,
            anchors,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        self.conv.forward(x, training)
    }
}

/// YOLO v3 Backbone (Darknet-53)
pub struct Darknet53 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    blocks: Vec<DarknetBlock>,
}

struct DarknetBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    num_repeats: usize,
}

impl DarknetBlock {
    fn new(in_channels: usize, out_channels: usize, num_repeats: usize) -> Self {
        DarknetBlock {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3)).stride((2, 2)).padding((1, 1)),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels / 2, (1, 1)),
            bn2: BatchNorm2d::new(out_channels / 2),
            num_repeats,
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = LeakyReLU::new(0.1).forward(&out);
        
        for _ in 0..self.num_repeats {
            let identity = out.clone();
            
            out = self.conv2.forward(&out, training);
            out = self.bn2.forward(&out, training);
            out = LeakyReLU::new(0.1).forward(&out);
            
            // Add residual
            let out_data = out.data_f32();
            let id_data = identity.data_f32();
            let result: Vec<f32> = out_data.iter()
                .zip(id_data.iter())
                .map(|(&o, &i)| o + i)
                .collect();
            out = Tensor::from_slice(&result, out.dims()).unwrap();
        }
        
        out
    }
}

impl Darknet53 {
    pub fn new() -> Self {
        Darknet53 {
            conv1: Conv2d::new(3, 32, (3, 3)).padding((1, 1)),
            bn1: BatchNorm2d::new(32),
            blocks: vec![
                DarknetBlock::new(32, 64, 1),
                DarknetBlock::new(64, 128, 2),
                DarknetBlock::new(128, 256, 8),
                DarknetBlock::new(256, 512, 8),
                DarknetBlock::new(512, 1024, 4),
            ],
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = LeakyReLU::new(0.1).forward(&out);
        
        let mut feature_maps = Vec::new();
        
        for (i, block) in self.blocks.iter_mut().enumerate() {
            out = block.forward(&out, training);
            // Save feature maps at different scales
            if i >= 2 {
                feature_maps.push(out.clone());
            }
        }
        
        feature_maps
    }
}

/// YOLO v3 Complete Model
pub struct YOLOv3 {
    backbone: Darknet53,
    detection_layers: Vec<YOLOv3DetectionLayer>,
    num_classes: usize,
}

impl YOLOv3 {
    pub fn new(num_classes: usize) -> Self {
        // YOLO v3 uses 3 detection scales with 3 anchors each
        let anchors_scale1 = vec![(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)];
        let anchors_scale2 = vec![(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)];
        let anchors_scale3 = vec![(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)];
        
        YOLOv3 {
            backbone: Darknet53::new(),
            detection_layers: vec![
                YOLOv3DetectionLayer::new(1024, num_classes, anchors_scale1),
                YOLOv3DetectionLayer::new(512, num_classes, anchors_scale2),
                YOLOv3DetectionLayer::new(256, num_classes, anchors_scale3),
            ],
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let feature_maps = self.backbone.forward(x, training);
        
        let mut detections = Vec::new();
        for (i, layer) in self.detection_layers.iter_mut().enumerate() {
            if i < feature_maps.len() {
                let detection = layer.forward(&feature_maps[i], training);
                detections.push(detection);
            }
        }
        
        detections
    }
}

/// Region Proposal Network for Faster R-CNN
pub struct RegionProposalNetwork {
    conv: Conv2d,
    cls_logits: Conv2d,
    bbox_pred: Conv2d,
    num_anchors: usize,
}

impl RegionProposalNetwork {
    pub fn new(in_channels: usize, num_anchors: usize) -> Self {
        RegionProposalNetwork {
            conv: Conv2d::new(in_channels, 512, (3, 3)).padding((1, 1)),
            cls_logits: Conv2d::new(512, num_anchors * 2, (1, 1)), // 2 for objectness
            bbox_pred: Conv2d::new(512, num_anchors * 4, (1, 1)), // 4 for bbox coords
            num_anchors,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let mut out = self.conv.forward(x, training);
        out = ReLU::new().forward(&out);
        
        let objectness = self.cls_logits.forward(&out, training);
        let bbox_deltas = self.bbox_pred.forward(&out, training);
        
        (objectness, bbox_deltas)
    }
}

/// ROI Pooling Layer
pub struct ROIPooling {
    output_size: (usize, usize),
}

impl ROIPooling {
    pub fn new(output_size: (usize, usize)) -> Self {
        ROIPooling { output_size }
    }

    pub fn forward(&self, features: &Tensor, rois: &[(f32, f32, f32, f32)]) -> Tensor {
        // Simplified ROI pooling
        // In practice, this would pool features from specified regions
        let batch_size = rois.len();
        let channels = features.dims()[1];
        let (out_h, out_w) = self.output_size;
        
        let mut result = vec![0.0f32; batch_size * channels * out_h * out_w];
        
        // Placeholder implementation
        for i in 0..result.len() {
            result[i] = 0.1;
        }
        
        Tensor::from_slice(&result, &[batch_size, channels, out_h, out_w]).unwrap()
    }
}

/// Faster R-CNN Detection Head
pub struct FasterRCNNHead {
    fc1: Dense,
    fc2: Dense,
    cls_score: Dense,
    bbox_pred: Dense,
    num_classes: usize,
}

impl FasterRCNNHead {
    pub fn new(in_features: usize, num_classes: usize) -> Self {
        FasterRCNNHead {
            fc1: Dense::new(in_features, 1024),
            fc2: Dense::new(1024, 1024),
            cls_score: Dense::new(1024, num_classes),
            bbox_pred: Dense::new(1024, num_classes * 4),
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let mut out = self.fc1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.fc2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        let class_logits = self.cls_score.forward(&out, training);
        let bbox_deltas = self.bbox_pred.forward(&out, training);
        
        (class_logits, bbox_deltas)
    }
}

/// Faster R-CNN Complete Model
pub struct FasterRCNN {
    backbone: ResNetBackbone,
    rpn: RegionProposalNetwork,
    roi_pool: ROIPooling,
    head: FasterRCNNHead,
    num_classes: usize,
}

struct ResNetBackbone {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,
    layers: Vec<Vec<ResidualBlock>>,
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
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1)),
            layers: vec![
                vec![ResidualBlock::new(64, 64), ResidualBlock::new(64, 64)],
                vec![ResidualBlock::new(64, 128), ResidualBlock::new(128, 128)],
                vec![ResidualBlock::new(128, 256), ResidualBlock::new(256, 256)],
                vec![ResidualBlock::new(256, 512), ResidualBlock::new(512, 512)],
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = self.bn1.forward(&out, training);
        out = ReLU::new().forward(&out);
        out = self.maxpool.forward(&out);
        
        for layer_blocks in &mut self.layers {
            for block in layer_blocks {
                out = block.forward(&out, training);
            }
        }
        
        out
    }
}

impl FasterRCNN {
    pub fn new(num_classes: usize) -> Self {
        FasterRCNN {
            backbone: ResNetBackbone::new(),
            rpn: RegionProposalNetwork::new(512, 9),
            roi_pool: ROIPooling::new((7, 7)),
            head: FasterRCNNHead::new(512 * 7 * 7, num_classes),
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        // Extract features
        let features = self.backbone.forward(x, training);
        
        // Generate proposals
        let (objectness, bbox_deltas) = self.rpn.forward(&features, training);
        
        // Simplified: use dummy ROIs
        let dummy_rois = vec![(0.0, 0.0, 1.0, 1.0); 128];
        let pooled_features = self.roi_pool.forward(&features, &dummy_rois);
        
        // Flatten pooled features
        let batch_size = pooled_features.dims()[0];
        let flat_size = pooled_features.data_f32().len() / batch_size;
        let pooled_flat = Tensor::from_slice(pooled_features.data_f32(), &[batch_size, flat_size]).unwrap();
        
        // Detection head
        self.head.forward(&pooled_flat, training)
    }
}

/// Feature Pyramid Network
pub struct FeaturePyramidNetwork {
    lateral_convs: Vec<Conv2d>,
    fpn_convs: Vec<Conv2d>,
}

impl FeaturePyramidNetwork {
    pub fn new(in_channels_list: Vec<usize>, out_channels: usize) -> Self {
        let lateral_convs: Vec<Conv2d> = in_channels_list.iter()
            .map(|&in_ch| Conv2d::new(in_ch, out_channels, (1, 1)))
            .collect();
        
        let fpn_convs: Vec<Conv2d> = (0..in_channels_list.len())
            .map(|_| Conv2d::new(out_channels, out_channels, (3, 3)).padding((1, 1)))
            .collect();
        
        FeaturePyramidNetwork {
            lateral_convs,
            fpn_convs,
        }
    }

    pub fn forward(&mut self, features: Vec<Tensor>, training: bool) -> Vec<Tensor> {
        let mut laterals = Vec::new();
        
        // Apply lateral convolutions
        for (i, feat) in features.iter().enumerate() {
            let lateral = self.lateral_convs[i].forward(feat, training);
            laterals.push(lateral);
        }
        
        // Top-down pathway with lateral connections
        let mut fpn_features = vec![laterals[laterals.len() - 1].clone()];
        
        for i in (0..laterals.len() - 1).rev() {
            let upsampled = self.upsample(&fpn_features[0]);
            let merged = self.add_tensors(&upsampled, &laterals[i]);
            fpn_features.insert(0, merged);
        }
        
        // Apply FPN convolutions
        fpn_features.iter_mut()
            .enumerate()
            .map(|(i, feat)| self.fpn_convs[i].forward(feat, training))
            .collect()
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

    fn add_tensors(&self, x1: &Tensor, x2: &Tensor) -> Tensor {
        let data1 = x1.data_f32();
        let data2 = x2.data_f32();
        let result: Vec<f32> = data1.iter()
            .zip(data2.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Tensor::from_slice(&result, x1.dims()).unwrap()
    }
}

/// RetinaNet Detection Head
pub struct RetinaNetHead {
    cls_subnet: Vec<Conv2d>,
    bbox_subnet: Vec<Conv2d>,
    cls_score: Conv2d,
    bbox_pred: Conv2d,
    num_classes: usize,
    num_anchors: usize,
}

impl RetinaNetHead {
    pub fn new(in_channels: usize, num_classes: usize, num_anchors: usize) -> Self {
        let num_convs = 4;
        
        let cls_subnet: Vec<Conv2d> = (0..num_convs)
            .map(|_| Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)))
            .collect();
        
        let bbox_subnet: Vec<Conv2d> = (0..num_convs)
            .map(|_| Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)))
            .collect();
        
        RetinaNetHead {
            cls_subnet,
            bbox_subnet,
            cls_score: Conv2d::new(in_channels, num_anchors * num_classes, (3, 3)).padding((1, 1)),
            bbox_pred: Conv2d::new(in_channels, num_anchors * 4, (3, 3)).padding((1, 1)),
            num_classes,
            num_anchors,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        // Classification subnet
        let mut cls_out = x.clone();
        for conv in &mut self.cls_subnet {
            cls_out = conv.forward(&cls_out, training);
            cls_out = ReLU::new().forward(&cls_out);
        }
        let cls_logits = self.cls_score.forward(&cls_out, training);
        
        // Box regression subnet
        let mut bbox_out = x.clone();
        for conv in &mut self.bbox_subnet {
            bbox_out = conv.forward(&bbox_out, training);
            bbox_out = ReLU::new().forward(&bbox_out);
        }
        let bbox_pred = self.bbox_pred.forward(&bbox_out, training);
        
        (cls_logits, bbox_pred)
    }
}

/// RetinaNet Complete Model
pub struct RetinaNet {
    backbone: ResNetBackbone,
    fpn: FeaturePyramidNetwork,
    head: RetinaNetHead,
    num_classes: usize,
}

impl RetinaNet {
    pub fn new(num_classes: usize) -> Self {
        RetinaNet {
            backbone: ResNetBackbone::new(),
            fpn: FeaturePyramidNetwork::new(vec![256, 512, 1024, 2048], 256),
            head: RetinaNetHead::new(256, num_classes, 9),
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<(Tensor, Tensor)> {
        // Extract features
        let backbone_features = vec![self.backbone.forward(x, training)];
        
        // Build feature pyramid
        let fpn_features = self.fpn.forward(backbone_features, training);
        
        // Apply detection head to each FPN level
        fpn_features.iter()
            .map(|feat| self.head.forward(feat, training))
            .collect()
    }
}

/// SSD (Single Shot Detector) Extra Layers
pub struct SSDExtraLayers {
    layers: Vec<(Conv2d, BatchNorm2d, Conv2d, BatchNorm2d)>,
}

impl SSDExtraLayers {
    pub fn new() -> Self {
        SSDExtraLayers {
            layers: vec![
                (
                    Conv2d::new(1024, 256, (1, 1)),
                    BatchNorm2d::new(256),
                    Conv2d::new(256, 512, (3, 3)).stride((2, 2)).padding((1, 1)),
                    BatchNorm2d::new(512),
                ),
                (
                    Conv2d::new(512, 128, (1, 1)),
                    BatchNorm2d::new(128),
                    Conv2d::new(128, 256, (3, 3)).stride((2, 2)).padding((1, 1)),
                    BatchNorm2d::new(256),
                ),
            ],
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let mut features = vec![x.clone()];
        let mut out = x.clone();
        
        for (conv1, bn1, conv2, bn2) in &mut self.layers {
            out = conv1.forward(&out, training);
            out = bn1.forward(&out, training);
            out = ReLU::new().forward(&out);
            
            out = conv2.forward(&out, training);
            out = bn2.forward(&out, training);
            out = ReLU::new().forward(&out);
            
            features.push(out.clone());
        }
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolov3() {
        let mut model = YOLOv3::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 416 * 416], &[1, 3, 416, 416]).unwrap();
        let outputs = model.forward(&input, false);
        assert!(outputs.len() > 0);
    }

    #[test]
    fn test_faster_rcnn() {
        let mut model = FasterRCNN::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let (cls, bbox) = model.forward(&input, false);
        assert_eq!(cls.dims()[1], 80);
    }

    #[test]
    fn test_retinanet() {
        let mut model = RetinaNet::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let outputs = model.forward(&input, false);
        assert!(outputs.len() > 0);
    }
}

/// YOLO v4 CSPDarknet Backbone
pub struct CSPDarknet53 {
    stem: Conv2d,
    stages: Vec<CSPStage>,
}

struct CSPStage {
    downsample: Conv2d,
    split_conv: Conv2d,
    blocks: Vec<ResidualBlock>,
    concat_conv: Conv2d,
}

impl CSPStage {
    fn new(in_channels: usize, out_channels: usize, num_blocks: usize) -> Self {
        CSPStage {
            downsample: Conv2d::new(in_channels, out_channels, (3, 3)).stride((2, 2)).padding((1, 1)),
            split_conv: Conv2d::new(out_channels, out_channels / 2, (1, 1)),
            blocks: (0..num_blocks).map(|_| ResidualBlock::new(out_channels / 2, out_channels / 2)).collect(),
            concat_conv: Conv2d::new(out_channels, out_channels, (1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.downsample.forward(x, training);
        let split = self.split_conv.forward(&out, training);
        
        let mut block_out = split.clone();
        for block in &mut self.blocks {
            block_out = block.forward(&block_out, training);
        }
        
        let concat = self.concatenate(&split, &block_out);
        self.concat_conv.forward(&concat, training)
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

impl CSPDarknet53 {
    pub fn new() -> Self {
        CSPDarknet53 {
            stem: Conv2d::new(3, 32, (3, 3)).padding((1, 1)),
            stages: vec![
                CSPStage::new(32, 64, 1),
                CSPStage::new(64, 128, 2),
                CSPStage::new(128, 256, 8),
                CSPStage::new(256, 512, 8),
                CSPStage::new(512, 1024, 4),
            ],
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let mut out = self.stem.forward(x, training);
        let mut features = Vec::new();
        
        for (i, stage) in self.stages.iter_mut().enumerate() {
            out = stage.forward(&out, training);
            if i >= 2 {
                features.push(out.clone());
            }
        }
        
        features
    }
}

/// YOLO v4
pub struct YOLOv4 {
    backbone: CSPDarknet53,
    neck: PANet,
    heads: Vec<YOLOv3DetectionLayer>,
}

struct PANet {
    fpn_convs: Vec<Conv2d>,
    pan_convs: Vec<Conv2d>,
}

impl PANet {
    fn new() -> Self {
        PANet {
            fpn_convs: vec![
                Conv2d::new(1024, 512, (1, 1)),
                Conv2d::new(512, 256, (1, 1)),
            ],
            pan_convs: vec![
                Conv2d::new(256, 512, (3, 3)).padding((1, 1)),
                Conv2d::new(512, 1024, (3, 3)).padding((1, 1)),
            ],
        }
    }

    fn forward(&mut self, features: Vec<Tensor>, training: bool) -> Vec<Tensor> {
        // Simplified PANet
        features
    }
}

impl YOLOv4 {
    pub fn new(num_classes: usize) -> Self {
        let anchors1 = vec![(12.0, 16.0), (19.0, 36.0), (40.0, 28.0)];
        let anchors2 = vec![(36.0, 75.0), (76.0, 55.0), (72.0, 146.0)];
        let anchors3 = vec![(142.0, 110.0), (192.0, 243.0), (459.0, 401.0)];
        
        YOLOv4 {
            backbone: CSPDarknet53::new(),
            neck: PANet::new(),
            heads: vec![
                YOLOv3DetectionLayer::new(512, num_classes, anchors1),
                YOLOv3DetectionLayer::new(256, num_classes, anchors2),
                YOLOv3DetectionLayer::new(128, num_classes, anchors3),
            ],
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let features = self.backbone.forward(x, training);
        let neck_features = self.neck.forward(features, training);
        
        neck_features.iter()
            .zip(self.heads.iter_mut())
            .map(|(feat, head)| head.forward(feat, training))
            .collect()
    }
}

/// EfficientDet
pub struct EfficientDet {
    backbone: EfficientNetBackbone,
    bifpn: BiFPN,
    heads: Vec<EfficientDetHead>,
    num_classes: usize,
}

struct EfficientNetBackbone {
    conv1: Conv2d,
    blocks: Vec<MBConvBlock>,
}

struct MBConvBlock {
    expand: Conv2d,
    depthwise: Conv2d,
    project: Conv2d,
}

impl MBConvBlock {
    fn new(in_channels: usize, out_channels: usize, expand_ratio: usize) -> Self {
        let expanded = in_channels * expand_ratio;
        MBConvBlock {
            expand: Conv2d::new(in_channels, expanded, (1, 1)),
            depthwise: Conv2d::new(expanded, expanded, (3, 3)).padding((1, 1)),
            project: Conv2d::new(expanded, out_channels, (1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.expand.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.depthwise.forward(&out, training);
        out = ReLU::new().forward(&out);
        
        self.project.forward(&out, training)
    }
}

impl EfficientNetBackbone {
    fn new() -> Self {
        EfficientNetBackbone {
            conv1: Conv2d::new(3, 32, (3, 3)).stride((2, 2)).padding((1, 1)),
            blocks: vec![
                MBConvBlock::new(32, 16, 1),
                MBConvBlock::new(16, 24, 6),
                MBConvBlock::new(24, 40, 6),
                MBConvBlock::new(40, 80, 6),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let mut out = self.conv1.forward(x, training);
        let mut features = Vec::new();
        
        for block in &mut self.blocks {
            out = block.forward(&out, training);
            features.push(out.clone());
        }
        
        features
    }
}

struct BiFPN {
    layers: Vec<BiFPNLayer>,
}

struct BiFPNLayer {
    convs: Vec<Conv2d>,
}

impl BiFPNLayer {
    fn new(num_channels: usize, num_levels: usize) -> Self {
        BiFPNLayer {
            convs: (0..num_levels).map(|_| Conv2d::new(num_channels, num_channels, (3, 3)).padding((1, 1))).collect(),
        }
    }

    fn forward(&mut self, features: Vec<Tensor>, training: bool) -> Vec<Tensor> {
        features.iter()
            .zip(self.convs.iter_mut())
            .map(|(feat, conv)| conv.forward(feat, training))
            .collect()
    }
}

impl BiFPN {
    fn new(num_channels: usize, num_levels: usize, num_layers: usize) -> Self {
        BiFPN {
            layers: (0..num_layers).map(|_| BiFPNLayer::new(num_channels, num_levels)).collect(),
        }
    }

    fn forward(&mut self, mut features: Vec<Tensor>, training: bool) -> Vec<Tensor> {
        for layer in &mut self.layers {
            features = layer.forward(features, training);
        }
        features
    }
}

struct EfficientDetHead {
    cls_convs: Vec<Conv2d>,
    box_convs: Vec<Conv2d>,
}

impl EfficientDetHead {
    fn new(in_channels: usize, num_classes: usize, num_anchors: usize) -> Self {
        EfficientDetHead {
            cls_convs: vec![
                Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)),
                Conv2d::new(in_channels, num_anchors * num_classes, (3, 3)).padding((1, 1)),
            ],
            box_convs: vec![
                Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)),
                Conv2d::new(in_channels, num_anchors * 4, (3, 3)).padding((1, 1)),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let mut cls_out = x.clone();
        for conv in &mut self.cls_convs {
            cls_out = conv.forward(&cls_out, training);
        }
        
        let mut box_out = x.clone();
        for conv in &mut self.box_convs {
            box_out = conv.forward(&box_out, training);
        }
        
        (cls_out, box_out)
    }
}

impl EfficientDet {
    pub fn new(num_classes: usize) -> Self {
        EfficientDet {
            backbone: EfficientNetBackbone::new(),
            bifpn: BiFPN::new(64, 5, 3),
            heads: (0..5).map(|_| EfficientDetHead::new(64, num_classes, 9)).collect(),
            num_classes,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<(Tensor, Tensor)> {
        let features = self.backbone.forward(x, training);
        let bifpn_features = self.bifpn.forward(features, training);
        
        bifpn_features.iter()
            .zip(self.heads.iter_mut())
            .map(|(feat, head)| head.forward(feat, training))
            .collect()
    }
}

/// DETR (Detection Transformer)
pub struct DETR {
    backbone: ResNetBackbone,
    transformer: DETRTransformer,
    class_embed: Dense,
    bbox_embed: Dense,
    num_queries: usize,
}

struct DETRTransformer {
    encoder_layers: Vec<TransformerEncoderLayer>,
    decoder_layers: Vec<TransformerDecoderLayer>,
}

struct TransformerEncoderLayer {
    self_attn: Dense,
    ffn: Vec<Dense>,
}

impl TransformerEncoderLayer {
    fn new(d_model: usize) -> Self {
        TransformerEncoderLayer {
            self_attn: Dense::new(d_model, d_model),
            ffn: vec![
                Dense::new(d_model, d_model * 4),
                Dense::new(d_model * 4, d_model),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let attn_out = self.self_attn.forward(x, training);
        
        let mut ffn_out = attn_out;
        for (i, layer) in self.ffn.iter_mut().enumerate() {
            ffn_out = layer.forward(&ffn_out, training);
            if i == 0 {
                ffn_out = ReLU::new().forward(&ffn_out);
            }
        }
        
        ffn_out
    }
}

struct TransformerDecoderLayer {
    self_attn: Dense,
    cross_attn: Dense,
    ffn: Vec<Dense>,
}

impl TransformerDecoderLayer {
    fn new(d_model: usize) -> Self {
        TransformerDecoderLayer {
            self_attn: Dense::new(d_model, d_model),
            cross_attn: Dense::new(d_model, d_model),
            ffn: vec![
                Dense::new(d_model, d_model * 4),
                Dense::new(d_model * 4, d_model),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, memory: &Tensor, training: bool) -> Tensor {
        let self_attn_out = self.self_attn.forward(x, training);
        let cross_attn_out = self.cross_attn.forward(&self_attn_out, training);
        
        let mut ffn_out = cross_attn_out;
        for (i, layer) in self.ffn.iter_mut().enumerate() {
            ffn_out = layer.forward(&ffn_out, training);
            if i == 0 {
                ffn_out = ReLU::new().forward(&ffn_out);
            }
        }
        
        ffn_out
    }
}

impl DETRTransformer {
    fn new(d_model: usize, num_encoder_layers: usize, num_decoder_layers: usize) -> Self {
        DETRTransformer {
            encoder_layers: (0..num_encoder_layers).map(|_| TransformerEncoderLayer::new(d_model)).collect(),
            decoder_layers: (0..num_decoder_layers).map(|_| TransformerDecoderLayer::new(d_model)).collect(),
        }
    }

    fn forward(&mut self, src: &Tensor, query_embed: &Tensor, training: bool) -> Tensor {
        let mut memory = src.clone();
        for layer in &mut self.encoder_layers {
            memory = layer.forward(&memory, training);
        }
        
        let mut tgt = query_embed.clone();
        for layer in &mut self.decoder_layers {
            tgt = layer.forward(&tgt, &memory, training);
        }
        
        tgt
    }
}

impl DETR {
    pub fn new(num_classes: usize, num_queries: usize) -> Self {
        let d_model = 256;
        
        DETR {
            backbone: ResNetBackbone::new(),
            transformer: DETRTransformer::new(d_model, 6, 6),
            class_embed: Dense::new(d_model, num_classes + 1),
            bbox_embed: Dense::new(d_model, 4),
            num_queries,
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor) {
        let features = self.backbone.forward(x, training);
        
        // Create query embeddings
        let query_embed = Tensor::from_slice(&vec![0.1f32; self.num_queries * 256], 
                                             &[self.num_queries, 256]).unwrap();
        
        let hs = self.transformer.forward(&features, &query_embed, training);
        
        let class_logits = self.class_embed.forward(&hs, training);
        let bbox_pred = self.bbox_embed.forward(&hs, training);
        
        (class_logits, bbox_pred)
    }
}

#[cfg(test)]
mod tests_extended {
    use super::*;

    #[test]
    fn test_yolov4() {
        let mut model = YOLOv4::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 416 * 416], &[1, 3, 416, 416]).unwrap();
        let outputs = model.forward(&input, false);
        assert!(outputs.len() > 0);
    }

    #[test]
    fn test_efficientdet() {
        let mut model = EfficientDet::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 512 * 512], &[1, 3, 512, 512]).unwrap();
        let outputs = model.forward(&input, false);
        assert!(outputs.len() > 0);
    }

    #[test]
    fn test_detr() {
        let mut model = DETR::new(80, 100);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 224 * 224], &[1, 3, 224, 224]).unwrap();
        let (cls, bbox) = model.forward(&input, false);
        assert_eq!(cls.dims()[0], 100);
    }
}


/// YOLO v5
pub struct YOLOv5 {
    backbone: YOLOv5Backbone,
    neck: YOLOv5Neck,
    heads: Vec<YOLOv5Head>,
}

struct YOLOv5Backbone {
    focus: FocusLayer,
    csp_blocks: Vec<CSPBlock>,
}

struct FocusLayer {
    conv: Conv2d,
}

impl FocusLayer {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        FocusLayer {
            conv: Conv2d::new(in_channels * 4, out_channels, (3, 3)).padding((1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let sliced = self.slice_tensor(x);
        self.conv.forward(&sliced, training)
    }

    fn slice_tensor(&self, x: &Tensor) -> Tensor {
        // Simplified focus operation
        x.clone()
    }
}

struct CSPBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    bottlenecks: Vec<Bottleneck>,
}

struct Bottleneck {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl Bottleneck {
    fn new(channels: usize) -> Self {
        Bottleneck {
            conv1: Conv2d::new(channels, channels, (1, 1)),
            conv2: Conv2d::new(channels, channels, (3, 3)).padding((1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let identity = x.clone();
        
        let mut out = self.conv1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        out = ReLU::new().forward(&out);
        
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

impl CSPBlock {
    fn new(in_channels: usize, out_channels: usize, num_bottlenecks: usize) -> Self {
        CSPBlock {
            conv1: Conv2d::new(in_channels, out_channels / 2, (1, 1)),
            conv2: Conv2d::new(in_channels, out_channels / 2, (1, 1)),
            bottlenecks: (0..num_bottlenecks).map(|_| Bottleneck::new(out_channels / 2)).collect(),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let part1 = self.conv1.forward(x, training);
        
        let mut part2 = self.conv2.forward(x, training);
        for bottleneck in &mut self.bottlenecks {
            part2 = bottleneck.forward(&part2, training);
        }
        
        self.concatenate(&part1, &part2)
    }

    fn concatenate(&self, a: &Tensor, b: &Tensor) -> Tensor {
        a.clone() // Simplified
    }
}

impl YOLOv5Backbone {
    fn new() -> Self {
        YOLOv5Backbone {
            focus: FocusLayer::new(3, 64),
            csp_blocks: vec![
                CSPBlock::new(64, 128, 3),
                CSPBlock::new(128, 256, 9),
                CSPBlock::new(256, 512, 9),
                CSPBlock::new(512, 1024, 3),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let mut out = self.focus.forward(x, training);
        let mut features = Vec::new();
        
        for block in &mut self.csp_blocks {
            out = block.forward(&out, training);
            features.push(out.clone());
        }
        
        features
    }
}

struct YOLOv5Neck {
    spp: SPPLayer,
    upsamples: Vec<Conv2d>,
}

struct SPPLayer {
    pools: Vec<MaxPool2d>,
}

impl SPPLayer {
    fn new() -> Self {
        SPPLayer {
            pools: vec![
                MaxPool2d::new((5, 5), (1, 1), (2, 2)),
                MaxPool2d::new((9, 9), (1, 1), (4, 4)),
                MaxPool2d::new((13, 13), (1, 1), (6, 6)),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut outputs = vec![x.clone()];
        
        for pool in &mut self.pools {
            outputs.push(pool.forward(x));
        }
        
        self.concatenate(&outputs)
    }

    fn concatenate(&self, tensors: &[Tensor]) -> Tensor {
        tensors[0].clone() // Simplified
    }
}

impl YOLOv5Neck {
    fn new() -> Self {
        YOLOv5Neck {
            spp: SPPLayer::new(),
            upsamples: vec![
                Conv2d::new(1024, 512, (1, 1)),
                Conv2d::new(512, 256, (1, 1)),
            ],
        }
    }

    fn forward(&mut self, features: Vec<Tensor>, training: bool) -> Vec<Tensor> {
        let mut out = self.spp.forward(&features[features.len() - 1]);
        let mut neck_features = vec![out.clone()];
        
        for upsample in &mut self.upsamples {
            out = upsample.forward(&out, training);
            neck_features.push(out.clone());
        }
        
        neck_features
    }
}

struct YOLOv5Head {
    conv: Conv2d,
}

impl YOLOv5Head {
    fn new(in_channels: usize, num_classes: usize, num_anchors: usize) -> Self {
        YOLOv5Head {
            conv: Conv2d::new(in_channels, num_anchors * (5 + num_classes), (1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        self.conv.forward(x, training)
    }
}

impl YOLOv5 {
    pub fn new(num_classes: usize) -> Self {
        YOLOv5 {
            backbone: YOLOv5Backbone::new(),
            neck: YOLOv5Neck::new(),
            heads: vec![
                YOLOv5Head::new(256, num_classes, 3),
                YOLOv5Head::new(512, num_classes, 3),
                YOLOv5Head::new(1024, num_classes, 3),
            ],
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let backbone_features = self.backbone.forward(x, training);
        let neck_features = self.neck.forward(backbone_features, training);
        
        neck_features.iter()
            .zip(self.heads.iter_mut())
            .map(|(feat, head)| head.forward(feat, training))
            .collect()
    }
}

/// YOLOX
pub struct YOLOX {
    backbone: YOLOXBackbone,
    neck: YOLOXNeck,
    head: YOLOXHead,
}

struct YOLOXBackbone {
    stem: Conv2d,
    dark_blocks: Vec<DarkBlock>,
}

struct DarkBlock {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl DarkBlock {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        DarkBlock {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3)).stride((2, 2)).padding((1, 1)),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3)).padding((1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = self.conv1.forward(x, training);
        out = ReLU::new().forward(&out);
        
        out = self.conv2.forward(&out, training);
        ReLU::new().forward(&out)
    }
}

impl YOLOXBackbone {
    fn new() -> Self {
        YOLOXBackbone {
            stem: Conv2d::new(3, 32, (3, 3)).padding((1, 1)),
            dark_blocks: vec![
                DarkBlock::new(32, 64),
                DarkBlock::new(64, 128),
                DarkBlock::new(128, 256),
                DarkBlock::new(256, 512),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Vec<Tensor> {
        let mut out = self.stem.forward(x, training);
        out = ReLU::new().forward(&out);
        
        let mut features = Vec::new();
        for block in &mut self.dark_blocks {
            out = block.forward(&out, training);
            features.push(out.clone());
        }
        
        features
    }
}

struct YOLOXNeck {
    lateral_convs: Vec<Conv2d>,
}

impl YOLOXNeck {
    fn new() -> Self {
        YOLOXNeck {
            lateral_convs: vec![
                Conv2d::new(512, 256, (1, 1)),
                Conv2d::new(256, 128, (1, 1)),
            ],
        }
    }

    fn forward(&mut self, features: Vec<Tensor>, training: bool) -> Vec<Tensor> {
        features.iter()
            .zip(self.lateral_convs.iter_mut())
            .map(|(feat, conv)| conv.forward(feat, training))
            .collect()
    }
}

struct YOLOXHead {
    cls_convs: Vec<Conv2d>,
    reg_convs: Vec<Conv2d>,
    cls_pred: Conv2d,
    reg_pred: Conv2d,
    obj_pred: Conv2d,
}

impl YOLOXHead {
    fn new(in_channels: usize, num_classes: usize) -> Self {
        YOLOXHead {
            cls_convs: vec![
                Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)),
                Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)),
            ],
            reg_convs: vec![
                Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)),
                Conv2d::new(in_channels, in_channels, (3, 3)).padding((1, 1)),
            ],
            cls_pred: Conv2d::new(in_channels, num_classes, (1, 1)),
            reg_pred: Conv2d::new(in_channels, 4, (1, 1)),
            obj_pred: Conv2d::new(in_channels, 1, (1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let mut cls_feat = x.clone();
        for conv in &mut self.cls_convs {
            cls_feat = conv.forward(&cls_feat, training);
            cls_feat = ReLU::new().forward(&cls_feat);
        }
        
        let mut reg_feat = x.clone();
        for conv in &mut self.reg_convs {
            reg_feat = conv.forward(&reg_feat, training);
            reg_feat = ReLU::new().forward(&reg_feat);
        }
        
        let cls_output = self.cls_pred.forward(&cls_feat, training);
        let reg_output = self.reg_pred.forward(&reg_feat, training);
        let obj_output = self.obj_pred.forward(&reg_feat, training);
        
        (cls_output, reg_output, obj_output)
    }
}

impl YOLOX {
    pub fn new(num_classes: usize) -> Self {
        YOLOX {
            backbone: YOLOXBackbone::new(),
            neck: YOLOXNeck::new(),
            head: YOLOXHead::new(256, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> Vec<(Tensor, Tensor, Tensor)> {
        let backbone_features = self.backbone.forward(x, training);
        let neck_features = self.neck.forward(backbone_features, training);
        
        neck_features.iter()
            .map(|feat| self.head.forward(feat, training))
            .collect()
    }
}

/// CenterNet
pub struct CenterNet {
    backbone: CenterNetBackbone,
    head: CenterNetHead,
}

struct CenterNetBackbone {
    layers: Vec<Conv2d>,
}

impl CenterNetBackbone {
    fn new() -> Self {
        CenterNetBackbone {
            layers: vec![
                Conv2d::new(3, 64, (7, 7)).stride((2, 2)).padding((3, 3)),
                Conv2d::new(64, 128, (3, 3)).stride((2, 2)).padding((1, 1)),
                Conv2d::new(128, 256, (3, 3)).stride((2, 2)).padding((1, 1)),
            ],
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Tensor {
        let mut out = x.clone();
        
        for layer in &mut self.layers {
            out = layer.forward(&out, training);
            out = ReLU::new().forward(&out);
        }
        
        out
    }
}

struct CenterNetHead {
    heatmap_head: Conv2d,
    wh_head: Conv2d,
    offset_head: Conv2d,
}

impl CenterNetHead {
    fn new(in_channels: usize, num_classes: usize) -> Self {
        CenterNetHead {
            heatmap_head: Conv2d::new(in_channels, num_classes, (1, 1)),
            wh_head: Conv2d::new(in_channels, 2, (1, 1)),
            offset_head: Conv2d::new(in_channels, 2, (1, 1)),
        }
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let heatmap = self.heatmap_head.forward(x, training);
        let wh = self.wh_head.forward(x, training);
        let offset = self.offset_head.forward(x, training);
        
        (heatmap, wh, offset)
    }
}

impl CenterNet {
    pub fn new(num_classes: usize) -> Self {
        CenterNet {
            backbone: CenterNetBackbone::new(),
            head: CenterNetHead::new(256, num_classes),
        }
    }

    pub fn forward(&mut self, x: &Tensor, training: bool) -> (Tensor, Tensor, Tensor) {
        let features = self.backbone.forward(x, training);
        self.head.forward(&features, training)
    }
}

#[cfg(test)]
mod tests_yolo_variants {
    use super::*;

    #[test]
    fn test_yolov5() {
        let mut model = YOLOv5::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 640 * 640], &[1, 3, 640, 640]).unwrap();
        let outputs = model.forward(&input, false);
        assert!(outputs.len() > 0);
    }

    #[test]
    fn test_yolox() {
        let mut model = YOLOX::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 640 * 640], &[1, 3, 640, 640]).unwrap();
        let outputs = model.forward(&input, false);
        assert!(outputs.len() > 0);
    }

    #[test]
    fn test_centernet() {
        let mut model = CenterNet::new(80);
        let input = Tensor::from_slice(&vec![0.5f32; 1 * 3 * 512 * 512], &[1, 3, 512, 512]).unwrap();
        let (heatmap, wh, offset) = model.forward(&input, false);
        assert_eq!(heatmap.dims()[1], 80);
    }
}
