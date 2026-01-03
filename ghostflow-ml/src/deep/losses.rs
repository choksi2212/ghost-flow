//! Loss Functions - MSE, CrossEntropy, BCE, Huber, etc.

use ghostflow_core::Tensor;

/// Loss function trait
pub trait Loss: Send + Sync {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32;
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}

/// Mean Squared Error Loss
pub struct MSELoss {
    pub reduction: Reduction,
}

#[derive(Clone, Copy)]
pub enum Reduction {
    Mean,
    Sum,
    None,
}

impl MSELoss {
    pub fn new() -> Self {
        MSELoss { reduction: Reduction::Mean }
    }

    pub fn reduction(mut self, r: Reduction) -> Self {
        self.reduction = r;
        self
    }
}

impl Default for MSELoss {
    fn default() -> Self { Self::new() }
}

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        
        let sum: f32 = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum();

        match self.reduction {
            Reduction::Mean => sum / pred.len() as f32,
            Reduction::Sum => sum,
            Reduction::None => sum,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let n = pred.len();

        let grad: Vec<f32> = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let g = 2.0 * (p - t);
                match self.reduction {
                    Reduction::Mean => g / n as f32,
                    _ => g,
                }
            })
            .collect();

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

/// Cross Entropy Loss (for classification)
pub struct CrossEntropyLoss {
    pub reduction: Reduction,
    pub label_smoothing: f32,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss {
            reduction: Reduction::Mean,
            label_smoothing: 0.0,
        }
    }

    pub fn label_smoothing(mut self, ls: f32) -> Self {
        self.label_smoothing = ls.clamp(0.0, 1.0);
        self
    }

    fn softmax(logits: &[f32], n_classes: usize) -> Vec<f32> {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
        logits.iter().map(|&x| (x - max_val).exp() / exp_sum).collect()
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self { Self::new() }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let batch_size = predictions.dims()[0];
        let n_classes = predictions.dims()[1];
        let eps = 1e-10f32;

        let mut total_loss = 0.0f32;

        for b in 0..batch_size {
            let logits = &pred[b * n_classes..(b + 1) * n_classes];
            let probs = Self::softmax(logits, n_classes);
            let target_class = targ[b] as usize;

            if self.label_smoothing > 0.0 {
                let smooth = self.label_smoothing / n_classes as f32;
                for c in 0..n_classes {
                    let target_prob = if c == target_class {
                        1.0 - self.label_smoothing + smooth
                    } else {
                        smooth
                    };
                    total_loss -= target_prob * (probs[c] + eps).ln();
                }
            } else {
                total_loss -= (probs[target_class] + eps).ln();
            }
        }

        match self.reduction {
            Reduction::Mean => total_loss / batch_size as f32,
            Reduction::Sum => total_loss,
            Reduction::None => total_loss,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let batch_size = predictions.dims()[0];
        let n_classes = predictions.dims()[1];

        let mut grad = vec![0.0f32; batch_size * n_classes];

        for b in 0..batch_size {
            let logits = &pred[b * n_classes..(b + 1) * n_classes];
            let probs = Self::softmax(logits, n_classes);
            let target_class = targ[b] as usize;

            for c in 0..n_classes {
                let target_prob = if c == target_class { 1.0 } else { 0.0 };
                grad[b * n_classes + c] = probs[c] - target_prob;
            }
        }

        if matches!(self.reduction, Reduction::Mean) {
            for g in &mut grad {
                *g /= batch_size as f32;
            }
        }

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

/// Binary Cross Entropy Loss
pub struct BCELoss {
    pub reduction: Reduction,
}

impl BCELoss {
    pub fn new() -> Self {
        BCELoss { reduction: Reduction::Mean }
    }
}

impl Default for BCELoss {
    fn default() -> Self { Self::new() }
}

impl Loss for BCELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let eps = 1e-10f32;

        let sum: f32 = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let p_clipped = p.clamp(eps, 1.0 - eps);
                -t * p_clipped.ln() - (1.0 - t) * (1.0 - p_clipped).ln()
            })
            .sum();

        match self.reduction {
            Reduction::Mean => sum / pred.len() as f32,
            Reduction::Sum => sum,
            Reduction::None => sum,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let n = pred.len();
        let eps = 1e-10f32;

        let grad: Vec<f32> = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let p_clipped = p.clamp(eps, 1.0 - eps);
                let g = -t / p_clipped + (1.0 - t) / (1.0 - p_clipped);
                match self.reduction {
                    Reduction::Mean => g / n as f32,
                    _ => g,
                }
            })
            .collect();

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

/// BCE with Logits Loss (numerically stable)
pub struct BCEWithLogitsLoss {
    pub reduction: Reduction,
    pub pos_weight: Option<f32>,
}

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        BCEWithLogitsLoss {
            reduction: Reduction::Mean,
            pos_weight: None,
        }
    }

    pub fn pos_weight(mut self, w: f32) -> Self {
        self.pos_weight = Some(w);
        self
    }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self { Self::new() }
}

impl Loss for BCEWithLogitsLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();

        let sum: f32 = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let max_val = p.max(0.0);
                let loss = max_val - p * t + ((-max_val).exp() + (p - max_val).exp()).ln();
                if let Some(w) = self.pos_weight {
                    t * w * loss + (1.0 - t) * loss
                } else {
                    loss
                }
            })
            .sum();

        match self.reduction {
            Reduction::Mean => sum / pred.len() as f32,
            Reduction::Sum => sum,
            Reduction::None => sum,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let n = pred.len();

        let grad: Vec<f32> = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let sig = Self::sigmoid(p);
                let g = sig - t;
                match self.reduction {
                    Reduction::Mean => g / n as f32,
                    _ => g,
                }
            })
            .collect();

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

/// Huber Loss (Smooth L1)
pub struct HuberLoss {
    pub delta: f32,
    pub reduction: Reduction,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        HuberLoss {
            delta,
            reduction: Reduction::Mean,
        }
    }
}

impl Loss for HuberLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();

        let sum: f32 = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let diff = (p - t).abs();
                if diff <= self.delta {
                    0.5 * diff * diff
                } else {
                    self.delta * (diff - 0.5 * self.delta)
                }
            })
            .sum();

        match self.reduction {
            Reduction::Mean => sum / pred.len() as f32,
            Reduction::Sum => sum,
            Reduction::None => sum,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let n = pred.len();

        let grad: Vec<f32> = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                let g = if diff.abs() <= self.delta {
                    diff
                } else {
                    self.delta * diff.signum()
                };
                match self.reduction {
                    Reduction::Mean => g / n as f32,
                    _ => g,
                }
            })
            .collect();

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

/// L1 Loss (Mean Absolute Error)
pub struct L1Loss {
    pub reduction: Reduction,
}

impl L1Loss {
    pub fn new() -> Self {
        L1Loss { reduction: Reduction::Mean }
    }
}

impl Default for L1Loss {
    fn default() -> Self { Self::new() }
}

impl Loss for L1Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();

        let sum: f32 = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum();

        match self.reduction {
            Reduction::Mean => sum / pred.len() as f32,
            Reduction::Sum => sum,
            Reduction::None => sum,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let n = pred.len();

        let grad: Vec<f32> = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let g = (p - t).signum();
                match self.reduction {
                    Reduction::Mean => g / n as f32,
                    _ => g,
                }
            })
            .collect();

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

/// Hinge Loss (for SVM-style classification)
pub struct HingeLoss {
    pub margin: f32,
    pub reduction: Reduction,
}

impl HingeLoss {
    pub fn new() -> Self {
        HingeLoss {
            margin: 1.0,
            reduction: Reduction::Mean,
        }
    }
}

impl Default for HingeLoss {
    fn default() -> Self { Self::new() }
}

impl Loss for HingeLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();

        let sum: f32 = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                // Convert 0/1 to -1/+1
                let t_signed = if t > 0.5 { 1.0 } else { -1.0 };
                (self.margin - t_signed * p).max(0.0)
            })
            .sum();

        match self.reduction {
            Reduction::Mean => sum / pred.len() as f32,
            Reduction::Sum => sum,
            Reduction::None => sum,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let n = pred.len();

        let grad: Vec<f32> = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                let t_signed = if t > 0.5 { 1.0 } else { -1.0 };
                let g = if self.margin - t_signed * p > 0.0 {
                    -t_signed
                } else {
                    0.0
                };
                match self.reduction {
                    Reduction::Mean => g / n as f32,
                    _ => g,
                }
            })
            .collect();

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

/// KL Divergence Loss
pub struct KLDivLoss {
    pub reduction: Reduction,
    pub log_target: bool,
}

impl KLDivLoss {
    pub fn new() -> Self {
        KLDivLoss {
            reduction: Reduction::Mean,
            log_target: false,
        }
    }
}

impl Default for KLDivLoss {
    fn default() -> Self { Self::new() }
}

impl Loss for KLDivLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let pred = predictions.data_f32(); // log probabilities
        let targ = targets.data_f32();
        let eps = 1e-10f32;

        let sum: f32 = pred.iter().zip(targ.iter())
            .map(|(&p, &t)| {
                if self.log_target {
                    (t.exp()) * (t - p)
                } else {
                    t * ((t + eps).ln() - p)
                }
            })
            .sum();

        match self.reduction {
            Reduction::Mean => sum / pred.len() as f32,
            Reduction::Sum => sum,
            Reduction::None => sum,
        }
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let pred = predictions.data_f32();
        let targ = targets.data_f32();
        let n = pred.len();

        let grad: Vec<f32> = pred.iter().zip(targ.iter())
            .map(|(&_p, &t)| {
                let g = if self.log_target { -t.exp() } else { -t };
                match self.reduction {
                    Reduction::Mean => g / n as f32,
                    _ => g,
                }
            })
            .collect();

        Tensor::from_slice(&grad, predictions.dims()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let targ = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let loss = MSELoss::new();
        assert!((loss.forward(&pred, &targ) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss() {
        let pred = Tensor::from_slice(&[0.9, 0.1], &[2]).unwrap();
        let targ = Tensor::from_slice(&[1.0, 0.0], &[2]).unwrap();
        let loss = BCELoss::new();
        let l = loss.forward(&pred, &targ);
        assert!(l > 0.0 && l < 1.0);
    }
}


/// Focal Loss - for addressing class imbalance
/// Focuses training on hard examples
pub struct FocalLoss {
    pub alpha: f32,
    pub gamma: f32,
}

impl FocalLoss {
    pub fn new(alpha: f32, gamma: f32) -> Self {
        FocalLoss { alpha, gamma }
    }

    pub fn forward(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        let eps = 1e-7f32;
        let mut loss = 0.0f32;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_clipped = pred.clamp(eps, 1.0 - eps);
            let pt = if target > 0.5 { pred_clipped } else { 1.0 - pred_clipped };
            let focal_weight = (1.0 - pt).powf(self.gamma);
            let ce = -target * pred_clipped.ln() - (1.0 - target) * (1.0 - pred_clipped).ln();
            loss += self.alpha * focal_weight * ce;
        }

        loss / predictions.len() as f32
    }

    pub fn backward(&self, predictions: &[f32], targets: &[f32]) -> Vec<f32> {
        let eps = 1e-7f32;
        let mut grads = Vec::with_capacity(predictions.len());

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_clipped = pred.clamp(eps, 1.0 - eps);
            let pt = if target > 0.5 { pred_clipped } else { 1.0 - pred_clipped };
            let focal_weight = (1.0 - pt).powf(self.gamma);
            
            // Gradient computation
            let grad = if target > 0.5 {
                self.alpha * focal_weight * (self.gamma * pt.ln() * (1.0 - pt) - 1.0) / pred_clipped
            } else {
                self.alpha * focal_weight * (self.gamma * (1.0 - pt).ln() * pt + 1.0) / (1.0 - pred_clipped)
            };
            
            grads.push(grad / predictions.len() as f32);
        }

        grads
    }
}

/// Dice Loss - for segmentation tasks
pub struct DiceLoss {
    pub smooth: f32,
}

impl DiceLoss {
    pub fn new() -> Self {
        DiceLoss { smooth: 1.0 }
    }

    pub fn smooth(mut self, s: f32) -> Self {
        self.smooth = s;
        self
    }

    pub fn forward(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        let mut intersection = 0.0f32;
        let mut pred_sum = 0.0f32;
        let mut target_sum = 0.0f32;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            intersection += pred * target;
            pred_sum += pred;
            target_sum += target;
        }

        1.0 - (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
    }

    pub fn backward(&self, predictions: &[f32], targets: &[f32]) -> Vec<f32> {
        let mut intersection = 0.0f32;
        let mut pred_sum = 0.0f32;
        let mut target_sum = 0.0f32;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            intersection += pred * target;
            pred_sum += pred;
            target_sum += target;
        }

        let denominator = pred_sum + target_sum + self.smooth;
        let numerator = 2.0 * intersection + self.smooth;

        predictions.iter().zip(targets.iter())
            .map(|(&pred, &target)| {
                -2.0 * (target * denominator - numerator) / (denominator * denominator)
            })
            .collect()
    }
}

impl Default for DiceLoss {
    fn default() -> Self { Self::new() }
}

/// Tversky Loss - generalization of Dice loss
pub struct TverskyLoss {
    pub alpha: f32,
    pub beta: f32,
    pub smooth: f32,
}

impl TverskyLoss {
    pub fn new(alpha: f32, beta: f32) -> Self {
        TverskyLoss { alpha, beta, smooth: 1.0 }
    }

    pub fn forward(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        let mut tp = 0.0f32;
        let mut fp = 0.0f32;
        let mut fn_count = 0.0f32;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            tp += pred * target;
            fp += pred * (1.0 - target);
            fn_count += (1.0 - pred) * target;
        }

        1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn_count + self.smooth)
    }
}

/// Lovász-Softmax Loss - for segmentation
pub struct LovaszLoss {
    pub per_image: bool,
}

impl LovaszLoss {
    pub fn new() -> Self {
        LovaszLoss { per_image: false }
    }

    pub fn forward(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        // Simplified Lovász hinge loss
        let mut errors: Vec<(f32, bool)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let error = if target > 0.5 { 1.0 - pred } else { pred };
                (error, target > 0.5)
            })
            .collect();

        errors.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut loss = 0.0f32;
        let mut tp = 0.0f32;
        let mut fp = 0.0f32;

        for (error, is_positive) in errors {
            if is_positive {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let jaccard_grad = fp / (tp + fp);
            loss += error * jaccard_grad;
        }

        loss / predictions.len() as f32
    }
}

impl Default for LovaszLoss {
    fn default() -> Self { Self::new() }
}

/// Contrastive Loss - for metric learning
pub struct ContrastiveLoss {
    pub margin: f32,
}

impl ContrastiveLoss {
    pub fn new(margin: f32) -> Self {
        ContrastiveLoss { margin }
    }

    pub fn forward(&self, embeddings1: &[f32], embeddings2: &[f32], labels: &[f32]) -> f32 {
        let n = embeddings1.len();
        let mut loss = 0.0f32;

        for i in 0..n {
            let dist_sq = (embeddings1[i] - embeddings2[i]).powi(2);
            
            if labels[i] > 0.5 {
                // Similar pair
                loss += dist_sq;
            } else {
                // Dissimilar pair
                loss += (self.margin - dist_sq.sqrt()).max(0.0).powi(2);
            }
        }

        loss / n as f32
    }
}

/// Triplet Loss - for metric learning
pub struct TripletLoss {
    pub margin: f32,
}

impl TripletLoss {
    pub fn new(margin: f32) -> Self {
        TripletLoss { margin }
    }

    pub fn forward(&self, anchor: &[f32], positive: &[f32], negative: &[f32]) -> f32 {
        let n = anchor.len();
        
        let mut pos_dist = 0.0f32;
        let mut neg_dist = 0.0f32;

        for i in 0..n {
            pos_dist += (anchor[i] - positive[i]).powi(2);
            neg_dist += (anchor[i] - negative[i]).powi(2);
        }

        (pos_dist - neg_dist + self.margin).max(0.0)
    }
}

/// Center Loss - for face recognition
pub struct CenterLoss {
    pub alpha: f32,
    pub num_classes: usize,
    centers: Vec<Vec<f32>>,
}

impl CenterLoss {
    pub fn new(num_classes: usize, feature_dim: usize, alpha: f32) -> Self {
        CenterLoss {
            alpha,
            num_classes,
            centers: vec![vec![0.0; feature_dim]; num_classes],
        }
    }

    pub fn forward(&self, features: &[f32], labels: &[usize], feature_dim: usize) -> f32 {
        let batch_size = labels.len();
        let mut loss = 0.0f32;

        for (i, &label) in labels.iter().enumerate() {
            if label < self.num_classes {
                let feat_start = i * feature_dim;
                let feat = &features[feat_start..feat_start + feature_dim];
                let center = &self.centers[label];

                for (f, c) in feat.iter().zip(center.iter()) {
                    loss += (f - c).powi(2);
                }
            }
        }

        loss / (2.0 * batch_size as f32)
    }

    pub fn update_centers(&mut self, features: &[f32], labels: &[usize], feature_dim: usize) {
        let batch_size = labels.len();
        let mut counts = vec![0; self.num_classes];
        let mut deltas = vec![vec![0.0f32; feature_dim]; self.num_classes];

        for (i, &label) in labels.iter().enumerate() {
            if label < self.num_classes {
                counts[label] += 1;
                let feat_start = i * feature_dim;
                let feat = &features[feat_start..feat_start + feature_dim];

                for (j, &f) in feat.iter().enumerate() {
                    deltas[label][j] += f - self.centers[label][j];
                }
            }
        }

        for c in 0..self.num_classes {
            if counts[c] > 0 {
                for j in 0..feature_dim {
                    self.centers[c][j] += self.alpha * deltas[c][j] / counts[c] as f32;
                }
            }
        }
    }
}

/// Label Smoothing Cross Entropy
pub struct LabelSmoothingLoss {
    pub smoothing: f32,
    pub num_classes: usize,
}

impl LabelSmoothingLoss {
    pub fn new(num_classes: usize, smoothing: f32) -> Self {
        LabelSmoothingLoss { smoothing, num_classes }
    }

    pub fn forward(&self, predictions: &[f32], targets: &[usize]) -> f32 {
        let batch_size = targets.len();
        let mut loss = 0.0f32;
        let eps = 1e-7f32;
        let confidence = 1.0 - self.smoothing;
        let smooth_value = self.smoothing / self.num_classes as f32;

        for (i, &target) in targets.iter().enumerate() {
            for c in 0..self.num_classes {
                let pred = predictions[i * self.num_classes + c].clamp(eps, 1.0 - eps);
                let target_prob = if c == target { confidence } else { smooth_value };
                loss -= target_prob * pred.ln();
            }
        }

        loss / batch_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focal_loss() {
        let focal = FocalLoss::new(0.25, 2.0);
        let preds = vec![0.9, 0.1, 0.8, 0.2];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        let loss = focal.forward(&preds, &targets);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_dice_loss() {
        let dice = DiceLoss::new();
        let preds = vec![0.9, 0.8, 0.7, 0.6];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let loss = dice.forward(&preds, &targets);
        assert!(loss >= 0.0 && loss <= 1.0);
    }

    #[test]
    fn test_triplet_loss() {
        let triplet = TripletLoss::new(1.0);
        let anchor = vec![1.0, 2.0, 3.0];
        let positive = vec![1.1, 2.1, 3.1];
        let negative = vec![5.0, 6.0, 7.0];
        let loss = triplet.forward(&anchor, &positive, &negative);
        assert!(loss >= 0.0);
    }
}
