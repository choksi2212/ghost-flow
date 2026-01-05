//! Loss functions

use ghostflow_core::Tensor;

/// Mean Squared Error loss
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = input.sub(target).unwrap();
    let squared = diff.pow(2.0);
    squared.mean()
}

/// Mean Absolute Error loss
pub fn l1_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = input.sub(target).unwrap();
    diff.abs().mean()
}

/// Smooth L1 loss (Huber loss)
pub fn smooth_l1_loss(input: &Tensor, target: &Tensor, beta: f32) -> Tensor {
    let diff = input.sub(target).unwrap().abs();
    let diff_data = diff.data_f32();
    
    let result: Vec<f32> = diff_data.iter()
        .map(|&x| {
            if x < beta {
                0.5 * x * x / beta
            } else {
                x - 0.5 * beta
            }
        })
        .collect();
    
    let tensor = Tensor::from_slice(&result, diff.dims()).unwrap();
    tensor.mean()
}

/// Binary Cross Entropy loss (input should be probabilities)
pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    let input_data = input.data_f32();
    let target_data = target.data_f32();
    
    let eps = 1e-7f32;
    
    let result: Vec<f32> = input_data.iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| {
            let p = p.clamp(eps, 1.0 - eps);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .collect();
    
    let tensor = Tensor::from_slice(&result, input.dims()).unwrap();
    tensor.mean()
}

/// Binary Cross Entropy with Logits (more numerically stable)
pub fn binary_cross_entropy_with_logits(input: &Tensor, target: &Tensor) -> Tensor {
    let input_data = input.data_f32();
    let target_data = target.data_f32();
    
    let result: Vec<f32> = input_data.iter()
        .zip(target_data.iter())
        .map(|(&x, &t)| {
            // max(x, 0) - x * t + log(1 + exp(-|x|))
            let max_val = x.max(0.0);
            max_val - x * t + (1.0 + (-x.abs()).exp()).ln()
        })
        .collect();
    
    let tensor = Tensor::from_slice(&result, input.dims()).unwrap();
    tensor.mean()
}

/// Cross Entropy loss (input should be logits, target should be class indices)
pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    // input: [batch, num_classes]
    // target: [batch] (class indices as floats)
    
    let dims = input.dims();
    let batch_size = dims[0];
    let num_classes = dims[1];
    
    let input_data = input.data_f32();
    let target_data = target.data_f32();
    
    let mut total_loss = 0.0f32;
    
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        let start = b * num_classes;
        let logits = &input_data[start..start + num_classes];
        let target_class = target_data[b] as usize;
        
        // Compute log_softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_softmax = logits[target_class] - max_logit - exp_sum.ln();
        
        total_loss -= log_softmax;
    }
    
    Tensor::from_slice(&[total_loss / batch_size as f32], &[]).unwrap()
}

/// Negative Log Likelihood loss (input should be log probabilities)
pub fn nll_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let dims = input.dims();
    let batch_size = dims[0];
    let num_classes = dims[1];
    
    let input_data = input.data_f32();
    let target_data = target.data_f32();
    
    let mut total_loss = 0.0f32;
    
    for b in 0..batch_size {
        let target_class = target_data[b] as usize;
        let log_prob = input_data[b * num_classes + target_class];
        total_loss -= log_prob;
    }
    
    Tensor::from_slice(&[total_loss / batch_size as f32], &[]).unwrap()
}

/// Cosine Embedding Loss
pub fn cosine_embedding_loss(x1: &Tensor, x2: &Tensor, target: &Tensor, margin: f32) -> Tensor {
    // target: 1 for similar, -1 for dissimilar
    let x1_data = x1.data_f32();
    let x2_data = x2.data_f32();
    let target_data = target.data_f32();
    
    let batch_size = x1.dims()[0];
    let dim = x1.numel() / batch_size;
    
    let mut total_loss = 0.0f32;
    
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        let start = b * dim;
        let end = start + dim;
        
        let v1 = &x1_data[start..end];
        let v2 = &x2_data[start..end];
        
        // Compute cosine similarity
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm1 * norm2 + 1e-8);
        
        let y = target_data[b];
        let loss = if y > 0.0 {
            1.0 - cos_sim
        } else {
            (cos_sim - margin).max(0.0)
        };
        
        total_loss += loss;
    }
    
    Tensor::from_slice(&[total_loss / batch_size as f32], &[]).unwrap()
}

/// Triplet Margin Loss
pub fn triplet_margin_loss(anchor: &Tensor, positive: &Tensor, negative: &Tensor, margin: f32) -> Tensor {
    let anchor_data = anchor.data_f32();
    let positive_data = positive.data_f32();
    let negative_data = negative.data_f32();
    
    let batch_size = anchor.dims()[0];
    let dim = anchor.numel() / batch_size;
    
    let mut total_loss = 0.0f32;
    
    for b in 0..batch_size {
        let start = b * dim;
        let end = start + dim;
        
        let a = &anchor_data[start..end];
        let p = &positive_data[start..end];
        let n = &negative_data[start..end];
        
        // Compute distances
        let dist_ap: f32 = a.iter().zip(p.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>().sqrt();
        let dist_an: f32 = a.iter().zip(n.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>().sqrt();
        
        let loss = (dist_ap - dist_an + margin).max(0.0);
        total_loss += loss;
    }
    
    Tensor::from_slice(&[total_loss / batch_size as f32], &[]).unwrap()
}

/// Focal Loss
/// Addresses class imbalance by down-weighting easy examples
/// FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
pub fn focal_loss(input: &Tensor, target: &Tensor, alpha: f32, gamma: f32) -> Tensor {
    // input: [batch, num_classes] (logits)
    // target: [batch] (class indices)
    
    let dims = input.dims();
    let batch_size = dims[0];
    let num_classes = dims[1];
    
    let input_data = input.data_f32();
    let target_data = target.data_f32();
    
    let mut total_loss = 0.0f32;
    
    #[allow(clippy::needless_range_loop)]
    for b in 0..batch_size {
        let start = b * num_classes;
        let logits = &input_data[start..start + num_classes];
        let target_class = target_data[b] as usize;
        
        // Compute softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let p_t = (logits[target_class] - max_logit).exp() / exp_sum;
        
        // Focal loss formula
        let focal_weight = (1.0 - p_t).powf(gamma);
        let ce_loss = -(p_t.ln());
        let loss = alpha * focal_weight * ce_loss;
        
        total_loss += loss;
    }
    
    Tensor::from_slice(&[total_loss / batch_size as f32], &[]).unwrap()
}

/// Contrastive Loss
/// Used for learning embeddings where similar pairs should be close and dissimilar pairs should be far
pub fn contrastive_loss(x1: &Tensor, x2: &Tensor, target: &Tensor, margin: f32) -> Tensor {
    // target: 1 for similar pairs, 0 for dissimilar pairs
    let x1_data = x1.data_f32();
    let x2_data = x2.data_f32();
    let target_data = target.data_f32();
    
    let batch_size = x1.dims()[0];
    let dim = x1.numel() / batch_size;
    
    let mut total_loss = 0.0f32;
    
    for b in 0..batch_size {
        let start = b * dim;
        let end = start + dim;
        
        let v1 = &x1_data[start..end];
        let v2 = &x2_data[start..end];
        
        // Compute Euclidean distance
        let dist: f32 = v1.iter().zip(v2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        
        let y = target_data[b];
        let loss = if y > 0.5 {
            // Similar pair: minimize distance
            dist.powi(2)
        } else {
            // Dissimilar pair: maximize distance up to margin
            (margin - dist).max(0.0).powi(2)
        };
        
        total_loss += loss;
    }
    
    Tensor::from_slice(&[total_loss / batch_size as f32], &[]).unwrap()
}

/// Huber Loss (Smooth L1 Loss variant)
/// Combines MSE for small errors and MAE for large errors
/// Less sensitive to outliers than MSE
pub fn huber_loss(input: &Tensor, target: &Tensor, delta: f32) -> Tensor {
    let diff = input.sub(target).unwrap();
    let diff_data = diff.data_f32();
    
    let result: Vec<f32> = diff_data.iter()
        .map(|&x| {
            let abs_x = x.abs();
            if abs_x <= delta {
                0.5 * x * x
            } else {
                delta * (abs_x - 0.5 * delta)
            }
        })
        .collect();
    
    let tensor = Tensor::from_slice(&result, diff.dims()).unwrap();
    tensor.mean()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let target = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        
        let loss = mse_loss(&input, &target);
        assert!(loss.data_f32()[0].abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy() {
        let input = Tensor::from_slice(&[2.0f32, 1.0, 0.1], &[1, 3]).unwrap();
        let target = Tensor::from_slice(&[0.0f32], &[1]).unwrap();
        
        let loss = cross_entropy(&input, &target);
        assert!(loss.data_f32()[0] > 0.0);
    }

    #[test]
    fn test_bce_with_logits() {
        let input = Tensor::from_slice(&[0.0f32], &[1]).unwrap();
        let target = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
        
        let loss = binary_cross_entropy_with_logits(&input, &target);
        // log(2) ≈ 0.693
        assert!((loss.data_f32()[0] - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_focal_loss() {
        let input = Tensor::from_slice(&[2.0f32, 1.0, 0.1], &[1, 3]).unwrap();
        let target = Tensor::from_slice(&[0.0f32], &[1]).unwrap();
        
        let loss = focal_loss(&input, &target, 1.0, 2.0);
        assert!(loss.data_f32()[0] > 0.0);
    }

    #[test]
    fn test_contrastive_loss() {
        let x1 = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
        let x2 = Tensor::from_slice(&[1.1f32, 2.1, 3.1], &[1, 3]).unwrap();
        let target = Tensor::from_slice(&[1.0f32], &[1]).unwrap(); // Similar pair
        
        let loss = contrastive_loss(&x1, &x2, &target, 1.0);
        assert!(loss.data_f32()[0] >= 0.0);
    }

    #[test]
    fn test_huber_loss() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let target = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        
        let loss = huber_loss(&input, &target, 1.0);
        assert!(loss.data_f32()[0].abs() < 1e-6);
    }

    #[test]
    fn test_triplet_margin_loss() {
        let anchor = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
        let positive = Tensor::from_slice(&[1.1f32, 2.1, 3.1], &[1, 3]).unwrap();
        let negative = Tensor::from_slice(&[5.0f32, 6.0, 7.0], &[1, 3]).unwrap();
        
        let loss = triplet_margin_loss(&anchor, &positive, &negative, 0.5);
        assert!(loss.data_f32()[0] >= 0.0);
    }
}
