//! Model evaluation metrics

use ghostflow_core::Tensor;

// ============ Classification Metrics ============

/// Compute accuracy score
pub fn accuracy_score(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    let correct: usize = y_true_data.iter()
        .zip(y_pred_data.iter())
        .filter(|(&t, &p)| (t - p).abs() < 0.5)
        .count();
    
    correct as f32 / y_true_data.len() as f32
}

/// Compute precision score for binary classification
pub fn precision_score(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    let mut tp = 0usize;
    let mut fp = 0usize;
    
    for (&t, &p) in y_true_data.iter().zip(y_pred_data.iter()) {
        let t_pos = t > 0.5;
        let p_pos = p > 0.5;
        
        if p_pos && t_pos { tp += 1; }
        if p_pos && !t_pos { fp += 1; }
    }
    
    if tp + fp == 0 { 0.0 } else { tp as f32 / (tp + fp) as f32 }
}

/// Compute recall score for binary classification
pub fn recall_score(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    let mut tp = 0usize;
    let mut fn_ = 0usize;
    
    for (&t, &p) in y_true_data.iter().zip(y_pred_data.iter()) {
        let t_pos = t > 0.5;
        let p_pos = p > 0.5;
        
        if p_pos && t_pos { tp += 1; }
        if !p_pos && t_pos { fn_ += 1; }
    }
    
    if tp + fn_ == 0 { 0.0 } else { tp as f32 / (tp + fn_) as f32 }
}

/// Compute F1 score for binary classification
pub fn f1_score(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let precision = precision_score(y_true, y_pred);
    let recall = recall_score(y_true, y_pred);
    
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Compute confusion matrix
pub fn confusion_matrix(y_true: &Tensor, y_pred: &Tensor, n_classes: usize) -> Vec<Vec<usize>> {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    let mut matrix = vec![vec![0usize; n_classes]; n_classes];
    
    for (&t, &p) in y_true_data.iter().zip(y_pred_data.iter()) {
        let t_class = t.round() as usize;
        let p_class = p.round() as usize;
        
        if t_class < n_classes && p_class < n_classes {
            matrix[t_class][p_class] += 1;
        }
    }
    
    matrix
}

/// Compute ROC AUC score for binary classification
pub fn roc_auc_score(y_true: &Tensor, y_scores: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_scores_data = y_scores.data_f32();
    
    // Sort by scores descending
    let mut pairs: Vec<(f32, f32)> = y_true_data.iter()
        .zip(y_scores_data.iter())
        .map(|(&t, &s)| (t, s))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let n_pos: f32 = y_true_data.iter().filter(|&&t| t > 0.5).count() as f32;
    let n_neg: f32 = y_true_data.len() as f32 - n_pos;
    
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    
    let mut auc = 0.0f32;
    let mut tp = 0.0f32;
    let mut _fp = 0.0f32;
    let mut prev_tp = 0.0f32;
    
    for (label, _) in pairs {
        if label > 0.5 {
            tp += 1.0;
        } else {
            _fp += 1.0;
            auc += (tp + prev_tp) / 2.0;
        }
        prev_tp = tp;
    }
    
    auc / (n_pos * n_neg)
}

/// Classification report as a struct
pub struct ClassificationReport {
    pub precision: Vec<f32>,
    pub recall: Vec<f32>,
    pub f1: Vec<f32>,
    pub support: Vec<usize>,
    pub accuracy: f32,
}

pub fn classification_report(y_true: &Tensor, y_pred: &Tensor, n_classes: usize) -> ClassificationReport {
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    
    let mut precision = vec![0.0f32; n_classes];
    let mut recall = vec![0.0f32; n_classes];
    let mut f1 = vec![0.0f32; n_classes];
    let mut support = vec![0usize; n_classes];
    
    for c in 0..n_classes {
        let tp = cm[c][c];
        let fp: usize = (0..n_classes).map(|i| cm[i][c]).sum::<usize>() - tp;
        let fn_: usize = cm[c].iter().sum::<usize>() - tp;
        
        support[c] = cm[c].iter().sum();
        
        precision[c] = if tp + fp == 0 { 0.0 } else { tp as f32 / (tp + fp) as f32 };
        recall[c] = if tp + fn_ == 0 { 0.0 } else { tp as f32 / (tp + fn_) as f32 };
        f1[c] = if precision[c] + recall[c] == 0.0 {
            0.0
        } else {
            2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
        };
    }
    
    ClassificationReport {
        precision,
        recall,
        f1,
        support,
        accuracy: accuracy_score(y_true, y_pred),
    }
}


// ============ Regression Metrics ============

/// Mean Squared Error
pub fn mean_squared_error(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    y_true_data.iter()
        .zip(y_pred_data.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f32>() / y_true_data.len() as f32
}

/// Root Mean Squared Error
pub fn root_mean_squared_error(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    mean_squared_error(y_true, y_pred).sqrt()
}

/// Mean Absolute Error
pub fn mean_absolute_error(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    y_true_data.iter()
        .zip(y_pred_data.iter())
        .map(|(&t, &p)| (t - p).abs())
        .sum::<f32>() / y_true_data.len() as f32
}

/// R² Score (Coefficient of Determination)
pub fn r2_score(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    let y_mean: f32 = y_true_data.iter().sum::<f32>() / y_true_data.len() as f32;
    
    let ss_res: f32 = y_true_data.iter()
        .zip(y_pred_data.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    
    let ss_tot: f32 = y_true_data.iter()
        .map(|&t| (t - y_mean).powi(2))
        .sum();
    
    1.0 - ss_res / ss_tot.max(1e-10)
}

/// Mean Absolute Percentage Error
pub fn mean_absolute_percentage_error(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    
    y_true_data.iter()
        .zip(y_pred_data.iter())
        .map(|(&t, &p)| ((t - p) / t.abs().max(1e-10)).abs())
        .sum::<f32>() / y_true_data.len() as f32 * 100.0
}

/// Explained Variance Score
pub fn explained_variance_score(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    let n = y_true_data.len() as f32;
    
    // Compute residuals
    let residuals: Vec<f32> = y_true_data.iter()
        .zip(y_pred_data.iter())
        .map(|(&t, &p)| t - p)
        .collect();
    
    let res_mean: f32 = residuals.iter().sum::<f32>() / n;
    let res_var: f32 = residuals.iter().map(|&r| (r - res_mean).powi(2)).sum::<f32>() / n;
    
    let y_mean: f32 = y_true_data.iter().sum::<f32>() / n;
    let y_var: f32 = y_true_data.iter().map(|&y| (y - y_mean).powi(2)).sum::<f32>() / n;
    
    1.0 - res_var / y_var.max(1e-10)
}

// ============ Clustering Metrics ============

/// Silhouette Score
pub fn silhouette_score(x: &Tensor, labels: &Tensor) -> f32 {
    let x_data = x.data_f32();
    let labels_data = labels.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];
    
    if n_samples < 2 {
        return 0.0;
    }
    
    let n_clusters = labels_data.iter().map(|&l| l as usize).max().unwrap_or(0) + 1;
    
    let mut total_silhouette = 0.0f32;
    
    for i in 0..n_samples {
        let xi = &x_data[i * n_features..(i + 1) * n_features];
        let cluster_i = labels_data[i] as usize;
        
        // Compute a(i): mean distance to points in same cluster
        let mut a_sum = 0.0f32;
        let mut a_count = 0usize;
        
        for j in 0..n_samples {
            if i != j && labels_data[j] as usize == cluster_i {
                let xj = &x_data[j * n_features..(j + 1) * n_features];
                let dist: f32 = xi.iter().zip(xj.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                a_sum += dist;
                a_count += 1;
            }
        }
        
        let a = if a_count > 0 { a_sum / a_count as f32 } else { 0.0 };
        
        // Compute b(i): min mean distance to points in other clusters
        let mut b = f32::INFINITY;
        
        for c in 0..n_clusters {
            if c == cluster_i { continue; }
            
            let mut b_sum = 0.0f32;
            let mut b_count = 0usize;
            
            for j in 0..n_samples {
                if labels_data[j] as usize == c {
                    let xj = &x_data[j * n_features..(j + 1) * n_features];
                    let dist: f32 = xi.iter().zip(xj.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    b_sum += dist;
                    b_count += 1;
                }
            }
            
            if b_count > 0 {
                b = b.min(b_sum / b_count as f32);
            }
        }
        
        if b.is_infinite() { b = 0.0; }
        
        let s = if a.max(b) > 0.0 { (b - a) / a.max(b) } else { 0.0 };
        total_silhouette += s;
    }
    
    total_silhouette / n_samples as f32
}

/// Davies-Bouldin Index (lower is better)
pub fn davies_bouldin_score(x: &Tensor, labels: &Tensor) -> f32 {
    let x_data = x.data_f32();
    let labels_data = labels.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];
    
    let n_clusters = labels_data.iter().map(|&l| l as usize).max().unwrap_or(0) + 1;
    
    // Compute cluster centroids
    let mut centroids = vec![vec![0.0f32; n_features]; n_clusters];
    let mut counts = vec![0usize; n_clusters];
    
    for i in 0..n_samples {
        let cluster = labels_data[i] as usize;
        counts[cluster] += 1;
        for j in 0..n_features {
            centroids[cluster][j] += x_data[i * n_features + j];
        }
    }
    
    for c in 0..n_clusters {
        if counts[c] > 0 {
            for j in 0..n_features {
                centroids[c][j] /= counts[c] as f32;
            }
        }
    }
    
    // Compute scatter for each cluster
    let mut scatter = vec![0.0f32; n_clusters];
    for i in 0..n_samples {
        let cluster = labels_data[i] as usize;
        let xi = &x_data[i * n_features..(i + 1) * n_features];
        let dist: f32 = xi.iter().zip(centroids[cluster].iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        scatter[cluster] += dist;
    }
    
    for c in 0..n_clusters {
        if counts[c] > 0 {
            scatter[c] /= counts[c] as f32;
        }
    }
    
    // Compute DB index
    let mut db_sum = 0.0f32;
    for i in 0..n_clusters {
        let mut max_ratio = 0.0f32;
        for j in 0..n_clusters {
            if i != j {
                let centroid_dist: f32 = centroids[i].iter().zip(centroids[j].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                
                if centroid_dist > 0.0 {
                    let ratio = (scatter[i] + scatter[j]) / centroid_dist;
                    max_ratio = max_ratio.max(ratio);
                }
            }
        }
        db_sum += max_ratio;
    }
    
    db_sum / n_clusters as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let y_true = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4]).unwrap();
        let y_pred = Tensor::from_slice(&[0.0, 1.0, 0.0, 0.0], &[4]).unwrap();
        
        let acc = accuracy_score(&y_true, &y_pred);
        assert!((acc - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_mse() {
        let y_true = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let y_pred = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        
        let mse = mean_squared_error(&y_true, &y_pred);
        assert!(mse < 0.01);
    }

    #[test]
    fn test_r2() {
        let y_true = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let y_pred = Tensor::from_slice(&[1.1, 2.1, 2.9, 3.9], &[4]).unwrap();
        
        let r2 = r2_score(&y_true, &y_pred);
        assert!(r2 > 0.9);
    }
}
