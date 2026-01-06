//! Advanced Metrics - Log Loss, Hinge Loss, Cohen's Kappa, Matthews Correlation, etc.

use ghostflow_core::Tensor;

/// Log Loss (Cross-Entropy Loss)
pub fn log_loss(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    let n = y_true_data.len();
    let eps = 1e-15f32;

    let loss: f32 = y_true_data.iter().zip(y_pred_data.iter())
        .map(|(&yt, &yp)| {
            let yp_clipped = yp.clamp(eps, 1.0 - eps);
            -yt * yp_clipped.ln() - (1.0 - yt) * (1.0 - yp_clipped).ln()
        })
        .sum();

    loss / n as f32
}

/// Multiclass Log Loss
pub fn log_loss_multiclass(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    let n_samples = y_true.dims()[0];
    let n_classes = y_pred.dims()[1];
    let eps = 1e-15f32;

    let mut loss = 0.0f32;
    for i in 0..n_samples {
        let true_class = y_true_data[i] as usize;
        if true_class < n_classes {
            let pred = y_pred_data[i * n_classes + true_class].clamp(eps, 1.0 - eps);
            loss -= pred.ln();
        }
    }

    loss / n_samples as f32
}

/// Hinge Loss
pub fn hinge_loss(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    let n = y_true_data.len();

    let loss: f32 = y_true_data.iter().zip(y_pred_data.iter())
        .map(|(&yt, &yp)| {
            // Convert 0/1 labels to -1/+1
            let yt_signed = if yt > 0.5 { 1.0 } else { -1.0 };
            (1.0 - yt_signed * yp).max(0.0)
        })
        .sum();

    loss / n as f32
}

/// Squared Hinge Loss
pub fn squared_hinge_loss(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    let n = y_true_data.len();

    let loss: f32 = y_true_data.iter().zip(y_pred_data.iter())
        .map(|(&yt, &yp)| {
            let yt_signed = if yt > 0.5 { 1.0 } else { -1.0 };
            (1.0 - yt_signed * yp).max(0.0).powi(2)
        })
        .sum();

    loss / n as f32
}

/// Cohen's Kappa Score
pub fn cohen_kappa_score(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();
    let n = y_true_data.len() as f32;

    // Find unique classes
    let mut classes: Vec<i32> = y_true_data.iter()
        .chain(y_pred_data.iter())
        .map(|&x| x as i32)
        .collect();
    classes.sort();
    classes.dedup();
    let n_classes = classes.len();

    // Build confusion matrix
    let mut confusion = vec![0.0f32; n_classes * n_classes];
    for (&yt, &yp) in y_true_data.iter().zip(y_pred_data.iter()) {
        let i = classes.iter().position(|&c| c == yt as i32).unwrap();
        let j = classes.iter().position(|&c| c == yp as i32).unwrap();
        confusion[i * n_classes + j] += 1.0;
    }

    // Compute observed agreement
    let po: f32 = (0..n_classes).map(|i| confusion[i * n_classes + i]).sum::<f32>() / n;

    // Compute expected agreement
    let row_sums: Vec<f32> = (0..n_classes)
        .map(|i| (0..n_classes).map(|j| confusion[i * n_classes + j]).sum())
        .collect();
    let col_sums: Vec<f32> = (0..n_classes)
        .map(|j| (0..n_classes).map(|i| confusion[i * n_classes + j]).sum())
        .collect();

    let pe: f32 = row_sums.iter().zip(col_sums.iter())
        .map(|(&r, &c)| r * c)
        .sum::<f32>() / (n * n);

    if (1.0 - pe).abs() < 1e-10 {
        1.0
    } else {
        (po - pe) / (1.0 - pe)
    }
}

/// Matthews Correlation Coefficient
pub fn matthews_corrcoef(y_true: &Tensor, y_pred: &Tensor) -> f32 {
    let y_true_data = y_true.data_f32();
    let y_pred_data = y_pred.data_f32();

    let mut tp = 0.0f32;
    let mut tn = 0.0f32;
    let mut fp = 0.0f32;
    let mut fn_ = 0.0f32;

    for (&yt, &yp) in y_true_data.iter().zip(y_pred_data.iter()) {
        let yt_bool = yt > 0.5;
        let yp_bool = yp > 0.5;

        match (yt_bool, yp_bool) {
            (true, true) => tp += 1.0,
            (false, false) => tn += 1.0,
            (false, true) => fp += 1.0,
            (true, false) => fn_ += 1.0,
        }
    }

    let numerator = tp * tn - fp * fn_;
    let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();

    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Adjusted Rand Index
pub fn adjusted_rand_score(labels_true: &Tensor, labels_pred: &Tensor) -> f32 {
    let true_data = labels_true.data_f32();
    let pred_data = labels_pred.data_f32();
    let n = true_data.len();

    // Build contingency table
    let true_classes: Vec<i32> = true_data.iter().map(|&x| x as i32).collect();
    let pred_classes: Vec<i32> = pred_data.iter().map(|&x| x as i32).collect();
    
    let mut unique_true: Vec<i32> = true_classes.clone();
    unique_true.sort();
    unique_true.dedup();
    
    let mut unique_pred: Vec<i32> = pred_classes.clone();
    unique_pred.sort();
    unique_pred.dedup();

    let n_true = unique_true.len();
    let n_pred = unique_pred.len();

    let mut contingency = vec![0i32; n_true * n_pred];
    for i in 0..n {
        let ti = unique_true.iter().position(|&c| c == true_classes[i]).unwrap();
        let pi = unique_pred.iter().position(|&c| c == pred_classes[i]).unwrap();
        contingency[ti * n_pred + pi] += 1;
    }

    // Compute sums
    let row_sums: Vec<i32> = (0..n_true)
        .map(|i| (0..n_pred).map(|j| contingency[i * n_pred + j]).sum())
        .collect();
    let col_sums: Vec<i32> = (0..n_pred)
        .map(|j| (0..n_true).map(|i| contingency[i * n_pred + j]).sum())
        .collect();

    // Compute index
    fn comb2(n: i32) -> i64 {
        if n < 2 { 0 } else { (n as i64 * (n as i64 - 1)) / 2 }
    }

    let sum_comb_c: i64 = contingency.iter().map(|&c| comb2(c)).sum();
    let sum_comb_a: i64 = row_sums.iter().map(|&a| comb2(a)).sum();
    let sum_comb_b: i64 = col_sums.iter().map(|&b| comb2(b)).sum();
    let comb_n = comb2(n as i32);

    let expected = (sum_comb_a * sum_comb_b) as f64 / comb_n.max(1) as f64;
    let max_index = (sum_comb_a + sum_comb_b) as f64 / 2.0;

    if (max_index - expected).abs() < 1e-10 {
        1.0
    } else {
        ((sum_comb_c as f64 - expected) / (max_index - expected)) as f32
    }
}

/// Normalized Mutual Information
pub fn normalized_mutual_info_score(labels_true: &Tensor, labels_pred: &Tensor) -> f32 {
    let true_data = labels_true.data_f32();
    let pred_data = labels_pred.data_f32();
    let n = true_data.len() as f32;

    // Build contingency table
    let mut unique_true: Vec<i32> = true_data.iter().map(|&x| x as i32).collect();
    unique_true.sort();
    unique_true.dedup();
    
    let mut unique_pred: Vec<i32> = pred_data.iter().map(|&x| x as i32).collect();
    unique_pred.sort();
    unique_pred.dedup();

    let n_true = unique_true.len();
    let n_pred = unique_pred.len();

    let mut contingency = vec![0.0f32; n_true * n_pred];
    for i in 0..true_data.len() {
        let ti = unique_true.iter().position(|&c| c == true_data[i] as i32).unwrap();
        let pi = unique_pred.iter().position(|&c| c == pred_data[i] as i32).unwrap();
        contingency[ti * n_pred + pi] += 1.0;
    }

    // Compute marginals
    let row_sums: Vec<f32> = (0..n_true)
        .map(|i| (0..n_pred).map(|j| contingency[i * n_pred + j]).sum())
        .collect();
    let col_sums: Vec<f32> = (0..n_pred)
        .map(|j| (0..n_true).map(|i| contingency[i * n_pred + j]).sum())
        .collect();

    // Compute mutual information
    let mut mi = 0.0f32;
    for i in 0..n_true {
        for j in 0..n_pred {
            let nij = contingency[i * n_pred + j];
            if nij > 0.0 {
                mi += nij / n * (n * nij / (row_sums[i] * col_sums[j])).ln();
            }
        }
    }

    // Compute entropies
    let h_true: f32 = row_sums.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -(p / n) * (p / n).ln())
        .sum();
    let h_pred: f32 = col_sums.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -(p / n) * (p / n).ln())
        .sum();

    // Normalized MI (arithmetic mean)
    let denom = (h_true + h_pred) / 2.0;
    if denom < 1e-10 {
        0.0
    } else {
        mi / denom
    }
}

/// Fowlkes-Mallows Score
pub fn fowlkes_mallows_score(labels_true: &Tensor, labels_pred: &Tensor) -> f32 {
    let true_data = labels_true.data_f32();
    let pred_data = labels_pred.data_f32();
    let n = true_data.len();

    // Count pairs
    let mut tp = 0i64; // Same cluster in both
    let mut fp = 0i64; // Same in pred, different in true
    let mut fn_ = 0i64; // Different in pred, same in true

    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = (true_data[i] - true_data[j]).abs() < 0.5;
            let same_pred = (pred_data[i] - pred_data[j]).abs() < 0.5;

            match (same_true, same_pred) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                _ => {}
            }
        }
    }

    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_ > 0 { tp as f32 / (tp + fn_) as f32 } else { 0.0 };

    (precision * recall).sqrt()
}

/// Silhouette Score
pub fn silhouette_score(x: &Tensor, labels: &Tensor) -> f32 {
    let x_data = x.data_f32();
    let labels_data = labels.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];

    if n_samples < 2 {
        return 0.0;
    }

    let mut silhouettes = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let label_i = labels_data[i] as i32;
        let xi = &x_data[i * n_features..(i + 1) * n_features];

        // Compute mean intra-cluster distance (a)
        let mut intra_sum = 0.0f32;
        let mut intra_count = 0;

        // Compute mean nearest-cluster distance (b)
        let mut cluster_dists: std::collections::HashMap<i32, (f32, usize)> = std::collections::HashMap::new();

        for j in 0..n_samples {
            if i == j { continue; }
            
            let label_j = labels_data[j] as i32;
            let xj = &x_data[j * n_features..(j + 1) * n_features];
            
            let dist: f32 = xi.iter().zip(xj.iter())
                .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt();

            if label_j == label_i {
                intra_sum += dist;
                intra_count += 1;
            } else {
                let entry = cluster_dists.entry(label_j).or_insert((0.0, 0));
                entry.0 += dist;
                entry.1 += 1;
            }
        }

        let a = if intra_count > 0 { intra_sum / intra_count as f32 } else { 0.0 };
        
        let b = cluster_dists.values()
            .map(|(sum, count)| if *count > 0 { sum / *count as f32 } else { f32::MAX })
            .fold(f32::MAX, f32::min);

        let s = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };

        silhouettes.push(s);
    }

    silhouettes.iter().sum::<f32>() / silhouettes.len() as f32
}

/// Calinski-Harabasz Index (Variance Ratio Criterion)
pub fn calinski_harabasz_score(x: &Tensor, labels: &Tensor) -> f32 {
    let x_data = x.data_f32();
    let labels_data = labels.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];

    // Find unique labels
    let mut unique_labels: Vec<i32> = labels_data.iter().map(|&l| l as i32).collect();
    unique_labels.sort();
    unique_labels.dedup();
    let n_clusters = unique_labels.len();

    if n_clusters < 2 || n_samples <= n_clusters {
        return 0.0;
    }

    // Compute overall centroid
    let overall_centroid: Vec<f32> = (0..n_features)
        .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
        .collect();

    // Compute cluster centroids and sizes
    let mut cluster_centroids: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
    let mut cluster_sizes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();

    for &label in &unique_labels {
        let mut centroid = vec![0.0f32; n_features];
        let mut count = 0;

        for i in 0..n_samples {
            if labels_data[i] as i32 == label {
                for j in 0..n_features {
                    centroid[j] += x_data[i * n_features + j];
                }
                count += 1;
            }
        }

        for c in &mut centroid {
            *c /= count as f32;
        }

        cluster_centroids.insert(label, centroid);
        cluster_sizes.insert(label, count);
    }

    // Between-cluster dispersion
    let mut bgss = 0.0f32;
    for (&label, centroid) in &cluster_centroids {
        let size = cluster_sizes[&label] as f32;
        let dist_sq: f32 = centroid.iter().zip(overall_centroid.iter())
            .map(|(&c, &o)| (c - o).powi(2)).sum();
        bgss += size * dist_sq;
    }

    // Within-cluster dispersion
    let mut wgss = 0.0f32;
    for i in 0..n_samples {
        let label = labels_data[i] as i32;
        let centroid = &cluster_centroids[&label];
        let dist_sq: f32 = (0..n_features)
            .map(|j| (x_data[i * n_features + j] - centroid[j]).powi(2)).sum();
        wgss += dist_sq;
    }

    if wgss < 1e-10 {
        return 0.0;
    }

    (bgss / (n_clusters - 1) as f32) / (wgss / (n_samples - n_clusters) as f32)
}

/// Davies-Bouldin Index
pub fn davies_bouldin_score(x: &Tensor, labels: &Tensor) -> f32 {
    let x_data = x.data_f32();
    let labels_data = labels.data_f32();
    let n_samples = x.dims()[0];
    let n_features = x.dims()[1];

    let mut unique_labels: Vec<i32> = labels_data.iter().map(|&l| l as i32).collect();
    unique_labels.sort();
    unique_labels.dedup();
    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return 0.0;
    }

    // Compute cluster centroids
    let mut cluster_centroids: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
    let mut cluster_sizes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();

    for &label in &unique_labels {
        let mut centroid = vec![0.0f32; n_features];
        let mut count = 0;

        for i in 0..n_samples {
            if labels_data[i] as i32 == label {
                for j in 0..n_features {
                    centroid[j] += x_data[i * n_features + j];
                }
                count += 1;
            }
        }

        for c in &mut centroid {
            *c /= count.max(1) as f32;
        }

        cluster_centroids.insert(label, centroid);
        cluster_sizes.insert(label, count);
    }

    // Compute intra-cluster distances (scatter)
    let mut scatters: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
    for &label in &unique_labels {
        let centroid = &cluster_centroids[&label];
        let mut scatter = 0.0f32;
        let mut count = 0;

        for i in 0..n_samples {
            if labels_data[i] as i32 == label {
                let dist: f32 = (0..n_features)
                    .map(|j| (x_data[i * n_features + j] - centroid[j]).powi(2)).sum::<f32>().sqrt();
                scatter += dist;
                count += 1;
            }
        }

        scatters.insert(label, scatter / count.max(1) as f32);
    }

    // Compute Davies-Bouldin index
    let mut db_sum = 0.0f32;
    for (i, &label_i) in unique_labels.iter().enumerate() {
        let mut max_ratio = 0.0f32;

        for (j, &label_j) in unique_labels.iter().enumerate() {
            if i == j { continue; }

            let centroid_i = &cluster_centroids[&label_i];
            let centroid_j = &cluster_centroids[&label_j];

            let centroid_dist: f32 = centroid_i.iter().zip(centroid_j.iter())
                .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt();

            if centroid_dist > 1e-10 {
                let ratio = (scatters[&label_i] + scatters[&label_j]) / centroid_dist;
                max_ratio = max_ratio.max(ratio);
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
    fn test_log_loss() {
        let y_true = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 1.0], &[4]).unwrap();
        let y_pred = Tensor::from_slice(&[0.9f32, 0.1, 0.8, 0.7], &[4]).unwrap();
        let loss = log_loss(&y_true, &y_pred);
        assert!(loss > 0.0 && loss < 1.0);
    }

    #[test]
    fn test_cohen_kappa() {
        let y_true = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0], &[6]).unwrap();
        let y_pred = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0], &[6]).unwrap();
        let kappa = cohen_kappa_score(&y_true, &y_pred);
        assert!((kappa - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matthews_corrcoef() {
        let y_true = Tensor::from_slice(&[1.0f32, 1.0, 0.0, 0.0], &[4]).unwrap();
        let y_pred = Tensor::from_slice(&[1.0f32, 1.0, 0.0, 0.0], &[4]).unwrap();
        let mcc = matthews_corrcoef(&y_true, &y_pred);
        assert!((mcc - 1.0).abs() < 1e-5);
    }
}


