//! Incremental Decomposition - IncrementalPCA, MiniBatchDictionaryLearning

use ghostflow_core::Tensor;

/// Incremental PCA - for large datasets that don't fit in memory
pub struct IncrementalPCA {
    pub n_components: usize,
    pub batch_size: Option<usize>,
    pub whiten: bool,
    components_: Option<Vec<Vec<f32>>>,
    mean_: Option<Vec<f32>>,
    var_: Option<Vec<f32>>,
    singular_values_: Option<Vec<f32>>,
    explained_variance_: Option<Vec<f32>>,
    explained_variance_ratio_: Option<Vec<f32>>,
    n_samples_seen_: usize,
    n_features_: usize,
}

impl IncrementalPCA {
    pub fn new(n_components: usize) -> Self {
        IncrementalPCA {
            n_components,
            batch_size: None,
            whiten: false,
            components_: None,
            mean_: None,
            var_: None,
            singular_values_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            n_samples_seen_: 0,
            n_features_: 0,
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn whiten(mut self, w: bool) -> Self {
        self.whiten = w;
        self
    }

    /// Partial fit on a batch of data
    pub fn partial_fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        if self.n_samples_seen_ == 0 {
            self.n_features_ = n_features;
            self.mean_ = Some(vec![0.0; n_features]);
            self.var_ = Some(vec![0.0; n_features]);
        }

        let mean = self.mean_.as_mut().unwrap();
        let var = self.var_.as_mut().unwrap();

        // Update mean and variance incrementally (Welford's algorithm)
        let old_n = self.n_samples_seen_ as f32;
        let new_n = (self.n_samples_seen_ + n_samples) as f32;

        // Compute batch statistics
        let batch_mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();

        let batch_var: Vec<f32> = (0..n_features)
            .map(|j| {
                let m = batch_mean[j];
                (0..n_samples).map(|i| (x_data[i * n_features + j] - m).powi(2)).sum::<f32>() / n_samples as f32
            })
            .collect();

        // Update global statistics
        for j in 0..n_features {
            let delta = batch_mean[j] - mean[j];
            let new_mean = mean[j] + delta * n_samples as f32 / new_n;
            
            // Combined variance
            let m_a = var[j] * old_n;
            let m_b = batch_var[j] * n_samples as f32;
            let m2 = m_a + m_b + delta * delta * old_n * n_samples as f32 / new_n;
            
            mean[j] = new_mean;
            var[j] = m2 / new_n;
        }

        // Center the batch data
        let x_centered: Vec<f32> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - mean[j]).collect::<Vec<_>>())
            .collect();

        // Incremental SVD update
        if self.components_.is_none() {
            // First batch: compute SVD directly
            let (_u, s, vt) = self.svd(&x_centered, n_samples, n_features);
            
            let n_comp = self.n_components.min(n_samples).min(n_features);
            self.components_ = Some(vt.into_iter().take(n_comp).collect());
            self.singular_values_ = Some(s.into_iter().take(n_comp).collect());
        } else {
            // Subsequent batches: incremental update
            let components = self.components_.as_ref().unwrap();
            let singular_values = self.singular_values_.as_ref().unwrap();
            let n_comp = components.len();

            // Project old components scaled by singular values
            // Combine with new data for SVD
            let mut combined = Vec::with_capacity((n_comp + n_samples) * n_features);
            
            // Add scaled components
            for k in 0..n_comp {
                for j in 0..n_features {
                    combined.push(singular_values[k] * components[k][j]);
                }
            }
            
            // Add centered batch data
            combined.extend_from_slice(&x_centered);

            // SVD on combined matrix
            let (_, s, vt) = self.svd(&combined, n_comp + n_samples, n_features);
            
            let new_n_comp = self.n_components.min(n_comp + n_samples).min(n_features);
            self.components_ = Some(vt.into_iter().take(new_n_comp).collect());
            self.singular_values_ = Some(s.into_iter().take(new_n_comp).collect());
        }

        self.n_samples_seen_ += n_samples;

        // Update explained variance
        self.update_explained_variance();
    }

    fn svd(&self, x: &[f32], n_rows: usize, n_cols: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
        // Compute X^T X for eigendecomposition
        let mut xtx = vec![0.0f32; n_cols * n_cols];
        for i in 0..n_cols {
            for j in 0..n_cols {
                for k in 0..n_rows {
                    xtx[i * n_cols + j] += x[k * n_cols + i] * x[k * n_cols + j];
                }
            }
        }

        // Power iteration for top eigenvectors
        let n_components = self.n_components.min(n_rows).min(n_cols);
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();

        for _ in 0..n_components {
            let mut v = vec![1.0f32; n_cols];
            normalize(&mut v);

            for _ in 0..100 {
                let mut av = vec![0.0f32; n_cols];
                for i in 0..n_cols {
                    for j in 0..n_cols {
                        av[i] += xtx[i * n_cols + j] * v[j];
                    }
                }

                let norm: f32 = av.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 { break; }

                let prev_v = v.clone();
                for (vi, avi) in v.iter_mut().zip(av.iter()) {
                    *vi = *avi / norm;
                }

                let diff: f32 = v.iter().zip(prev_v.iter()).map(|(&a, &b)| (a - b).abs()).sum();
                if diff < 1e-6 { break; }
            }

            // Compute eigenvalue
            let mut eigenvalue = 0.0f32;
            for i in 0..n_cols {
                let mut av_i = 0.0f32;
                for j in 0..n_cols {
                    av_i += xtx[i * n_cols + j] * v[j];
                }
                eigenvalue += v[i] * av_i;
            }

            eigenvalues.push(eigenvalue.max(0.0).sqrt());
            eigenvectors.push(v.clone());

            // Deflate
            for i in 0..n_cols {
                for j in 0..n_cols {
                    xtx[i * n_cols + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        // Compute U = X V S^-1
        let mut u = Vec::new();
        for k in 0..eigenvalues.len() {
            if eigenvalues[k] > 1e-10 {
                let mut u_k = vec![0.0f32; n_rows];
                for i in 0..n_rows {
                    for j in 0..n_cols {
                        u_k[i] += x[i * n_cols + j] * eigenvectors[k][j];
                    }
                    u_k[i] /= eigenvalues[k];
                }
                u.push(u_k);
            }
        }

        (u, eigenvalues, eigenvectors)
    }

    fn update_explained_variance(&mut self) {
        if let Some(ref sv) = self.singular_values_ {
            let n = self.n_samples_seen_ as f32;
            let explained_var: Vec<f32> = sv.iter().map(|&s| s * s / (n - 1.0).max(1.0)).collect();
            let total_var: f32 = explained_var.iter().sum();
            let explained_ratio: Vec<f32> = explained_var.iter()
                .map(|&v| v / total_var.max(1e-10))
                .collect();

            self.explained_variance_ = Some(explained_var);
            self.explained_variance_ratio_ = Some(explained_ratio);
        }
    }

    /// Fit the model with all data at once (convenience method)
    pub fn fit(&mut self, x: &Tensor) {
        let batch_size = self.batch_size.unwrap_or(x.dims()[0]);
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];
        let x_data = x.data_f32();

        // Reset state
        self.n_samples_seen_ = 0;
        self.components_ = None;
        self.mean_ = None;
        self.var_ = None;

        // Process in batches
        let mut start = 0;
        while start < n_samples {
            let end = (start + batch_size).min(n_samples);
            let batch_data: Vec<f32> = x_data[start * n_features..end * n_features].to_vec();
            let batch = Tensor::from_slice(&batch_data, &[end - start, n_features]).unwrap();
            self.partial_fit(&batch);
            start = end;
        }
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let components = self.components_.as_ref().expect("Model not fitted");
        let mean = self.mean_.as_ref().unwrap();
        let n_components = components.len();

        let mut result = vec![0.0f32; n_samples * n_components];

        for i in 0..n_samples {
            for k in 0..n_components {
                for j in 0..n_features {
                    result[i * n_components + k] += 
                        (x_data[i * n_features + j] - mean[j]) * components[k][j];
                }

                if self.whiten {
                    if let Some(ref sv) = self.singular_values_ {
                        let scale = (self.n_samples_seen_ as f32 - 1.0).sqrt() / sv[k].max(1e-10);
                        result[i * n_components + k] *= scale;
                    }
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_components]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_components = x.dims()[1];

        let components = self.components_.as_ref().expect("Model not fitted");
        let mean = self.mean_.as_ref().unwrap();
        let n_features = self.n_features_;

        let mut result = vec![0.0f32; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                result[i * n_features + j] = mean[j];
                for k in 0..n_components {
                    let mut val = x_data[i * n_components + k];
                    
                    if self.whiten {
                        if let Some(ref sv) = self.singular_values_ {
                            val *= sv[k] / (self.n_samples_seen_ as f32 - 1.0).sqrt().max(1e-10);
                        }
                    }
                    
                    result[i * n_features + j] += val * components[k][j];
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }

    pub fn explained_variance_ratio(&self) -> Option<&Vec<f32>> {
        self.explained_variance_ratio_.as_ref()
    }

    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen_
    }
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for vi in v.iter_mut() { *vi /= norm; }
    }
}

/// Mini-Batch Sparse PCA
pub struct MiniBatchSparsePCA {
    pub n_components: usize,
    pub alpha: f32,
    pub batch_size: usize,
    pub max_iter: usize,
    components_: Option<Vec<Vec<f32>>>,
    mean_: Option<Vec<f32>>,
    n_iter_: usize,
}

impl MiniBatchSparsePCA {
    pub fn new(n_components: usize) -> Self {
        MiniBatchSparsePCA {
            n_components,
            alpha: 1.0,
            batch_size: 100,
            max_iter: 100,
            components_: None,
            mean_: None,
            n_iter_: 0,
        }
    }

    pub fn alpha(mut self, a: f32) -> Self { self.alpha = a; self }
    pub fn batch_size(mut self, b: usize) -> Self { self.batch_size = b; self }

    pub fn fit(&mut self, x: &Tensor) {
        use rand::prelude::*;
        
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute mean
        let mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();

        // Initialize components randomly
        let mut rng = thread_rng();
        let mut components: Vec<Vec<f32>> = (0..self.n_components)
            .map(|_| {
                let mut v: Vec<f32> = (0..n_features).map(|_| rng.gen::<f32>() - 0.5).collect();
                normalize(&mut v);
                v
            })
            .collect();

        // Mini-batch optimization
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for iter in 0..self.max_iter {
            indices.shuffle(&mut rng);

            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_samples);
                let batch_indices = &indices[batch_start..batch_end];

                // Extract and center batch
                let batch: Vec<f32> = batch_indices.iter()
                    .flat_map(|&i| (0..n_features).map(|j| x_data[i * n_features + j] - mean[j]).collect::<Vec<_>>())
                    .collect();
                let batch_size = batch_indices.len();

                // Update each component
                for k in 0..self.n_components {
                    // Compute gradient
                    let mut grad = vec![0.0f32; n_features];
                    
                    for i in 0..batch_size {
                        // Project sample onto component
                        let mut proj = 0.0f32;
                        for j in 0..n_features {
                            proj += batch[i * n_features + j] * components[k][j];
                        }

                        // Gradient of reconstruction error
                        for j in 0..n_features {
                            grad[j] += proj * batch[i * n_features + j];
                        }
                    }

                    // Update with soft thresholding (L1)
                    let lr = 0.01 / (iter + 1) as f32;
                    for j in 0..n_features {
                        let g = grad[j] / batch_size as f32;
                        components[k][j] += lr * g;
                        
                        // Soft thresholding
                        let sign = components[k][j].signum();
                        components[k][j] = (components[k][j].abs() - lr * self.alpha).max(0.0) * sign;
                    }

                    normalize(&mut components[k]);
                }
            }

            self.n_iter_ = iter + 1;
        }

        self.components_ = Some(components);
        self.mean_ = Some(mean);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let components = self.components_.as_ref().expect("Model not fitted");
        let mean = self.mean_.as_ref().unwrap();

        let mut result = vec![0.0f32; n_samples * self.n_components];

        for i in 0..n_samples {
            for k in 0..self.n_components {
                for j in 0..n_features {
                    result[i * self.n_components + k] += 
                        (x_data[i * n_features + j] - mean[j]) * components[k][j];
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, self.n_components]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_pca() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ], &[4, 3]).unwrap();

        let mut ipca = IncrementalPCA::new(2).batch_size(2);
        let result = ipca.fit_transform(&x);
        assert_eq!(result.dims(), &[4, 2]);
    }

    #[test]
    fn test_incremental_pca_partial_fit() {
        let mut ipca = IncrementalPCA::new(2);
        
        let batch1 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let batch2 = Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]).unwrap();
        
        ipca.partial_fit(&batch1);
        ipca.partial_fit(&batch2);
        
        assert_eq!(ipca.n_samples_seen(), 4);
    }
}


