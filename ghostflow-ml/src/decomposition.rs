//! Dimensionality reduction - PCA, SVD, NMF

use ghostflow_core::Tensor;

/// Principal Component Analysis using power iteration for eigendecomposition
pub struct PCA {
    pub n_components: usize,
    pub whiten: bool,
    pub components_: Option<Vec<Vec<f32>>>,
    pub explained_variance_: Option<Vec<f32>>,
    pub explained_variance_ratio_: Option<Vec<f32>>,
    pub mean_: Option<Vec<f32>>,
    pub n_features_: usize,
    pub n_samples_: usize,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            whiten: false,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            mean_: None,
            n_features_: 0,
            n_samples_: 0,
        }
    }

    pub fn whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    fn compute_mean(&self, x: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let mut mean = vec![0.0f32; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += x[i * n_features + j];
            }
        }
        for m in &mut mean {
            *m /= n_samples as f32;
        }
        mean
    }

    fn center_data(&self, x: &[f32], mean: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let mut centered = vec![0.0f32; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                centered[i * n_features + j] = x[i * n_features + j] - mean[j];
            }
        }
        centered
    }

    fn compute_covariance(&self, x_centered: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let mut cov = vec![0.0f32; n_features * n_features];
        
        // C = X^T X / (n - 1)
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0f32;
                for k in 0..n_samples {
                    sum += x_centered[k * n_features + i] * x_centered[k * n_features + j];
                }
                cov[i * n_features + j] = sum / (n_samples - 1).max(1) as f32;
            }
        }
        
        cov
    }

    /// Power iteration to find dominant eigenvector
    fn power_iteration(&self, matrix: &[f32], n: usize, max_iter: usize, tol: f32) -> (Vec<f32>, f32) {
        let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
        let mut eigenvalue = 0.0f32;

        for _ in 0..max_iter {
            // w = A * v
            let mut w = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += matrix[i * n + j] * v[j];
                }
            }

            // Compute eigenvalue (Rayleigh quotient)
            let new_eigenvalue: f32 = w.iter().zip(v.iter()).map(|(&wi, &vi)| wi * vi).sum();

            // Normalize
            let norm: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                break;
            }
            for wi in &mut w {
                *wi /= norm;
            }

            // Check convergence
            let diff: f32 = v.iter().zip(w.iter()).map(|(&vi, &wi)| (vi - wi).abs()).sum();
            v = w;
            eigenvalue = new_eigenvalue;

            if diff < tol {
                break;
            }
        }

        (v, eigenvalue)
    }

    /// Deflation: remove contribution of found eigenvector
    fn deflate(&self, matrix: &mut [f32], eigenvector: &[f32], eigenvalue: f32, n: usize) {
        for i in 0..n {
            for j in 0..n {
                matrix[i * n + j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        self.n_samples_ = n_samples;
        self.n_features_ = n_features;

        // Compute mean and center data
        let mean = self.compute_mean(&x_data, n_samples, n_features);
        let x_centered = self.center_data(&x_data, &mean, n_samples, n_features);

        // Compute covariance matrix
        let mut cov = self.compute_covariance(&x_centered, n_samples, n_features);

        // Find top k eigenvectors using power iteration with deflation
        let k = self.n_components.min(n_features);
        let mut components = Vec::with_capacity(k);
        let mut eigenvalues = Vec::with_capacity(k);

        for _ in 0..k {
            let (eigenvector, eigenvalue) = self.power_iteration(&cov, n_features, 1000, 1e-6);
            
            if eigenvalue < 1e-10 {
                break;
            }

            components.push(eigenvector.clone());
            eigenvalues.push(eigenvalue);

            // Deflate
            self.deflate(&mut cov, &eigenvector, eigenvalue, n_features);
        }

        // Compute explained variance ratio
        let total_variance: f32 = eigenvalues.iter().sum();
        let explained_variance_ratio: Vec<f32> = eigenvalues.iter()
            .map(|&e| e / total_variance.max(1e-10))
            .collect();

        self.components_ = Some(components);
        self.explained_variance_ = Some(eigenvalues);
        self.explained_variance_ratio_ = Some(explained_variance_ratio);
        self.mean_ = Some(mean);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean = self.mean_.as_ref().expect("Model not fitted");
        let components = self.components_.as_ref().expect("Model not fitted");
        let k = components.len();

        let mut result = vec![0.0f32; n_samples * k];

        for i in 0..n_samples {
            for (c, component) in components.iter().enumerate() {
                let mut sum = 0.0f32;
                for j in 0..n_features {
                    sum += (x_data[i * n_features + j] - mean[j]) * component[j];
                }
                
                if self.whiten {
                    let var = self.explained_variance_.as_ref().unwrap()[c];
                    sum /= var.sqrt().max(1e-10);
                }
                
                result[i * k + c] = sum;
            }
        }

        Tensor::from_slice(&result, &[n_samples, k]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }

    pub fn inverse_transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let k = x.dims()[1];

        let mean = self.mean_.as_ref().expect("Model not fitted");
        let components = self.components_.as_ref().expect("Model not fitted");
        let n_features = self.n_features_;

        let mut result = vec![0.0f32; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut sum = mean[j];
                for c in 0..k {
                    let mut val = x_data[i * k + c];
                    if self.whiten {
                        let var = self.explained_variance_.as_ref().unwrap()[c];
                        val *= var.sqrt();
                    }
                    sum += val * components[c][j];
                }
                result[i * n_features + j] = sum;
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }
}

/// Singular Value Decomposition using power iteration
pub struct SVD {
    pub n_components: Option<usize>,
    pub u_: Option<Vec<Vec<f32>>>,
    pub s_: Option<Vec<f32>>,
    pub vt_: Option<Vec<Vec<f32>>>,
}

impl SVD {
    pub fn new(n_components: Option<usize>) -> Self {
        SVD {
            n_components,
            u_: None,
            s_: None,
            vt_: None,
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let m = x.dims()[0];
        let n = x.dims()[1];
        let k = self.n_components.unwrap_or(m.min(n));

        let mut a = x_data.clone();
        let mut u_vecs = Vec::with_capacity(k);
        let mut singular_values = Vec::with_capacity(k);
        let mut v_vecs = Vec::with_capacity(k);

        let max_iter = 100;
        let tol = 1e-6;

        for _ in 0..k {
            // Power iteration to find largest singular value
            let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
            let mut u = vec![0.0f32; m];
            let mut sigma = 0.0f32;

            for _ in 0..max_iter {
                // u = A * v
                for i in 0..m {
                    u[i] = 0.0;
                    for j in 0..n {
                        u[i] += a[i * n + j] * v[j];
                    }
                }

                // Normalize u
                let u_norm: f32 = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if u_norm < tol {
                    break;
                }
                for ui in &mut u {
                    *ui /= u_norm;
                }

                // v = A^T * u
                let mut new_v = vec![0.0f32; n];
                for j in 0..n {
                    for i in 0..m {
                        new_v[j] += a[i * n + j] * u[i];
                    }
                }

                // Compute sigma and normalize v
                let v_norm: f32 = new_v.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if v_norm < tol {
                    break;
                }
                sigma = v_norm;
                for vj in &mut new_v {
                    *vj /= v_norm;
                }

                // Check convergence
                let diff: f32 = v.iter().zip(new_v.iter())
                    .map(|(&old, &new)| (old - new).abs())
                    .sum();
                v = new_v;

                if diff < tol {
                    break;
                }
            }

            if sigma < tol {
                break;
            }

            singular_values.push(sigma);
            u_vecs.push(u.clone());
            v_vecs.push(v.clone());

            // Deflate: A = A - sigma * u * v^T
            for i in 0..m {
                for j in 0..n {
                    a[i * n + j] -= sigma * u[i] * v[j];
                }
            }
        }

        self.u_ = Some(u_vecs);
        self.s_ = Some(singular_values);
        self.vt_ = Some(v_vecs);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let vt = self.vt_.as_ref().expect("Model not fitted");
        let k = vt.len();

        let mut result = vec![0.0f32; n_samples * k];

        for i in 0..n_samples {
            for c in 0..k {
                let mut sum = 0.0f32;
                for j in 0..n_features {
                    sum += x_data[i * n_features + j] * vt[c][j];
                }
                result[i * k + c] = sum;
            }
        }

        Tensor::from_slice(&result, &[n_samples, k]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

/// Non-negative Matrix Factorization using multiplicative updates
pub struct NMF {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f32,
    pub alpha: f32,
    pub l1_ratio: f32,
    pub components_: Option<Vec<Vec<f32>>>,
    pub reconstruction_err_: Option<f32>,
    pub n_iter_: usize,
}

impl NMF {
    pub fn new(n_components: usize) -> Self {
        NMF {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            alpha: 0.0,
            l1_ratio: 0.0,
            components_: None,
            reconstruction_err_: None,
            n_iter_: 0,
        }
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    fn frobenius_norm(&self, x: &[f32], wh: &[f32]) -> f32 {
        x.iter().zip(wh.iter())
            .map(|(&xi, &whi)| (xi - whi).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Initialize W and H randomly (non-negative)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut w: Vec<Vec<f32>> = (0..n_samples)
            .map(|_| (0..self.n_components).map(|_| rng.gen::<f32>().abs() + 0.1).collect())
            .collect();
        
        let mut h: Vec<Vec<f32>> = (0..self.n_components)
            .map(|_| (0..n_features).map(|_| rng.gen::<f32>().abs() + 0.1).collect())
            .collect();

        let eps = 1e-10f32;
        let mut prev_error = f32::INFINITY;

        for iter in 0..self.max_iter {
            // Update H: H = H .* (W^T X) ./ (W^T W H + eps)
            // Compute W^T X
            let mut wt_x = vec![vec![0.0f32; n_features]; self.n_components];
            for k in 0..self.n_components {
                for j in 0..n_features {
                    for i in 0..n_samples {
                        wt_x[k][j] += w[i][k] * x_data[i * n_features + j];
                    }
                }
            }
            
            // Compute W^T W
            let mut wt_w = vec![vec![0.0f32; self.n_components]; self.n_components];
            for k1 in 0..self.n_components {
                for k2 in 0..self.n_components {
                    for i in 0..n_samples {
                        wt_w[k1][k2] += w[i][k1] * w[i][k2];
                    }
                }
            }
            
            // Update H
            for k in 0..self.n_components {
                for j in 0..n_features {
                    let mut denom = eps;
                    for k2 in 0..self.n_components {
                        denom += wt_w[k][k2] * h[k2][j];
                    }
                    h[k][j] *= wt_x[k][j] / denom;
                }
            }

            // Update W: W = W .* (X H^T) ./ (W H H^T + eps)
            // Compute X H^T
            let mut x_ht = vec![vec![0.0f32; self.n_components]; n_samples];
            for i in 0..n_samples {
                for k in 0..self.n_components {
                    for j in 0..n_features {
                        x_ht[i][k] += x_data[i * n_features + j] * h[k][j];
                    }
                }
            }
            
            // Compute H H^T
            let mut h_ht = vec![vec![0.0f32; self.n_components]; self.n_components];
            for k1 in 0..self.n_components {
                for k2 in 0..self.n_components {
                    for j in 0..n_features {
                        h_ht[k1][k2] += h[k1][j] * h[k2][j];
                    }
                }
            }
            
            // Update W
            for i in 0..n_samples {
                for k in 0..self.n_components {
                    let mut denom = eps;
                    for k2 in 0..self.n_components {
                        denom += w[i][k2] * h_ht[k2][k];
                    }
                    w[i][k] *= x_ht[i][k] / denom;
                }
            }

            // Check convergence
            if iter % 10 == 0 {
                // Compute W * H
                let mut wh = vec![0.0f32; n_samples * n_features];
                for i in 0..n_samples {
                    for j in 0..n_features {
                        for k in 0..self.n_components {
                            wh[i * n_features + j] += w[i][k] * h[k][j];
                        }
                    }
                }
                
                let error = self.frobenius_norm(&x_data, &wh);
                if (prev_error - error).abs() < self.tol {
                    self.n_iter_ = iter + 1;
                    break;
                }
                prev_error = error;
            }
            
            self.n_iter_ = iter + 1;
        }

        // Compute final reconstruction error
        let mut wh = vec![0.0f32; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                for k in 0..self.n_components {
                    wh[i * n_features + j] += w[i][k] * h[k][j];
                }
            }
        }
        self.reconstruction_err_ = Some(self.frobenius_norm(&x_data, &wh));
        self.components_ = Some(h);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let h = self.components_.as_ref().expect("Model not fitted");
        
        // Solve for W given H using multiplicative updates
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut w: Vec<Vec<f32>> = (0..n_samples)
            .map(|_| (0..self.n_components).map(|_| rng.gen::<f32>().abs() + 0.1).collect())
            .collect();

        let eps = 1e-10f32;

        for _ in 0..50 {
            // Compute X H^T
            let mut x_ht = vec![vec![0.0f32; self.n_components]; n_samples];
            for i in 0..n_samples {
                for k in 0..self.n_components {
                    for j in 0..n_features {
                        x_ht[i][k] += x_data[i * n_features + j] * h[k][j];
                    }
                }
            }
            
            // Compute H H^T
            let mut h_ht = vec![vec![0.0f32; self.n_components]; self.n_components];
            for k1 in 0..self.n_components {
                for k2 in 0..self.n_components {
                    for j in 0..n_features {
                        h_ht[k1][k2] += h[k1][j] * h[k2][j];
                    }
                }
            }
            
            // Update W
            for i in 0..n_samples {
                for k in 0..self.n_components {
                    let mut denom = eps;
                    for k2 in 0..self.n_components {
                        denom += w[i][k2] * h_ht[k2][k];
                    }
                    w[i][k] *= x_ht[i][k] / denom;
                }
            }
        }

        let mut result = vec![0.0f32; n_samples * self.n_components];
        for i in 0..n_samples {
            for k in 0..self.n_components {
                result[i * self.n_components + k] = w[i][k];
            }
        }

        Tensor::from_slice(&result, &[n_samples, self.n_components]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }

    pub fn inverse_transform(&self, w: &Tensor) -> Tensor {
        let w_data = w.data_f32();
        let n_samples = w.dims()[0];
        let k = w.dims()[1];

        let h = self.components_.as_ref().expect("Model not fitted");
        let n_features = h[0].len();

        let mut result = vec![0.0f32; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                for c in 0..k {
                    result[i * n_features + j] += w_data[i * k + c] * h[c][j];
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_features]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca() {
        let x = Tensor::from_slice(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ], &[4, 3]).unwrap();

        let mut pca = PCA::new(2);
        let transformed = pca.fit_transform(&x);

        assert_eq!(transformed.dims(), &[4, 2]);
    }

    #[test]
    fn test_svd() {
        let x = Tensor::from_slice(&[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ], &[3, 2]).unwrap();

        let mut svd = SVD::new(Some(2));
        svd.fit(&x);

        assert!(svd.s_.is_some());
        assert!(svd.u_.is_some());
        assert!(svd.vt_.is_some());
    }

    #[test]
    fn test_nmf() {
        let x = Tensor::from_slice(&[
            1.0, 2.0, 0.0,
            0.0, 1.0, 3.0,
            2.0, 0.0, 1.0,
        ], &[3, 3]).unwrap();

        let mut nmf = NMF::new(2).max_iter(100);
        let transformed = nmf.fit_transform(&x);

        assert_eq!(transformed.dims(), &[3, 2]);
        assert!(nmf.reconstruction_err_.unwrap() >= 0.0);
    }
}
