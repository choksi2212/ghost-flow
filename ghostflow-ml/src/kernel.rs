//! Kernel Methods - Kernel Ridge, Kernel PCA

use ghostflow_core::Tensor;

/// Kernel functions
#[derive(Clone)]
pub enum Kernel {
    Linear,
    Polynomial { degree: usize, coef0: f32 },
    RBF { gamma: f32 },
    Sigmoid { gamma: f32, coef0: f32 },
}

impl Kernel {
    pub fn compute(&self, x1: &[f32], x2: &[f32]) -> f32 {
        match self {
            Kernel::Linear => {
                x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum()
            }
            Kernel::Polynomial { degree, coef0 } => {
                let dot: f32 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
                (dot + coef0).powi(*degree as i32)
            }
            Kernel::RBF { gamma } => {
                let sq_dist: f32 = x1.iter().zip(x2.iter())
                    .map(|(&a, &b)| (a - b).powi(2)).sum();
                (-gamma * sq_dist).exp()
            }
            Kernel::Sigmoid { gamma, coef0 } => {
                let dot: f32 = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
                (gamma * dot + coef0).tanh()
            }
        }
    }

    pub fn rbf(gamma: f32) -> Self {
        Kernel::RBF { gamma }
    }

    pub fn polynomial(degree: usize) -> Self {
        Kernel::Polynomial { degree, coef0: 1.0 }
    }
}

/// Kernel Ridge Regression
pub struct KernelRidge {
    pub alpha: f32,
    pub kernel: Kernel,
    x_train_: Option<Vec<f32>>,
    dual_coef_: Option<Vec<f32>>,
    n_samples_: usize,
    n_features_: usize,
}

impl KernelRidge {
    pub fn new() -> Self {
        KernelRidge {
            alpha: 1.0,
            kernel: Kernel::RBF { gamma: 1.0 },
            x_train_: None,
            dual_coef_: None,
            n_samples_: 0,
            n_features_: 0,
        }
    }

    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) {
        let x_data = x.data_f32();
        let y_data = y.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute kernel matrix
        let mut k_matrix = vec![0.0f32; n_samples * n_samples];
        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            for j in 0..n_samples {
                let xj = &x_data[j * n_features..(j + 1) * n_features];
                k_matrix[i * n_samples + j] = self.kernel.compute(xi, xj);
            }
        }

        // Add regularization: K + alpha * I
        for i in 0..n_samples {
            k_matrix[i * n_samples + i] += self.alpha;
        }

        // Solve (K + alpha*I) * dual_coef = y
        let dual_coef = solve_linear_system(&k_matrix, &y_data, n_samples);

        self.x_train_ = Some(x_data.clone());
        self.dual_coef_ = Some(dual_coef);
        self.n_samples_ = n_samples;
        self.n_features_ = n_features;
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let x_train = self.x_train_.as_ref().expect("Model not fitted");
        let dual_coef = self.dual_coef_.as_ref().unwrap();

        let mut predictions = vec![0.0f32; n_samples];

        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            for j in 0..self.n_samples_ {
                let xj = &x_train[j * self.n_features_..(j + 1) * self.n_features_];
                predictions[i] += dual_coef[j] * self.kernel.compute(xi, xj);
            }
        }

        Tensor::from_slice(&predictions, &[n_samples]).unwrap()
    }
}

impl Default for KernelRidge {
    fn default() -> Self { Self::new() }
}

/// Kernel PCA
pub struct KernelPCA {
    pub n_components: usize,
    pub kernel: Kernel,
    x_train_: Option<Vec<f32>>,
    alphas_: Option<Vec<Vec<f32>>>,
    lambdas_: Option<Vec<f32>>,
    n_samples_: usize,
    n_features_: usize,
}

impl KernelPCA {
    pub fn new(n_components: usize) -> Self {
        KernelPCA {
            n_components,
            kernel: Kernel::RBF { gamma: 1.0 },
            x_train_: None,
            alphas_: None,
            lambdas_: None,
            n_samples_: 0,
            n_features_: 0,
        }
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        // Compute kernel matrix
        let mut k_matrix = vec![0.0f32; n_samples * n_samples];
        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            for j in 0..n_samples {
                let xj = &x_data[j * n_features..(j + 1) * n_features];
                k_matrix[i * n_samples + j] = self.kernel.compute(xi, xj);
            }
        }

        // Center kernel matrix
        let row_means: Vec<f32> = (0..n_samples)
            .map(|i| (0..n_samples).map(|j| k_matrix[i * n_samples + j]).sum::<f32>() / n_samples as f32)
            .collect();
        let total_mean: f32 = row_means.iter().sum::<f32>() / n_samples as f32;

        for i in 0..n_samples {
            for j in 0..n_samples {
                k_matrix[i * n_samples + j] = k_matrix[i * n_samples + j] 
                    - row_means[i] - row_means[j] + total_mean;
            }
        }

        // Eigendecomposition using power iteration
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();
        let mut k_deflated = k_matrix.clone();

        for _ in 0..self.n_components.min(n_samples) {
            let mut v = vec![1.0f32; n_samples];
            normalize(&mut v);

            for _ in 0..100 {
                let mut kv = vec![0.0f32; n_samples];
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        kv[i] += k_deflated[i * n_samples + j] * v[j];
                    }
                }

                let norm: f32 = kv.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 { break; }
                
                let prev_v = v.clone();
                for (vi, kvi) in v.iter_mut().zip(kv.iter()) {
                    *vi = *kvi / norm;
                }

                let diff: f32 = v.iter().zip(prev_v.iter())
                    .map(|(&a, &b)| (a - b).abs()).sum();
                if diff < 1e-6 { break; }
            }

            // Compute eigenvalue
            let mut eigenvalue = 0.0f32;
            for i in 0..n_samples {
                let mut kv_i = 0.0f32;
                for j in 0..n_samples {
                    kv_i += k_deflated[i * n_samples + j] * v[j];
                }
                eigenvalue += v[i] * kv_i;
            }

            if eigenvalue > 1e-10 {
                eigenvalues.push(eigenvalue);
                eigenvectors.push(v.clone());

                // Deflate
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        k_deflated[i * n_samples + j] -= eigenvalue * v[i] * v[j];
                    }
                }
            }
        }

        // Normalize eigenvectors by sqrt(eigenvalue)
        let mut alphas = Vec::new();
        for (i, ev) in eigenvectors.iter().enumerate() {
            let scale = 1.0 / eigenvalues[i].sqrt();
            let alpha: Vec<f32> = ev.iter().map(|&v| v * scale).collect();
            alphas.push(alpha);
        }

        self.x_train_ = Some(x_data.clone());
        self.alphas_ = Some(alphas);
        self.lambdas_ = Some(eigenvalues);
        self.n_samples_ = n_samples;
        self.n_features_ = n_features;
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let x_train = self.x_train_.as_ref().expect("Model not fitted");
        let alphas = self.alphas_.as_ref().unwrap();
        let n_train = self.n_samples_;

        // Compute kernel between new points and training points
        let mut k_new = vec![0.0f32; n_samples * n_train];
        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            for j in 0..n_train {
                let xj = &x_train[j * self.n_features_..(j + 1) * self.n_features_];
                k_new[i * n_train + j] = self.kernel.compute(xi, xj);
            }
        }

        // Project onto principal components
        let n_components = alphas.len();
        let mut result = vec![0.0f32; n_samples * n_components];

        for i in 0..n_samples {
            for (k, alpha) in alphas.iter().enumerate() {
                for j in 0..n_train {
                    result[i * n_components + k] += k_new[i * n_train + j] * alpha[j];
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_components]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

/// Nystrom approximation for large-scale kernel methods
pub struct Nystrom {
    pub n_components: usize,
    pub kernel: Kernel,
    component_indices_: Option<Vec<usize>>,
    components_: Option<Vec<f32>>,
    normalization_: Option<Vec<Vec<f32>>>,
    n_features_: usize,
}

impl Nystrom {
    pub fn new(n_components: usize) -> Self {
        Nystrom {
            n_components,
            kernel: Kernel::RBF { gamma: 1.0 },
            component_indices_: None,
            components_: None,
            normalization_: None,
            n_features_: 0,
        }
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn fit(&mut self, x: &Tensor) {
        use rand::prelude::*;
        
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let n_components = self.n_components.min(n_samples);

        // Random sampling of landmark points
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..n_samples)
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, n_components)
            .cloned()
            .collect();

        // Extract component data
        let components: Vec<f32> = indices.iter()
            .flat_map(|&i| x_data[i * n_features..(i + 1) * n_features].to_vec())
            .collect();

        // Compute kernel matrix between landmarks
        let mut k_mm = vec![0.0f32; n_components * n_components];
        for i in 0..n_components {
            let xi = &components[i * n_features..(i + 1) * n_features];
            for j in 0..n_components {
                let xj = &components[j * n_features..(j + 1) * n_features];
                k_mm[i * n_components + j] = self.kernel.compute(xi, xj);
            }
        }

        // Compute K_mm^(-1/2) via eigendecomposition
        let (eigenvalues, eigenvectors) = eigen_decomp(&k_mm, n_components);
        
        let mut normalization = vec![vec![0.0f32; n_components]; n_components];
        for i in 0..n_components {
            for j in 0..n_components {
                for k in 0..n_components {
                    if eigenvalues[k] > 1e-10 {
                        normalization[i][j] += eigenvectors[i][k] * eigenvectors[j][k] 
                            / eigenvalues[k].sqrt();
                    }
                }
            }
        }

        self.component_indices_ = Some(indices);
        self.components_ = Some(components);
        self.normalization_ = Some(normalization);
        self.n_features_ = n_features;
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let components = self.components_.as_ref().expect("Model not fitted");
        let normalization = self.normalization_.as_ref().unwrap();
        let n_components = self.n_components;

        // Compute kernel between x and landmarks
        let mut k_nm = vec![0.0f32; n_samples * n_components];
        for i in 0..n_samples {
            let xi = &x_data[i * n_features..(i + 1) * n_features];
            for j in 0..n_components {
                let xj = &components[j * self.n_features_..(j + 1) * self.n_features_];
                k_nm[i * n_components + j] = self.kernel.compute(xi, xj);
            }
        }

        // Apply normalization: K_nm @ K_mm^(-1/2)
        let mut result = vec![0.0f32; n_samples * n_components];
        for i in 0..n_samples {
            for j in 0..n_components {
                for k in 0..n_components {
                    result[i * n_components + j] += k_nm[i * n_components + k] * normalization[k][j];
                }
            }
        }

        Tensor::from_slice(&result, &[n_samples, n_components]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

// Helper functions
fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut aug = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k * (n + 1) + i].abs() > aug[max_row * (n + 1) + i].abs() {
                max_row = k;
            }
        }

        for j in 0..=n {
            aug.swap(i * (n + 1) + j, max_row * (n + 1) + j);
        }

        let pivot = aug[i * (n + 1) + i];
        if pivot.abs() < 1e-10 { continue; }

        for j in i..=n {
            aug[i * (n + 1) + j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k * (n + 1) + i];
                for j in i..=n {
                    aug[k * (n + 1) + j] -= factor * aug[i * (n + 1) + j];
                }
            }
        }
    }

    (0..n).map(|i| aug[i * (n + 1) + n]).collect()
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for vi in v.iter_mut() { *vi /= norm; }
    }
}

fn eigen_decomp(a: &[f32], n: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();
    let mut a_deflated = a.to_vec();

    for _ in 0..n {
        let mut v = vec![1.0f32; n];
        normalize(&mut v);

        for _ in 0..100 {
            let mut av = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    av[i] += a_deflated[i * n + j] * v[j];
                }
            }

            let norm: f32 = av.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 { break; }
            
            for (vi, avi) in v.iter_mut().zip(av.iter()) {
                *vi = *avi / norm;
            }
        }

        let mut eigenvalue = 0.0f32;
        for i in 0..n {
            let mut av_i = 0.0f32;
            for j in 0..n {
                av_i += a_deflated[i * n + j] * v[j];
            }
            eigenvalue += v[i] * av_i;
        }

        eigenvalues.push(eigenvalue.max(0.0));
        eigenvectors.push(v.clone());

        for i in 0..n {
            for j in 0..n {
                a_deflated[i * n + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_ridge() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        
        let mut kr = KernelRidge::new().kernel(Kernel::rbf(0.5));
        kr.fit(&x, &y);
        let pred = kr.predict(&x);
        assert_eq!(pred.dims(), &[3]);
    }

    #[test]
    fn test_kernel_pca() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        
        let mut kpca = KernelPCA::new(2).kernel(Kernel::rbf(1.0));
        let result = kpca.fit_transform(&x);
        assert_eq!(result.dims(), &[4, 2]);
    }
}


