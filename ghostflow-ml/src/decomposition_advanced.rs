//! Advanced Decomposition - Factor Analysis, ICA, Sparse PCA, Dictionary Learning

use ghostflow_core::Tensor;
use rand::prelude::*;

/// Factor Analysis
pub struct FactorAnalysis {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f32,
    components_: Option<Vec<Vec<f32>>>,
    noise_variance_: Option<Vec<f32>>,
    mean_: Option<Vec<f32>>,
    n_iter_: usize,
}

impl FactorAnalysis {
    pub fn new(n_components: usize) -> Self {
        FactorAnalysis {
            n_components,
            max_iter: 1000,
            tol: 1e-2,
            components_: None,
            noise_variance_: None,
            mean_: None,
            n_iter_: 0,
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();

        let x_centered: Vec<f32> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - mean[j]).collect::<Vec<_>>())
            .collect();

        let mut rng = thread_rng();
        let mut components: Vec<Vec<f32>> = (0..self.n_components)
            .map(|_| (0..n_features).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();

        let mut noise_var: Vec<f32> = (0..n_features)
            .map(|j| {
                (0..n_samples).map(|i| x_centered[i * n_features + j].powi(2)).sum::<f32>() 
                    / n_samples as f32
            })
            .collect();

        // EM algorithm
        for iter in 0..self.max_iter {
            let prev_components = components.clone();

            // E-step: compute M = (I + W^T Psi^-1 W)^-1
            let mut m = vec![vec![0.0f32; self.n_components]; self.n_components];
            for i in 0..self.n_components {
                m[i][i] = 1.0;
                for j in 0..self.n_components {
                    for k in 0..n_features {
                        m[i][j] += components[i][k] * components[j][k] / noise_var[k].max(1e-10);
                    }
                }
            }

            let m_inv = invert_matrix(&m, self.n_components);

            // Sufficient statistics
            let mut s1 = vec![vec![0.0f32; n_features]; self.n_components];
            let mut s2 = vec![vec![0.0f32; self.n_components]; self.n_components];

            for i in 0..n_samples {
                let mut ez = vec![0.0f32; self.n_components];
                for k in 0..self.n_components {
                    for j in 0..n_features {
                        ez[k] += components[k][j] * x_centered[i * n_features + j] / noise_var[j].max(1e-10);
                    }
                }
                
                let mut ez_final = vec![0.0f32; self.n_components];
                for k in 0..self.n_components {
                    for l in 0..self.n_components {
                        ez_final[k] += m_inv[k][l] * ez[l];
                    }
                }

                for k in 0..self.n_components {
                    for j in 0..n_features {
                        s1[k][j] += ez_final[k] * x_centered[i * n_features + j];
                    }
                    for l in 0..self.n_components {
                        s2[k][l] += m_inv[k][l] + ez_final[k] * ez_final[l];
                    }
                }
            }

            // M-step
            let s2_inv = invert_matrix(&s2, self.n_components);
            for k in 0..self.n_components {
                for j in 0..n_features {
                    components[k][j] = 0.0;
                    for l in 0..self.n_components {
                        components[k][j] += s1[l][j] * s2_inv[k][l];
                    }
                }
            }

            for j in 0..n_features {
                let mut var = 0.0f32;
                for i in 0..n_samples {
                    var += x_centered[i * n_features + j].powi(2);
                }
                var /= n_samples as f32;
                for k in 0..self.n_components {
                    var -= components[k][j] * s1[k][j] / n_samples as f32;
                }
                noise_var[j] = var.max(1e-6);
            }

            self.n_iter_ = iter + 1;

            let mut max_change = 0.0f32;
            for k in 0..self.n_components {
                for j in 0..n_features {
                    max_change = max_change.max((components[k][j] - prev_components[k][j]).abs());
                }
            }
            if max_change < self.tol {
                break;
            }
        }

        self.components_ = Some(components);
        self.noise_variance_ = Some(noise_var);
        self.mean_ = Some(mean);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let components = self.components_.as_ref().expect("Not fitted");
        let noise_var = self.noise_variance_.as_ref().unwrap();
        let mean = self.mean_.as_ref().unwrap();

        let mut result = vec![0.0f32; n_samples * self.n_components];
        for i in 0..n_samples {
            for k in 0..self.n_components {
                for j in 0..n_features {
                    result[i * self.n_components + k] += 
                        components[k][j] * (x_data[i * n_features + j] - mean[j]) / noise_var[j].max(1e-10);
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

/// FastICA - Independent Component Analysis
pub struct FastICA {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f32,
    pub fun: ICAFunction,
    components_: Option<Vec<Vec<f32>>>,
    mixing_: Option<Vec<Vec<f32>>>,
    mean_: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
pub enum ICAFunction {
    Logcosh,
    Exp,
    Cube,
}

impl FastICA {
    pub fn new(n_components: usize) -> Self {
        FastICA {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            fun: ICAFunction::Logcosh,
            components_: None,
            mixing_: None,
            mean_: None,
        }
    }

    fn g(&self, x: f32) -> (f32, f32) {
        match self.fun {
            ICAFunction::Logcosh => {
                let t = x.tanh();
                (t, 1.0 - t * t)
            }
            ICAFunction::Exp => {
                let e = (-x * x / 2.0).exp();
                (x * e, (1.0 - x * x) * e)
            }
            ICAFunction::Cube => (x.powi(3), 3.0 * x * x),
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();

        let x_centered: Vec<f32> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - mean[j]).collect::<Vec<_>>())
            .collect();

        // Whitening via PCA
        let (x_white, whitening) = self.whiten(&x_centered, n_samples, n_features);

        // Initialize W randomly
        let mut rng = thread_rng();
        let mut w = vec![vec![0.0f32; self.n_components]; self.n_components];
        for i in 0..self.n_components {
            for j in 0..self.n_components {
                w[i][j] = rng.gen::<f32>() - 0.5;
            }
        }
        orthogonalize(&mut w, self.n_components);

        // FastICA iterations
        for _ in 0..self.max_iter {
            let w_old = w.clone();

            for i in 0..self.n_components {
                let mut g_sum = vec![0.0f32; self.n_components];
                let mut gp_mean = 0.0f32;

                for s in 0..n_samples {
                    let mut wtx = 0.0f32;
                    for j in 0..self.n_components {
                        wtx += w[i][j] * x_white[s * self.n_components + j];
                    }
                    let (g_val, gp_val) = self.g(wtx);
                    gp_mean += gp_val;
                    for j in 0..self.n_components {
                        g_sum[j] += g_val * x_white[s * self.n_components + j];
                    }
                }

                gp_mean /= n_samples as f32;
                for j in 0..self.n_components {
                    w[i][j] = g_sum[j] / n_samples as f32 - gp_mean * w[i][j];
                }
            }

            orthogonalize(&mut w, self.n_components);

            let mut max_change = 0.0f32;
            for i in 0..self.n_components {
                for j in 0..self.n_components {
                    max_change = max_change.max((w[i][j] - w_old[i][j]).abs());
                }
            }
            if max_change < self.tol {
                break;
            }
        }

        // Compute mixing matrix
        let mut mixing = vec![vec![0.0f32; self.n_components]; n_features];
        for i in 0..n_features {
            for j in 0..self.n_components {
                for k in 0..self.n_components {
                    mixing[i][j] += whitening[k][i] * w[j][k];
                }
            }
        }

        self.components_ = Some(w);
        self.mixing_ = Some(mixing);
        self.mean_ = Some(mean);
    }

    fn whiten(&self, x: &[f32], n_samples: usize, n_features: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
        // Compute covariance
        let mut cov = vec![0.0f32; n_features * n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    cov[i * n_features + j] += x[k * n_features + i] * x[k * n_features + j];
                }
                cov[i * n_features + j] /= (n_samples - 1).max(1) as f32;
            }
        }

        // Power iteration for eigendecomposition
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();
        let mut a = cov.clone();

        for _ in 0..self.n_components.min(n_features) {
            let mut v = vec![1.0f32; n_features];
            normalize(&mut v);

            for _ in 0..100 {
                let mut av = vec![0.0f32; n_features];
                for i in 0..n_features {
                    for j in 0..n_features {
                        av[i] += a[i * n_features + j] * v[j];
                    }
                }
                let norm: f32 = av.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 { break; }
                for vi in &mut av { *vi /= norm; }
                v = av;
            }

            let mut eigenvalue = 0.0f32;
            for i in 0..n_features {
                let mut av = 0.0f32;
                for j in 0..n_features {
                    av += a[i * n_features + j] * v[j];
                }
                eigenvalue += v[i] * av;
            }

            eigenvalues.push(eigenvalue.max(1e-10));
            eigenvectors.push(v.clone());

            for i in 0..n_features {
                for j in 0..n_features {
                    a[i * n_features + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        // Whitening matrix
        let mut whitening = vec![vec![0.0f32; n_features]; self.n_components];
        for i in 0..self.n_components {
            let scale = 1.0 / eigenvalues[i].sqrt();
            for j in 0..n_features {
                whitening[i][j] = scale * eigenvectors[i][j];
            }
        }

        // Apply whitening
        let mut x_white = vec![0.0f32; n_samples * self.n_components];
        for i in 0..n_samples {
            for k in 0..self.n_components {
                for j in 0..n_features {
                    x_white[i * self.n_components + k] += whitening[k][j] * x[i * n_features + j];
                }
            }
        }

        (x_white, whitening)
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mixing = self.mixing_.as_ref().expect("Not fitted");
        let mean = self.mean_.as_ref().unwrap();

        // Compute pseudo-inverse of mixing for unmixing
        let mut result = vec![0.0f32; n_samples * self.n_components];
        for i in 0..n_samples {
            for k in 0..self.n_components {
                for j in 0..n_features {
                    result[i * self.n_components + k] += 
                        mixing[j][k] * (x_data[i * n_features + j] - mean[j]);
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

/// Sparse PCA
pub struct SparsePCA {
    pub n_components: usize,
    pub alpha: f32,
    pub max_iter: usize,
    pub tol: f32,
    components_: Option<Vec<Vec<f32>>>,
    mean_: Option<Vec<f32>>,
}

impl SparsePCA {
    pub fn new(n_components: usize) -> Self {
        SparsePCA {
            n_components,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            components_: None,
            mean_: None,
        }
    }

    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();

        let x_centered: Vec<f32> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - mean[j]).collect::<Vec<_>>())
            .collect();

        let mut rng = thread_rng();
        let mut components: Vec<Vec<f32>> = (0..self.n_components)
            .map(|_| {
                let mut v: Vec<f32> = (0..n_features).map(|_| rng.gen::<f32>() - 0.5).collect();
                normalize(&mut v);
                v
            })
            .collect();

        // Coordinate descent with L1 penalty
        for _ in 0..self.max_iter {
            let prev = components.clone();

            for k in 0..self.n_components {
                // Compute X^T X v_k
                let mut grad = vec![0.0f32; n_features];
                for j in 0..n_features {
                    for i in 0..n_samples {
                        let mut xv = 0.0f32;
                        for l in 0..n_features {
                            xv += x_centered[i * n_features + l] * components[k][l];
                        }
                        grad[j] += x_centered[i * n_features + j] * xv;
                    }
                }

                // Soft thresholding
                for j in 0..n_features {
                    let val = grad[j] / n_samples as f32;
                    components[k][j] = soft_threshold(val, self.alpha);
                }

                normalize(&mut components[k]);
            }

            let mut max_change = 0.0f32;
            for k in 0..self.n_components {
                for j in 0..n_features {
                    max_change = max_change.max((components[k][j] - prev[k][j]).abs());
                }
            }
            if max_change < self.tol {
                break;
            }
        }

        self.components_ = Some(components);
        self.mean_ = Some(mean);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let components = self.components_.as_ref().expect("Not fitted");
        let mean = self.mean_.as_ref().unwrap();

        let mut result = vec![0.0f32; n_samples * self.n_components];
        for i in 0..n_samples {
            for k in 0..self.n_components {
                for j in 0..n_features {
                    result[i * self.n_components + k] += 
                        components[k][j] * (x_data[i * n_features + j] - mean[j]);
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

/// Dictionary Learning
pub struct DictionaryLearning {
    pub n_components: usize,
    pub alpha: f32,
    pub max_iter: usize,
    pub tol: f32,
    components_: Option<Vec<Vec<f32>>>,
    mean_: Option<Vec<f32>>,
}

impl DictionaryLearning {
    pub fn new(n_components: usize) -> Self {
        DictionaryLearning {
            n_components,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            components_: None,
            mean_: None,
        }
    }

    pub fn fit(&mut self, x: &Tensor) {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let mean: Vec<f32> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| x_data[i * n_features + j]).sum::<f32>() / n_samples as f32)
            .collect();

        let x_centered: Vec<f32> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(|j| x_data[i * n_features + j] - mean[j]).collect::<Vec<_>>())
            .collect();

        let mut rng = thread_rng();
        
        // Initialize dictionary
        let mut dictionary: Vec<Vec<f32>> = (0..self.n_components)
            .map(|_| {
                let mut d: Vec<f32> = (0..n_features).map(|_| rng.gen::<f32>() - 0.5).collect();
                normalize(&mut d);
                d
            })
            .collect();

        // Alternating minimization
        for _ in 0..self.max_iter {
            let prev = dictionary.clone();

            // Sparse coding step (simplified OMP-like)
            let mut codes = vec![vec![0.0f32; self.n_components]; n_samples];
            for i in 0..n_samples {
                let x_i: Vec<f32> = (0..n_features).map(|j| x_centered[i * n_features + j]).collect();
                codes[i] = sparse_encode(&x_i, &dictionary, self.alpha);
            }

            // Dictionary update
            for k in 0..self.n_components {
                let mut num = vec![0.0f32; n_features];
                let mut denom = 0.0f32;

                for i in 0..n_samples {
                    if codes[i][k].abs() > 1e-10 {
                        for j in 0..n_features {
                            let mut residual = x_centered[i * n_features + j];
                            for l in 0..self.n_components {
                                if l != k {
                                    residual -= codes[i][l] * dictionary[l][j];
                                }
                            }
                            num[j] += codes[i][k] * residual;
                        }
                        denom += codes[i][k] * codes[i][k];
                    }
                }

                if denom > 1e-10 {
                    for j in 0..n_features {
                        dictionary[k][j] = num[j] / denom;
                    }
                    normalize(&mut dictionary[k]);
                }
            }

            let mut max_change = 0.0f32;
            for k in 0..self.n_components {
                for j in 0..n_features {
                    max_change = max_change.max((dictionary[k][j] - prev[k][j]).abs());
                }
            }
            if max_change < self.tol {
                break;
            }
        }

        self.components_ = Some(dictionary);
        self.mean_ = Some(mean);
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        let x_data = x.data_f32();
        let n_samples = x.dims()[0];
        let n_features = x.dims()[1];

        let dictionary = self.components_.as_ref().expect("Not fitted");
        let mean = self.mean_.as_ref().unwrap();

        let mut result = vec![0.0f32; n_samples * self.n_components];
        for i in 0..n_samples {
            let x_i: Vec<f32> = (0..n_features).map(|j| x_data[i * n_features + j] - mean[j]).collect();
            let code = sparse_encode(&x_i, dictionary, self.alpha);
            for k in 0..self.n_components {
                result[i * self.n_components + k] = code[k];
            }
        }

        Tensor::from_slice(&result, &[n_samples, self.n_components]).unwrap()
    }

    pub fn fit_transform(&mut self, x: &Tensor) -> Tensor {
        self.fit(x);
        self.transform(x)
    }
}

// Helper functions
fn invert_matrix(m: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
    let mut a: Vec<Vec<f32>> = m.to_vec();
    let mut inv = vec![vec![0.0f32; n]; n];
    for i in 0..n { inv[i][i] = 1.0; }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > a[max_row][i].abs() { max_row = k; }
        }
        a.swap(i, max_row);
        inv.swap(i, max_row);

        let diag = a[i][i];
        if diag.abs() > 1e-10 {
            for j in 0..n {
                a[i][j] /= diag;
                inv[i][j] /= diag;
            }
        }

        for k in 0..n {
            if k != i {
                let factor = a[k][i];
                for j in 0..n {
                    a[k][j] -= factor * a[i][j];
                    inv[k][j] -= factor * inv[i][j];
                }
            }
        }
    }
    inv
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for vi in v.iter_mut() { *vi /= norm; }
    }
}

fn orthogonalize(w: &mut [Vec<f32>], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let dot: f32 = (0..n).map(|k| w[i][k] * w[j][k]).sum();
            for k in 0..n {
                w[i][k] -= dot * w[j][k];
            }
        }
        normalize(&mut w[i]);
    }
}

fn soft_threshold(x: f32, lambda: f32) -> f32 {
    if x > lambda { x - lambda }
    else if x < -lambda { x + lambda }
    else { 0.0 }
}

fn sparse_encode(x: &[f32], dictionary: &[Vec<f32>], alpha: f32) -> Vec<f32> {
    let n_components = dictionary.len();
    let n_features = x.len();
    let mut code = vec![0.0f32; n_components];

    // Simple coordinate descent
    for _ in 0..100 {
        for k in 0..n_components {
            let mut residual_dot = 0.0f32;
            for j in 0..n_features {
                let mut residual = x[j];
                for l in 0..n_components {
                    if l != k {
                        residual -= code[l] * dictionary[l][j];
                    }
                }
                residual_dot += residual * dictionary[k][j];
            }
            code[k] = soft_threshold(residual_dot, alpha);
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_analysis() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();
        let mut fa = FactorAnalysis::new(2);
        let result = fa.fit_transform(&x);
        assert_eq!(result.dims(), &[3, 2]);
    }

    #[test]
    fn test_fast_ica() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let mut ica = FastICA::new(2);
        let result = ica.fit_transform(&x);
        assert_eq!(result.dims(), &[4, 2]);
    }

    #[test]
    fn test_sparse_pca() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let mut spca = SparsePCA::new(2);
        let result = spca.fit_transform(&x);
        assert_eq!(result.dims(), &[2, 2]);
    }
}
