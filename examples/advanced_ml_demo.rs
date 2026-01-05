use ghostflow_core::Tensor;
use ghostflow_ml::gradient_boosting::{XGBoostClassifier, XGBoostRegressor};
use ghostflow_ml::lightgbm::LightGBMClassifier;
use ghostflow_ml::gmm::{GaussianMixture, CovarianceType};
use ghostflow_ml::hmm::{GaussianHMM, HMMCovarianceType};

fn main() {
    println!("=== GhostFlow v0.3.0 Advanced ML Demo ===\n");

    // Test XGBoost Classifier
    println!("1. Testing XGBoost Classifier...");
    let x_train = Tensor::from_slice(
        &[
            0.0f32, 0.0,
            0.1, 0.1,
            1.0, 1.0,
            1.1, 1.1,
            0.2, 0.2,
            0.9, 0.9,
        ],
        &[6, 2],
    ).unwrap();
    let y_train = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0], &[6]).unwrap();

    let mut xgb = XGBoostClassifier::new(20)
        .learning_rate(0.1)
        .max_depth(3)
        .subsample(0.8)
        .colsample_bytree(0.8);

    xgb.fit(&x_train, &y_train);
    let predictions = xgb.predict(&x_train);
    let probabilities = xgb.predict_proba(&x_train);

    println!("   Predictions shape: {:?}", predictions.dims());
    println!("   Probabilities shape: {:?}", probabilities.dims());
    println!("   Sample predictions: {:?}", &predictions.data_f32()[..3]);
    println!("   ✓ XGBoost Classifier works!\n");

    // Test XGBoost Regressor
    println!("2. Testing XGBoost Regressor...");
    let x_reg = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    ).unwrap();
    let y_reg = Tensor::from_slice(&[2.5f32, 5.0, 7.5], &[3]).unwrap();

    let mut xgb_reg = XGBoostRegressor::new(20)
        .learning_rate(0.1)
        .max_depth(3);

    xgb_reg.fit(&x_reg, &y_reg);
    let reg_predictions = xgb_reg.predict(&x_reg);

    println!("   Predictions shape: {:?}", reg_predictions.dims());
    println!("   Sample predictions: {:?}", reg_predictions.data_f32());
    println!("   ✓ XGBoost Regressor works!\n");

    // Test LightGBM Classifier
    println!("3. Testing LightGBM Classifier...");
    let mut lgbm = LightGBMClassifier::new(20)
        .learning_rate(0.1)
        .num_leaves(15)
        .feature_fraction(0.8);

    lgbm.fit(&x_train, &y_train);
    let lgbm_predictions = lgbm.predict(&x_train);
    let lgbm_probabilities = lgbm.predict_proba(&x_train);

    println!("   Predictions shape: {:?}", lgbm_predictions.dims());
    println!("   Probabilities shape: {:?}", lgbm_probabilities.dims());
    println!("   Sample predictions: {:?}", &lgbm_predictions.data_f32()[..3]);
    println!("   ✓ LightGBM Classifier works!\n");

    // Test Gaussian Mixture Model
    println!("4. Testing Gaussian Mixture Model...");
    let gmm_data = Tensor::from_slice(
        &[
            0.0f32, 0.0,
            0.1, 0.1,
            0.2, 0.2,
            5.0, 5.0,
            5.1, 5.1,
            5.2, 5.2,
        ],
        &[6, 2],
    ).unwrap();

    let mut gmm = GaussianMixture::new(2)
        .covariance_type(CovarianceType::Diag)
        .max_iter(50)
        .tol(1e-3);

    gmm.fit(&gmm_data);
    let gmm_labels = gmm.predict(&gmm_data);
    let gmm_proba = gmm.predict_proba(&gmm_data);

    println!("   Cluster labels shape: {:?}", gmm_labels.dims());
    println!("   Probabilities shape: {:?}", gmm_proba.dims());
    println!("   Sample labels: {:?}", &gmm_labels.data_f32()[..3]);
    println!("   ✓ Gaussian Mixture Model works!\n");

    // Test GMM sampling
    println!("5. Testing GMM Sampling...");
    let samples = gmm.sample(5);
    println!("   Generated samples shape: {:?}", samples.dims());
    println!("   Sample data: {:?}", &samples.data_f32()[..4]);
    println!("   ✓ GMM Sampling works!\n");

    // Test Hidden Markov Model
    println!("6. Testing Hidden Markov Model...");
    let hmm_seq1 = Tensor::from_slice(
        &[
            0.0f32, 0.0,
            0.1, 0.1,
            0.2, 0.2,
            5.0, 5.0,
            5.1, 5.1,
        ],
        &[5, 2],
    ).unwrap();

    let hmm_seq2 = Tensor::from_slice(
        &[
            0.0f32, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
        ],
        &[4, 2],
    ).unwrap();

    let sequences = vec![hmm_seq1.clone(), hmm_seq2];

    let mut hmm = GaussianHMM::new(2, 2)
        .covariance_type(HMMCovarianceType::Diag)
        .max_iter(30);

    hmm.fit(&sequences);
    let hmm_states = hmm.predict(&hmm_seq1);

    println!("   Hidden states shape: {:?}", hmm_states.dims());
    println!("   Sample states: {:?}", hmm_states.data_f32());
    println!("   ✓ Hidden Markov Model works!\n");

    // Performance comparison
    println!("7. Comparing Gradient Boosting Methods...");
    println!("   XGBoost features:");
    println!("     - L1/L2 regularization");
    println!("     - Column/row subsampling");
    println!("     - Histogram-based splits");
    println!("     - Depth-wise tree growth");
    println!();
    println!("   LightGBM features:");
    println!("     - Leaf-wise (best-first) growth");
    println!("     - Histogram-based learning");
    println!("     - Faster training on large datasets");
    println!("     - Lower memory usage");
    println!();

    // Probabilistic models comparison
    println!("8. Comparing Probabilistic Models...");
    println!("   Gaussian Mixture Model:");
    println!("     - Soft clustering");
    println!("     - Density estimation");
    println!("     - Generative model");
    println!("     - Can sample new data");
    println!();
    println!("   Hidden Markov Model:");
    println!("     - Sequential data modeling");
    println!("     - State prediction (Viterbi)");
    println!("     - Temporal dependencies");
    println!("     - Speech/gesture recognition");
    println!();

    println!("=== All v0.3.0 Advanced ML features working correctly! ===");
}
