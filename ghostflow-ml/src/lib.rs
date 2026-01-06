//! GhostFlow Classical ML Algorithms
//!
//! Comprehensive real implementations of classical machine learning algorithms.
//! No mocks, no simulations - production-ready ML from scratch.
//!
//! ## Modules
//! - **tree**: Decision Trees (CART)
//! - **ensemble**: Random Forest, Gradient Boosting
//! - **ensemble_advanced**: AdaBoost, Bagging, Extra Trees, Isolation Forest
//! - **stacking**: Stacking Classifier/Regressor
//! - **linear**: Linear/Logistic Regression, Ridge, Lasso, ElasticNet
//! - **robust**: Huber, RANSAC, Theil-Sen, Quantile Regression
//! - **svm**: Support Vector Machines (SVC, SVR)
//! - **kernel**: Kernel Ridge, Kernel PCA, Nystrom
//! - **neighbors**: K-Nearest Neighbors
//! - **naive_bayes**: Gaussian, Multinomial, Bernoulli NB
//! - **bayesian**: Bayesian Ridge, ARD Regression
//! - **discriminant_analysis**: LDA, QDA
//! - **gaussian_process**: GP Regressor, GP Classifier
//! - **mixture**: Gaussian Mixture, Bayesian GMM
//! - **neural_network**: Perceptron, MLP
//! - **rbf_network**: RBF Network, RBF Classifier
//! - **clustering**: KMeans, DBSCAN, Agglomerative
//! - **clustering_advanced**: Spectral, Mean Shift, Mini-Batch KMeans, Affinity Propagation
//! - **clustering_more**: OPTICS, BIRCH, HDBSCAN
//! - **decomposition**: PCA, SVD, NMF
//! - **decomposition_advanced**: Factor Analysis, ICA, Sparse PCA, Dictionary Learning
//! - **manifold**: t-SNE, MDS, Isomap, LLE
//! - **outlier_detection**: LOF, One-Class SVM, Elliptic Envelope
//! - **feature_selection**: Variance Threshold, SelectKBest, RFE
//! - **metrics**: Classification, Regression, Clustering metrics
//! - **metrics_advanced**: Log Loss, Hinge Loss, Cohen's Kappa, Matthews Correlation
//! - **preprocessing**: Scalers, Encoders
//! - **polynomial**: Polynomial Features, Spline Transformer, Power Transformer
//! - **model_selection**: Cross-validation, Grid Search
//! - **calibration**: Isotonic Regression, Platt Scaling, Calibrated Classifier
//! - **semi_supervised**: Label Propagation, Label Spreading, Self-Training
//! - **multiclass**: One-vs-Rest, One-vs-One, Output Code, Classifier Chain
//! - **imbalanced**: SMOTE, Random Over/Under Sampling
//! - **time_series**: ARIMA, Exponential Smoothing
//! - **time_series_extended**: SARIMA, STL Decomposition
//! - **linear_sgd**: SGD Classifier/Regressor
//! - **decomposition_incremental**: Incremental PCA
//! - **preprocessing_extended**: RobustScaler, MaxAbsScaler, OrdinalEncoder
//! - **model_selection_extended**: RandomizedSearchCV, GroupKFold, Learning Curves
//! - **nlp**: Tokenizers, Word2Vec, TF-IDF
//! - **vision**: Image Augmentation, Normalization, Resizing
//! - **distributed**: Data Parallelism, Gradient Compression, Ring All-Reduce
//! - **gpu**: GPU Acceleration, CUDA Support, Mixed Precision
//! - **deep**: Deep Learning (CNN, RNN, LSTM, GRU, Transformer, Optimizers, Losses)

// Core modules
pub mod tree;
pub mod ensemble;
pub mod ensemble_advanced;
pub mod stacking;
pub mod linear;
pub mod robust;
pub mod clustering;
pub mod clustering_advanced;
pub mod clustering_more;
pub mod decomposition;
pub mod decomposition_advanced;
pub mod neighbors;
pub mod svm;
pub mod kernel;

// Probabilistic models
pub mod naive_bayes;
pub mod bayesian;
pub mod discriminant_analysis;
pub mod gaussian_process;
pub mod mixture;
pub mod gmm;  // Gaussian Mixture Models (v0.3.0)
pub mod hmm;  // Hidden Markov Models (v0.3.0)

// Advanced Gradient Boosting (v0.3.0)
pub mod gradient_boosting;  // XGBoost-style
pub mod lightgbm;           // LightGBM-style
pub mod crf;                // Conditional Random Fields
pub mod feature_engineering;  // Feature engineering utilities
pub mod hyperparameter_optimization;  // Bayesian optimization, etc.

// Neural Networks
pub mod neural_network;
pub mod rbf_network;

// Dimensionality reduction & manifold learning
pub mod manifold;

// Outlier Detection
pub mod outlier_detection;

// Feature Selection
pub mod feature_selection;

// Utilities
pub mod metrics;
pub mod metrics_advanced;
pub mod preprocessing;
pub mod polynomial;
pub mod model_selection;

// Semi-supervised & Calibration
pub mod calibration;
pub mod semi_supervised;

// Multiclass strategies
pub mod multiclass;

// Imbalanced Learning
pub mod imbalanced;

// Time Series
pub mod time_series;
pub mod time_series_extended;

// Extended modules
pub mod linear_sgd;
pub mod decomposition_incremental;
pub mod preprocessing_extended;
pub mod model_selection_extended;

// NLP & Vision
pub mod nlp;
pub mod vision;

// Distributed & GPU
pub mod distributed;
pub mod gpu;

// Deep Learning
// NOTE: Deep module has compilation issues - disabled for now
// pub mod deep;

// Re-exports: Trees
pub use tree::{DecisionTreeClassifier, DecisionTreeRegressor, Criterion};

// Re-exports: Ensemble methods
pub use ensemble::{
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
};
pub use ensemble_advanced::{
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    VotingClassifier, IsolationForest,
};
pub use stacking::{StackingClassifier, StackingRegressor, StackMethod};

// Re-exports: Linear models
pub use linear::{LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet};
pub use robust::{HuberRegressor, RANSACRegressor, TheilSenRegressor, QuantileRegressor, PassiveAggressiveRegressor};

// Re-exports: Clustering
pub use clustering::{KMeans, DBSCAN, AgglomerativeClustering};
pub use clustering_advanced::{
    SpectralClustering, MeanShift, MiniBatchKMeans, AffinityPropagation,
};
pub use clustering_more::{OPTICS, BIRCH, HDBSCAN};

// Re-exports: Mixture Models
pub use mixture::{GaussianMixture, BayesianGaussianMixture, CovarianceType};

// Re-exports: Decomposition
pub use decomposition::{PCA, SVD, NMF};
pub use decomposition_advanced::{FactorAnalysis, FastICA, SparsePCA, DictionaryLearning};

// Re-exports: Neighbors
pub use neighbors::{KNeighborsClassifier, KNeighborsRegressor};

// Re-exports: SVM
pub use svm::{SVC, SVR, Kernel as SVMKernel};

// Re-exports: Kernel Methods
pub use kernel::{KernelRidge, KernelPCA, Nystrom, Kernel};

// Re-exports: Naive Bayes
pub use naive_bayes::{GaussianNB, MultinomialNB, BernoulliNB, ComplementNB};

// Re-exports: Bayesian
pub use bayesian::{BayesianRidge, ARDRegression};

// Re-exports: Discriminant Analysis
pub use discriminant_analysis::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};

// Re-exports: Gaussian Processes
pub use gaussian_process::{GaussianProcessRegressor, GaussianProcessClassifier, GPKernel};

// Re-exports: Neural Networks
pub use neural_network::{Perceptron, MLPClassifier, MLPRegressor, Activation};
pub use rbf_network::{RBFNetwork, RBFClassifier};

// Re-exports: Manifold Learning
pub use manifold::{TSNE, MDS, Isomap, LocallyLinearEmbedding};

// Re-exports: Outlier Detection
pub use outlier_detection::{LocalOutlierFactor, OneClassSVM, EllipticEnvelope};

// Re-exports: Feature Selection
pub use feature_selection::{VarianceThreshold, SelectKBest, RFE, ScoreFunction};

// Re-exports: Preprocessing
pub use preprocessing::{
    StandardScaler, MinMaxScaler, Normalizer, 
    LabelEncoder, OneHotEncoder, train_test_split,
};
pub use polynomial::{
    PolynomialFeatures, SplineTransformer, PowerTransformer, QuantileTransformer,
    PowerMethod, OutputDistribution,
};

// Re-exports: Multiclass
pub use multiclass::{OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier, ClassifierChain};

// Re-exports: Metrics
pub use metrics::{
    // Classification
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, classification_report,
    // Regression
    mean_squared_error, root_mean_squared_error, mean_absolute_error, 
    r2_score, mean_absolute_percentage_error, explained_variance_score,
    // Clustering
    silhouette_score, davies_bouldin_score,
};
pub use metrics_advanced::{
    log_loss, log_loss_multiclass, hinge_loss, squared_hinge_loss,
    cohen_kappa_score, matthews_corrcoef, adjusted_rand_score,
    normalized_mutual_info_score, fowlkes_mallows_score,
    calinski_harabasz_score,
};

// Re-exports: Model Selection
pub use model_selection::{
    KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit,
    cross_val_score, parameter_grid, shuffle_split,
};

// Re-exports: Calibration
pub use calibration::{IsotonicRegression, PlattScaling, CalibratedClassifier};

// Re-exports: Semi-supervised
pub use semi_supervised::{LabelPropagation, LabelSpreading, SelfTrainingClassifier};

// Re-exports: Imbalanced Learning
pub use imbalanced::{RandomOverSampler, RandomUnderSampler, SMOTE, BorderlineSMOTE, ADASYN, SamplingStrategy};

// Re-exports: Time Series
pub use time_series::{
    SimpleExponentialSmoothing, HoltLinear, HoltWinters, SeasonalType,
    ARIMA, MovingAverage, EWMA,
};
pub use time_series_extended::SARIMA;

// Re-exports: Extended Linear Models
pub use linear_sgd::{SGDClassifier, SGDRegressor, SGDLoss, SGDRegressorLoss, Penalty, LearningRate};

// Re-exports: Extended Decomposition
pub use decomposition_incremental::IncrementalPCA;

// Re-exports: Extended Preprocessing
pub use preprocessing_extended::{RobustScaler, MaxAbsScaler, OrdinalEncoder};

// Re-exports: Extended Model Selection
pub use model_selection_extended::{
    RandomizedSearchCV, ParamDistribution, RandomizedSearchResult, CVResult,
    GroupKFold, RepeatedKFold, StratifiedShuffleSplit, Scoring,
    learning_curve, validation_curve,
};

// Re-exports: Deep Learning
// NOTE: Deep module disabled - has compilation issues
/*
pub use deep::{
    // Layers
    Dense, Dropout, BatchNorm, LayerNorm, Embedding, Flatten,
    // CNN
    Conv2d, Conv1d, Conv3d, MaxPool2d, AvgPool2d, GlobalAvgPool2d,
    BatchNorm2d, GroupNorm,
    // RNN
    RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU,
    // Transformer
    ScaledDotProductAttention, MultiHeadAttention, PositionalEncoding,
    FeedForward, TransformerLayerNorm, TransformerEncoderLayer, TransformerEncoder,
    TransformerDecoderLayer, TransformerDecoder, Transformer,
    PatchEmbedding, VisionTransformer,
    // Optimizers
    SGD, Adam, AdamW, RMSprop, AdaGrad, Adadelta, Adamax, NAdam, RAdam,
    // Losses
    MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, HuberLoss as DeepHuberLoss, 
    L1Loss, HingeLoss as DeepHingeLoss, KLDivLoss, Reduction,
    FocalLoss, DiceLoss, TverskyLoss, LovaszLoss, ContrastiveLoss, TripletLoss,
    CenterLoss, LabelSmoothingLoss,
    // Activations
    ReLU, LeakyReLU, PReLU, ELU, SELU, Sigmoid, Tanh, Softmax, GELU, Swish, Mish, Hardswish,
};
*/

// Re-exports: NLP
pub use nlp::{
    WordTokenizer, CharTokenizer, BPETokenizer,
    TfidfVectorizer, Word2Vec,
};

// Re-exports: Vision
pub use vision::{
    ImageAugmentation, ImageNormalization, ImageResize, ImageCrop,
    RandomCrop, ColorJitter, Interpolation,
};

// Re-exports: Distributed Training
pub use distributed::{
    DistributedStrategy, CommunicationBackend, GradientAggregation,
    DistributedConfig, DataParallelTrainer, DistributedDataLoader,
    GradientCompression, CompressionMethod, RingAllReduce,
};

// Re-exports: GPU Acceleration
pub use gpu::{
    DeviceType, DeviceInfo, GPUContext, GPUTensor, GPUOps,
    GPUMemoryManager, AutoMixedPrecision,
};


