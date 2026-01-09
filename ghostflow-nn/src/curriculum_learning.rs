//! Curriculum Learning
//!
//! Implements training strategies that gradually increase task difficulty:
//! - Easy-to-hard curriculum
//! - Self-paced learning
//! - Teacher-student curriculum
//! - Competence-based curriculum
//! - Dynamic difficulty adjustment

use std::collections::HashMap;

/// Curriculum learning strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurriculumStrategy {
    /// Fixed curriculum (predefined difficulty order)
    Fixed,
    /// Self-paced learning (model chooses samples)
    SelfPaced,
    /// Teacher-student (teacher guides difficulty)
    TeacherStudent,
    /// Competence-based (adjust based on performance)
    CompetenceBased,
    /// Anti-curriculum (hard-to-easy)
    AntiCurriculum,
}

/// Difficulty scoring method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifficultyMetric {
    /// Loss-based difficulty
    Loss,
    /// Prediction confidence
    Confidence,
    /// Sample complexity (length, features, etc.)
    Complexity,
    /// Custom scoring function
    Custom,
}

/// Curriculum learning configuration
#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    /// Curriculum strategy
    pub strategy: CurriculumStrategy,
    /// Difficulty metric
    pub difficulty_metric: DifficultyMetric,
    /// Initial difficulty threshold (0.0 = easiest, 1.0 = hardest)
    pub initial_threshold: f32,
    /// Final difficulty threshold
    pub final_threshold: f32,
    /// Number of epochs to reach final threshold
    pub warmup_epochs: usize,
    /// Pacing function (linear, exponential, etc.)
    pub pacing_function: PacingFunction,
    /// Minimum samples per batch
    pub min_samples_per_batch: usize,
}

/// Pacing function for curriculum progression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PacingFunction {
    /// Linear progression
    Linear,
    /// Exponential progression
    Exponential,
    /// Step-wise progression
    Step,
    /// Root progression (slower at start)
    Root,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        CurriculumConfig {
            strategy: CurriculumStrategy::Fixed,
            difficulty_metric: DifficultyMetric::Loss,
            initial_threshold: 0.3,
            final_threshold: 1.0,
            warmup_epochs: 10,
            pacing_function: PacingFunction::Linear,
            min_samples_per_batch: 8,
        }
    }
}

impl CurriculumConfig {
    /// Self-paced learning configuration
    pub fn self_paced(warmup_epochs: usize) -> Self {
        CurriculumConfig {
            strategy: CurriculumStrategy::SelfPaced,
            warmup_epochs,
            ..Default::default()
        }
    }
    
    /// Competence-based configuration
    pub fn competence_based(warmup_epochs: usize) -> Self {
        CurriculumConfig {
            strategy: CurriculumStrategy::CompetenceBased,
            warmup_epochs,
            ..Default::default()
        }
    }
    
    /// Anti-curriculum (hard-to-easy)
    pub fn anti_curriculum() -> Self {
        CurriculumConfig {
            strategy: CurriculumStrategy::AntiCurriculum,
            initial_threshold: 1.0,
            final_threshold: 0.0,
            ..Default::default()
        }
    }
}

/// Sample with difficulty score
#[derive(Debug, Clone)]
pub struct ScoredSample {
    /// Sample index
    pub index: usize,
    /// Difficulty score (0.0 = easy, 1.0 = hard)
    pub difficulty: f32,
    /// Sample loss (if available)
    pub loss: Option<f32>,
    /// Sample metadata
    pub metadata: HashMap<String, f32>,
}

/// Curriculum learning trainer
pub struct CurriculumLearning {
    config: CurriculumConfig,
    /// Current epoch
    current_epoch: usize,
    /// Sample difficulty scores
    sample_scores: Vec<ScoredSample>,
    /// Current difficulty threshold
    current_threshold: f32,
    /// Performance history
    performance_history: Vec<f32>,
}

impl CurriculumLearning {
    /// Create new curriculum learning trainer
    pub fn new(config: CurriculumConfig) -> Self {
        CurriculumLearning {
            current_threshold: config.initial_threshold,
            config,
            current_epoch: 0,
            sample_scores: Vec::new(),
            performance_history: Vec::new(),
        }
    }
    
    /// Initialize sample difficulties
    pub fn initialize_samples(&mut self, num_samples: usize, difficulties: Vec<f32>) {
        self.sample_scores = difficulties.into_iter()
            .enumerate()
            .map(|(i, difficulty)| ScoredSample {
                index: i,
                difficulty,
                loss: None,
                metadata: HashMap::new(),
            })
            .collect();
    }
    
    /// Update difficulty threshold for current epoch
    pub fn update_threshold(&mut self) {
        self.current_threshold = self.compute_threshold(self.current_epoch);
    }
    
    /// Compute threshold based on pacing function
    fn compute_threshold(&self, epoch: usize) -> f32 {
        if epoch >= self.config.warmup_epochs {
            return self.config.final_threshold;
        }
        
        let progress = epoch as f32 / self.config.warmup_epochs as f32;
        let start = self.config.initial_threshold;
        let end = self.config.final_threshold;
        
        match self.config.pacing_function {
            PacingFunction::Linear => {
                start + (end - start) * progress
            }
            PacingFunction::Exponential => {
                start + (end - start) * progress.powi(2)
            }
            PacingFunction::Step => {
                let num_steps = 5;
                let step = (progress * num_steps as f32).floor() / num_steps as f32;
                start + (end - start) * step
            }
            PacingFunction::Root => {
                start + (end - start) * progress.sqrt()
            }
        }
    }
    
    /// Select samples for current curriculum stage
    pub fn select_samples(&self) -> Vec<usize> {
        match self.config.strategy {
            CurriculumStrategy::Fixed => self.select_fixed_curriculum(),
            CurriculumStrategy::SelfPaced => self.select_self_paced(),
            CurriculumStrategy::CompetenceBased => self.select_competence_based(),
            CurriculumStrategy::TeacherStudent => self.select_teacher_student(),
            CurriculumStrategy::AntiCurriculum => self.select_anti_curriculum(),
        }
    }
    
    /// Fixed curriculum: select samples below threshold
    fn select_fixed_curriculum(&self) -> Vec<usize> {
        self.sample_scores.iter()
            .filter(|s| s.difficulty <= self.current_threshold)
            .map(|s| s.index)
            .collect()
    }
    
    /// Self-paced learning: select based on loss
    fn select_self_paced(&self) -> Vec<usize> {
        let mut scored: Vec<_> = self.sample_scores.iter()
            .filter(|s| s.loss.is_some())
            .collect();
        
        scored.sort_by(|a, b| {
            a.loss.unwrap().partial_cmp(&b.loss.unwrap()).unwrap()
        });
        
        let num_select = (scored.len() as f32 * self.current_threshold) as usize;
        let num_select = num_select.max(self.config.min_samples_per_batch);
        
        scored.iter()
            .take(num_select)
            .map(|s| s.index)
            .collect()
    }
    
    /// Competence-based: adjust based on recent performance
    fn select_competence_based(&self) -> Vec<usize> {
        let recent_performance = self.get_recent_performance();
        
        // Adjust threshold based on performance
        let adjusted_threshold = if recent_performance > 0.8 {
            // Doing well, increase difficulty
            (self.current_threshold + 0.1).min(1.0)
        } else if recent_performance < 0.5 {
            // Struggling, decrease difficulty
            (self.current_threshold - 0.1).max(0.0)
        } else {
            self.current_threshold
        };
        
        self.sample_scores.iter()
            .filter(|s| s.difficulty <= adjusted_threshold)
            .map(|s| s.index)
            .collect()
    }
    
    /// Teacher-student curriculum
    fn select_teacher_student(&self) -> Vec<usize> {
        // Similar to fixed but with teacher guidance
        // In practice, teacher would provide difficulty scores
        self.select_fixed_curriculum()
    }
    
    /// Anti-curriculum: hard-to-easy
    fn select_anti_curriculum(&self) -> Vec<usize> {
        self.sample_scores.iter()
            .filter(|s| s.difficulty >= self.current_threshold)
            .map(|s| s.index)
            .collect()
    }
    
    /// Update sample losses after training step
    pub fn update_sample_losses(&mut self, indices: &[usize], losses: &[f32]) {
        for (idx, &loss) in indices.iter().zip(losses.iter()) {
            if let Some(sample) = self.sample_scores.iter_mut().find(|s| s.index == *idx) {
                sample.loss = Some(loss);
            }
        }
    }
    
    /// Update performance history
    pub fn update_performance(&mut self, performance: f32) {
        self.performance_history.push(performance);
    }
    
    /// Get recent performance (average of last N epochs)
    fn get_recent_performance(&self) -> f32 {
        let window = 3;
        let recent = self.performance_history.iter()
            .rev()
            .take(window)
            .copied()
            .collect::<Vec<_>>();
        
        if recent.is_empty() {
            0.5 // Default
        } else {
            recent.iter().sum::<f32>() / recent.len() as f32
        }
    }
    
    /// Advance to next epoch
    pub fn next_epoch(&mut self) {
        self.current_epoch += 1;
        self.update_threshold();
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> CurriculumStats {
        let selected = self.select_samples();
        let avg_difficulty = if !selected.is_empty() {
            selected.iter()
                .filter_map(|&idx| self.sample_scores.get(idx))
                .map(|s| s.difficulty)
                .sum::<f32>() / selected.len() as f32
        } else {
            0.0
        };
        
        CurriculumStats {
            current_epoch: self.current_epoch,
            current_threshold: self.current_threshold,
            num_selected_samples: selected.len(),
            total_samples: self.sample_scores.len(),
            avg_difficulty: avg_difficulty,
            recent_performance: self.get_recent_performance(),
        }
    }
}

/// Curriculum learning statistics
#[derive(Debug, Clone)]
pub struct CurriculumStats {
    pub current_epoch: usize,
    pub current_threshold: f32,
    pub num_selected_samples: usize,
    pub total_samples: usize,
    pub avg_difficulty: f32,
    pub recent_performance: f32,
}

/// Difficulty scorer for samples
pub struct DifficultyScorer {
    metric: DifficultyMetric,
}

impl DifficultyScorer {
    /// Create new difficulty scorer
    pub fn new(metric: DifficultyMetric) -> Self {
        DifficultyScorer { metric }
    }
    
    /// Score sample difficulty
    pub fn score(&self, loss: f32, confidence: f32, complexity: f32) -> f32 {
        match self.metric {
            DifficultyMetric::Loss => {
                // Normalize loss to [0, 1]
                loss.min(10.0) / 10.0
            }
            DifficultyMetric::Confidence => {
                // Low confidence = high difficulty
                1.0 - confidence
            }
            DifficultyMetric::Complexity => {
                complexity
            }
            DifficultyMetric::Custom => {
                // Combine multiple metrics
                (loss * 0.4 + (1.0 - confidence) * 0.3 + complexity * 0.3).min(1.0)
            }
        }
    }
    
    /// Batch score multiple samples
    pub fn score_batch(&self, losses: &[f32], confidences: &[f32], complexities: &[f32]) -> Vec<f32> {
        losses.iter()
            .zip(confidences.iter())
            .zip(complexities.iter())
            .map(|((&loss, &conf), &comp)| self.score(loss, conf, comp))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_curriculum_config() {
        let config = CurriculumConfig::default();
        assert_eq!(config.strategy, CurriculumStrategy::Fixed);
        assert_eq!(config.initial_threshold, 0.3);
        
        let self_paced = CurriculumConfig::self_paced(20);
        assert_eq!(self_paced.strategy, CurriculumStrategy::SelfPaced);
        assert_eq!(self_paced.warmup_epochs, 20);
    }
    
    #[test]
    fn test_curriculum_initialization() {
        let config = CurriculumConfig::default();
        let mut curriculum = CurriculumLearning::new(config);
        
        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        curriculum.initialize_samples(5, difficulties);
        
        assert_eq!(curriculum.sample_scores.len(), 5);
    }
    
    #[test]
    fn test_threshold_computation() {
        let config = CurriculumConfig {
            initial_threshold: 0.2,
            final_threshold: 1.0,
            warmup_epochs: 10,
            pacing_function: PacingFunction::Linear,
            ..Default::default()
        };
        let curriculum = CurriculumLearning::new(config);
        
        let threshold_0 = curriculum.compute_threshold(0);
        let threshold_5 = curriculum.compute_threshold(5);
        let threshold_10 = curriculum.compute_threshold(10);
        
        assert_eq!(threshold_0, 0.2);
        assert!((threshold_5 - 0.6).abs() < 0.01);
        assert_eq!(threshold_10, 1.0);
    }
    
    #[test]
    fn test_fixed_curriculum_selection() {
        let config = CurriculumConfig {
            initial_threshold: 0.5,
            ..Default::default()
        };
        let mut curriculum = CurriculumLearning::new(config);
        
        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        curriculum.initialize_samples(5, difficulties);
        
        let selected = curriculum.select_samples();
        assert_eq!(selected.len(), 3); // 0.1, 0.3, 0.5
    }
    
    #[test]
    fn test_self_paced_selection() {
        let config = CurriculumConfig {
            strategy: CurriculumStrategy::SelfPaced,
            initial_threshold: 0.6,
            ..Default::default()
        };
        let mut curriculum = CurriculumLearning::new(config);
        
        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        curriculum.initialize_samples(5, difficulties);
        
        // Update losses
        curriculum.update_sample_losses(&[0, 1, 2, 3, 4], &[0.5, 0.3, 0.8, 0.2, 0.9]);
        
        let selected = curriculum.select_samples();
        assert!(selected.len() >= 2); // Should select easier samples
    }
    
    #[test]
    fn test_pacing_functions() {
        let linear_config = CurriculumConfig {
            pacing_function: PacingFunction::Linear,
            warmup_epochs: 10,
            ..Default::default()
        };
        let linear_curriculum = CurriculumLearning::new(linear_config);
        
        let exp_config = CurriculumConfig {
            pacing_function: PacingFunction::Exponential,
            warmup_epochs: 10,
            ..Default::default()
        };
        let exp_curriculum = CurriculumLearning::new(exp_config);
        
        let linear_mid = linear_curriculum.compute_threshold(5);
        let exp_mid = exp_curriculum.compute_threshold(5);
        
        // Exponential should be slower at midpoint
        assert!(exp_mid < linear_mid);
    }
    
    #[test]
    fn test_competence_based_adjustment() {
        let config = CurriculumConfig {
            strategy: CurriculumStrategy::CompetenceBased,
            ..Default::default()
        };
        let mut curriculum = CurriculumLearning::new(config);
        
        let difficulties = vec![0.2, 0.4, 0.6, 0.8];
        curriculum.initialize_samples(4, difficulties);
        
        // Simulate good performance
        curriculum.update_performance(0.9);
        curriculum.update_performance(0.85);
        curriculum.update_performance(0.88);
        
        let selected = curriculum.select_samples();
        // Should select more samples due to good performance
        assert!(selected.len() > 0);
    }
    
    #[test]
    fn test_anti_curriculum() {
        let config = CurriculumConfig::anti_curriculum();
        let mut curriculum = CurriculumLearning::new(config);
        
        assert_eq!(curriculum.current_threshold, 1.0);
        
        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        curriculum.initialize_samples(5, difficulties);
        
        // Update threshold to select some samples
        curriculum.current_threshold = 0.8;
        
        let selected = curriculum.select_samples();
        // Should select hardest samples (>= 0.8)
        assert!(selected.contains(&4)); // 0.9 difficulty
        assert!(!selected.contains(&0)); // 0.1 difficulty should not be selected
    }
    
    #[test]
    fn test_difficulty_scorer() {
        let scorer = DifficultyScorer::new(DifficultyMetric::Loss);
        
        let score = scorer.score(2.0, 0.8, 0.5);
        assert!(score >= 0.0 && score <= 1.0);
        
        let batch_scores = scorer.score_batch(
            &[1.0, 2.0, 3.0],
            &[0.9, 0.7, 0.5],
            &[0.3, 0.5, 0.7],
        );
        assert_eq!(batch_scores.len(), 3);
    }
    
    #[test]
    fn test_epoch_progression() {
        let config = CurriculumConfig {
            initial_threshold: 0.2,
            final_threshold: 1.0,
            warmup_epochs: 5,
            ..Default::default()
        };
        let mut curriculum = CurriculumLearning::new(config);
        
        assert_eq!(curriculum.current_epoch, 0);
        assert_eq!(curriculum.current_threshold, 0.2);
        
        curriculum.next_epoch();
        assert_eq!(curriculum.current_epoch, 1);
        assert!(curriculum.current_threshold > 0.2);
        
        for _ in 0..10 {
            curriculum.next_epoch();
        }
        assert_eq!(curriculum.current_threshold, 1.0);
    }
    
    #[test]
    fn test_curriculum_stats() {
        let config = CurriculumConfig::default();
        let mut curriculum = CurriculumLearning::new(config);
        
        let difficulties = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        curriculum.initialize_samples(5, difficulties);
        
        let stats = curriculum.get_stats();
        assert_eq!(stats.total_samples, 5);
        assert!(stats.num_selected_samples > 0);
    }
}
