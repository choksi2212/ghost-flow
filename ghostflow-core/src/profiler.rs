//! Profiling tools for performance analysis
//!
//! This module provides utilities for profiling tensor operations and model execution.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

/// Operation profiler for tracking performance
pub struct Profiler {
    records: Arc<Mutex<Vec<ProfileRecord>>>,
    enabled: bool,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
            enabled: true,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Start profiling an operation
    pub fn start(&self, name: &str) -> ProfileScope {
        if !self.enabled {
            return ProfileScope::disabled();
        }

        ProfileScope {
            name: name.to_string(),
            start: Instant::now(),
            records: Some(Arc::clone(&self.records)),
        }
    }

    /// Get all profile records
    pub fn records(&self) -> Vec<ProfileRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Get summary statistics
    pub fn summary(&self) -> ProfileSummary {
        let records = self.records.lock().unwrap();
        let mut op_stats: HashMap<String, OpStats> = HashMap::new();

        for record in records.iter() {
            let stats = op_stats.entry(record.name.clone()).or_insert_with(OpStats::default);
            stats.count += 1;
            stats.total_time += record.duration;
            stats.min_time = stats.min_time.min(record.duration);
            stats.max_time = stats.max_time.max(record.duration);
        }

        // Calculate averages
        for stats in op_stats.values_mut() {
            stats.avg_time = stats.total_time / stats.count as u32;
        }

        ProfileSummary {
            total_operations: records.len(),
            op_stats,
        }
    }

    /// Clear all records
    pub fn clear(&self) {
        self.records.lock().unwrap().clear();
    }

    /// Print summary to stdout
    pub fn print_summary(&self) {
        let summary = self.summary();
        println!("\n=== Profiler Summary ===");
        println!("Total operations: {}", summary.total_operations);
        println!("\nOperation Statistics:");
        println!("{:<30} {:>10} {:>15} {:>15} {:>15}", 
                 "Operation", "Count", "Total (ms)", "Avg (ms)", "Max (ms)");
        println!("{}", "-".repeat(85));

        let mut ops: Vec<_> = summary.op_stats.iter().collect();
        ops.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        for (name, stats) in ops {
            println!("{:<30} {:>10} {:>15.3} {:>15.3} {:>15.3}",
                     name,
                     stats.count,
                     stats.total_time.as_secs_f64() * 1000.0,
                     stats.avg_time.as_secs_f64() * 1000.0,
                     stats.max_time.as_secs_f64() * 1000.0);
        }
        println!();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile scope for RAII-style profiling
pub struct ProfileScope {
    name: String,
    start: Instant,
    records: Option<Arc<Mutex<Vec<ProfileRecord>>>>,
}

impl ProfileScope {
    fn disabled() -> Self {
        Self {
            name: String::new(),
            start: Instant::now(),
            records: None,
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        if let Some(records) = &self.records {
            let duration = self.start.elapsed();
            records.lock().unwrap().push(ProfileRecord {
                name: self.name.clone(),
                duration,
            });
        }
    }
}

/// Single profile record
#[derive(Debug, Clone)]
pub struct ProfileRecord {
    pub name: String,
    pub duration: Duration,
}

/// Statistics for a single operation type
#[derive(Debug, Clone)]
pub struct OpStats {
    pub count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl Default for OpStats {
    fn default() -> Self {
        Self {
            count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
        }
    }
}

/// Profile summary
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub total_operations: usize,
    pub op_stats: HashMap<String, OpStats>,
}

/// Benchmark utility for measuring performance
pub struct Benchmark {
    name: String,
    warmup_iterations: usize,
    iterations: usize,
}

impl Benchmark {
    /// Create a new benchmark
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            warmup_iterations: 3,
            iterations: 10,
        }
    }

    /// Set warmup iterations
    pub fn warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set benchmark iterations
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Run the benchmark
    pub fn run<F>(&self, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }

        // Benchmark
        let mut times = Vec::with_capacity(self.iterations);
        for _ in 0..self.iterations {
            let start = Instant::now();
            f();
            times.push(start.elapsed());
        }

        BenchmarkResult::new(&self.name, times)
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub median_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_dev: f64,
}

impl BenchmarkResult {
    fn new(name: &str, mut times: Vec<Duration>) -> Self {
        times.sort();
        
        let iterations = times.len();
        let total_time: Duration = times.iter().sum();
        let mean_time = total_time / iterations as u32;
        let median_time = times[iterations / 2];
        let min_time = times[0];
        let max_time = times[iterations - 1];

        // Calculate standard deviation
        let mean_secs = mean_time.as_secs_f64();
        let variance: f64 = times.iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum::<f64>() / iterations as f64;
        let std_dev = variance.sqrt();

        Self {
            name: name.to_string(),
            iterations,
            total_time,
            mean_time,
            median_time,
            min_time,
            max_time,
            std_dev,
        }
    }

    /// Print the benchmark result
    pub fn print(&self) {
        println!("\n=== Benchmark: {} ===", self.name);
        println!("Iterations: {}", self.iterations);
        println!("Total time: {:.3} ms", self.total_time.as_secs_f64() * 1000.0);
        println!("Mean time:   {:.3} ms", self.mean_time.as_secs_f64() * 1000.0);
        println!("Median time: {:.3} ms", self.median_time.as_secs_f64() * 1000.0);
        println!("Min time:    {:.3} ms", self.min_time.as_secs_f64() * 1000.0);
        println!("Max time:    {:.3} ms", self.max_time.as_secs_f64() * 1000.0);
        println!("Std dev:     {:.3} ms", self.std_dev * 1000.0);
        println!();
    }
}

/// Global profiler instance
static mut GLOBAL_PROFILER: Option<Profiler> = None;

/// Get the global profiler
pub fn global_profiler() -> &'static Profiler {
    unsafe {
        GLOBAL_PROFILER.get_or_insert_with(Profiler::new)
    }
}

/// Profile a code block
#[macro_export]
macro_rules! profile {
    ($name:expr, $code:block) => {{
        let _scope = $crate::profiler::global_profiler().start($name);
        $code
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler() {
        let profiler = Profiler::new();
        
        {
            let _scope = profiler.start("test_op");
            thread::sleep(Duration::from_millis(10));
        }
        
        let records = profiler.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "test_op");
        assert!(records[0].duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_summary() {
        let profiler = Profiler::new();
        
        for _ in 0..5 {
            let _scope = profiler.start("op1");
            thread::sleep(Duration::from_millis(1));
        }
        
        for _ in 0..3 {
            let _scope = profiler.start("op2");
            thread::sleep(Duration::from_millis(1));
        }
        
        let summary = profiler.summary();
        assert_eq!(summary.total_operations, 8);
        assert_eq!(summary.op_stats.len(), 2);
        assert_eq!(summary.op_stats["op1"].count, 5);
        assert_eq!(summary.op_stats["op2"].count, 3);
    }

    #[test]
    fn test_benchmark() {
        let result = Benchmark::new("test")
            .warmup(2)
            .iterations(5)
            .run(|| {
                thread::sleep(Duration::from_millis(1));
            });
        
        assert_eq!(result.iterations, 5);
        assert!(result.mean_time >= Duration::from_millis(1));
    }

    #[test]
    fn test_disabled_profiler() {
        let mut profiler = Profiler::new();
        profiler.disable();
        
        {
            let _scope = profiler.start("test");
            thread::sleep(Duration::from_millis(10));
        }
        
        let records = profiler.records();
        assert_eq!(records.len(), 0);
    }
}
