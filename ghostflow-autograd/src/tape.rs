//! Gradient tape for recording operations

use std::cell::RefCell;

thread_local! {
    static GRAD_TAPE: RefCell<Option<GradTape>> = const { RefCell::new(None) };
}

/// Gradient tape that records operations for backward pass
#[derive(Debug, Default)]
pub struct GradTape {
    operations: Vec<RecordedOp>,
    enabled: bool,
}

/// Type alias for backward function
type BackwardFn = Box<dyn Fn(&[f32], &[Vec<f32>]) -> Vec<Vec<f32>> + Send + Sync>;

/// A recorded operation in the tape
pub struct RecordedOp {
    pub op_name: &'static str,
    pub input_ids: Vec<usize>,
    pub output_id: usize,
    pub backward_fn: BackwardFn,
}

impl std::fmt::Debug for RecordedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordedOp")
            .field("op_name", &self.op_name)
            .field("input_ids", &self.input_ids)
            .field("output_id", &self.output_id)
            .field("backward_fn", &"<closure>")
            .finish()
    }
}

impl GradTape {
    /// Create a new gradient tape
    pub fn new() -> Self {
        GradTape {
            operations: Vec::new(),
            enabled: true,
        }
    }

    /// Check if tape is recording
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable recording
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable recording
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Record an operation
    pub fn record(&mut self, op: RecordedOp) {
        if self.enabled {
            self.operations.push(op);
        }
    }

    /// Get recorded operations
    pub fn operations(&self) -> &[RecordedOp] {
        &self.operations
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.operations.clear();
    }
}

/// Context manager for gradient tape
pub struct GradTapeContext;

impl GradTapeContext {
    /// Start recording gradients
    pub fn new() -> Self {
        GRAD_TAPE.with(|tape| {
            *tape.borrow_mut() = Some(GradTape::new());
        });
        GradTapeContext
    }
}

impl Default for GradTapeContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GradTapeContext {
    fn drop(&mut self) {
        GRAD_TAPE.with(|tape| {
            *tape.borrow_mut() = None;
        });
    }
}

/// Check if we're currently recording
pub fn is_recording() -> bool {
    GRAD_TAPE.with(|tape| {
        tape.borrow().as_ref().is_some_and(|t| t.is_enabled())
    })
}

/// Record an operation to the current tape
pub fn record_op(op: RecordedOp) {
    GRAD_TAPE.with(|tape| {
        if let Some(ref mut t) = *tape.borrow_mut() {
            t.record(op);
        }
    });
}
