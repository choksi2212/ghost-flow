//! Data types supported by GhostFlow tensors

use std::fmt;

/// Supported data types for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    /// 16-bit floating point (half precision)
    F16,
    /// Brain floating point (truncated f32)
    BF16,
    /// 32-bit floating point (single precision) - DEFAULT
    #[default]
    F32,
    /// 64-bit floating point (double precision)
    F64,
    /// 8-bit signed integer (for quantization)
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

impl DType {
    /// Size in bytes of this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Check if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8)
    }

    /// Check if this is a signed type
    pub fn is_signed(&self) -> bool {
        !matches!(self, DType::U8 | DType::Bool)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "float16"),
            DType::BF16 => write!(f, "bfloat16"),
            DType::F32 => write!(f, "float32"),
            DType::F64 => write!(f, "float64"),
            DType::I8 => write!(f, "int8"),
            DType::I16 => write!(f, "int16"),
            DType::I32 => write!(f, "int32"),
            DType::I64 => write!(f, "int64"),
            DType::U8 => write!(f, "uint8"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

/// Trait for types that can be stored in tensors
pub trait TensorElement: Copy + Send + Sync + 'static {
    const DTYPE: DType;
    fn zero() -> Self;
    fn one() -> Self;
}

impl TensorElement for f32 {
    const DTYPE: DType = DType::F32;
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl TensorElement for f64 {
    const DTYPE: DType = DType::F64;
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl TensorElement for i32 {
    const DTYPE: DType = DType::I32;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorElement for i64 {
    const DTYPE: DType = DType::I64;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorElement for i8 {
    const DTYPE: DType = DType::I8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorElement for u8 {
    const DTYPE: DType = DType::U8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorElement for bool {
    const DTYPE: DType = DType::Bool;
    fn zero() -> Self { false }
    fn one() -> Self { true }
}
