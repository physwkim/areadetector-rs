pub mod base;
pub mod file_base;

use std::sync::Arc;

use crate::ndarray::NDArray;

/// Policy for handling backpressure when the plugin queue is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    DropOldest,
    DropNewest,
    LatestOnly,
}

/// Trait for NDArray plugin receivers.
pub trait NDPluginDriver: Send + Sync {
    fn name(&self) -> &str;
    fn push_array(&self, array: Arc<NDArray>);
}
