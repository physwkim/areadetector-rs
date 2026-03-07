use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ad_core::ndarray::NDArray;
use ad_core::plugin::NDPluginDriver;

/// Scatter plugin: distributes arrays round-robin to N output plugins.
pub struct ScatterPlugin {
    name: String,
    outputs: Vec<Arc<dyn NDPluginDriver>>,
    next_output: AtomicUsize,
}

impl ScatterPlugin {
    pub fn new(port_name: &str, outputs: Vec<Arc<dyn NDPluginDriver>>) -> Self {
        Self {
            name: port_name.to_string(),
            outputs,
            next_output: AtomicUsize::new(0),
        }
    }

    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }
}

impl NDPluginDriver for ScatterPlugin {
    fn name(&self) -> &str { &self.name }

    fn push_array(&self, array: Arc<NDArray>) {
        if self.outputs.is_empty() {
            return;
        }
        let idx = self.next_output.fetch_add(1, Ordering::Relaxed) % self.outputs.len();
        self.outputs[idx].push_array(array);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use ad_core::ndarray::{NDDataType, NDDimension};

    struct CountPlugin {
        name: String,
        count: AtomicU32,
    }
    impl CountPlugin {
        fn new(name: &str) -> Self {
            Self { name: name.into(), count: AtomicU32::new(0) }
        }
    }
    impl NDPluginDriver for CountPlugin {
        fn name(&self) -> &str { &self.name }
        fn push_array(&self, _: Arc<NDArray>) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn make_array(id: i32) -> Arc<NDArray> {
        let mut arr = NDArray::new(vec![NDDimension::new(4)], NDDataType::UInt8);
        arr.unique_id = id;
        Arc::new(arr)
    }

    #[test]
    fn test_round_robin() {
        let p1 = Arc::new(CountPlugin::new("out1"));
        let p2 = Arc::new(CountPlugin::new("out2"));
        let p3 = Arc::new(CountPlugin::new("out3"));

        let scatter = ScatterPlugin::new(
            "test:scatter",
            vec![p1.clone(), p2.clone(), p3.clone()],
        );

        for i in 0..9 {
            scatter.push_array(make_array(i));
        }

        assert_eq!(p1.count.load(Ordering::Relaxed), 3);
        assert_eq!(p2.count.load(Ordering::Relaxed), 3);
        assert_eq!(p3.count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_scatter_empty_outputs() {
        let scatter = ScatterPlugin::new("test:scatter", vec![]);
        scatter.push_array(make_array(1)); // should not panic
    }
}
