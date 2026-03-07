use std::sync::Arc;

use ad_core::ndarray::NDArray;
use ad_core::plugin::{DropPolicy, NDPluginDriver};
use ad_core::plugin::base::PluginWorker;
use parking_lot::Mutex;

/// Gather plugin: subscribes to multiple source ports and merges arrays into a single stream.
pub struct GatherPlugin {
    name: String,
    worker: PluginWorker,
    latest_output: Arc<Mutex<Option<Arc<NDArray>>>>,
    count: Arc<std::sync::atomic::AtomicU64>,
}

impl GatherPlugin {
    pub fn new(port_name: &str, queue_size: usize) -> Self {
        let latest_output: Arc<Mutex<Option<Arc<NDArray>>>> = Arc::new(Mutex::new(None));
        let latest = latest_output.clone();
        let count = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let count2 = count.clone();

        let worker = PluginWorker::new(
            port_name,
            queue_size,
            DropPolicy::DropNewest,
            move |array: Arc<NDArray>| {
                count2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                *latest.lock() = Some(array);
            },
        );

        Self {
            name: port_name.to_string(),
            worker,
            latest_output,
            count,
        }
    }

    pub fn latest_output(&self) -> Option<Arc<NDArray>> {
        self.latest_output.lock().clone()
    }

    pub fn total_received(&self) -> u64 {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl NDPluginDriver for GatherPlugin {
    fn name(&self) -> &str { &self.name }
    fn push_array(&self, array: Arc<NDArray>) {
        self.worker.push(array, DropPolicy::DropNewest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_core::ndarray::{NDDataType, NDDimension};

    fn make_array(id: i32) -> Arc<NDArray> {
        let mut arr = NDArray::new(vec![NDDimension::new(4)], NDDataType::UInt8);
        arr.unique_id = id;
        Arc::new(arr)
    }

    #[test]
    fn test_gather_from_multiple_sources() {
        let gather = Arc::new(GatherPlugin::new("test:gather", 20));

        // Simulate two sources pushing to the same gather plugin
        gather.push_array(make_array(1));
        gather.push_array(make_array(2));
        gather.push_array(make_array(3));

        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(gather.total_received(), 3);
    }
}
