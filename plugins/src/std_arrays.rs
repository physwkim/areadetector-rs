use std::sync::Arc;

use ad_core::ndarray::NDArray;
use ad_core::plugin::base::PluginWorker;
use ad_core::plugin::{DropPolicy, NDPluginDriver};
use parking_lot::Mutex;
use tokio::sync::mpsc;

/// StdArrays plugin: stores the latest NDArray and notifies via I/O Intr.
pub struct StdArraysPlugin {
    name: String,
    worker: PluginWorker,
    latest_data: Arc<Mutex<Option<Arc<NDArray>>>>,
}

impl StdArraysPlugin {
    pub fn new(port_name: &str, io_intr_tx: mpsc::Sender<()>) -> Self {
        let latest_data: Arc<Mutex<Option<Arc<NDArray>>>> = Arc::new(Mutex::new(None));
        let latest = latest_data.clone();

        let worker = PluginWorker::new(
            port_name,
            1,
            DropPolicy::LatestOnly,
            move |array: Arc<NDArray>| {
                *latest.lock() = Some(array);
                let _ = io_intr_tx.try_send(());
            },
        );

        Self {
            name: port_name.to_string(),
            worker,
            latest_data,
        }
    }

    pub fn latest_data(&self) -> Option<Arc<NDArray>> {
        self.latest_data.lock().clone()
    }
}

impl NDPluginDriver for StdArraysPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn push_array(&self, array: Arc<NDArray>) {
        self.worker.push(array, DropPolicy::LatestOnly);
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
    fn test_stores_latest() {
        let (tx, _rx) = mpsc::channel(16);
        let plugin = StdArraysPlugin::new("test:image1", tx);

        assert!(plugin.latest_data().is_none());

        plugin.push_array(make_array(1));
        plugin.push_array(make_array(2));
        plugin.push_array(make_array(3));

        std::thread::sleep(std::time::Duration::from_millis(50));

        let latest = plugin.latest_data().unwrap();
        assert!(latest.unique_id >= 1);
    }

    #[test]
    fn test_triggers_io_intr() {
        let (tx, mut rx) = mpsc::channel(16);
        let plugin = StdArraysPlugin::new("test:image1", tx);

        plugin.push_array(make_array(1));
        std::thread::sleep(std::time::Duration::from_millis(50));

        assert!(rx.try_recv().is_ok());
    }
}
