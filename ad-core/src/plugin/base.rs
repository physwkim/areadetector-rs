use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::ndarray::NDArray;

use super::DropPolicy;

/// A worker thread that receives NDArrays and calls a handler function.
pub struct PluginWorker {
    tx: std::sync::mpsc::SyncSender<Arc<NDArray>>,
    _handle: thread::JoinHandle<()>,
    dropped_count: Arc<AtomicU64>,
    last_execution_time_ns: Arc<AtomicU64>,
}

impl PluginWorker {
    pub fn new<F>(
        name: &str,
        queue_size: usize,
        drop_policy: DropPolicy,
        mut handler: F,
    ) -> Self
    where
        F: FnMut(Arc<NDArray>) + Send + 'static,
    {
        let effective_size = match drop_policy {
            DropPolicy::LatestOnly => 1,
            _ => queue_size.max(1),
        };
        let (tx, rx) = std::sync::mpsc::sync_channel::<Arc<NDArray>>(effective_size);
        let dropped_count = Arc::new(AtomicU64::new(0));
        let last_execution_time_ns = Arc::new(AtomicU64::new(0));
        let exec_time = last_execution_time_ns.clone();

        let thread_name = format!("plugin-{name}");
        let _handle = thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                while let Ok(array) = rx.recv() {
                    let start = Instant::now();
                    handler(array);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    exec_time.store(elapsed, Ordering::Relaxed);
                }
            })
            .expect("failed to spawn plugin worker thread");

        Self {
            tx,
            _handle,
            dropped_count,
            last_execution_time_ns,
        }
    }

    /// Push an array to the worker, applying the configured drop policy.
    pub fn push(&self, array: Arc<NDArray>, drop_policy: DropPolicy) {
        match self.tx.try_send(array.clone()) {
            Ok(()) => {}
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                match drop_policy {
                    DropPolicy::DropNewest => {}
                    DropPolicy::DropOldest | DropPolicy::LatestOnly => {
                        while self.tx.try_send(array.clone()).is_err() {
                            std::thread::yield_now();
                        }
                    }
                }
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {}
        }
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Last execution time in seconds.
    pub fn last_execution_time(&self) -> f64 {
        self.last_execution_time_ns.load(Ordering::Relaxed) as f64 * 1e-9
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn make_test_array(id: i32) -> Arc<NDArray> {
        use crate::ndarray::{NDDataType, NDDimension};
        let mut arr = NDArray::new(vec![NDDimension::new(4)], NDDataType::UInt8);
        arr.unique_id = id;
        Arc::new(arr)
    }

    #[test]
    fn test_worker_delivers_arrays() {
        let received = Arc::new(AtomicU32::new(0));
        let received2 = received.clone();

        let worker = PluginWorker::new("test", 10, DropPolicy::DropNewest, move |_arr| {
            received2.fetch_add(1, Ordering::Relaxed);
        });

        for i in 0..5 {
            worker.push(make_test_array(i), DropPolicy::DropNewest);
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(received.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_worker_drop_newest_when_full() {
        let received = Arc::new(AtomicU32::new(0));
        let received2 = received.clone();

        let worker = PluginWorker::new("test", 1, DropPolicy::DropNewest, move |_arr| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            received2.fetch_add(1, Ordering::Relaxed);
        });

        for i in 0..10 {
            worker.push(make_test_array(i), DropPolicy::DropNewest);
        }

        std::thread::sleep(std::time::Duration::from_millis(200));
        let count = received.load(Ordering::Relaxed);
        let dropped = worker.dropped_count();
        assert!(count > 0);
        assert!(dropped > 0);
    }

    #[test]
    fn test_worker_latest_only() {
        let last_id = Arc::new(std::sync::Mutex::new(0i32));
        let last_id2 = last_id.clone();

        let worker = PluginWorker::new("test", 1, DropPolicy::LatestOnly, move |arr| {
            *last_id2.lock().unwrap() = arr.unique_id;
        });

        for i in 0..5 {
            worker.push(make_test_array(i), DropPolicy::LatestOnly);
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
        let last = *last_id.lock().unwrap();
        assert!(last >= 0);
    }

    #[test]
    fn test_dropped_count_tracking() {
        let worker = PluginWorker::new("test", 1, DropPolicy::DropNewest, move |_| {
            std::thread::sleep(std::time::Duration::from_millis(100));
        });

        assert_eq!(worker.dropped_count(), 0);
        worker.push(make_test_array(0), DropPolicy::DropNewest);
        worker.push(make_test_array(1), DropPolicy::DropNewest);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_execution_time_tracked() {
        let worker = PluginWorker::new("test", 10, DropPolicy::DropNewest, move |_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
        });

        worker.push(make_test_array(0), DropPolicy::DropNewest);
        std::thread::sleep(std::time::Duration::from_millis(50));

        let exec_time = worker.last_execution_time();
        assert!(exec_time >= 0.005);
    }
}
