use std::sync::Arc;

use ad_core::ndarray::{NDArray, NDDataBuffer, NDDataType, NDDimension};
use ad_core::driver::ad_driver::ADDriverBase;
use ad_plugins::stats::StatsPlugin;
use ad_plugins::std_arrays::StdArraysPlugin;

#[test]
fn test_driver_to_stats_pipeline() {
    let (tx, _rx) = tokio::sync::mpsc::channel(16);
    let stats = Arc::new(StatsPlugin::new("STATS1", 10, tx));

    let mut driver = ADDriverBase::new("SIM1", 64, 64, 10_000_000).unwrap();
    driver.register_plugin(stats.clone());

    // Create and publish a test array
    let mut arr = driver.pool.alloc(
        vec![NDDimension::new(64), NDDimension::new(64)],
        NDDataType::UInt8,
    ).unwrap();

    if let NDDataBuffer::U8(ref mut v) = arr.data {
        for i in 0..v.len() {
            v[i] = (i % 256) as u8;
        }
    }

    driver.publish_array(Arc::new(arr)).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(100));

    let result = stats.latest_stats();
    assert_eq!(result.num_elements, 64 * 64);
    assert!(result.max > 0.0);
}

#[test]
fn test_driver_to_std_arrays_pipeline() {
    let (tx, _rx) = tokio::sync::mpsc::channel(16);
    let image = Arc::new(StdArraysPlugin::new("IMAGE1", tx));

    let mut driver = ADDriverBase::new("SIM1", 32, 32, 10_000_000).unwrap();
    driver.register_plugin(image.clone());

    let arr = driver.pool.alloc(
        vec![NDDimension::new(32), NDDimension::new(32)],
        NDDataType::UInt16,
    ).unwrap();

    let id = arr.unique_id;
    driver.publish_array(Arc::new(arr)).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(100));

    let latest = image.latest_data().unwrap();
    assert_eq!(latest.unique_id, id);
}

#[test]
fn test_pool_reuse_in_pipeline() {
    let pool = Arc::new(ad_core::ndarray_pool::NDArrayPool::new(10_000_000));

    // Allocate, use, release, reallocate
    let arr1 = pool.alloc(vec![NDDimension::new(1000)], NDDataType::UInt8).unwrap();
    let bytes_after_first = pool.allocated_bytes();
    pool.release(arr1);
    assert_eq!(pool.num_free_buffers(), 1);

    let arr2 = pool.alloc(vec![NDDimension::new(500)], NDDataType::UInt8).unwrap();
    assert_eq!(pool.num_free_buffers(), 0);
    // Should have reused buffer, allocated_bytes unchanged
    assert_eq!(pool.allocated_bytes(), bytes_after_first);
}
