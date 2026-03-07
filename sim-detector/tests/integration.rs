use std::sync::Arc;

use ad_core::driver::{ColorMode, ImageMode};
use asyn_rs::port::PortDriver;
use asyn_rs::user::AsynUser;
use sim_detector::SimDetector;

#[test]
fn test_single_mode_one_frame() {
    let mut det = SimDetector::new("INT_SINGLE", 64, 64, 10_000_000).unwrap();
    det.ad.port_base.set_int32_param(det.ad.params.image_mode, 0, ImageMode::Single as i32).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_time, 0, 0.001).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_period, 0, 0.001).unwrap();

    let driver = Arc::new(parking_lot::Mutex::new(det));
    SimDetector::start_thread(driver.clone());

    // Start
    {
        let mut drv = driver.lock();
        let reason = drv.ad.params.acquire;
        let mut user = AsynUser::new(reason);
        drv.write_int32(&mut user, 1).unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(500));

    let drv = driver.lock();
    let acquire = drv.ad.port_base.get_int32_param(drv.ad.params.acquire, 0).unwrap();
    assert_eq!(acquire, 0);
    let counter = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_counter, 0).unwrap();
    assert_eq!(counter, 1);
}

#[test]
fn test_mode_switch_during_continuous() {
    let mut det = SimDetector::new("INT_SWITCH", 32, 32, 10_000_000).unwrap();
    det.ad.port_base.set_int32_param(det.ad.params.image_mode, 0, ImageMode::Continuous as i32).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_time, 0, 0.001).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_period, 0, 0.002).unwrap();
    // Start with LinearRamp
    det.ad.port_base.set_int32_param(det.sim_params.sim_mode, 0, 0).unwrap();

    let driver = Arc::new(parking_lot::Mutex::new(det));
    SimDetector::start_thread(driver.clone());

    // Start
    {
        let mut drv = driver.lock();
        let reason = drv.ad.params.acquire;
        let mut user = AsynUser::new(reason);
        drv.write_int32(&mut user, 1).unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(30));

    // Switch to Peaks mode
    {
        let mut drv = driver.lock();
        let reason = drv.sim_params.sim_mode;
        let mut user = AsynUser::new(reason);
        drv.write_int32(&mut user, 1).unwrap(); // Peaks
    }

    std::thread::sleep(std::time::Duration::from_millis(30));

    // Stop
    {
        let mut drv = driver.lock();
        let reason = drv.ad.params.acquire;
        let mut user = AsynUser::new(reason);
        drv.write_int32(&mut user, 0).unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(50));

    let drv = driver.lock();
    let counter = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_counter, 0).unwrap();
    assert!(counter >= 2, "should have produced frames across mode switch, got {}", counter);
}

#[test]
fn test_rgb1_mode_acquisition() {
    let mut det = SimDetector::new("INT_RGB", 16, 16, 10_000_000).unwrap();
    det.ad.port_base.set_int32_param(det.ad.params.image_mode, 0, ImageMode::Single as i32).unwrap();
    det.ad.port_base.set_int32_param(det.ad.params.base.color_mode, 0, ColorMode::RGB1 as i32).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_time, 0, 0.001).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_period, 0, 0.001).unwrap();
    // Need to set dirty for color mode change
    det.dirty.reallocate_buffers = true;

    let driver = Arc::new(parking_lot::Mutex::new(det));
    SimDetector::start_thread(driver.clone());

    {
        let mut drv = driver.lock();
        let reason = drv.ad.params.acquire;
        let mut user = AsynUser::new(reason);
        drv.write_int32(&mut user, 1).unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(500));

    let drv = driver.lock();
    let counter = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_counter, 0).unwrap();
    assert_eq!(counter, 1);
    // RGB1: array_size_z should be 3
    let z = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_size_z, 0).unwrap();
    assert_eq!(z, 3);
}
