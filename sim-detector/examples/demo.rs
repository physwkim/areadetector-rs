use std::sync::Arc;

use ad_core::driver::ImageMode;
use asyn_rs::port::PortDriver;
use asyn_rs::user::AsynUser;
use sim_detector::SimDetector;

fn main() {
    println!("SimDetector Demo");
    println!("================");

    let mut det = SimDetector::new("SIM_DEMO", 256, 256, 50_000_000).unwrap();

    // Configure: UInt16, Continuous, LinearRamp
    det.ad.port_base.set_int32_param(det.ad.params.base.data_type, 0, 3).unwrap(); // UInt16
    det.ad.port_base.set_int32_param(det.ad.params.image_mode, 0, ImageMode::Continuous as i32).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_time, 0, 0.01).unwrap();
    det.ad.port_base.set_float64_param(det.ad.params.acquire_period, 0, 0.05).unwrap();
    det.dirty.reallocate_buffers = true;

    let driver = Arc::new(parking_lot::Mutex::new(det));
    SimDetector::start_thread(driver.clone());

    // Start acquisition
    {
        let mut drv = driver.lock();
        let reason = drv.ad.params.acquire;
        let mut user = AsynUser::new(reason);
        drv.write_int32(&mut user, 1).unwrap();
    }

    println!("Acquiring 10 frames...");

    // Wait and print stats
    for i in 0..10 {
        std::thread::sleep(std::time::Duration::from_millis(60));
        let drv = driver.lock();
        let counter = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_counter, 0).unwrap();
        let size_x = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_size_x, 0).unwrap();
        let size_y = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_size_y, 0).unwrap();
        let total = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_size, 0).unwrap();
        println!(
            "  Frame {}: counter={}, size={}x{}, bytes={}",
            i + 1, counter, size_x, size_y, total
        );
    }

    // Stop
    {
        let mut drv = driver.lock();
        let reason = drv.ad.params.acquire;
        let mut user = AsynUser::new(reason);
        drv.write_int32(&mut user, 0).unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let drv = driver.lock();
    let final_count = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_counter, 0).unwrap();
    println!("\nAcquisition stopped. Total frames: {}", final_count);
}
