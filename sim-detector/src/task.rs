use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use ad_core::driver::{ADStatus, ImageMode};
use ad_core::ndarray::{NDArray, NDDataBuffer};

use crate::color_layout::ColorLayout;
use crate::compute::{self, SineState};
use crate::driver::SimDetector;
use crate::params::SimConfigSnapshot;
use crate::roi::crop_roi;
use crate::types::DirtyFlags;

const MIN_DELAY_SECS: f64 = 1e-5;

/// Cached param indices to avoid borrow conflicts with port_base.
struct ParamIndices {
    status: usize,
    status_message: usize,
    acquire: usize,
    num_images_counter: usize,
    array_counter: usize,
}

struct TaskState {
    rng: StdRng,
    raw_buf: NDDataBuffer,
    background_buf: NDDataBuffer,
    ramp_buf: NDDataBuffer,
    peak_buf: NDDataBuffer,
    sine_state: SineState,
    use_background: bool,
}

impl TaskState {
    fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
            raw_buf: NDDataBuffer::zeros(ad_core::ndarray::NDDataType::UInt8, 0),
            background_buf: NDDataBuffer::zeros(ad_core::ndarray::NDDataType::UInt8, 0),
            ramp_buf: NDDataBuffer::zeros(ad_core::ndarray::NDDataType::UInt8, 0),
            peak_buf: NDDataBuffer::zeros(ad_core::ndarray::NDDataType::UInt8, 0),
            sine_state: SineState::new(),
            use_background: false,
        }
    }

    fn apply_dirty(&mut self, dirty: &DirtyFlags, config: &SimConfigSnapshot) {
        if dirty.reallocate_buffers {
            let layout = ColorLayout {
                color_mode: config.color_mode,
                size_x: config.max_size_x,
                size_y: config.max_size_y,
            };
            let n = layout.num_elements();
            self.raw_buf = NDDataBuffer::zeros(config.data_type, n);
            self.background_buf = NDDataBuffer::zeros(config.data_type, n);
            self.ramp_buf = NDDataBuffer::zeros(config.data_type, n);
            self.peak_buf = NDDataBuffer::zeros(config.data_type, n);
            self.use_background = false;
        }

        let needs_rebuild = dirty.rebuild_background || dirty.reallocate_buffers;
        if needs_rebuild {
            self.use_background = config.noise != 0.0 || config.offset != 0.0;
        }
    }

    fn compute_frame(&mut self, config: &SimConfigSnapshot, reset: bool) -> NDArray {
        let layout = ColorLayout {
            color_mode: config.color_mode,
            size_x: config.max_size_x,
            size_y: config.max_size_y,
        };

        compute::compute_frame(
            &mut self.raw_buf,
            &mut self.background_buf,
            &mut self.ramp_buf,
            &mut self.peak_buf,
            &mut self.sine_state,
            &layout,
            config.sim_mode,
            &config.gains,
            &config.peak,
            &config.sine,
            config.offset,
            config.noise,
            self.use_background,
            reset,
            &mut self.rng,
        );

        let min_x = config.min_x.min(config.max_size_x.saturating_sub(1));
        let min_y = config.min_y.min(config.max_size_y.saturating_sub(1));
        let size_x = config.size_x.min(config.max_size_x - min_x).max(1);
        let size_y = config.size_y.min(config.max_size_y - min_y).max(1);

        crop_roi(&self.raw_buf, &layout, min_x, min_y, size_x, size_y)
    }
}

fn wait_for_signal(signal: &Arc<(Mutex<bool>, Condvar)>) {
    let (lock, cvar) = &**signal;
    let mut flag = lock.lock().unwrap();
    while !*flag {
        flag = cvar.wait(flag).unwrap();
    }
    *flag = false;
}

fn wait_for_signal_timeout(signal: &Arc<(Mutex<bool>, Condvar)>, duration: std::time::Duration) -> bool {
    let (lock, cvar) = &**signal;
    let mut flag = lock.lock().unwrap();
    if *flag {
        *flag = false;
        return true;
    }
    let result = cvar.wait_timeout(flag, duration).unwrap();
    flag = result.0;
    if *flag {
        *flag = false;
        true
    } else {
        false
    }
}

pub fn start_acquisition_thread(driver: Arc<parking_lot::Mutex<SimDetector>>) {
    let start_signal;
    let stop_signal;
    let pi;
    {
        let drv = driver.lock();
        start_signal = drv.start_signal.clone();
        stop_signal = drv.stop_signal.clone();
        pi = ParamIndices {
            status: drv.ad.params.status,
            status_message: drv.ad.params.status_message,
            acquire: drv.ad.params.acquire,
            num_images_counter: drv.ad.params.num_images_counter,
            array_counter: drv.ad.params.base.array_counter,
        };
    }

    std::thread::Builder::new()
        .name("SimDetTask".into())
        .spawn(move || {
            let mut task_state = TaskState::new();

            loop {
                wait_for_signal(&start_signal);

                // Initialize counters
                {
                    let mut drv = driver.lock();
                    drv.ad.port_base.set_int32_param(pi.num_images_counter, 0, 0).ok();
                    drv.ad.port_base.set_int32_param(pi.status, 0, ADStatus::Acquire as i32).ok();
                    drv.ad.port_base.call_param_callbacks(0).ok();
                }

                loop {
                    let start_time = Instant::now();

                    let (config, dirty) = {
                        let mut drv = driver.lock();
                        let config = SimConfigSnapshot::read_from(
                            &drv.ad.port_base,
                            &drv.ad.params,
                            &drv.sim_params,
                        );
                        let dirty = drv.dirty.take();
                        match config {
                            Ok(cfg) => (cfg, dirty),
                            Err(_) => break,
                        }
                    };

                    let reset = dirty.any();
                    task_state.apply_dirty(&dirty, &config);

                    let mut frame = task_state.compute_frame(&config, reset);

                    // Exposure time sleep with stop interruption
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let delay = (config.acquire_time - elapsed).max(MIN_DELAY_SECS);
                    if wait_for_signal_timeout(
                        &stop_signal,
                        std::time::Duration::from_secs_f64(delay),
                    ) {
                        let mut drv = driver.lock();
                        drv.ad.port_base.set_int32_param(pi.status, 0, ADStatus::Idle as i32).ok();
                        drv.ad.port_base.set_int32_param(pi.acquire, 0, 0).ok();
                        drv.ad.port_base.call_param_callbacks(0).ok();
                        break;
                    }

                    // Publish frame
                    let should_stop = {
                        let mut drv = driver.lock();

                        let counter = drv.ad.port_base.get_int32_param(pi.array_counter, 0).unwrap_or(0);
                        let num_counter = drv.ad.port_base.get_int32_param(pi.num_images_counter, 0).unwrap_or(0) + 1;
                        drv.ad.port_base.set_int32_param(pi.num_images_counter, 0, num_counter).ok();

                        frame.unique_id = counter + 1;
                        frame.timestamp = ad_core::timestamp::EpicsTimestamp::now();

                        drv.ad.publish_array(Arc::new(frame)).ok();

                        let image_mode = config.image_mode;
                        let num_images = config.num_images;
                        if image_mode == ImageMode::Single
                            || (image_mode == ImageMode::Multiple && num_counter >= num_images)
                        {
                            drv.ad.port_base.set_int32_param(pi.status, 0, ADStatus::Idle as i32).ok();
                            drv.ad.port_base.set_int32_param(pi.acquire, 0, 0).ok();
                            drv.ad.port_base.set_string_param(pi.status_message, 0, "Waiting for acquisition".into()).ok();
                            drv.ad.port_base.call_param_callbacks(0).ok();
                            true
                        } else {
                            false
                        }
                    };

                    if should_stop {
                        break;
                    }

                    // Period delay with stop interruption
                    let total_elapsed = start_time.elapsed().as_secs_f64();
                    let period_delay = config.acquire_period - total_elapsed;
                    if period_delay > 0.0 {
                        if wait_for_signal_timeout(
                            &stop_signal,
                            std::time::Duration::from_secs_f64(period_delay),
                        ) {
                            let mut drv = driver.lock();
                            drv.ad.port_base.set_int32_param(pi.status, 0, ADStatus::Idle as i32).ok();
                            drv.ad.port_base.set_int32_param(pi.acquire, 0, 0).ok();
                            drv.ad.port_base.call_param_callbacks(0).ok();
                            break;
                        }
                    }
                }
            }
        })
        .expect("failed to spawn SimDetTask thread");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::SimDetector;
    use asyn_rs::port::PortDriver;
    use asyn_rs::user::AsynUser;

    #[test]
    fn test_single_mode_auto_stop() {
        let mut det = SimDetector::new("SIM_TEST", 32, 32, 1_000_000).unwrap();
        det.ad.port_base.set_int32_param(det.ad.params.image_mode, 0, ImageMode::Single as i32).unwrap();
        det.ad.port_base.set_float64_param(det.ad.params.acquire_time, 0, 0.001).unwrap();
        det.ad.port_base.set_float64_param(det.ad.params.acquire_period, 0, 0.001).unwrap();

        let driver = Arc::new(parking_lot::Mutex::new(det));
        SimDetector::start_thread(driver.clone());

        {
            let mut drv = driver.lock();
            let reason = drv.ad.params.acquire;
            let mut user = AsynUser::new(reason);
            drv.write_int32(&mut user, 1).unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(200));

        let drv = driver.lock();
        let acquire = drv.ad.port_base.get_int32_param(drv.ad.params.acquire, 0).unwrap();
        assert_eq!(acquire, 0, "acquire should be 0 after Single mode completes");
        let counter = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_counter, 0).unwrap();
        assert!(counter >= 1, "should have produced at least 1 frame");
    }

    #[test]
    fn test_continuous_mode_produces_frames() {
        let mut det = SimDetector::new("SIM_CONT", 16, 16, 1_000_000).unwrap();
        det.ad.port_base.set_int32_param(det.ad.params.image_mode, 0, ImageMode::Continuous as i32).unwrap();
        det.ad.port_base.set_float64_param(det.ad.params.acquire_time, 0, 0.001).unwrap();
        det.ad.port_base.set_float64_param(det.ad.params.acquire_period, 0, 0.002).unwrap();

        let driver = Arc::new(parking_lot::Mutex::new(det));
        SimDetector::start_thread(driver.clone());

        {
            let mut drv = driver.lock();
            let reason = drv.ad.params.acquire;
            let mut user = AsynUser::new(reason);
            drv.write_int32(&mut user, 1).unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(100));

        {
            let mut drv = driver.lock();
            let reason = drv.ad.params.acquire;
            let mut user = AsynUser::new(reason);
            drv.write_int32(&mut user, 0).unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(50));

        let drv = driver.lock();
        let counter = drv.ad.port_base.get_int32_param(drv.ad.params.base.array_counter, 0).unwrap();
        assert!(counter >= 2, "should have produced multiple frames, got {}", counter);
    }

    #[test]
    fn test_stop_during_acquisition() {
        let mut det = SimDetector::new("SIM_STOP", 8, 8, 1_000_000).unwrap();
        det.ad.port_base.set_int32_param(det.ad.params.image_mode, 0, ImageMode::Continuous as i32).unwrap();
        det.ad.port_base.set_float64_param(det.ad.params.acquire_time, 0, 0.5).unwrap();
        det.ad.port_base.set_float64_param(det.ad.params.acquire_period, 0, 1.0).unwrap();

        let driver = Arc::new(parking_lot::Mutex::new(det));
        SimDetector::start_thread(driver.clone());

        {
            let mut drv = driver.lock();
            let reason = drv.ad.params.acquire;
            let mut user = AsynUser::new(reason);
            drv.write_int32(&mut user, 1).unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
        {
            let mut drv = driver.lock();
            let reason = drv.ad.params.acquire;
            let mut user = AsynUser::new(reason);
            drv.write_int32(&mut user, 0).unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(100));

        let drv = driver.lock();
        let acquire = drv.ad.port_base.get_int32_param(drv.ad.params.acquire, 0).unwrap();
        assert_eq!(acquire, 0);
    }
}
