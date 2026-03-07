use std::sync::{Arc, Condvar, Mutex};

use asyn_rs::error::AsynResult;
use asyn_rs::port::{PortDriver, PortDriverBase};
use asyn_rs::user::AsynUser;

use ad_core::driver::{ADDriver, ADDriverBase, ImageMode};

use crate::params::SimDetectorParams;
use crate::task::start_acquisition_thread;
use crate::types::DirtyFlags;

pub struct SimDetector {
    pub ad: ADDriverBase,
    pub sim_params: SimDetectorParams,
    pub dirty: DirtyFlags,
    pub start_signal: Arc<(Mutex<bool>, Condvar)>,
    pub stop_signal: Arc<(Mutex<bool>, Condvar)>,
}

impl SimDetector {
    pub fn new(port_name: &str, max_size_x: i32, max_size_y: i32, max_memory: usize) -> AsynResult<Self> {
        let mut ad = ADDriverBase::new(port_name, max_size_x, max_size_y, max_memory)?;
        let sim_params = SimDetectorParams::create(&mut ad.port_base)?;

        // Set default values matching C++ constructor
        let base = &mut ad.port_base;
        base.set_string_param(ad.params.base.manufacturer, 0, "Simulated detector".into())?;
        base.set_string_param(ad.params.base.model, 0, "Basic simulator".into())?;

        base.set_int32_param(ad.params.min_x, 0, 0)?;
        base.set_int32_param(ad.params.min_y, 0, 0)?;
        base.set_float64_param(ad.params.acquire_time, 0, 0.001)?;
        base.set_float64_param(ad.params.acquire_period, 0, 0.005)?;
        base.set_int32_param(ad.params.image_mode, 0, ImageMode::Continuous as i32)?;
        base.set_int32_param(ad.params.num_images, 0, 100)?;

        base.set_float64_param(sim_params.gain, 0, 1.0)?;
        base.set_float64_param(sim_params.gain_x, 0, 1.0)?;
        base.set_float64_param(sim_params.gain_y, 0, 1.0)?;
        base.set_float64_param(sim_params.gain_red, 0, 1.0)?;
        base.set_float64_param(sim_params.gain_green, 0, 1.0)?;
        base.set_float64_param(sim_params.gain_blue, 0, 1.0)?;
        base.set_float64_param(sim_params.offset, 0, 0.0)?;
        base.set_float64_param(sim_params.noise, 0, 0.0)?;
        base.set_int32_param(sim_params.sim_mode, 0, 0)?;
        base.set_int32_param(sim_params.peak_start_x, 0, 1)?;
        base.set_int32_param(sim_params.peak_start_y, 0, 1)?;
        base.set_int32_param(sim_params.peak_width_x, 0, 10)?;
        base.set_int32_param(sim_params.peak_width_y, 0, 20)?;
        base.set_int32_param(sim_params.peak_num_x, 0, 1)?;
        base.set_int32_param(sim_params.peak_num_y, 0, 1)?;
        base.set_int32_param(sim_params.peak_step_x, 0, 1)?;
        base.set_int32_param(sim_params.peak_step_y, 0, 1)?;
        base.set_float64_param(sim_params.peak_height_variation, 0, 0.0)?;
        base.set_int32_param(sim_params.x_sine_operation, 0, 0)?;
        base.set_int32_param(sim_params.y_sine_operation, 0, 0)?;
        base.set_float64_param(sim_params.x_sine1_amplitude, 0, 0.0)?;
        base.set_float64_param(sim_params.x_sine1_frequency, 0, 0.0)?;
        base.set_float64_param(sim_params.x_sine1_phase, 0, 0.0)?;
        base.set_float64_param(sim_params.x_sine2_amplitude, 0, 0.0)?;
        base.set_float64_param(sim_params.x_sine2_frequency, 0, 0.0)?;
        base.set_float64_param(sim_params.x_sine2_phase, 0, 0.0)?;
        base.set_float64_param(sim_params.y_sine1_amplitude, 0, 0.0)?;
        base.set_float64_param(sim_params.y_sine1_frequency, 0, 0.0)?;
        base.set_float64_param(sim_params.y_sine1_phase, 0, 0.0)?;
        base.set_float64_param(sim_params.y_sine2_amplitude, 0, 0.0)?;
        base.set_float64_param(sim_params.y_sine2_frequency, 0, 0.0)?;
        base.set_float64_param(sim_params.y_sine2_phase, 0, 0.0)?;

        // Reset image flag - triggers initial buffer allocation
        base.set_int32_param(sim_params.reset_image, 0, 1)?;

        let mut dirty = DirtyFlags::default();
        dirty.set_all();

        Ok(Self {
            ad,
            sim_params,
            dirty,
            start_signal: Arc::new((Mutex::new(false), Condvar::new())),
            stop_signal: Arc::new((Mutex::new(false), Condvar::new())),
        })
    }

    /// Start the acquisition background thread.
    /// Must be called after wrapping in `Arc<parking_lot::Mutex<..>>`.
    pub fn start_thread(driver: Arc<parking_lot::Mutex<Self>>) {
        start_acquisition_thread(driver);
    }

    fn signal_start(&self) {
        let (lock, cvar) = &*self.start_signal;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one();
    }

    fn signal_stop(&self) {
        let (lock, cvar) = &*self.stop_signal;
        let mut stopped = lock.lock().unwrap();
        *stopped = true;
        cvar.notify_one();
    }

    fn set_dirty_for_int32(&mut self, reason: usize) {
        if reason == self.ad.params.base.data_type || reason == self.ad.params.base.color_mode {
            self.dirty.reallocate_buffers = true;
            self.dirty.rebuild_background = true;
            self.dirty.reset_ramp = true;
            self.dirty.reset_peak_cache = true;
            self.dirty.reset_sine_state = true;
        } else if reason == self.sim_params.sim_mode {
            self.dirty.reset_ramp = true;
            self.dirty.reset_peak_cache = true;
            self.dirty.reset_sine_state = true;
            self.dirty.rebuild_background = true;
        } else if reason == self.sim_params.peak_start_x
            || reason == self.sim_params.peak_start_y
            || reason == self.sim_params.peak_width_x
            || reason == self.sim_params.peak_width_y
            || reason == self.sim_params.peak_num_x
            || reason == self.sim_params.peak_num_y
            || reason == self.sim_params.peak_step_x
            || reason == self.sim_params.peak_step_y
        {
            self.dirty.reset_peak_cache = true;
        } else if reason == self.sim_params.x_sine_operation
            || reason == self.sim_params.y_sine_operation
        {
            self.dirty.reset_sine_state = true;
        }
    }

    fn set_dirty_for_float64(&mut self, reason: usize) {
        if reason == self.sim_params.gain {
            self.dirty.reset_ramp = true;
            self.dirty.reset_peak_cache = true;
        } else if reason == self.sim_params.gain_x || reason == self.sim_params.gain_y {
            self.dirty.reset_ramp = true;
        } else if reason == self.sim_params.gain_red
            || reason == self.sim_params.gain_green
            || reason == self.sim_params.gain_blue
        {
            self.dirty.reset_ramp = true;
        } else if reason == self.sim_params.offset || reason == self.sim_params.noise {
            self.dirty.rebuild_background = true;
        } else if reason == self.sim_params.peak_height_variation {
            self.dirty.reset_peak_cache = true;
        } else if reason == self.sim_params.x_sine1_amplitude
            || reason == self.sim_params.x_sine1_frequency
            || reason == self.sim_params.x_sine1_phase
            || reason == self.sim_params.x_sine2_amplitude
            || reason == self.sim_params.x_sine2_frequency
            || reason == self.sim_params.x_sine2_phase
            || reason == self.sim_params.y_sine1_amplitude
            || reason == self.sim_params.y_sine1_frequency
            || reason == self.sim_params.y_sine1_phase
            || reason == self.sim_params.y_sine2_amplitude
            || reason == self.sim_params.y_sine2_frequency
            || reason == self.sim_params.y_sine2_phase
        {
            self.dirty.reset_sine_state = true;
        }
    }
}

impl PortDriver for SimDetector {
    fn base(&self) -> &PortDriverBase {
        &self.ad.port_base
    }

    fn base_mut(&mut self) -> &mut PortDriverBase {
        &mut self.ad.port_base
    }

    fn write_int32(&mut self, user: &mut AsynUser, value: i32) -> AsynResult<()> {
        let reason = user.reason;
        let acquire_idx = self.ad.params.acquire;
        let status_msg_idx = self.ad.params.status_message;

        if reason == acquire_idx {
            let acquiring = self.ad.port_base.get_int32_param(acquire_idx, 0).unwrap_or(0);
            if value != 0 && acquiring == 0 {
                self.ad.port_base.set_string_param(status_msg_idx, 0, "Acquiring data".into())?;
                self.ad.port_base.set_int32_param(acquire_idx, 0, value)?;
                self.signal_start();
            } else if value == 0 && acquiring != 0 {
                self.ad.port_base.set_string_param(status_msg_idx, 0, "Acquisition stopped".into())?;
                self.ad.port_base.set_int32_param(acquire_idx, 0, value)?;
                self.signal_stop();
            } else {
                self.ad.port_base.set_int32_param(acquire_idx, 0, value)?;
            }
        } else {
            self.ad.port_base.params.set_int32(reason, user.addr, value)?;
            self.set_dirty_for_int32(reason);
        }

        self.ad.port_base.call_param_callbacks(0)?;
        Ok(())
    }

    fn write_float64(&mut self, user: &mut AsynUser, value: f64) -> AsynResult<()> {
        let reason = user.reason;
        self.ad.port_base.params.set_float64(reason, user.addr, value)?;
        self.set_dirty_for_float64(reason);
        self.ad.port_base.call_param_callbacks(0)?;
        Ok(())
    }
}

impl ADDriver for SimDetector {
    fn ad_base(&self) -> &ADDriverBase {
        &self.ad
    }

    fn ad_base_mut(&mut self) -> &mut ADDriverBase {
        &mut self.ad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_default_values() {
        let det = SimDetector::new("SIM1", 256, 256, 10_000_000).unwrap();
        let base = &det.ad.port_base;
        assert_eq!(base.get_int32_param(det.ad.params.max_size_x, 0).unwrap(), 256);
        assert_eq!(base.get_int32_param(det.ad.params.max_size_y, 0).unwrap(), 256);
        assert_eq!(base.get_float64_param(det.sim_params.gain_x, 0).unwrap(), 1.0);
        assert_eq!(base.get_float64_param(det.sim_params.gain_y, 0).unwrap(), 1.0);
        assert_eq!(base.get_int32_param(det.sim_params.peak_width_x, 0).unwrap(), 10);
        assert_eq!(base.get_int32_param(det.sim_params.peak_width_y, 0).unwrap(), 20);
        assert_eq!(
            base.get_float64_param(det.ad.params.acquire_time, 0).unwrap(),
            0.001
        );
    }

    #[test]
    fn test_dirty_flags_on_data_type_change() {
        let mut det = SimDetector::new("SIM1", 64, 64, 1_000_000).unwrap();
        det.dirty.clear();
        let reason = det.ad.params.base.data_type;
        let mut user = AsynUser::new(reason);
        det.write_int32(&mut user, 3).unwrap();
        assert!(det.dirty.reallocate_buffers);
    }

    #[test]
    fn test_dirty_flags_on_gain_change() {
        let mut det = SimDetector::new("SIM1", 64, 64, 1_000_000).unwrap();
        det.dirty.clear();
        let reason = det.sim_params.gain;
        let mut user = AsynUser::new(reason);
        det.write_float64(&mut user, 2.0).unwrap();
        assert!(det.dirty.reset_ramp);
        assert!(det.dirty.reset_peak_cache);
        assert!(!det.dirty.reallocate_buffers);
    }

    #[test]
    fn test_dirty_flags_on_offset_change() {
        let mut det = SimDetector::new("SIM1", 64, 64, 1_000_000).unwrap();
        det.dirty.clear();
        let reason = det.sim_params.offset;
        let mut user = AsynUser::new(reason);
        det.write_float64(&mut user, 5.0).unwrap();
        assert!(det.dirty.rebuild_background);
        assert!(!det.dirty.reset_ramp);
    }

    #[test]
    fn test_dirty_flags_on_sine_param_change() {
        let mut det = SimDetector::new("SIM1", 64, 64, 1_000_000).unwrap();
        det.dirty.clear();
        let reason = det.sim_params.x_sine1_amplitude;
        let mut user = AsynUser::new(reason);
        det.write_float64(&mut user, 100.0).unwrap();
        assert!(det.dirty.reset_sine_state);
        assert!(!det.dirty.reset_ramp);
    }

    #[test]
    fn test_dirty_flags_on_mode_change() {
        let mut det = SimDetector::new("SIM1", 64, 64, 1_000_000).unwrap();
        det.dirty.clear();
        let reason = det.sim_params.sim_mode;
        let mut user = AsynUser::new(reason);
        det.write_int32(&mut user, 2).unwrap();
        assert!(det.dirty.reset_ramp);
        assert!(det.dirty.reset_peak_cache);
        assert!(det.dirty.reset_sine_state);
        assert!(det.dirty.rebuild_background);
    }
}
