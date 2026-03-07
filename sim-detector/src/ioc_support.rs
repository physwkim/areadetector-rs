use std::collections::HashMap;
use std::sync::Arc;

use asyn_rs::adapter::AsynDeviceSupport;
use asyn_rs::port_handle::PortHandle;
use epics_base_rs::error::CaResult;
use epics_base_rs::server::device_support::{DeviceSupport, WriteCompletion};
use epics_base_rs::server::record::{Record, ScanType};
use epics_base_rs::types::EpicsValue;

use ad_core::params::ADBaseParams;
use crate::params::SimDetectorParams;
use crate::SimDetector;

#[derive(Clone, Copy)]
pub enum ParamType {
    Int32,
    Float64,
    OctetString,
}

#[derive(Clone)]
pub struct ParamInfo {
    pub param_index: usize,
    pub param_type: ParamType,
    pub drv_info: String,
}

impl ParamInfo {
    pub fn int32(index: usize, drv_info: &str) -> Self {
        Self { param_index: index, param_type: ParamType::Int32, drv_info: drv_info.to_string() }
    }
    pub fn float64(index: usize, drv_info: &str) -> Self {
        Self { param_index: index, param_type: ParamType::Float64, drv_info: drv_info.to_string() }
    }
    pub fn string(index: usize, drv_info: &str) -> Self {
        Self { param_index: index, param_type: ParamType::OctetString, drv_info: drv_info.to_string() }
    }
}

/// Registry mapping record name suffixes to sim-detector parameters.
pub type ParamRegistry = HashMap<String, ParamInfo>;

/// Build the parameter registry from a SimDetector instance.
pub fn build_param_registry(det: &SimDetector) -> ParamRegistry {
    build_param_registry_from_params(&det.ad.params, &det.sim_params)
}

/// Build the parameter registry from param indices (for use after driver is consumed).
pub fn build_param_registry_from_params(ad: &ADBaseParams, sim: &SimDetectorParams) -> ParamRegistry {
    let mut map = HashMap::new();
    let base = &ad.base;

    // Float64 writable params (ao records)
    map.insert("Gain".into(), ParamInfo::float64(sim.gain, "SIM_GAIN"));
    map.insert("GainX".into(), ParamInfo::float64(sim.gain_x, "SIM_GAIN_X"));
    map.insert("GainY".into(), ParamInfo::float64(sim.gain_y, "SIM_GAIN_Y"));
    map.insert("GainRed".into(), ParamInfo::float64(sim.gain_red, "SIM_GAIN_RED"));
    map.insert("GainGreen".into(), ParamInfo::float64(sim.gain_green, "SIM_GAIN_GREEN"));
    map.insert("GainBlue".into(), ParamInfo::float64(sim.gain_blue, "SIM_GAIN_BLUE"));
    map.insert("Offset".into(), ParamInfo::float64(sim.offset, "SIM_OFFSET"));
    map.insert("Noise".into(), ParamInfo::float64(sim.noise, "SIM_NOISE"));
    map.insert("AcquireTime".into(), ParamInfo::float64(ad.acquire_time, "ACQ_TIME"));
    map.insert("AcquirePeriod".into(), ParamInfo::float64(ad.acquire_period, "ACQ_PERIOD"));
    map.insert("PeakHeightVariation".into(), ParamInfo::float64(sim.peak_height_variation, "SIM_PEAK_HEIGHT_VAR"));

    // Int32 writable params (longout/bo records)
    map.insert("Acquire".into(), ParamInfo::int32(ad.acquire, "ACQUIRE"));
    map.insert("SimMode".into(), ParamInfo::int32(sim.sim_mode, "SIM_MODE"));
    map.insert("ImageMode".into(), ParamInfo::int32(ad.image_mode, "IMAGE_MODE"));
    map.insert("DataType".into(), ParamInfo::int32(base.data_type, "DATA_TYPE"));
    map.insert("ColorMode".into(), ParamInfo::int32(base.color_mode, "COLOR_MODE"));
    map.insert("NumImages".into(), ParamInfo::int32(ad.num_images, "NUM_IMAGES"));
    map.insert("ResetImage".into(), ParamInfo::int32(sim.reset_image, "SIM_RESET_IMAGE"));
    map.insert("PeakStartX".into(), ParamInfo::int32(sim.peak_start_x, "SIM_PEAK_START_X"));
    map.insert("PeakStartY".into(), ParamInfo::int32(sim.peak_start_y, "SIM_PEAK_START_Y"));
    map.insert("PeakWidthX".into(), ParamInfo::int32(sim.peak_width_x, "SIM_PEAK_WIDTH_X"));
    map.insert("PeakWidthY".into(), ParamInfo::int32(sim.peak_width_y, "SIM_PEAK_WIDTH_Y"));
    map.insert("PeakNumX".into(), ParamInfo::int32(sim.peak_num_x, "SIM_PEAK_NUM_X"));
    map.insert("PeakNumY".into(), ParamInfo::int32(sim.peak_num_y, "SIM_PEAK_NUM_Y"));
    map.insert("PeakStepX".into(), ParamInfo::int32(sim.peak_step_x, "SIM_PEAK_STEP_X"));
    map.insert("PeakStepY".into(), ParamInfo::int32(sim.peak_step_y, "SIM_PEAK_STEP_Y"));

    // Readback versions of writable params (_RBV)
    map.insert("Gain_RBV".into(), ParamInfo::float64(sim.gain, "SIM_GAIN"));
    map.insert("GainX_RBV".into(), ParamInfo::float64(sim.gain_x, "SIM_GAIN_X"));
    map.insert("GainY_RBV".into(), ParamInfo::float64(sim.gain_y, "SIM_GAIN_Y"));
    map.insert("GainRed_RBV".into(), ParamInfo::float64(sim.gain_red, "SIM_GAIN_RED"));
    map.insert("GainGreen_RBV".into(), ParamInfo::float64(sim.gain_green, "SIM_GAIN_GREEN"));
    map.insert("GainBlue_RBV".into(), ParamInfo::float64(sim.gain_blue, "SIM_GAIN_BLUE"));
    map.insert("Offset_RBV".into(), ParamInfo::float64(sim.offset, "SIM_OFFSET"));
    map.insert("Noise_RBV".into(), ParamInfo::float64(sim.noise, "SIM_NOISE"));
    map.insert("AcquireTime_RBV".into(), ParamInfo::float64(ad.acquire_time, "ACQ_TIME"));
    map.insert("AcquirePeriod_RBV".into(), ParamInfo::float64(ad.acquire_period, "ACQ_PERIOD"));
    map.insert("SimMode_RBV".into(), ParamInfo::int32(sim.sim_mode, "SIM_MODE"));
    map.insert("ImageMode_RBV".into(), ParamInfo::int32(ad.image_mode, "IMAGE_MODE"));
    map.insert("DataType_RBV".into(), ParamInfo::int32(base.data_type, "DATA_TYPE"));
    map.insert("ColorMode_RBV".into(), ParamInfo::int32(base.color_mode, "COLOR_MODE"));
    map.insert("NumImages_RBV".into(), ParamInfo::int32(ad.num_images, "NUM_IMAGES"));

    // Read-only int32 params (longin records)
    map.insert("MaxSizeX_RBV".into(), ParamInfo::int32(ad.max_size_x, "MAX_SIZE_X"));
    map.insert("MaxSizeY_RBV".into(), ParamInfo::int32(ad.max_size_y, "MAX_SIZE_Y"));
    map.insert("ArrayCounter_RBV".into(), ParamInfo::int32(base.array_counter, "ARRAY_COUNTER"));
    map.insert("ArraySizeX_RBV".into(), ParamInfo::int32(base.array_size_x, "ARRAY_SIZE_X"));
    map.insert("ArraySizeY_RBV".into(), ParamInfo::int32(base.array_size_y, "ARRAY_SIZE_Y"));
    map.insert("ArraySizeZ_RBV".into(), ParamInfo::int32(base.array_size_z, "ARRAY_SIZE_Z"));
    map.insert("NumImagesCounter_RBV".into(), ParamInfo::int32(ad.num_images_counter, "NUM_IMAGES_COUNTER"));
    map.insert("Acquire_RBV".into(), ParamInfo::int32(ad.acquire, "ACQUIRE"));
    map.insert("Status_RBV".into(), ParamInfo::int32(ad.status, "STATUS"));

    // Read-only string params (stringin records)
    map.insert("StatusMessage_RBV".into(), ParamInfo::string(ad.status_message, "STATUS_MESSAGE"));
    map.insert("Manufacturer_RBV".into(), ParamInfo::string(base.manufacturer, "MANUFACTURER"));
    map.insert("Model_RBV".into(), ParamInfo::string(base.model, "MODEL"));

    map
}

/// Device support bridge between epics-base-rs records and SimDetector.
/// Wraps AsynDeviceSupport for PortHandle-based access.
pub struct SimDeviceSupport {
    inner: AsynDeviceSupport,
    registry: Arc<ParamRegistry>,
}

impl SimDeviceSupport {
    /// Create from a legacy `Arc<Mutex<SimDetector>>` (direct locking).
    pub fn new(
        driver: Arc<parking_lot::Mutex<SimDetector>>,
        registry: Arc<ParamRegistry>,
    ) -> Self {
        use asyn_rs::adapter::AsynLink;
        let link = AsynLink {
            port_name: String::new(),
            addr: 0,
            timeout: std::time::Duration::from_secs(1),
            drv_info: String::new(),
        };
        Self {
            inner: AsynDeviceSupport::new(driver, link, "asynInt32"),
            registry,
        }
    }

    /// Create from a [`PortHandle`] (actor model).
    pub fn from_handle(
        handle: PortHandle,
        registry: Arc<ParamRegistry>,
    ) -> Self {
        use asyn_rs::adapter::AsynLink;
        let link = AsynLink {
            port_name: String::new(),
            addr: 0,
            timeout: std::time::Duration::from_secs(1),
            drv_info: String::new(),
        };
        Self {
            inner: AsynDeviceSupport::from_handle(handle, link, "asynInt32"),
            registry,
        }
    }
}

impl DeviceSupport for SimDeviceSupport {
    fn dtyp(&self) -> &str {
        "asynSimDetector"
    }

    fn set_record_info(&mut self, name: &str, scan: ScanType) {
        // Extract suffix after last ':'
        let suffix = name.rsplit(':').next().unwrap_or(name);
        if let Some(info) = self.registry.get(suffix) {
            self.inner.set_drv_info(&info.drv_info);
            let iface = match info.param_type {
                ParamType::Int32 => "asynInt32",
                ParamType::Float64 => "asynFloat64",
                ParamType::OctetString => "asynOctet",
            };
            self.inner.set_iface_type(iface);
        } else {
            eprintln!("asynSimDetector: no param mapping for record suffix '{suffix}' (record: {name})");
        }
        self.inner.set_record_info(name, scan);
    }

    fn init(&mut self, record: &mut dyn Record) -> CaResult<()> {
        self.inner.init(record)
    }

    fn read(&mut self, record: &mut dyn Record) -> CaResult<()> {
        self.inner.read(record)
    }

    fn write(&mut self, record: &mut dyn Record) -> CaResult<()> {
        self.inner.write(record)
    }

    fn write_begin(&mut self, record: &mut dyn Record) -> CaResult<Option<Box<dyn WriteCompletion>>> {
        self.inner.write_begin(record)
    }

    fn last_alarm(&self) -> Option<(u16, u16)> {
        self.inner.last_alarm()
    }

    fn last_timestamp(&self) -> Option<std::time::SystemTime> {
        self.inner.last_timestamp()
    }

    fn io_intr_receiver(&mut self) -> Option<tokio::sync::mpsc::Receiver<()>> {
        self.inner.io_intr_receiver()
    }
}
