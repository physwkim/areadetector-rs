use std::collections::HashMap;
use std::sync::Arc;

use asyn_rs::port::PortDriver;
use asyn_rs::user::AsynUser;
use epics_base_rs::error::CaResult;
use epics_base_rs::server::device_support::DeviceSupport;
use epics_base_rs::server::record::{Record, ScanType};
use epics_base_rs::types::EpicsValue;

use crate::SimDetector;

#[derive(Clone, Copy)]
pub enum ParamType {
    Int32,
    Float64,
    OctetString,
}

#[derive(Clone, Copy)]
pub struct ParamInfo {
    pub param_index: usize,
    pub param_type: ParamType,
}

impl ParamInfo {
    pub fn int32(index: usize) -> Self {
        Self { param_index: index, param_type: ParamType::Int32 }
    }
    pub fn float64(index: usize) -> Self {
        Self { param_index: index, param_type: ParamType::Float64 }
    }
    pub fn string(index: usize) -> Self {
        Self { param_index: index, param_type: ParamType::OctetString }
    }
}

/// Registry mapping record name suffixes to sim-detector parameters.
pub type ParamRegistry = HashMap<String, ParamInfo>;

/// Build the parameter registry from a SimDetector instance.
pub fn build_param_registry(det: &SimDetector) -> ParamRegistry {
    let mut map = HashMap::new();
    let ad = &det.ad.params;
    let sim = &det.sim_params;
    let base = &ad.base;

    // Float64 writable params (ao records)
    map.insert("Gain".into(), ParamInfo::float64(sim.gain));
    map.insert("GainX".into(), ParamInfo::float64(sim.gain_x));
    map.insert("GainY".into(), ParamInfo::float64(sim.gain_y));
    map.insert("GainRed".into(), ParamInfo::float64(sim.gain_red));
    map.insert("GainGreen".into(), ParamInfo::float64(sim.gain_green));
    map.insert("GainBlue".into(), ParamInfo::float64(sim.gain_blue));
    map.insert("Offset".into(), ParamInfo::float64(sim.offset));
    map.insert("Noise".into(), ParamInfo::float64(sim.noise));
    map.insert("AcquireTime".into(), ParamInfo::float64(ad.acquire_time));
    map.insert("AcquirePeriod".into(), ParamInfo::float64(ad.acquire_period));
    map.insert("PeakHeightVariation".into(), ParamInfo::float64(sim.peak_height_variation));

    // Int32 writable params (longout/bo records)
    map.insert("Acquire".into(), ParamInfo::int32(ad.acquire));
    map.insert("SimMode".into(), ParamInfo::int32(sim.sim_mode));
    map.insert("ImageMode".into(), ParamInfo::int32(ad.image_mode));
    map.insert("DataType".into(), ParamInfo::int32(base.data_type));
    map.insert("ColorMode".into(), ParamInfo::int32(base.color_mode));
    map.insert("NumImages".into(), ParamInfo::int32(ad.num_images));
    map.insert("ResetImage".into(), ParamInfo::int32(sim.reset_image));
    map.insert("PeakStartX".into(), ParamInfo::int32(sim.peak_start_x));
    map.insert("PeakStartY".into(), ParamInfo::int32(sim.peak_start_y));
    map.insert("PeakWidthX".into(), ParamInfo::int32(sim.peak_width_x));
    map.insert("PeakWidthY".into(), ParamInfo::int32(sim.peak_width_y));
    map.insert("PeakNumX".into(), ParamInfo::int32(sim.peak_num_x));
    map.insert("PeakNumY".into(), ParamInfo::int32(sim.peak_num_y));
    map.insert("PeakStepX".into(), ParamInfo::int32(sim.peak_step_x));
    map.insert("PeakStepY".into(), ParamInfo::int32(sim.peak_step_y));

    // Readback versions of writable params (_RBV)
    map.insert("Gain_RBV".into(), ParamInfo::float64(sim.gain));
    map.insert("GainX_RBV".into(), ParamInfo::float64(sim.gain_x));
    map.insert("GainY_RBV".into(), ParamInfo::float64(sim.gain_y));
    map.insert("GainRed_RBV".into(), ParamInfo::float64(sim.gain_red));
    map.insert("GainGreen_RBV".into(), ParamInfo::float64(sim.gain_green));
    map.insert("GainBlue_RBV".into(), ParamInfo::float64(sim.gain_blue));
    map.insert("Offset_RBV".into(), ParamInfo::float64(sim.offset));
    map.insert("Noise_RBV".into(), ParamInfo::float64(sim.noise));
    map.insert("AcquireTime_RBV".into(), ParamInfo::float64(ad.acquire_time));
    map.insert("AcquirePeriod_RBV".into(), ParamInfo::float64(ad.acquire_period));
    map.insert("SimMode_RBV".into(), ParamInfo::int32(sim.sim_mode));
    map.insert("ImageMode_RBV".into(), ParamInfo::int32(ad.image_mode));
    map.insert("DataType_RBV".into(), ParamInfo::int32(base.data_type));
    map.insert("ColorMode_RBV".into(), ParamInfo::int32(base.color_mode));
    map.insert("NumImages_RBV".into(), ParamInfo::int32(ad.num_images));

    // Read-only int32 params (longin records)
    map.insert("MaxSizeX_RBV".into(), ParamInfo::int32(ad.max_size_x));
    map.insert("MaxSizeY_RBV".into(), ParamInfo::int32(ad.max_size_y));
    map.insert("ArrayCounter_RBV".into(), ParamInfo::int32(base.array_counter));
    map.insert("ArraySizeX_RBV".into(), ParamInfo::int32(base.array_size_x));
    map.insert("ArraySizeY_RBV".into(), ParamInfo::int32(base.array_size_y));
    map.insert("ArraySizeZ_RBV".into(), ParamInfo::int32(base.array_size_z));
    map.insert("NumImagesCounter_RBV".into(), ParamInfo::int32(ad.num_images_counter));
    map.insert("Acquire_RBV".into(), ParamInfo::int32(ad.acquire));
    map.insert("Status_RBV".into(), ParamInfo::int32(ad.status));

    // Read-only string params (stringin records)
    map.insert("StatusMessage_RBV".into(), ParamInfo::string(ad.status_message));
    map.insert("Manufacturer_RBV".into(), ParamInfo::string(base.manufacturer));
    map.insert("Model_RBV".into(), ParamInfo::string(base.model));

    map
}

/// Device support bridge between epics-base-rs records and SimDetector.
pub struct SimDeviceSupport {
    driver: Arc<parking_lot::Mutex<SimDetector>>,
    registry: Arc<ParamRegistry>,
    mapping: Option<ParamInfo>,
    record_name: String,
}

impl SimDeviceSupport {
    pub fn new(
        driver: Arc<parking_lot::Mutex<SimDetector>>,
        registry: Arc<ParamRegistry>,
    ) -> Self {
        Self {
            driver,
            registry,
            mapping: None,
            record_name: String::new(),
        }
    }
}

impl DeviceSupport for SimDeviceSupport {
    fn dtyp(&self) -> &str {
        "asynSimDetector"
    }

    fn set_record_info(&mut self, name: &str, _scan: ScanType) {
        self.record_name = name.to_string();
        // Extract suffix after last ':'
        let suffix = name.rsplit(':').next().unwrap_or(name);
        if let Some(info) = self.registry.get(suffix) {
            self.mapping = Some(*info);
        } else {
            eprintln!("asynSimDetector: no param mapping for record suffix '{suffix}' (record: {name})");
        }
    }

    fn read(&mut self, record: &mut dyn Record) -> CaResult<()> {
        let info = match self.mapping {
            Some(info) => info,
            None => return Ok(()),
        };
        let drv = self.driver.lock();
        let base = &drv.ad.port_base;
        match info.param_type {
            ParamType::Int32 => {
                let val = base.get_int32_param(info.param_index, 0)
                    .map_err(|e| epics_base_rs::error::CaError::InvalidValue(e.to_string()))?;
                record.set_val(EpicsValue::Long(val))?;
            }
            ParamType::Float64 => {
                let val = base.get_float64_param(info.param_index, 0)
                    .map_err(|e| epics_base_rs::error::CaError::InvalidValue(e.to_string()))?;
                record.set_val(EpicsValue::Double(val))?;
            }
            ParamType::OctetString => {
                let val = base.get_string_param(info.param_index, 0)
                    .map_err(|e| epics_base_rs::error::CaError::InvalidValue(e.to_string()))?;
                record.set_val(EpicsValue::String(val.to_string()))?;
            }
        }
        Ok(())
    }

    fn write(&mut self, record: &mut dyn Record) -> CaResult<()> {
        let info = match self.mapping {
            Some(info) => info,
            None => return Ok(()),
        };
        let val = record.val()
            .ok_or_else(|| epics_base_rs::error::CaError::InvalidValue("no VAL".into()))?;

        let mut drv = self.driver.lock();
        match info.param_type {
            ParamType::Int32 => {
                let v = val.to_f64()
                    .ok_or_else(|| epics_base_rs::error::CaError::InvalidValue("cannot convert to i32".into()))? as i32;
                let mut user = AsynUser::new(info.param_index);
                drv.write_int32(&mut user, v)
                    .map_err(|e| epics_base_rs::error::CaError::InvalidValue(e.to_string()))?;
            }
            ParamType::Float64 => {
                let v = val.to_f64()
                    .ok_or_else(|| epics_base_rs::error::CaError::InvalidValue("cannot convert to f64".into()))?;
                let mut user = AsynUser::new(info.param_index);
                drv.write_float64(&mut user, v)
                    .map_err(|e| epics_base_rs::error::CaError::InvalidValue(e.to_string()))?;
            }
            ParamType::OctetString => {
                // String params are read-only from device side
            }
        }
        Ok(())
    }
}
