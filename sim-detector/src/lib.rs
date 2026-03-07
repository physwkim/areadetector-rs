pub mod types;
pub mod pixel_cast;
pub mod color_layout;
pub mod params;
pub mod compute;
pub mod roi;
pub mod driver;
pub mod task;

#[cfg(feature = "ioc")]
pub mod ioc_support;

pub use driver::{SimDetector, SimDetectorRuntime, create_sim_detector};

/// Path to this crate's Db/ directory (set at compile time).
/// Use in st.cmd as: `dbLoadRecords("$(SIM_DETECTOR)/Db/simDetector.db", ...)`
pub const DB_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Db");
