pub mod error;
pub mod timestamp;
pub mod ndarray;
pub mod attributes;
pub mod ndarray_handle;
pub mod ndarray_pool;
pub mod codec;
pub mod color;
pub mod params;
pub mod driver;
pub mod plugin;

/// Path to this crate's Db/ directory (set at compile time).
/// Use in st.cmd as: `dbLoadRecords("$(AD_CORE)/Db/ADBase.db", ...)`
pub const DB_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Db");
