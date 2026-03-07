//! SimDetector IOC binary.
//!
//! Uses IocApplication for st.cmd-style startup matching the C++ EPICS pattern.
//!
//! Usage:
//!   cargo run --bin sim_ioc --features ioc -- st.cmd
//!   cargo run --bin sim_ioc --features ioc -- ioc/st.cmd
//!
//! The st.cmd script can use:
//!   epicsEnvSet, dbLoadRecords, simDetectorConfig

use std::sync::Arc;

use epics_base_rs::error::CaResult;
use epics_base_rs::server::ioc_app::IocApplication;
use epics_base_rs::server::iocsh::registry::*;

use sim_detector::ioc_support::{build_param_registry, SimDeviceSupport};
use sim_detector::SimDetector;

/// Shared state between simDetectorConfig command and device support factory.
struct DriverHolder {
    driver: std::sync::Mutex<Option<Arc<parking_lot::Mutex<SimDetector>>>>,
    registry: std::sync::Mutex<Option<Arc<sim_detector::ioc_support::ParamRegistry>>>,
}

impl DriverHolder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            driver: std::sync::Mutex::new(None),
            registry: std::sync::Mutex::new(None),
        })
    }
}

/// simDetectorConfig command handler (used during st.cmd execution).
struct SimDetectorConfigHandler {
    holder: Arc<DriverHolder>,
}

impl CommandHandler for SimDetectorConfigHandler {
    fn call(&self, args: &[ArgValue], _ctx: &CommandContext) -> CommandResult {
        let port_name = match &args[0] {
            ArgValue::String(s) => s.clone(),
            _ => return Err("portName required".into()),
        };
        let size_x = match &args[1] {
            ArgValue::Int(n) => *n as i32,
            _ => 256,
        };
        let size_y = match &args[2] {
            ArgValue::Int(n) => *n as i32,
            _ => 256,
        };
        let max_memory = match &args[3] {
            ArgValue::Int(n) => *n as usize,
            _ => 50_000_000,
        };

        println!("simDetectorConfig: port={port_name}, size={size_x}x{size_y}, maxMemory={max_memory}");

        let det = SimDetector::new(&port_name, size_x, size_y, max_memory)
            .map_err(|e| format!("failed to create SimDetector: {e}"))?;

        let registry = Arc::new(build_param_registry(&det));
        let driver = Arc::new(parking_lot::Mutex::new(det));
        SimDetector::start_thread(driver.clone());

        *self.holder.driver.lock().unwrap() = Some(driver);
        *self.holder.registry.lock().unwrap() = Some(registry);

        Ok(CommandOutcome::Continue)
    }
}

/// simDetectorReport command handler (used in interactive shell).
struct ReportHandler {
    holder: Arc<DriverHolder>,
}

impl CommandHandler for ReportHandler {
    fn call(&self, _args: &[ArgValue], _ctx: &CommandContext) -> CommandResult {
        let guard = self.holder.driver.lock().unwrap();
        let driver = match guard.as_ref() {
            Some(d) => d,
            None => {
                println!("No SimDetector configured");
                return Ok(CommandOutcome::Continue);
            }
        };

        let d = driver.lock();
        let base = &d.ad.port_base;
        let ad = &d.ad.params;
        let sim = &d.sim_params;

        let acquire = base.get_int32_param(ad.acquire, 0).unwrap_or(0);
        let counter = base.get_int32_param(ad.base.array_counter, 0).unwrap_or(0);
        let size_x = base.get_int32_param(ad.max_size_x, 0).unwrap_or(0);
        let size_y = base.get_int32_param(ad.max_size_y, 0).unwrap_or(0);
        let mode = base.get_int32_param(sim.sim_mode, 0).unwrap_or(0);
        let image_mode = base.get_int32_param(ad.image_mode, 0).unwrap_or(0);
        let status = base.get_int32_param(ad.status, 0).unwrap_or(0);

        let mode_str = match mode {
            0 => "LinearRamp", 1 => "Peaks", 2 => "Sine", 3 => "OffsetNoise", _ => "Unknown",
        };
        let image_mode_str = match image_mode {
            0 => "Single", 1 => "Multiple", 2 => "Continuous", _ => "Unknown",
        };
        let status_str = match status {
            0 => "Idle", 1 => "Acquire", _ => "Unknown",
        };

        println!("SimDetector Report");
        println!("  Size:         {}x{}", size_x, size_y);
        println!("  SimMode:      {} ({})", mode, mode_str);
        println!("  ImageMode:    {} ({})", image_mode, image_mode_str);
        println!("  Acquire:      {}", acquire);
        println!("  Status:       {} ({})", status, status_str);
        println!("  ArrayCounter: {}", counter);

        Ok(CommandOutcome::Continue)
    }
}

#[tokio::main]
async fn main() -> CaResult<()> {
    let args: Vec<String> = std::env::args().collect();

    // Set module paths as env vars (like C++ EPICS envPaths)
    // Allows st.cmd to use: dbLoadRecords("$(SIM_DETECTOR)/Db/simDetector.db", ...)
    unsafe {
        std::env::set_var("SIM_DETECTOR", sim_detector::DB_DIR.trim_end_matches("/Db"));
    }

    let script = if args.len() > 1 && !args[1].starts_with('-') {
        args[1].clone()
    } else {
        eprintln!("Usage: sim_ioc <st.cmd>");
        eprintln!();
        eprintln!("The st.cmd script should contain:");
        eprintln!(r#"  epicsEnvSet("PREFIX", "SIM1:")"#);
        eprintln!(r#"  simDetectorConfig("SIM1", 256, 256, 50000000)"#);
        eprintln!(r#"  dbLoadRecords("$(SIM_DETECTOR)/Db/simDetector.db", "P=$(PREFIX),R=cam1:")"#);
        std::process::exit(1);
    };

    // Shared holder: simDetectorConfig fills it, device support factory reads it
    let holder = DriverHolder::new();

    let holder_for_config = holder.clone();
    let holder_for_factory = holder.clone();
    let holder_for_report = holder.clone();

    IocApplication::new()
        .port(
            std::env::var("EPICS_CA_SERVER_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5064),
        )
        // Phase 1: startup script command
        .register_startup_command(CommandDef::new(
            "simDetectorConfig",
            vec![
                ArgDesc { name: "portName", arg_type: ArgType::String, optional: false },
                ArgDesc { name: "sizeX", arg_type: ArgType::Int, optional: true },
                ArgDesc { name: "sizeY", arg_type: ArgType::Int, optional: true },
                ArgDesc { name: "maxMemory", arg_type: ArgType::Int, optional: true },
            ],
            "simDetectorConfig portName [sizeX] [sizeY] [maxMemory] - Configure SimDetector driver",
            SimDetectorConfigHandler { holder: holder_for_config },
        ))
        // Device support: wired during iocInit (Phase 2)
        .register_device_support("asynSimDetector", move || {
            let driver = holder_for_factory.driver.lock().unwrap()
                .as_ref()
                .expect("simDetectorConfig must be called before iocInit")
                .clone();
            let registry = holder_for_factory.registry.lock().unwrap()
                .as_ref()
                .expect("simDetectorConfig must be called before iocInit")
                .clone();
            Box::new(SimDeviceSupport::new(driver, registry))
        })
        // Phase 3: interactive shell commands
        .register_shell_command(CommandDef::new(
            "simDetectorReport",
            vec![ArgDesc { name: "level", arg_type: ArgType::Int, optional: true }],
            "simDetectorReport [level] - Report SimDetector status",
            ReportHandler { holder: holder_for_report },
        ))
        .startup_script(&script)
        .run()
        .await
}
