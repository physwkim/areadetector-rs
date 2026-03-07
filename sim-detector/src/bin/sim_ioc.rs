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

use asyn_rs::port_handle::PortHandle;
use epics_base_rs::error::CaResult;
use epics_base_rs::server::ioc_app::IocApplication;
use epics_base_rs::server::iocsh::registry::*;

use ad_core::plugin::channel::NDArrayOutput;
use sim_detector::driver::{create_sim_detector, SimDetectorRuntime};
use sim_detector::ioc_support::{build_param_registry_from_params, SimDeviceSupport};

/// Shared state between simDetectorConfig command and device support factory.
struct DriverHolder {
    port_handle: std::sync::Mutex<Option<PortHandle>>,
    registry: std::sync::Mutex<Option<Arc<sim_detector::ioc_support::ParamRegistry>>>,
    /// Keep runtime alive to prevent shutdown
    _runtime: std::sync::Mutex<Option<SimDetectorRuntime>>,
}

impl DriverHolder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            port_handle: std::sync::Mutex::new(None),
            registry: std::sync::Mutex::new(None),
            _runtime: std::sync::Mutex::new(None),
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

        let array_output = NDArrayOutput::new();
        let runtime = create_sim_detector(&port_name, size_x, size_y, max_memory, array_output)
            .map_err(|e| format!("failed to create SimDetector: {e}"))?;

        let registry = Arc::new(build_param_registry_from_params(&runtime.ad_params, &runtime.sim_params));
        let port_handle = runtime.port_handle().clone();

        *self.holder.port_handle.lock().unwrap() = Some(port_handle);
        *self.holder.registry.lock().unwrap() = Some(registry);
        *self.holder._runtime.lock().unwrap() = Some(runtime);

        Ok(CommandOutcome::Continue)
    }
}

/// simDetectorReport command handler (used in interactive shell).
struct ReportHandler {
    holder: Arc<DriverHolder>,
}

impl CommandHandler for ReportHandler {
    fn call(&self, _args: &[ArgValue], _ctx: &CommandContext) -> CommandResult {
        let guard = self.holder.port_handle.lock().unwrap();
        if guard.is_none() {
            println!("No SimDetector configured");
            return Ok(CommandOutcome::Continue);
        }

        println!("SimDetector Report (PortHandle-based, use record reads for status)");
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
            let handle = holder_for_factory.port_handle.lock().unwrap()
                .as_ref()
                .expect("simDetectorConfig must be called before iocInit")
                .clone();
            let registry = holder_for_factory.registry.lock().unwrap()
                .as_ref()
                .expect("simDetectorConfig must be called before iocInit")
                .clone();
            Box::new(SimDeviceSupport::from_handle(handle, registry))
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
