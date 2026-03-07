#!../../target/debug/sim_ioc
#============================================================
# st.cmd — SimDetector IOC startup script
#
# This is an iocsh script executed by sim_ioc, matching the
# C++ ADSimDetector IOC startup structure.
#
# Usage:
#   cargo run --bin sim_ioc --features ioc -- ioc/st.cmd
#============================================================

# Environment
epicsEnvSet("PREFIX", "SIM1:")
epicsEnvSet("CAM",    "cam1:")

# Create the SimDetector driver
# simDetectorConfig(portName, sizeX, sizeY, maxMemory)
simDetectorConfig("SIM1", 256, 256, 50000000)

# Load the detector database
# Path is relative to working directory (typically the workspace root)
dbLoadRecords("$(SIM_DETECTOR)/Db/simDetector.db", "P=$(PREFIX),R=$(CAM)")

# iocInit is called automatically by IocApplication after this script completes.
iocInit()

# After init, the interactive iocsh shell starts.
#
# Example interactive commands:
#   dbl                                # List all PVs
#   dbpf SIM1:cam1:Acquire 1           # Start acquisition
#   dbgf SIM1:cam1:ArrayCounter_RBV    # Read frame counter
#   dbpf SIM1:cam1:SimMode 1           # Switch to Peaks mode
#   dbpf SIM1:cam1:Acquire 0           # Stop acquisition
#   simDetectorReport                  # Show detector status
