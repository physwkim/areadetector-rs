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

# Load the detector databases
# AD base records (image size, acquire control, shutter, temperature)
dbLoadRecords("$(AD_CORE)/Db/ADBase.db", "P=$(PREFIX),R=$(CAM)")
# NDArray base records (array info, pool stats, attributes)
dbLoadRecords("$(AD_CORE)/Db/NDArrayBase.db", "P=$(PREFIX),R=$(CAM)")
# File I/O records
dbLoadRecords("$(AD_CORE)/Db/NDFile.db", "P=$(PREFIX),R=$(CAM)")
# SimDetector-specific records (gains, peaks, sine, sim mode)
dbLoadRecords("$(SIM_DETECTOR)/Db/simDetector.db", "P=$(PREFIX),R=$(CAM)")

# iocInit is called automatically by IocApplication after this script completes.
#
# After init, the interactive iocsh shell starts.
#
# Example interactive commands:
#   dbl                                # List all PVs
#   dbpf SIM1:cam1:Acquire 1           # Start acquisition
#   dbgf SIM1:cam1:ArrayCounter_RBV    # Read frame counter
#   dbpf SIM1:cam1:SimMode 1           # Switch to Peaks mode
#   dbpf SIM1:cam1:Acquire 0           # Stop acquisition
#   simDetectorReport                  # Show detector status
