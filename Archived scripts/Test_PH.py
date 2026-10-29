import ctypes
from pathlib import Path

dll_path = Path(r"C:\Users\SSPDs\conda-envs\hydra_env_test\DLLs\phlib64.dll")
_phlib = ctypes.WinDLL(str(dll_path))

print("Probing PicoHarp Devices (Skipping dev_idx 0)...")

# Start at 1 to completely avoid the running device!
for i in range(1, 4): 
    serial = ctypes.create_string_buffer(32)
    ret = _phlib.PH_OpenDevice(ctypes.c_int(i), serial)
    
    
    if ret == 0:
        print(f"✅ Success: Device available at index {i} (Serial: {serial.value.decode()})")
        _phlib.PH_CloseDevice(ctypes.c_int(i)) # Close it immediately so we don't lock it
    elif ret == -1:
        print(f"❌ Index {i}: ERROR_DEVICE_OPEN_FAIL (Device missing or already in use)")
    else:
        print(f"❌ Index {i}: Failed with unknown error {ret}")