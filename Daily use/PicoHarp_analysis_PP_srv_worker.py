"""Worker launcher for the shared pump-probe analysis server.

This keeps the analysis implementation in PicoHarp_analysis_PP_srv.py and only
changes the default Pyro port for a second parallel server process.
"""

import os
import runpy
from pathlib import Path


os.environ.setdefault("PICOHARP_PP_PORT", "9113")

server_script = Path(__file__).with_name("PicoHarp_analysis_PP_srv.py")
runpy.run_path(str(server_script), run_name="__main__")
