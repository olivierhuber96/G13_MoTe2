# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:43:00 2025

@author: ohuber
"""

import numpy as np
import os
import threading
import glob 
import time
import Hydraharp_srv as hydra

def run_parallel(hydra, acq_time, chunk_time, filename_prefix, dp_merged):
        # Ensure output folders exist
        
        stop_event = threading.Event()
        folder_name = dp_merged['data_path']
        hydra.create_new_folder(folder_name)
        
        def analysis_loop():
            processed_chunks = set()
            while not stop_event.is_set():
                # Look for all chunk files so far
                chunk_files = sorted(glob.glob(f"{filename_prefix}_chunk*.b"))
                new_chunks = [f for f in chunk_files if f not in processed_chunks]
                if new_chunks:
                    print(f"[ANALYSIS] Found {len(new_chunks)} new chunks.")
                    hydra.merge_chunk_files(filename_prefix)
                    hydra.correlate_timestamps(dp_merged)
                    processed_chunks.update(new_chunks)
                time.sleep(chunk_time*1.01/1e3)  #wait until the next chunk is generated
                print('sleeping time is ', chunk_time*1.01/1e3,'s')

            # Process any leftover chunks one final time
            chunk_files = sorted(glob.glob(f"{filename_prefix}_chunk*.b"))
            new_chunks = [f for f in chunk_files if f not in processed_chunks]
            if new_chunks:
                hydra.merge_chunk_files(filename_prefix)
                hydra.correlate_timestamps(dp_merged)
            print(chunk_files)
            print("[ANALYSIS] Analysis finished.")

        # Start analysis in background thread
        analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        analysis_thread.start()

        # Run acquisition in main process
        hydra.measure_timestamps_T2_chunked(
            acq_time=acq_time,
            filename_prefix=filename_prefix,
            chunk_time=chunk_time
        )

        # When acquisition is done, signal analysis to stop
        stop_event.set()
        analysis_thread.join()