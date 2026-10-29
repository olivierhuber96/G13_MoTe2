# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:55:01 2017

@author: Thomas Fink, Quantum Photonics Group, ETH Zurich
"""

#from __future__ import division,print_function,unicode_literals,absolute_import

import ctypes
import numpy as np
import functools
import logging
import sys
import itertools
import Pyro4
import Pyro4.naming
import time
import shutil
from RemoteCorrelationAnalysis import correlate_parallel
from RemoteCorrelationAnalysis import correlate_timestamps as c_timestamps
import glob
import threading 
import os
import hashlib

# Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle', 'json', 'marshal', 'serpent']) #Kilian changed this to the line below for debugging
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}


@Pyro4.expose
class Analysis(object):
    def merge_chunk_files(self, filename_prefix):
        """
        Merge all chunk files starting with filename_prefix into a single file
        called 'chunk_merged.b' in the 'merged_chunks' subfolder. Overwrites existing file.
        Returns the merged file path and verifies integrity by comparing hashes.
        """
        folder = os.path.dirname(filename_prefix)
        if os.path.dirname(filename_prefix) is not None:
            os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)
        base_prefix = os.path.basename(filename_prefix)

        # Find all matching chunk files (fix: use base_prefix to match properly)
        chunk_files = glob.glob(os.path.join(folder, f"{base_prefix}_chunk*.b"))

        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found with prefix '{filename_prefix}'.")

        # Sort files by chunk index
        def chunk_index(filename):
            base = os.path.basename(filename)
            idx_part = base.split("_chunk")[-1].split(".b")[0]
            return int(idx_part)

        chunk_files.sort(key=chunk_index)

        merged_file_path = os.path.join(folder, "chunk_merged.b")

        # Initialize a hash object to verify merged content
        merged_hash = hashlib.sha256()

        with open(merged_file_path, 'wb') as outfile:
            for chunk_file in chunk_files:
                with open(chunk_file, 'rb') as infile:
                    while True:
                        buf = infile.read(1024 * 1024)  # 1 MB chunks
                        if not buf:
                            break
                        outfile.write(buf)
                        merged_hash.update(buf)

        # Verify integrity: recompute hash from merged file and compare
        verify_hash = hashlib.sha256()
        with open(merged_file_path, 'rb') as merged:
            for block in iter(lambda: merged.read(1024 * 1024), b""):
                verify_hash.update(block)

        if merged_hash.digest() != verify_hash.digest():
            raise ValueError("Integrity check failed: merged file does not match input chunks.")

        total_input_size = sum(os.path.getsize(f) for f in chunk_files)
        merged_size = os.path.getsize(merged_file_path)
        print(total_input_size,merged_size)
        if total_input_size != merged_size:
            raise ValueError("Merged file size does not match total input size.")
        
        return merged_file_path
        
        
    def correlate_timestamps(self, arguments):
        correlate_parallel(arguments)
        
    def correlate_timestamps_seq(self, arguments):
        c_timestamps(arguments)
    
    def create_new_folder(self, folder_name):
        try:
            print('Create a new folder for the measurement.')
            print(folder_name)
            if folder_name is not None:
                os.makedirs(folder_name, exist_ok=True)
            
        except:
            print('New folder was not created')
    
    def look_for_files(self,filename_prefix):
        return sorted(glob.glob(f"{filename_prefix}_chunk*.b"))
		
Analysis_ = Analysis()
daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9090)
uri = daemon.register(Analysis_, objectId='analysis_g13')
print('Obj URI:', uri)
daemon.requestLoop()