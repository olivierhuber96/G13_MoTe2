import os, time, threading
import Pyro4       
import numpy as np     
import pickle
import glob, ast
import traceback
from collections import deque
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
from time import gmtime, strftime
import time
def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())

from RemoteCorrelationAnalysis_edited_Kilian import (
    merge_chunk_files, look_for_files, correlate_timestamps_Kilian, get_countrates_chunk
)

@Pyro4.expose
class AnalysisService(object):
        
    def __init__(self):
        self._log = deque(maxlen=10000)
        self._cv = threading.Condition()

    def _append_log(self, msg):
        print(msg, flush=True) # comment out to suppress server side log dump
        with self._cv:
            self._log.append(msg)
            self._cv.notify_all()


    def get_logs(self, last_idx=0, timeout=5.0):
        """Client polls for new logs; waits up to `timeout` seconds."""
        end = time.time() + timeout
        with self._cv:
            while last_idx >= len(self._log):
                remaining = end - time.time()
                if remaining <= 0:
                    return last_idx, []
                self._cv.wait(remaining)
            new = list(self._log)[last_idx:]
            return len(self._log), new
                    
    def ping(self):
        self._append_log("Analysis server reached.")

        

    @staticmethod
    def run_with_timeout(func, timeout_s, *args, **kwargs):
        """Run func with a timeout. If it fails, raise a built-in exception only."""
        result = [None]
        exc_text = [None]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception:
                # capture full remote traceback as plain text (safe to pickle)
                exc_text[0] = traceback.format_exc()

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout_s)

        if t.is_alive():
            # don't return an Exception object; raise a built-in
            raise TimeoutError(f"{func.__name__} timed out after {timeout_s:.1f}s")

        if exc_text[0] is not None:
            # raise a built-in carrying the remote traceback as text
            raise RuntimeError(f"Remote error in {func.__name__}:\n{exc_text[0]}")

        return result[0]

    def list_chunks(self, filename_prefix):
        files = look_for_files(filename_prefix)
        return files

    def merge(self, filename_prefix):
        msg = f"[ANALYSIS SERVER - - {curtime()}] Received merge file job for filename {filename_prefix}"
        self._append_log(msg)
        return merge_chunk_files(filename_prefix)


    def analyze_merged(self, dp_merged, timeout_s=600.0, parallel = False):
        msg = f"[ANALYSIS SERVER - - {curtime()}] Received analyze merged files job with timeout {timeout_s/60} min in {dp_merged['mode']} mode."
        
        with open(os.path.join(dp_merged['data_path'],'analysis_params.pkl'), 'wb') as f:
            pickle.dump(dp_merged, f)
        
        dp_merged['log'] = self._append_log
        self._append_log(msg)      
        start_time = time.time()
        
        out = self.run_with_timeout(correlate_timestamps_Kilian, timeout_s, dp_merged, parallel)
        
        
        msg = f"[ANALYSIS SERVER - - {curtime()}] Finished analyze merged files job after time {(time.time() - start_time)/60} min."
        self._append_log(msg)
        # if isinstance(out, Exception):
        #     raise out
        if dp_merged['symmetric']:
            tau_bins = np.arange(-dp_merged['tau_max']*4, dp_merged['tau_max']*4,  dp_merged['tau_binning']*4, dtype=np.int64)
        else:
            tau_bins = np.arange(dp_merged['tau_min']*4, dp_merged['tau_max']*4, dp_merged['tau_binning']*4, dtype=np.uint32)
        np.save(os.path.join(dp_merged['data_path'],"tau_bins.npy"), tau_bins)
        
        if dp_merged['debug']:
            histograms, time_per_chunk, post_selection_indices = out
            # Save the correlated data to the folder
            np.save(os.path.join(dp_merged['data_path'],"histogram.npy"), histograms)
            np.save(os.path.join(dp_merged['data_path'],"time_per_chunk.npy"), time_per_chunk)
            np.save(os.path.join(dp_merged['data_path'],"post_selection_indices.npy"), post_selection_indices)
        else: 
            histograms, time_per_chunk = out
            # Save the correlated data to the folder
            np.save(os.path.join(dp_merged['data_path'],"histogram.npy"), histograms)
            np.save(os.path.join(dp_merged['data_path'],"time_per_chunk.npy"), time_per_chunk)
        
        
        return out, tau_bins
    
    def get_chunk_countrates(self, folder, chunksize=2**11):
        c = get_countrates_chunk(folder, chunksize, chunk_bins=None,
                     log = self._append_log)
        np.save(os.path.join(folder,f"time_per_chunk_chunksize_2^{int(np.log2(chunksize)):d}.npy"), c)
        return c
    
    def get_existing_chunk_countrates(self, folder, chunksize_power):
        c = np.load(os.path.join(folder,f"time_per_chunk_chunksize_2^{int(chunksize_power):d}.npy"))
        return c


    def read_existing(self,folder):
        self._append_log(f"[ANALYSIS - {curtime()}] Reading file from folder {folder}")
        h_dat = np.load(os.path.join(folder,"histogram.npy"))
        tpc_dat = np.load(os.path.join(folder,"time_per_chunk.npy"))
        tau_bins = np.load(os.path.join(folder,"tau_bins.npy"))

        with open(os.path.join(folder,'analysis_params.pkl'),'rb') as f:
            loaded_analysis_dict = pickle.load(f)

        try:
            psi_dat = np.load(os.path.join(folder,"post_selection_indices.npy"))
            return h_dat, tpc_dat, psi_dat, tau_bins, loaded_analysis_dict
        except:
            self._append_log(f"[ANALYSIS - {curtime()}] No Post selection indices file found, returning histogram and time_per_chunk files.")
            return h_dat, tpc_dat, tau_bins, loaded_analysis_dict
        
    def get_tacq_existing(self, metadata_path,appendix = None):
        
        tacq = None
        if appendix is not None: 
            filename = f"metadata_{appendix}.txt"
        else:
            filename = "metadata.txt"
        with open(os.path.join(metadata_path,), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = [part.strip() for part in line.split(":", 1)]
                if key == "tacq":
                    tacq = float(val)
                    break
        if tacq is not None: 
            return tacq
        else: 
            raise ValueError("No metadata dictionary found with key 'tacq'")

    def get_all_metadata(self, folder):
        """
        Reads all metadata_{index}.txt files in a folder.
        Returns a dict mapping index -> metadata_dict.
        Automatically converts numeric and list-like values.
        """
        metadata_files = sorted(glob.glob(os.path.join(folder, "metadata_*.txt")))
        if not metadata_files:
            metadata_files = sorted(glob.glob(os.path.join(folder, "metadata.txt")))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata_*.txt files found in {folder}")

        all_metadata = {}
        for path in metadata_files:
            name = os.path.basename(path)
            idx_str = name.split("metadata_")[-1].split(".txt")[0]
            try:
                idx = int(idx_str)
            except ValueError:
                idx = idx_str  # fallback if non-numeric suffix

            mdict = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    key, val = [part.strip() for part in line.split(":", 1)]
                    # try to parse numeric or list values safely
                    try:
                        parsed_val = ast.literal_eval(val)
                    except Exception:
                        parsed_val = val
                    mdict[key] = parsed_val

            all_metadata[idx] = mdict

        return all_metadata
    
if __name__ == "__main__":
    svc = AnalysisService()
    daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9091)
    uri = daemon.register(svc, objectId="Analysis_g13")
    print("Analysis server URI:", uri)
    daemon.requestLoop()
    