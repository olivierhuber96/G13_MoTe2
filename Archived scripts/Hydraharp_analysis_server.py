import os
import time
import threading
import pickle
import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from time import gmtime, strftime

import numpy as np
import Pyro4
from numba import njit

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}


def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())


# HydraHarp 400 T2 records use a 25-bit timetag at the base resolution.
T2WRAPAROUND = 33554432
BASE_RES = 4e-12


def decode_words(words):
    """Decode HydraHarp T2 uint32 records."""
    special = ((words >> 31) & 0x1).astype(np.bool_, copy=False)
    channels = ((words >> 25) & 0x3F).astype(np.uint8, copy=False)
    timetags = (words & 0x1FFFFFF).astype(np.uint32, copy=False)
    is_overflow = special & (channels == 63)
    return special, channels, timetags, is_overflow


@njit(nogil=True, fastmath=True)
def correlate_numba(ref_times, stop_times, tau_min, tau_max, bin_width, hist):
    """
    Count pairs with tau_min < (stop - ref) <= tau_max.
    ref_times and stop_times must be sorted ascending uint64 arrays.
    """
    nbins = hist.shape[0]
    n_stop = stop_times.size
    j_ge = 0

    for i in range(ref_times.size):
        t0 = ref_times[i]

        while j_ge < n_stop and stop_times[j_ge] < t0 + tau_min:
            j_ge += 1

        tmax = t0 + tau_max
        k = j_ge
        while k < n_stop:
            t1 = stop_times[k]
            if t1 > tmax:
                break
            dt = t1 - t0 - tau_min
            if dt != 0:
                b = dt // bin_width
                if b < nbins:
                    hist[b] += 1
            k += 1


def _append_block_photons(channels, times, time_lists):
    photon_mask = channels < 63
    if not photon_mask.any():
        return 0

    photon_channels = channels[photon_mask]
    photon_times = times[photon_mask]
    n_photons = int(photon_times.size)

    for channel in np.unique(photon_channels):
        channel = int(channel)
        time_lists.setdefault(channel, []).append(photon_times[photon_channels == channel])

    return n_photons


def read_hydra_photon_times(filepath, total_records, block_records, log=None):
    """
    Stream-decode a HydraHarp T2 .b file.

    Returns a dict channel -> sorted uint64 timestamps. Special records are not
    included. Overflow records increment by their timetag field, matching the
    HydraHarp TTTR format.
    """
    memmap = np.memmap(filepath, dtype=np.uint32, mode="r", shape=(total_records,))
    time_lists = {}
    overflow_total = np.uint64(0)
    total_photons = 0
    overflow_records = 0

    t0 = time.time()
    for rec_start in range(0, total_records, block_records):
        rec_stop = min(total_records, rec_start + block_records)
        words = memmap[rec_start:rec_stop]
        special, channels, timetags, is_overflow = decode_words(words)

        overflow_increments = np.where(is_overflow, timetags, 0).astype(np.uint64)
        local_overflows = np.cumsum(overflow_increments, dtype=np.uint64)
        unwrapped = timetags.astype(np.uint64) + (overflow_total + local_overflows) * np.uint64(T2WRAPAROUND)

        normal_mask = ~special
        if normal_mask.any():
            normal_channels = channels[normal_mask]
            normal_times = unwrapped[normal_mask]
            total_photons += _append_block_photons(normal_channels, normal_times, time_lists)

        if local_overflows.size:
            overflow_total += local_overflows[-1]
        overflow_records += int(is_overflow.sum())

        if log and (rec_stop == total_records or rec_stop // block_records != rec_start // block_records):
            done = 100.0 * rec_stop / total_records
            log(f"\r[decode] {done:5.1f}% ({rec_stop}/{total_records}) records")

    result = {}
    for channel, chunks in time_lists.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks).astype(np.uint64, copy=False)
        arr.sort()
        result[channel] = arr

    if log:
        counts = ", ".join(f"ch{ch}={arr.size}" for ch, arr in sorted(result.items()))
        log(f"Decoded {total_photons} photons in {time.time() - t0:.1f}s; overflow records={overflow_records}; {counts}")

    return result


def select_mode_times(times_by_channel, mode):
    if mode == "X01":
        return times_by_channel.get(0, np.empty(0, dtype=np.uint64)), times_by_channel.get(1, np.empty(0, dtype=np.uint64))
    if mode == "X10":
        return times_by_channel.get(1, np.empty(0, dtype=np.uint64)), times_by_channel.get(0, np.empty(0, dtype=np.uint64))
    if mode == "A0":
        t = times_by_channel.get(0, np.empty(0, dtype=np.uint64))
        return t, t
    if mode == "A1":
        t = times_by_channel.get(1, np.empty(0, dtype=np.uint64))
        return t, t
    if mode == "all":
        chunks = [
            times_by_channel[channel]
            for channel in (0, 1)
            if channel in times_by_channel
        ]
        if not chunks:
            t = np.empty(0, dtype=np.uint64)
        else:
            t = np.concatenate(chunks).astype(np.uint64, copy=False)
            t.sort()
        return t, t
    raise ValueError(f"Unknown mode: {mode}")


def correlate_sorted_parallel(ref_times, stop_times, tau_min, tau_max, bin_width, nbins, n_workers, log=None):
    hist_accum = np.zeros(nbins, dtype=np.uint64)
    if ref_times.size == 0 or stop_times.size == 0:
        return hist_accum

    chunks = [chunk for chunk in np.array_split(ref_times, n_workers) if chunk.size]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        for chunk in chunks:
            futures.append(ex.submit(_correlate_chunk, chunk, stop_times, tau_min, tau_max, bin_width, nbins))

        for i, fut in enumerate(as_completed(futures), 1):
            hist_accum += fut.result()
            if log and (i % 2 == 0 or i == len(futures)):
                frac = i / len(futures)
                elapsed = time.time() - t0
                eta = elapsed / frac - elapsed if frac else 0.0
                log(f"\r[correlate] {100.0 * frac:5.1f}% done, elapsed {elapsed:.1f}s ETA {eta:.1f}s")

    return hist_accum


def _correlate_chunk(ref_chunk, stop_times, tau_min, tau_max, bin_width, nbins):
    hist = np.zeros(nbins, dtype=np.uint64)
    correlate_numba(ref_chunk, stop_times, np.uint64(tau_min), np.uint64(tau_max), np.uint64(bin_width), hist)
    return hist


def _next_pow_two(n):
    return 1 << (int(n) - 1).bit_length()


def compute_classical_autocorrelation(counts, dt_s):
    counts = np.asarray(counts, dtype=np.float64)
    nonzero = counts > 0
    if not nonzero.any():
        return {
            "lag_s": np.array([], dtype=np.float64),
            "lag_min": np.array([], dtype=np.float64),
            "autocorr": np.array([], dtype=np.float64),
            "counts_start_index": np.array(-1, dtype=np.int64),
            "counts_stop_index": np.array(-1, dtype=np.int64),
        }

    start = int(np.argmax(nonzero))
    stop = int(len(nonzero) - np.argmax(nonzero[::-1]))
    signal = counts[start:stop]
    mean = float(np.mean(signal))
    if signal.size < 2 or mean == 0.0:
        autocorr = np.ones(1, dtype=np.float64)
        lag_s = np.zeros(1, dtype=np.float64)
    else:
        signal = (signal - mean) / mean
        n = signal.size
        nfft = _next_pow_two(2 * n - 1)
        spectrum = np.fft.rfft(signal, n=nfft)
        raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:n]
        norm = np.arange(n, 0, -1, dtype=np.float64)
        autocorr = raw / norm
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]
        lag_s = np.arange(n, dtype=np.float64) * float(dt_s)

    return {
        "lag_s": lag_s,
        "lag_min": lag_s / 60.0,
        "autocorr": autocorr,
        "counts_start_index": np.array(start, dtype=np.int64),
        "counts_stop_index": np.array(stop, dtype=np.int64),
    }


def save_classical_autocorrelation(out_dir, base, tau_bin_ms, counts, dt_s):
    autocorr = compute_classical_autocorrelation(counts, dt_s)
    path = os.path.join(out_dir, f"{base}_classical_autocorr_dt{tau_bin_ms}ms.npz")
    np.savez_compressed(
        path,
        lag_s=autocorr["lag_s"],
        lag_min=autocorr["lag_min"],
        autocorr=autocorr["autocorr"],
        counts_start_index=autocorr["counts_start_index"],
        counts_stop_index=autocorr["counts_stop_index"],
        dt_s=np.array(float(dt_s), dtype=np.float64),
        tau_bin_ms=np.array(float(tau_bin_ms), dtype=np.float64),
    )
    return path


def read_classical_autocorrelation(folder, tau_bin_ms=None, tol=1e-12):
    candidates = glob.glob(os.path.join(folder, "*_classical_autocorr_dt*ms.npz"))
    if tau_bin_ms is not None:
        filtered = []
        for path in candidates:
            stem = os.path.basename(path)[:-4]
            pos_dt = stem.rfind("_dt")
            pos_ms = stem.rfind("ms")
            if pos_dt == -1 or pos_ms == -1 or pos_ms <= pos_dt:
                continue
            try:
                dt_val = float(stem[pos_dt + len("_dt"):pos_ms])
            except ValueError:
                continue
            if abs(dt_val - float(tau_bin_ms)) <= tol:
                filtered.append(path)
        candidates = filtered

    if not candidates:
        return None

    path = sorted(candidates)[0]
    with np.load(path, allow_pickle=False) as z:
        return {
            "path": path,
            "lag_s": z["lag_s"],
            "lag_min": z["lag_min"],
            "autocorr": z["autocorr"],
            "counts_start_index": z["counts_start_index"],
            "counts_stop_index": z["counts_stop_index"],
            "dt_s": float(z["dt_s"]),
            "tau_bin_ms": float(z["tau_bin_ms"]),
        }


@Pyro4.expose
class AnalysisService(object):
    def __init__(self):
        self._log = deque(maxlen=10000)
        self._cv = threading.Condition()

    def _append_log(self, msg):
        with self._cv:
            self._log.append(msg)
            self._cv.notify_all()

    def get_logs(self, last_idx=0, timeout=5.0):
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
        self._append_log("HydraHarp analysis server reached.")

    def convert_to_ticks(self, tau_min_us, tau_max_us, tau_bin_ps):
        bin_width_ticks = np.uint64(max(1, round((tau_bin_ps * 1e-12) / BASE_RES)))
        tau_min_ticks = np.uint64(round((tau_min_us * 1e-6) / BASE_RES))
        tau_max_ticks = np.uint64(round((tau_max_us * 1e-6) / BASE_RES))
        if tau_max_ticks <= tau_min_ticks:
            raise ValueError("tau_max_us must be larger than tau_min_us")

        nbins = int(((tau_max_ticks - tau_min_ticks) + bin_width_ticks - np.uint64(1)) // bin_width_ticks)
        idx = np.arange(nbins, dtype=np.uint64)
        ticks = idx * bin_width_ticks
        tau = ticks.astype(np.float64) * np.float64(BASE_RES)
        return tau, tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins

    def correlate_b_file_streaming(self, filepath, mode="X01", chunk_ms=10.0,
                                   tau_min_us=0.0, tau_max_us=10.0, tau_bin_ps=16.0,
                                   n_workers=None, show_progress=True, block_records=10_000_000):
        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("Empty file")
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        tau, tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins = self.convert_to_ticks(
            tau_min_us, tau_max_us, tau_bin_ps
        )

        analysis_params = {
            "filepath": filepath,
            "record_format": "HydraHarp400_T2",
            "mode": mode,
            "chunk_ms": chunk_ms,
            "tau_min_us": tau_min_us,
            "tau_max_us": tau_max_us,
            "tau_bin_ps": tau_bin_ps,
            "n_workers": n_workers,
            "BASE_RES": BASE_RES,
            "T2WRAPAROUND": T2WRAPAROUND,
        }

        if show_progress:
            self._append_log(
                f"[ANALYSIS - {curtime()}] HydraHarp file {filepath}: {total_records} records\n"
                f"mode={mode}, tau_min={tau_min_us}us, tau_max={tau_max_us}us, bin={tau_bin_ps}ps\n"
                f"ticks: min={tau_min_ticks}, max={tau_max_ticks}, bin={bin_width_ticks}, nbins={nbins}"
            )

        log = self._append_log if show_progress else None
        times_by_channel = read_hydra_photon_times(filepath, total_records, block_records, log=log)
        ref_times, stop_times = select_mode_times(times_by_channel, mode)

        if show_progress:
            self._append_log(f"Selected mode={mode}: ref={ref_times.size}, stop={stop_times.size}; using {n_workers} workers.")

        t0 = time.time()
        hist_accum = correlate_sorted_parallel(
            ref_times, stop_times, tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins, n_workers, log=log
        )
        if show_progress:
            self._append_log(f"Correlation done in {time.time() - t0:.1f}s; coincidences={int(hist_accum.sum())}")

        base = os.path.basename(filepath)[:-2]
        folder = os.path.dirname(filepath)
        np.save(os.path.join(folder, f"histogram{base}_{mode}.npy"), hist_accum)
        np.save(os.path.join(folder, f"tau_bins{base}_{mode}.npy"), tau + tau_min_us * 1e-6)
        with open(os.path.join(folder, f"analysis_params{base}_{mode}.pkl"), "wb") as f:
            pickle.dump(analysis_params, f)

        self._append_log("Correlation done.")
        return tau + tau_min_us * 1e-6, hist_accum, total_records

    def correlate_b_file_highperf(self, filepath, mode="X01", chunk_ms=10.0, tau_min_us=0.0,
                                  tau_max_us=10.0, tau_bin_ps=16.0, n_workers=None,
                                  only_n_events=False, show_progress=True):
        if only_n_events:
            return (), (), os.path.getsize(filepath) // 4
        return self.correlate_b_file_streaming(
            filepath=filepath,
            mode=mode,
            chunk_ms=chunk_ms,
            tau_min_us=tau_min_us,
            tau_max_us=tau_max_us,
            tau_bin_ps=tau_bin_ps,
            n_workers=n_workers,
            show_progress=show_progress,
        )

    def correlate_b_file_classical(self, filepath, tau_bin_ms=0.01, chunk_minutes=30,
                                   block_records=10_000_000, include_channels=None,
                                   show_progress=True, compute_autocorr=True):
        """
        Bin HydraHarp photon arrivals versus measurement time.

        Writes compressed NPZ chunks next to the .b file in
        <basename>_classical_bins and returns (out_dir, out_files).
        """
        dt_s = float(tau_bin_ms) * 1e-3
        dt_ticks = int(round(dt_s / BASE_RES))
        if dt_ticks <= 0:
            dt_ticks = 1

        chunk_s = float(chunk_minutes) * 60.0
        chunk_ticks = np.uint64(max(1, round(chunk_s / BASE_RES)))
        n_bins_chunk = int(max(1, np.ceil(chunk_s / dt_s)))

        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("Empty file")

        base = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = os.path.join(os.path.dirname(filepath), f"{base}_classical_bins")
        os.makedirs(out_dir, exist_ok=True)

        analysis_params = {
            "filepath": filepath,
            "record_format": "HydraHarp400_T2",
            "tau_bin_ms": float(tau_bin_ms),
            "chunk_minutes": float(chunk_minutes),
            "block_records": int(block_records),
            "BASE_RES": float(BASE_RES),
            "T2WRAPAROUND": int(T2WRAPAROUND),
            "include_channels": include_channels,
            "compute_autocorr": bool(compute_autocorr),
        }
        with open(os.path.join(out_dir, "analysis_params_classical.pkl"), "wb") as f:
            pickle.dump(analysis_params, f)

        if show_progress:
            self._append_log(f"[classical] HydraHarp file {filepath}: {total_records} records")
            self._append_log(f"[classical] dt={tau_bin_ms} ms, chunk={chunk_minutes} min, bins/chunk={n_bins_chunk}")

        def save_chunk(chunk_idx, counts, t0_abs_ticks):
            fn = os.path.join(out_dir, f"{base}_bins_dt{tau_bin_ms}ms_chunk{chunk_idx:05d}.npz")
            np.savez_compressed(
                fn,
                counts=counts.astype(np.uint32, copy=False),
                dt_s=np.array(dt_s, dtype=float),
                chunk_minutes=np.array(chunk_minutes, dtype=float),
                chunk_index=np.array(chunk_idx, dtype=np.int64),
                t0_abs_ticks=np.array(np.uint64(t0_abs_ticks), dtype=np.uint64),
                wrap_ticks=np.array(T2WRAPAROUND, dtype=np.uint64),
            )
            return fn

        memmap = np.memmap(filepath, dtype=np.uint32, mode="r", shape=(total_records,))
        overflow_total = np.uint64(0)
        include_arr = None if include_channels is None else np.array(include_channels, dtype=np.uint8)

        chunk_idx = 0
        have_open_chunk = False
        t0_chunk = np.uint64(0)
        t1_chunk = np.uint64(0)
        counts = None
        out_files = []
        all_counts = []
        t_start = time.time()

        for rec_start in range(0, total_records, block_records):
            rec_stop = min(total_records, rec_start + block_records)
            words = memmap[rec_start:rec_stop]
            special, channels, timetags, is_overflow = decode_words(words)

            overflow_increments = np.where(is_overflow, timetags, 0).astype(np.uint64)
            local_overflows = np.cumsum(overflow_increments, dtype=np.uint64)
            unwrapped = timetags.astype(np.uint64) + (overflow_total + local_overflows) * np.uint64(T2WRAPAROUND)
            if local_overflows.size:
                overflow_total += local_overflows[-1]

            photon_mask = ~special
            if include_arr is not None:
                photon_mask &= np.isin(channels, include_arr)
            if not photon_mask.any():
                continue

            times = unwrapped[photon_mask]
            times.sort()

            if not have_open_chunk:
                t0_chunk = np.uint64(times[0])
                t1_chunk = t0_chunk + chunk_ticks
                counts = np.zeros(n_bins_chunk, dtype=np.uint32)
                have_open_chunk = True

            p = 0
            n_times = times.size
            while p < n_times:
                if times[p] < t0_chunk:
                    p = int(np.searchsorted(times, t0_chunk, side="left"))
                    if p >= n_times:
                        break

                while times[p] >= t1_chunk:
                    out_files.append(save_chunk(chunk_idx, counts, t0_chunk))
                    all_counts.append(counts.copy())
                    chunk_idx += 1
                    t0_chunk = t1_chunk
                    t1_chunk = t0_chunk + chunk_ticks
                    counts[:] = 0

                q = int(np.searchsorted(times, t1_chunk, side="left"))
                if q > p:
                    dticks = times[p:q] - t0_chunk
                    idx = (dticks // dt_ticks).astype(np.int64, copy=False)
                    idx = idx[(idx >= 0) & (idx < n_bins_chunk)]
                    if idx.size:
                        bincount = np.bincount(idx, minlength=n_bins_chunk)
                        counts += bincount.astype(np.uint32, copy=False)
                p = q

            if show_progress:
                done = 100.0 * rec_stop / total_records
                self._append_log(f"\r[classical] processed {done:5.1f}% ({rec_stop}/{total_records}) records")

        if have_open_chunk:
            out_files.append(save_chunk(chunk_idx, counts, t0_chunk))
            all_counts.append(counts.copy())

        if compute_autocorr and all_counts:
            if show_progress:
                self._append_log("[classical] computing FFT autocorrelation of classical bins...")
            concatenated_counts = np.concatenate(all_counts)
            autocorr_path = save_classical_autocorrelation(out_dir, base, tau_bin_ms, concatenated_counts, dt_s)
            if show_progress:
                self._append_log(f"[classical] saved autocorrelation: {autocorr_path}")

        if show_progress:
            self._append_log(f"[classical] done in {time.time() - t_start:.1f}s. Wrote {len(out_files)} chunks to: {out_dir}")

        return out_dir, out_files

    def read_existing_classical(self, folder, chunk_range=None, tau_bin_ms=None, tol=1e-12,
                                include_channel=None, return_autocorr=False):
        candidates = glob.glob(os.path.join(folder, "*_bins_dt*ms_chunk*.npz"))
        items = []
        for path in candidates:
            base = os.path.basename(path)
            stem = base[:-4]
            pos_chunk = stem.rfind("chunk")
            if pos_chunk == -1:
                continue
            idx_str = stem[pos_chunk + len("chunk"):]
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)

            if tau_bin_ms is not None:
                pos_dt = stem.rfind("_bins_dt")
                pos_ms = stem.rfind("ms_chunk")
                if pos_dt == -1 or pos_ms == -1 or pos_ms <= pos_dt:
                    continue
                try:
                    dt_val = float(stem[pos_dt + len("_bins_dt"):pos_ms])
                except ValueError:
                    continue
                if abs(dt_val - float(tau_bin_ms)) > tol:
                    continue

            items.append((idx, path))

        if not items:
            base_return = ([], [], [], [], [], [], None, None)
            if return_autocorr:
                return base_return + (read_classical_autocorrelation(folder, tau_bin_ms=tau_bin_ms, tol=tol),)
            return base_return

        items.sort(key=lambda item: item[0])
        if chunk_range is not None:
            lo, hi = chunk_range
            items = [(idx, path) for idx, path in items if lo <= idx <= hi]
            if not items:
                base_return = ([], [], [], [], [], [], None, None)
                if return_autocorr:
                    return base_return + (read_classical_autocorrelation(folder, tau_bin_ms=tau_bin_ms, tol=tol),)
                return base_return

        counts_list = []
        dt_s_list = []
        chunk_minutes_list = []
        chunk_index_list = []
        t0_abs_ticks_list = []
        wrap_ticks_list = []
        for idx, path in items:
            with np.load(path, allow_pickle=False) as z:
                counts_list.append(z["counts"])
                dt_s_list.append(float(z["dt_s"]))
                chunk_minutes_list.append(float(z["chunk_minutes"]))
                chunk_index_list.append(int(z["chunk_index"]))
                t0_abs_ticks_list.append(z["t0_abs_ticks"])
                wrap_ticks_list.append(z["wrap_ticks"])

        idxs = [idx for idx, _ in items]
        base_return = (counts_list, dt_s_list, chunk_minutes_list, chunk_index_list,
                       t0_abs_ticks_list, wrap_ticks_list, min(idxs), max(idxs))
        if return_autocorr:
            return base_return + (read_classical_autocorrelation(folder, tau_bin_ms=tau_bin_ms, tol=tol),)
        return base_return

    def read_existing_classical_autocorr(self, folder, tau_bin_ms=None, tol=1e-12):
        return read_classical_autocorrelation(folder, tau_bin_ms=tau_bin_ms, tol=tol)

    def read_existing(self, folder, name, mode="X01"):
        def _name_part_from_hist(base, mode):
            return base[len("histogram"):-len(f"_{mode}.npy")]

        def _strip_filepath(d):
            d = dict(d)
            d.pop("filepath", None)
            return d

        if name == "all":
            patt = os.path.join(folder, f"histogram*_{mode}.npy")
            h_files = glob.glob(patt)
            if not h_files:
                raise FileNotFoundError(f"No files matching {patt}")

            items = []
            for hpath in h_files:
                base = os.path.basename(hpath)
                if not (base.startswith("histogram") and base.endswith(f"_{mode}.npy")):
                    continue
                name_part = _name_part_from_hist(base, mode)
                tpath = os.path.join(folder, f"tau_bins{name_part}_{mode}.npy")
                ppath = os.path.join(folder, f"analysis_params{name_part}_{mode}.pkl")
                if os.path.exists(tpath) and os.path.exists(ppath):
                    items.append((name_part, hpath, tpath, ppath))

            if not items:
                raise FileNotFoundError(f"No complete (hist, tau, params) triples found in {folder} for mode {mode}")

            name0, h0p, t0p, p0p = items[0]
            h0 = np.load(h0p)
            tau0 = np.load(t0p)
            with open(p0p, "rb") as f:
                p0 = pickle.load(f)
            p0_norm = _strip_filepath(p0)

            h_sum = np.zeros_like(h0, dtype=np.uint64)
            for nm, hp, tp, pp in items:
                h = np.load(hp)
                tau = np.load(tp)
                with open(pp, "rb") as f:
                    p = pickle.load(f)
                if h.shape != h0.shape:
                    raise ValueError(f"Incompatible histogram shapes for {nm}: {h.shape} vs {h0.shape}")
                if not np.array_equal(tau, tau0):
                    raise ValueError(f"Incompatible tau bins for {nm}")
                if _strip_filepath(p) != p0_norm:
                    raise ValueError(f"Analysis parameters not identical (except filepath) for {nm}")
                h_sum += h.astype(np.uint64, copy=False)

            self._append_log(f"[ANALYSIS - {curtime()}] Combined {len(items)} chunks in mode {mode}")
            return h_sum, tau0, p0

        if isinstance(name, str):
            self._append_log(f"[ANALYSIS - {curtime()}] Reading file with name {name} in mode {mode} from folder {folder}")
            h_dat = np.load(os.path.join(folder, f"histogram{name}_{mode}.npy"))
            tau_bins = np.load(os.path.join(folder, f"tau_bins{name}_{mode}.npy"))
            with open(os.path.join(folder, f"analysis_params{name}_{mode}.pkl"), "rb") as f:
                loaded_analysis_dict = pickle.load(f)
            return h_dat, tau_bins, loaded_analysis_dict

        if isinstance(name, list) and all(isinstance(n, str) for n in name):
            self._append_log(f"[ANALYSIS - {curtime()}] Reading files with names {name} in mode {mode} from folder {folder}")
            h_dat = []
            tau_bins = []
            loaded_analysis_dicts = []
            for n in name:
                h_dat.append(np.load(os.path.join(folder, f"histogram{n}_{mode}.npy")))
                tau_bins.append(np.load(os.path.join(folder, f"tau_bins{n}_{mode}.npy")))
                with open(os.path.join(folder, f"analysis_params{n}_{mode}.pkl"), "rb") as f:
                    loaded_analysis_dicts.append(pickle.load(f))
            return h_dat, tau_bins, loaded_analysis_dicts

        raise ValueError("Value Error in read_existing: name argument neither a string nor a list of strings!")


if __name__ == "__main__":
    svc = AnalysisService()
    daemon = Pyro4.Daemon(host="g13-sspd.dhcp-int.phys.ethz.ch", port=9099)
    uri = daemon.register(svc, objectId="Analysis_g13")
    print("HydraHarp analysis server URI:", uri)
    daemon.requestLoop()
