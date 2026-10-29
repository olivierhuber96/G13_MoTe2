import glob
import json
import os
import pickle
import threading
import time
from collections import deque
from time import gmtime, strftime

import numpy as np
import Pyro4

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.REQUIRE_EXPOSE = False


def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())


# PicoHarp 300 T2 records use a 28-bit timetag at the base resolution.
T2WRAPAROUND = 210698240
BASE_RES = 4e-12


def decode_words(words):
    """Decode PicoHarp T2 uint32 records."""
    channels = ((words >> 28) & 0xF).astype(np.uint8, copy=False)
    timetags = (words & 0x0FFFFFFF).astype(np.uint32, copy=False)
    is_special = channels == 0xF
    low4 = timetags & 0xF
    is_overflow = is_special & (low4 == 0)
    return is_special, channels, timetags, is_overflow


def metadata_path_for(filepath):
    root, _ = os.path.splitext(filepath)
    return root + "_metadata.json"


def load_metadata(filepath):
    metadata_path = metadata_path_for(filepath)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def period_from_metadata(metadata):
    meta = metadata["metadata"]
    pulse_metadata = meta.get("pulse_metadata", {})
    pulse_parameters = meta.get("pulse_parameters", {})
    for key_source in (
        (pulse_metadata, "burst_period"),
        (pulse_parameters, "burst_period"),
        (meta, "folding_period_s"),
        (pulse_metadata, "cycle_period"),
    ):
        source, key = key_source
        if key in source and source[key] is not None:
            return float(source[key])
    raise KeyError("Could not find burst_period, folding_period_s, or cycle_period in metadata")


def folded_output_folder(filepath):
    root, _ = os.path.splitext(filepath)
    return root + "_pump_probe_folded"


def folded_filename(base, period_s, phase_bin_us, block_s, include_channels, phase_offset_s):
    block_part = "noblocks" if block_s is None else f"block{float(block_s):.9g}s"
    if include_channels is None:
        channel_part = "allch"
    else:
        channel_part = "ch" + "-".join(str(int(ch)) for ch in include_channels)
    return (
        f"{base}_fold_period{float(period_s) * 1e6:.9g}us_"
        f"bin{float(phase_bin_us):.9g}us_{block_part}_"
        f"{channel_part}_offset{float(phase_offset_s) * 1e6:.9g}us.npz"
    )


def _normalize_time_ranges(ranges):
    if ranges is None:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(ranges, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = arr.reshape(-1, 2)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Measurement time ranges must be finite")
    if np.any(arr[:, 1] <= arr[:, 0]):
        raise ValueError("Measurement time ranges must be [start, stop] with stop > start")
    return arr[np.argsort(arr[:, 0])]


def _safe_label_token(label):
    if label is None:
        return ""
    label = str(label).strip()
    if not label:
        return ""
    safe = []
    for ch in label:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "_" + "".join(safe)[:80]


def _time_filter_token(include_ranges_s, exclude_ranges_s, label=None):
    label_token = _safe_label_token(label)
    if label_token:
        return label_token
    parts = []
    if include_ranges_s.size:
        parts.append(
            "include"
            + "-".join(f"{start:.9g}s_to_{stop:.9g}s" for start, stop in include_ranges_s)
        )
    if exclude_ranges_s.size:
        parts.append(
            "exclude"
            + "-".join(f"{start:.9g}s_to_{stop:.9g}s" for start, stop in exclude_ranges_s)
        )
    return "" if not parts else "_" + "_".join(parts)


def _intervals_overlap_ranges(start_s, stop_s, ranges_s):
    if ranges_s.size == 0:
        return np.zeros_like(start_s, dtype=bool)
    overlap = np.zeros_like(start_s, dtype=bool)
    for range_start, range_stop in ranges_s:
        overlap |= (start_s < range_stop) & (stop_s > range_start)
    return overlap


def _points_in_ranges(t_s, ranges_s):
    if ranges_s.size == 0:
        return np.ones_like(t_s, dtype=bool)
    inside = np.zeros_like(t_s, dtype=bool)
    for range_start, range_stop in ranges_s:
        inside |= (t_s >= range_start) & (t_s < range_stop)
    return inside


def sync_folded_filename(base, sync_channel, signal_channels, bin_us, sync_tick_stride=1,
                         include_ranges_s=None, exclude_ranges_s=None, measurement_filter_label=None):
    signal_part = "ch" + "-".join(str(int(ch)) for ch in signal_channels)
    stride_part = "" if int(sync_tick_stride) == 1 else f"_stride{int(sync_tick_stride)}"
    include_ranges_s = _normalize_time_ranges(include_ranges_s)
    exclude_ranges_s = _normalize_time_ranges(exclude_ranges_s)
    time_part = _time_filter_token(include_ranges_s, exclude_ranges_s, measurement_filter_label)
    return (
        f"{base}_syncfold_syncch{int(sync_channel)}_{signal_part}_"
        f"ch1bounded{stride_part}{time_part}_bin{float(bin_us):.9g}us.npz"
    )


def count_rate_filename(base, channels, bin_s):
    if channels is None:
        channel_part = "allch"
    else:
        channel_part = "ch" + "-".join(str(int(ch)) for ch in channels)
    return f"{base}_countrate_{channel_part}_bin{float(bin_s):.9g}s.npz"


def _unwrap_t2_block(words, overflow_total):
    is_special, channels, timetags, is_overflow = decode_words(words)
    is_wrap = is_overflow.astype(np.uint8)
    if is_wrap.size:
        cum_wraps = np.cumsum(is_wrap, dtype=np.uint64)
        prefix_wraps = cum_wraps - is_wrap.astype(np.uint64)
        block_wraps = cum_wraps[-1]
    else:
        prefix_wraps = np.zeros(0, dtype=np.uint64)
        block_wraps = np.uint64(0)

    unwrapped = timetags.astype(np.uint64) + (overflow_total + prefix_wraps) * np.uint64(T2WRAPAROUND)
    return is_special, channels, unwrapped, is_overflow, block_wraps


@Pyro4.expose
class PumpProbeAnalysisService(object):
    def __init__(self):
        self._logs = deque(maxlen=5000)
        self._cond = threading.Condition()

    def _append_log(self, msg):
        with self._cond:
            self._logs.append(str(msg))
            self._cond.notify_all()

    @Pyro4.expose
    def get_logs(self, last_idx=0, timeout=5.0):
        end = time.time() + timeout
        with self._cond:
            while len(self._logs) <= last_idx:
                remaining = end - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            lines = list(self._logs)[last_idx:]
            return last_idx + len(lines), lines

    @Pyro4.expose
    def ping(self):
        self._append_log("PicoHarp pump-probe analysis server reached.")
        return True

    @Pyro4.expose
    def count_rate_over_time(self, filepath, bin_s=0.1, channels=None,
                             block_records=1_000_000, force_recompute=False,
                             show_progress=True):
        """
        Bin photon counts over lab time for drift/fluctuation diagnostics.

        By default this bins CH0 in 100 ms bins. The returned NPZ contains
        time_edges_s, time_centers_s, counts, rate_hz, counts_by_channel, and
        rate_by_channel_hz.
        """
        filepath = os.path.abspath(filepath)
        metadata = load_metadata(filepath)
        bin_s = float(bin_s)
        if bin_s <= 0:
            raise ValueError("bin_s must be positive")

        if channels is None:
            channels_saved = [0]
        else:
            channels_saved = [int(ch) for ch in channels]
        channel_arr = np.array(channels_saved, dtype=np.uint8)

        base = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = folded_output_folder(filepath)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, count_rate_filename(base, channels_saved, bin_s))
        params_path = os.path.join(out_dir, "analysis_params_count_rate.pkl")
        if os.path.exists(out_path) and not force_recompute:
            self._append_log(f"[count-rate] using existing count-rate analysis: {out_path}")
            return self._summary_from_npz(out_path)

        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("empty .b file")

        params = {
            "filepath": filepath,
            "metadata_path": metadata_path_for(filepath),
            "bin_s": bin_s,
            "channels": channels_saved,
            "block_records": int(block_records),
            "created_utc": curtime(),
            "BASE_RES": float(BASE_RES),
        }
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

        self._append_log(
            f"[count-rate] start {filepath}, bin={bin_s:.6g} s, "
            f"channels={channels_saved}, records={total_records:,}"
        )

        memmap = np.memmap(filepath, dtype=np.uint32, mode="r", shape=(total_records,))
        bin_ticks = max(1, int(round(bin_s / BASE_RES)))
        overflow_total = np.uint64(0)
        first_tick = None
        last_tick = None
        counts_by_channel = np.zeros((len(channels_saved), 0), dtype=np.uint64)
        total_counts = np.zeros(0, dtype=np.uint64)
        channel_counts = {int(ch): 0 for ch in channels_saved}
        included_photons = 0
        overflow_records = 0
        processed = 0
        t0 = time.time()

        def ensure_bins(n_bins):
            nonlocal counts_by_channel, total_counts
            if n_bins <= total_counts.size:
                return
            grow_total = np.zeros(n_bins, dtype=np.uint64)
            grow_total[:total_counts.size] = total_counts
            total_counts = grow_total

            grow_by_channel = np.zeros((len(channels_saved), n_bins), dtype=np.uint64)
            if counts_by_channel.size:
                grow_by_channel[:, :counts_by_channel.shape[1]] = counts_by_channel
            counts_by_channel = grow_by_channel

        for rec_start in range(0, total_records, block_records):
            rec_stop = min(total_records, rec_start + block_records)
            words = memmap[rec_start:rec_stop]
            is_special, channels_decoded, unwrapped, is_overflow, block_wraps = _unwrap_t2_block(words, overflow_total)
            overflow_total += block_wraps
            overflow_records += int(np.count_nonzero(is_overflow))

            photon_mask = ~is_special
            if not np.any(photon_mask):
                processed = rec_stop
                continue

            photon_channels = channels_decoded[photon_mask]
            photon_ticks = unwrapped[photon_mask]
            selected_mask = np.isin(photon_channels, channel_arr)
            if np.any(selected_mask):
                selected_channels = photon_channels[selected_mask]
                selected_ticks = photon_ticks[selected_mask]
                if first_tick is None:
                    first_tick = int(selected_ticks[0])

                last_tick = int(selected_ticks[-1])
                relative_ticks = selected_ticks - np.uint64(first_tick)
                bin_idx = (relative_ticks // np.uint64(bin_ticks)).astype(np.int64)
                n_bins = int(bin_idx.max()) + 1
                ensure_bins(n_bins)

                total_counts[:n_bins] += np.bincount(bin_idx, minlength=n_bins).astype(np.uint64, copy=False)
                for ch_i, ch in enumerate(channels_saved):
                    ch_mask = selected_channels == ch
                    if not np.any(ch_mask):
                        continue
                    ch_bins = bin_idx[ch_mask]
                    counts_by_channel[ch_i, :n_bins] += np.bincount(
                        ch_bins,
                        minlength=n_bins,
                    ).astype(np.uint64, copy=False)
                    n_ch = int(np.count_nonzero(ch_mask))
                    channel_counts[int(ch)] += n_ch
                    included_photons += n_ch

            processed = rec_stop
            if show_progress and (rec_start == 0 or rec_stop == total_records or time.time() - t0 > 2.0):
                self._append_log(
                    f"[count-rate] processed {processed:,}/{total_records:,} records, "
                    f"included={included_photons:,}"
                )
                t0 = time.time()

        if first_tick is None:
            raise ValueError(f"no photons found on channels {channels_saved}")

        n_bins = total_counts.size
        duration_s = max(0.0, (int(last_tick) - int(first_tick)) * BASE_RES)
        time_edges_s = np.arange(n_bins + 1, dtype=np.float64) * bin_s
        if duration_s > 0:
            time_edges_s[-1] = max(time_edges_s[-1], duration_s)
        time_centers_s = 0.5 * (time_edges_s[:-1] + time_edges_s[1:])
        bin_widths_s = np.diff(time_edges_s)
        rate_hz = total_counts.astype(np.float64) / bin_widths_s
        rate_by_channel_hz = counts_by_channel.astype(np.float64) / bin_widths_s[np.newaxis, :]

        np.savez_compressed(
            out_path,
            analysis_type=np.array("count_rate_over_time"),
            time_edges_s=time_edges_s,
            time_centers_s=time_centers_s,
            bin_widths_s=bin_widths_s,
            counts=total_counts,
            rate_hz=rate_hz,
            counts_by_channel=counts_by_channel,
            rate_by_channel_hz=rate_by_channel_hz,
            bin_s=np.array(bin_s, dtype=np.float64),
            total_records=np.array(total_records, dtype=np.int64),
            included_photons=np.array(included_photons, dtype=np.int64),
            overflow_records=np.array(overflow_records, dtype=np.int64),
            first_photon_tick=np.array(first_tick, dtype=np.uint64),
            last_photon_tick=np.array(last_tick, dtype=np.uint64),
            duration_s=np.array(duration_s, dtype=np.float64),
            channel_ids=np.array(channels_saved, dtype=np.int64),
            channel_counts=np.array([channel_counts[ch] for ch in channels_saved], dtype=np.int64),
            metadata_json=np.array(json.dumps(metadata) if metadata is not None else ""),
        )
        self._append_log(
            f"[count-rate] done {out_path}, bins={n_bins}, "
            f"included={included_photons:,}, duration={duration_s:.3f} s"
        )
        return self._summary_from_npz(out_path)

    @Pyro4.expose
    def fold_b_file_periodic(self, filepath, period_s=None, phase_bin_us=1.0, block_s=1.0,
                             include_channels=None, phase_offset_s=0.0,
                             block_records=1_000_000, force_recompute=False,
                             show_progress=True):
        """
        Fold PicoHarp T2 photon timestamps modulo a periodic pump-probe period.

        The method writes a compressed NPZ file next to the raw .b file and
        returns a small summary dictionary. If block_s is not None, it also
        writes a phase-vs-lab-time count matrix with one row per time block.
        """
        filepath = os.path.abspath(filepath)
        if period_s is None:
            metadata = load_metadata(filepath)
            if metadata is None:
                raise ValueError("period_s was not supplied and no metadata JSON was found")
            period_s = period_from_metadata(metadata)
        else:
            metadata = load_metadata(filepath)

        period_s = float(period_s)
        phase_bin_us = float(phase_bin_us)
        phase_offset_s = float(phase_offset_s)
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        if phase_bin_us <= 0:
            raise ValueError("phase_bin_us must be positive")
        if block_s is not None and float(block_s) <= 0:
            raise ValueError("block_s must be positive or None")

        period_ticks = int(round(period_s / BASE_RES))
        bin_ticks = max(1, int(round((phase_bin_us * 1e-6) / BASE_RES)))
        phase_offset_ticks = int(round(phase_offset_s / BASE_RES))
        n_phase_bins = int((period_ticks + bin_ticks - 1) // bin_ticks)
        phase_edges_s = np.arange(n_phase_bins + 1, dtype=np.float64) * bin_ticks * BASE_RES
        phase_edges_s[-1] = period_s
        phase_centers_s = 0.5 * (phase_edges_s[:-1] + phase_edges_s[1:])

        include_arr = None if include_channels is None else np.array(include_channels, dtype=np.uint8)
        include_saved = None if include_channels is None else [int(ch) for ch in include_channels]

        base = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = folded_output_folder(filepath)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            folded_filename(base, period_s, phase_bin_us, block_s, include_saved, phase_offset_s),
        )
        params_path = os.path.join(out_dir, "analysis_params_pump_probe.pkl")
        if os.path.exists(out_path) and not force_recompute:
            self._append_log(f"[pump-probe] using existing folded analysis: {out_path}")
            return self._summary_from_npz(out_path)

        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("Empty file")

        analysis_params = {
            "filepath": filepath,
            "metadata_path": metadata_path_for(filepath),
            "record_format": "PicoHarp300_T2",
            "period_s": period_s,
            "phase_bin_us": phase_bin_us,
            "block_s": None if block_s is None else float(block_s),
            "include_channels": include_saved,
            "phase_offset_s": phase_offset_s,
            "block_records": int(block_records),
            "BASE_RES": float(BASE_RES),
            "T2WRAPAROUND": int(T2WRAPAROUND),
        }
        with open(params_path, "wb") as f:
            pickle.dump(analysis_params, f)

        if show_progress:
            self._append_log(f"[pump-probe] PicoHarp file {filepath}: {total_records} records")
            self._append_log(
                f"[pump-probe] period={period_s * 1e6:.6f} us, "
                f"phase bin={phase_bin_us:.6f} us, bins={n_phase_bins}, block_s={block_s}"
            )

        phase_counts = np.zeros(n_phase_bins, dtype=np.uint64)
        counts_by_block = None
        block_edges_s = None
        block_ticks = None if block_s is None else max(1, int(round(float(block_s) / BASE_RES)))

        memmap = np.memmap(filepath, dtype=np.uint32, mode="r", shape=(total_records,))
        overflow_total = np.uint64(0)
        overflow_records = 0
        first_photon_tick = None
        last_photon_tick = None
        included_photons = 0
        channel_counts = {}
        t_start = time.time()

        for rec_start in range(0, total_records, block_records):
            rec_stop = min(total_records, rec_start + block_records)
            words = memmap[rec_start:rec_stop]
            is_special, channels, timetags, is_overflow = decode_words(words)

            is_wrap = is_overflow.astype(np.uint8)
            if is_wrap.size:
                cum_wraps = np.cumsum(is_wrap, dtype=np.uint64)
                prefix_wraps = cum_wraps - is_wrap.astype(np.uint64)
                block_wraps = cum_wraps[-1]
            else:
                prefix_wraps = np.zeros(0, dtype=np.uint64)
                block_wraps = np.uint64(0)

            unwrapped = timetags.astype(np.uint64) + (overflow_total + prefix_wraps) * np.uint64(T2WRAPAROUND)
            overflow_total += block_wraps
            overflow_records += int(is_overflow.sum())

            photon_mask = ~is_special
            if include_arr is not None:
                photon_mask &= np.isin(channels, include_arr)
            if not photon_mask.any():
                continue

            times = unwrapped[photon_mask]
            photon_channels = channels[photon_mask]
            vals, cnts = np.unique(photon_channels, return_counts=True)
            for ch, cnt in zip(vals, cnts):
                channel_counts[int(ch)] = channel_counts.get(int(ch), 0) + int(cnt)

            if first_photon_tick is None:
                first_photon_tick = np.uint64(times[0])
            last_photon_tick = np.uint64(times[-1])
            included_photons += int(times.size)

            phase_ticks = (times.astype(np.int64) + phase_offset_ticks) % period_ticks
            phase_idx = (phase_ticks // bin_ticks).astype(np.int64, copy=False)
            phase_idx = phase_idx[(phase_idx >= 0) & (phase_idx < n_phase_bins)]
            if phase_idx.size:
                phase_counts += np.bincount(phase_idx, minlength=n_phase_bins).astype(np.uint64, copy=False)

            if block_ticks is not None and phase_idx.size:
                rel_ticks = times.astype(np.uint64) - np.uint64(first_photon_tick)
                block_idx = (rel_ticks // np.uint64(block_ticks)).astype(np.int64, copy=False)
                max_block = int(block_idx.max())
                if counts_by_block is None:
                    counts_by_block = np.zeros((max_block + 1, n_phase_bins), dtype=np.uint64)
                elif max_block >= counts_by_block.shape[0]:
                    grown = np.zeros((max_block + 1, n_phase_bins), dtype=np.uint64)
                    grown[:counts_by_block.shape[0], :] = counts_by_block
                    counts_by_block = grown

                linear_idx = block_idx * n_phase_bins + phase_idx
                block_counts = np.bincount(
                    linear_idx,
                    minlength=(max_block + 1) * n_phase_bins,
                ).astype(np.uint64, copy=False)
                counts_by_block[:max_block + 1, :] += block_counts.reshape(max_block + 1, n_phase_bins)

            if show_progress:
                done = 100.0 * rec_stop / total_records
                self._append_log(f"\r[pump-probe] processed {done:5.1f}% ({rec_stop}/{total_records}) records")

        if first_photon_tick is None:
            duration_s = 0.0
            n_periods = 0.0
        else:
            duration_s = (int(last_photon_tick) - int(first_photon_tick)) * BASE_RES
            n_periods = duration_s / period_s if period_s > 0 else 0.0

        phase_bin_s = phase_bin_us * 1e-6
        if n_periods > 0 and phase_bin_s > 0:
            phase_rate_hz = phase_counts.astype(np.float64) / (n_periods * phase_bin_s)
        else:
            phase_rate_hz = np.zeros_like(phase_counts, dtype=np.float64)

        if counts_by_block is None:
            counts_by_block = np.zeros((0, n_phase_bins), dtype=np.uint64)
            block_edges_s = np.array([], dtype=np.float64)
        else:
            block_edges_s = np.arange(counts_by_block.shape[0] + 1, dtype=np.float64) * float(block_s)

        np.savez_compressed(
            out_path,
            phase_edges_s=phase_edges_s,
            phase_centers_s=phase_centers_s,
            phase_counts=phase_counts,
            phase_rate_hz=phase_rate_hz,
            counts_by_block=counts_by_block,
            block_edges_s=block_edges_s,
            period_s=np.array(period_s, dtype=np.float64),
            phase_bin_us=np.array(phase_bin_us, dtype=np.float64),
            phase_offset_s=np.array(phase_offset_s, dtype=np.float64),
            block_s=np.array(np.nan if block_s is None else float(block_s), dtype=np.float64),
            total_records=np.array(total_records, dtype=np.int64),
            included_photons=np.array(included_photons, dtype=np.int64),
            overflow_records=np.array(overflow_records, dtype=np.int64),
            first_photon_tick=np.array(0 if first_photon_tick is None else first_photon_tick, dtype=np.uint64),
            last_photon_tick=np.array(0 if last_photon_tick is None else last_photon_tick, dtype=np.uint64),
            duration_s=np.array(duration_s, dtype=np.float64),
            n_periods=np.array(n_periods, dtype=np.float64),
            channel_ids=np.array(sorted(channel_counts), dtype=np.int64),
            channel_counts=np.array([channel_counts[ch] for ch in sorted(channel_counts)], dtype=np.int64),
            metadata_json=np.array(json.dumps(metadata) if metadata is not None else ""),
        )

        if show_progress:
            self._append_log(
                f"[pump-probe] done in {time.time() - t_start:.1f}s; "
                f"photons={included_photons}, duration={duration_s:.6f}s"
            )
            self._append_log(f"[pump-probe] saved folded analysis: {out_path}")

        return self._summary_from_npz(out_path)

    @Pyro4.expose
    def fold_b_file_by_sync(self, filepath, sync_channel=1, bin_us=1.0, signal_channels=None,
                            sync_tick_stride=1,
                            measurement_include_ranges_s=None,
                            measurement_exclude_ranges_s=None,
                            measurement_filter_label=None,
                            block_records=1_000_000, force_recompute=False,
                            show_progress=True):
        """
        Fold signal-channel photons between consecutive sync-channel ticks.

        CH1 (or sync_channel) is the only clock. For every signal photon, phase is
        signal_time - previous selected sync tick. sync_tick_stride=1 uses every
        tick; sync_tick_stride=N uses every Nth tick. Photons before the first
        selected sync tick and after the final selected sync tick are excluded.
        bin_us sets the native histogram width; no user-supplied period is used.
        """
        filepath = os.path.abspath(filepath)
        metadata = load_metadata(filepath)
        sync_channel = int(sync_channel)
        bin_us = float(bin_us)
        sync_tick_stride = int(sync_tick_stride)
        if bin_us <= 0:
            raise ValueError("bin_us must be positive")
        if sync_tick_stride <= 0:
            raise ValueError("sync_tick_stride must be a positive integer")
        if signal_channels is None:
            signal_channels = [0]
        signal_channels = [int(ch) for ch in signal_channels]
        if sync_channel in signal_channels:
            raise ValueError("sync_channel must not be included in signal_channels")
        measurement_include_ranges_s = _normalize_time_ranges(measurement_include_ranges_s)
        measurement_exclude_ranges_s = _normalize_time_ranges(measurement_exclude_ranges_s)

        bin_ticks = max(1, int(round((bin_us * 1e-6) / BASE_RES)))
        signal_arr = np.array(signal_channels, dtype=np.uint8)

        base = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = folded_output_folder(filepath)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            sync_folded_filename(
                base,
                sync_channel,
                signal_channels,
                bin_us,
                sync_tick_stride,
                measurement_include_ranges_s,
                measurement_exclude_ranges_s,
                measurement_filter_label,
            ),
        )
        params_path = os.path.join(out_dir, "analysis_params_pump_probe_sync.pkl")
        if os.path.exists(out_path) and not force_recompute:
            self._append_log(f"[pump-probe sync] using existing folded analysis: {out_path}")
            return self._summary_from_npz(out_path)

        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("Empty file")

        analysis_params = {
            "filepath": filepath,
            "metadata_path": metadata_path_for(filepath),
            "record_format": "PicoHarp300_T2",
            "fold_mode": "sync_channel",
            "period_s": None,
            "phase_bin_us": bin_us,
            "block_s": None,
            "sync_channel": int(sync_channel),
            "sync_tick_stride": sync_tick_stride,
            "signal_channels": signal_channels,
            "measurement_include_ranges_s": measurement_include_ranges_s.tolist(),
            "measurement_exclude_ranges_s": measurement_exclude_ranges_s.tolist(),
            "phase_offset_s": 0.0,
            "block_records": int(block_records),
            "BASE_RES": float(BASE_RES),
            "T2WRAPAROUND": int(T2WRAPAROUND),
        }
        with open(params_path, "wb") as f:
            pickle.dump(analysis_params, f)

        if show_progress:
            self._append_log(f"[pump-probe sync] PicoHarp file {filepath}: {total_records} records")
            self._append_log(
                f"[pump-probe sync] sync=ch{sync_channel}, signal={signal_channels}, "
                f"sync tick stride={sync_tick_stride}, histogram bin={bin_us:g} us"
            )
            if measurement_include_ranges_s.size or measurement_exclude_ranges_s.size:
                self._append_log(
                    f"[pump-probe sync] measurement include={measurement_include_ranges_s.tolist()}, "
                    f"exclude={measurement_exclude_ranges_s.tolist()} s"
                )

        # Pass 1: gather sync ticks and channel counts. CH1 defines all cycle boundaries.
        memmap = np.memmap(filepath, dtype=np.uint32, mode="r", shape=(total_records,))
        overflow_total = np.uint64(0)
        overflow_records = 0
        channel_counts = {}
        sync_chunks = []
        measurement_zero_tick = None
        t_start = time.time()

        for rec_start in range(0, total_records, block_records):
            rec_stop = min(total_records, rec_start + block_records)
            words = memmap[rec_start:rec_stop]
            is_special, channels, unwrapped, is_overflow, block_wraps = _unwrap_t2_block(words, overflow_total)
            overflow_total += block_wraps
            overflow_records += int(is_overflow.sum())

            photon_mask = ~is_special
            if photon_mask.any():
                photon_channels = channels[photon_mask]
                photon_ticks = unwrapped[photon_mask]
                if measurement_zero_tick is None:
                    measurement_zero_tick = np.uint64(photon_ticks[0])
                vals, cnts = np.unique(photon_channels, return_counts=True)
                for ch, cnt in zip(vals, cnts):
                    channel_counts[int(ch)] = channel_counts.get(int(ch), 0) + int(cnt)

            sync_times = unwrapped[(~is_special) & (channels == sync_channel)]
            if sync_times.size:
                sync_chunks.append(sync_times.copy())

            if show_progress:
                done = 100.0 * rec_stop / total_records
                self._append_log(f"\r[pump-probe sync] pass 1/2 {done:5.1f}% ({rec_stop}/{total_records}) records")

        if not sync_chunks:
            raise ValueError(f"No sync ticks found on channel {sync_channel}")
        raw_sync_times_all = np.concatenate(sync_chunks).astype(np.uint64, copy=False)
        if measurement_zero_tick is None:
            measurement_zero_tick = np.uint64(raw_sync_times_all[0])
        raw_sync_tick_count = int(raw_sync_times_all.size)
        sync_times_all = raw_sync_times_all[::sync_tick_stride]
        if sync_times_all.size < 2:
            raise ValueError(
                f"Need at least two selected sync ticks on channel {sync_channel}; "
                f"found {raw_sync_tick_count} raw ticks with stride {sync_tick_stride}"
            )

        sync_intervals_ticks_all = np.diff(sync_times_all.astype(np.int64))
        if np.any(sync_intervals_ticks_all <= 0):
            raise ValueError("Sync timestamps are not strictly increasing")
        cycle_start_s = (sync_times_all[:-1].astype(np.float64) - float(measurement_zero_tick)) * BASE_RES
        cycle_stop_s = (sync_times_all[1:].astype(np.float64) - float(measurement_zero_tick)) * BASE_RES
        valid_cycles = np.ones(sync_intervals_ticks_all.shape, dtype=bool)
        if measurement_include_ranges_s.size:
            valid_cycles &= _intervals_overlap_ranges(cycle_start_s, cycle_stop_s, measurement_include_ranges_s)
        if measurement_exclude_ranges_s.size:
            valid_cycles &= ~_intervals_overlap_ranges(cycle_start_s, cycle_stop_s, measurement_exclude_ranges_s)
        if not np.any(valid_cycles):
            raise ValueError("No sync-to-sync cycles remain after applying measurement time filters")
        sync_intervals_ticks = sync_intervals_ticks_all[valid_cycles]
        sync_tick_count = int(sync_times_all.size)
        first_sync_tick = np.uint64(sync_times_all[0])
        last_sync_tick = np.uint64(sync_times_all[-1])
        max_interval_ticks = int(sync_intervals_ticks.max())
        n_phase_bins = int((max_interval_ticks + bin_ticks - 1) // bin_ticks)
        phase_edges_s = np.arange(n_phase_bins + 1, dtype=np.float64) * bin_ticks * BASE_RES
        phase_edges_s[-1] = max_interval_ticks * BASE_RES
        phase_centers_s = 0.5 * (phase_edges_s[:-1] + phase_edges_s[1:])
        phase_counts = np.zeros(n_phase_bins, dtype=np.uint64)

        # Pass 2: fold signal photons relative to the latest preceding sync tick.
        overflow_total = np.uint64(0)
        included_photons = 0
        excluded_before_first_sync = 0
        excluded_after_last_sync = 0
        excluded_by_measurement_time = 0
        excluded_by_invalid_cycle = 0
        first_signal_tick = None
        last_signal_tick = None

        for rec_start in range(0, total_records, block_records):
            rec_stop = min(total_records, rec_start + block_records)
            words = memmap[rec_start:rec_stop]
            is_special, channels, unwrapped, _is_overflow, block_wraps = _unwrap_t2_block(words, overflow_total)
            overflow_total += block_wraps

            signal_times = unwrapped[(~is_special) & np.isin(channels, signal_arr)]
            if signal_times.size:
                if first_signal_tick is None:
                    first_signal_tick = np.uint64(signal_times[0])
                last_signal_tick = np.uint64(signal_times[-1])

                idx = np.searchsorted(sync_times_all, signal_times, side="right") - 1
                valid = (idx >= 0) & (idx < sync_times_all.size - 1)
                excluded_before_first_sync += int((idx < 0).sum())
                excluded_after_last_sync += int((idx >= sync_times_all.size - 1).sum())
                if valid.any() and (measurement_include_ranges_s.size or measurement_exclude_ranges_s.size):
                    signal_t_s = (signal_times.astype(np.float64) - float(measurement_zero_tick)) * BASE_RES
                    in_time = np.ones(signal_times.shape, dtype=bool)
                    if measurement_include_ranges_s.size:
                        in_time &= _points_in_ranges(signal_t_s, measurement_include_ranges_s)
                    if measurement_exclude_ranges_s.size:
                        in_time &= ~_points_in_ranges(signal_t_s, measurement_exclude_ranges_s)
                    excluded_by_measurement_time += int(np.count_nonzero(valid & ~in_time))
                    valid &= in_time
                if valid.any():
                    cycle_is_valid = np.zeros(signal_times.shape, dtype=bool)
                    valid_indices = np.flatnonzero(valid)
                    cycle_is_valid[valid_indices] = valid_cycles[idx[valid_indices]]
                    excluded_by_invalid_cycle += int(np.count_nonzero(valid & ~cycle_is_valid))
                    valid &= cycle_is_valid
                if valid.any():
                    valid_signal_times = signal_times[valid]
                    valid_sync_times = sync_times_all[idx[valid]]
                    phase_ticks = valid_signal_times.astype(np.int64) - valid_sync_times.astype(np.int64)
                    phase_idx = (phase_ticks // bin_ticks).astype(np.int64, copy=False)
                    phase_idx = phase_idx[(phase_idx >= 0) & (phase_idx < n_phase_bins)]
                    if phase_idx.size:
                        included_photons += int(phase_idx.size)
                        phase_counts += np.bincount(phase_idx, minlength=n_phase_bins).astype(np.uint64, copy=False)

            if show_progress:
                done = 100.0 * rec_stop / total_records
                self._append_log(f"\r[pump-probe sync] pass 2/2 {done:5.1f}% ({rec_stop}/{total_records}) records")

        duration_s = (int(last_sync_tick) - int(first_sync_tick)) * BASE_RES
        n_periods = int(np.count_nonzero(valid_cycles))
        sync_interval_arr = sync_intervals_ticks.astype(np.float64) * BASE_RES
        sync_interval_median_s = float(np.median(sync_interval_arr))
        sync_interval_mean_s = float(np.mean(sync_interval_arr))
        phase_bin_us = bin_us
        phase_offset_s = 0.0

        exposure_cycles = np.searchsorted(
            np.sort(sync_intervals_ticks),
            np.arange(n_phase_bins, dtype=np.int64) * bin_ticks,
            side="right",
        )
        exposure_cycles = n_periods - exposure_cycles
        exposure_s = exposure_cycles.astype(np.float64) * (bin_us * 1e-6)
        phase_rate_hz = np.divide(
            phase_counts.astype(np.float64),
            exposure_s,
            out=np.zeros_like(phase_counts, dtype=np.float64),
            where=exposure_s > 0,
        )

        counts_by_block = np.zeros((0, n_phase_bins), dtype=np.uint64)
        block_edges_s = np.array([], dtype=np.float64)

        np.savez_compressed(
            out_path,
            phase_edges_s=phase_edges_s,
            phase_centers_s=phase_centers_s,
            phase_counts=phase_counts,
            phase_rate_hz=phase_rate_hz,
            counts_by_block=counts_by_block,
            block_edges_s=block_edges_s,
            phase_bin_us=np.array(phase_bin_us, dtype=np.float64),
            phase_offset_s=np.array(phase_offset_s, dtype=np.float64),
            block_s=np.array(np.nan, dtype=np.float64),
            total_records=np.array(total_records, dtype=np.int64),
            included_photons=np.array(included_photons, dtype=np.int64),
            overflow_records=np.array(overflow_records, dtype=np.int64),
            first_photon_tick=np.array(0 if first_signal_tick is None else first_signal_tick, dtype=np.uint64),
            last_photon_tick=np.array(0 if last_signal_tick is None else last_signal_tick, dtype=np.uint64),
            first_sync_tick=np.array(0 if first_sync_tick is None else first_sync_tick, dtype=np.uint64),
            last_sync_tick=np.array(0 if last_sync_tick is None else last_sync_tick, dtype=np.uint64),
            duration_s=np.array(duration_s, dtype=np.float64),
            n_periods=np.array(float(n_periods), dtype=np.float64),
            sync_channel=np.array(sync_channel, dtype=np.int64),
            sync_tick_stride=np.array(sync_tick_stride, dtype=np.int64),
            signal_channels=np.array(signal_channels, dtype=np.int64),
            sync_tick_count=np.array(sync_tick_count, dtype=np.int64),
            valid_cycle_count=np.array(n_periods, dtype=np.int64),
            raw_sync_tick_count=np.array(raw_sync_tick_count, dtype=np.int64),
            sync_interval_median_s=np.array(sync_interval_median_s, dtype=np.float64),
            sync_interval_mean_s=np.array(sync_interval_mean_s, dtype=np.float64),
            excluded_before_first_sync=np.array(excluded_before_first_sync, dtype=np.int64),
            excluded_after_last_sync=np.array(excluded_after_last_sync, dtype=np.int64),
            excluded_by_measurement_time=np.array(excluded_by_measurement_time, dtype=np.int64),
            excluded_by_invalid_cycle=np.array(excluded_by_invalid_cycle, dtype=np.int64),
            excluded_outside_period=np.array(0, dtype=np.int64),
            measurement_zero_tick=np.array(measurement_zero_tick, dtype=np.uint64),
            measurement_include_ranges_s=measurement_include_ranges_s.astype(np.float64),
            measurement_exclude_ranges_s=measurement_exclude_ranges_s.astype(np.float64),
            measurement_filter_label=np.array("" if measurement_filter_label is None else str(measurement_filter_label)),
            channel_ids=np.array(sorted(channel_counts), dtype=np.int64),
            channel_counts=np.array([channel_counts[ch] for ch in sorted(channel_counts)], dtype=np.int64),
            fold_mode=np.array("sync_channel"),
            metadata_json=np.array(json.dumps(metadata) if metadata is not None else ""),
        )

        if show_progress:
            self._append_log(
                f"[pump-probe sync] done in {time.time() - t_start:.1f}s; "
                f"selected_sync_ticks={sync_tick_count}/{raw_sync_tick_count}, "
                f"stride={sync_tick_stride}, signal_photons={included_photons}, duration={duration_s:.6f}s"
            )
            self._append_log(f"[pump-probe sync] saved folded analysis: {out_path}")

        return self._summary_from_npz(out_path)

    @Pyro4.expose
    def read_existing_folded(self, folder, pattern="*.npz"):
        candidates = sorted(glob.glob(os.path.join(folder, pattern)))
        return [self._summary_from_npz(path) for path in candidates]

    def _summary_from_npz(self, path):
        with np.load(path, allow_pickle=True) as z:
            summary = {
                "path": path,
                "total_records": int(z["total_records"]),
                "included_photons": int(z["included_photons"]),
                "duration_s": float(z["duration_s"]),
                "channel_counts": {
                    int(ch): int(cnt)
                    for ch, cnt in zip(z["channel_ids"].tolist(), z["channel_counts"].tolist())
                },
            }
            if "analysis_type" in z.files:
                summary["analysis_type"] = str(z["analysis_type"])
            if "n_periods" in z.files:
                summary["n_periods"] = float(z["n_periods"])
            if "phase_counts" in z.files:
                summary["n_phase_bins"] = int(z["phase_counts"].shape[0])
            if "counts_by_block" in z.files:
                summary["n_blocks"] = int(z["counts_by_block"].shape[0])
            if "counts" in z.files:
                summary["n_time_bins"] = int(z["counts"].shape[0])
            if "bin_s" in z.files:
                summary["bin_s"] = float(z["bin_s"])
            if "overflow_records" in z.files:
                summary["overflow_records"] = int(z["overflow_records"])
            if "period_s" in z.files:
                summary["period_s"] = float(z["period_s"])
            if "phase_bin_us" in z.files:
                summary["phase_bin_us"] = float(z["phase_bin_us"])
            if "phase_offset_s" in z.files:
                summary["phase_offset_s"] = float(z["phase_offset_s"])
            if "block_s" in z.files:
                block_s = float(z["block_s"])
                summary["block_s"] = None if np.isnan(block_s) else block_s
            if "fold_mode" in z.files:
                summary["fold_mode"] = str(z["fold_mode"])
            if "sync_channel" in z.files:
                summary["sync_channel"] = int(z["sync_channel"])
            if "sync_tick_stride" in z.files:
                summary["sync_tick_stride"] = int(z["sync_tick_stride"])
            if "signal_channels" in z.files:
                summary["signal_channels"] = [int(ch) for ch in z["signal_channels"].tolist()]
            if "sync_tick_count" in z.files:
                summary["sync_tick_count"] = int(z["sync_tick_count"])
            if "valid_cycle_count" in z.files:
                summary["valid_cycle_count"] = int(z["valid_cycle_count"])
            if "raw_sync_tick_count" in z.files:
                summary["raw_sync_tick_count"] = int(z["raw_sync_tick_count"])
            if "sync_interval_median_s" in z.files:
                summary["sync_interval_median_s"] = float(z["sync_interval_median_s"])
            if "sync_interval_mean_s" in z.files:
                summary["sync_interval_mean_s"] = float(z["sync_interval_mean_s"])
            if "excluded_before_first_sync" in z.files:
                summary["excluded_before_first_sync"] = int(z["excluded_before_first_sync"])
            if "excluded_after_last_sync" in z.files:
                summary["excluded_after_last_sync"] = int(z["excluded_after_last_sync"])
            if "excluded_by_measurement_time" in z.files:
                summary["excluded_by_measurement_time"] = int(z["excluded_by_measurement_time"])
            if "excluded_by_invalid_cycle" in z.files:
                summary["excluded_by_invalid_cycle"] = int(z["excluded_by_invalid_cycle"])
            if "excluded_outside_period" in z.files:
                summary["excluded_outside_period"] = int(z["excluded_outside_period"])
            if "measurement_include_ranges_s" in z.files:
                summary["measurement_include_ranges_s"] = z["measurement_include_ranges_s"].tolist()
            if "measurement_exclude_ranges_s" in z.files:
                summary["measurement_exclude_ranges_s"] = z["measurement_exclude_ranges_s"].tolist()
            if "measurement_filter_label" in z.files:
                summary["measurement_filter_label"] = str(z["measurement_filter_label"])
        return summary
        
    @Pyro4.expose
    def get_initial_timestamps(self, filepath, max_time_s=3.0, block_records=5_000_000):
        """Extracts the first few seconds of absolute photon timestamps to find t0."""
        filepath = os.path.abspath(filepath)
        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            return np.array([], dtype=np.float64)

        memmap = np.memmap(filepath, dtype=np.uint32, mode="r", shape=(total_records,))
        overflow_total = np.uint64(0)
        times_list = []
        max_ticks = np.uint64(max_time_s / BASE_RES)

        for rec_start in range(0, total_records, block_records):
            rec_stop = min(total_records, rec_start + block_records)
            words = memmap[rec_start:rec_stop]
            is_special, channels, timetags, is_overflow = decode_words(words)

            # Handle wraparounds
            is_wrap = is_overflow.astype(np.uint8)
            if is_wrap.size:
                cum_wraps = np.cumsum(is_wrap, dtype=np.uint64)
                prefix_wraps = cum_wraps - is_wrap.astype(np.uint64)
                block_wraps = cum_wraps[-1]
            else:
                prefix_wraps = np.zeros(0, dtype=np.uint64)
                block_wraps = np.uint64(0)

            unwrapped = timetags.astype(np.uint64) + (overflow_total + prefix_wraps) * np.uint64(T2WRAPAROUND)
            overflow_total += block_wraps

            # Filter for actual photon events
            photon_mask = ~is_special
            chunk_times = unwrapped[photon_mask]

            if chunk_times.size > 0:
                times_list.append(chunk_times)
                # Stop if we've passed the requested time window
                if chunk_times[-1] > max_ticks:
                    break

        if not times_list:
            return np.array([], dtype=np.float64)

        # Combine, truncate to max_time_s, and convert to seconds
        all_times = np.concatenate(times_list)
        all_times = all_times[all_times <= max_ticks]
        
        return all_times.astype(np.float64) * BASE_RES


if __name__ == "__main__":
    host = os.environ.get("PICOHARP_PP_HOST", "127.0.0.1")
    port = int(os.environ.get("PICOHARP_PP_PORT", "9112"))
    object_id = os.environ.get("PICOHARP_PP_OBJECT_ID", "PumpProbeAnalysis")
    daemon = Pyro4.Daemon(host=host, port=port)
    uri = daemon.register(PumpProbeAnalysisService, objectId=object_id)
    print("PicoHarp pump-probe analysis server URI:", uri)
    daemon.requestLoop()
