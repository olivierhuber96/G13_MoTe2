
    def get_raw_data(self, num_read: Optional[int] = None) -> np.ndarray:
        """
        Return raw records from the last measurement. If block-mode used, prefer getBlock()
        depending on workflow; getData() returns an NDArray[int].
        """
        data = self.raw.getData(numRead=num_read) if num_read is not None else self.raw.getData()
        return np.asarray(data, dtype=np.uint32)

    def get_block(self) -> np.ndarray:
        """Get one data block (block mode)."""
        data = self.raw.getBlock()
        return np.asarray(data, dtype=np.uint32)

    # ---- Decoding functions for T2 records ----
    def decode_t2(self, records: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode an array of Raw T2 records into (times_seconds, channels).
        Records that are 'special' or marker are skipped.

        Returns:
            times: 1D numpy array of absolute times in seconds (float64)
            channels: 1D numpy array of channel indices (int)
        """
        n = len(records)
        times = []
        chans = []

        # Loop in Python: necessary because decode functions are exposed via sn.raw.* and operate on ints
        for r in records:
            # skip special records (overflow/other)
            if self.raw.isSpecial(int(r)):
                continue
            # skip markers
            if self.raw.isMarker(int(r)):
                continue
            ch = int(self.raw.channel(int(r)))
            tt = int(self.raw.timeTag_T2(int(r)))  # timetag units (integer)
            t_sec = tt * self._resolution_sec
            times.append(t_sec)
            chans.append(ch)

        if len(times) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int32)

        times = np.array(times, dtype=np.float64)
        chans = np.array(chans, dtype=np.int32)
        # The times from timeTag_T2 are monotonically increasing (with overflow markers removed)
        return times, chans

    # ---- Correlation computation ----
    def cross_correlation(self,
                          times: np.ndarray,
                          channels: np.ndarray,
                          chan_a: int,
                          chan_b: int,
                          max_tau: float = 1e-6,
                          bin_width: float = 1e-9,
                          symmetric: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-correlation histogram (A vs B) using time-tag arrays.
        Efficient approach: separate times for A and B, for each event in A find B in window using searchsorted.

        Args:
            times: sorted 1D array of event times (seconds)
            channels: 1D array of integer channel indices, same length as times
            chan_a: integer channel index for "start" events
            chan_b: integer channel index for "stop" events
            max_tau: maximum |tau| to histogram (seconds)
            bin_width: histogram bin width (seconds)
            symmetric: if True, compute both positive and negative taus (centered at 0).
                       If False, compute only t_B - t_A (positive side)

        Returns:
            bins_centers (1D array), hist (1D array) -- hist counts for each bin
        """
        # Separate A and B times
        times_a = times[channels == chan_a]
        times_b = times[channels == chan_b]

        if times_a.size == 0 or times_b.size == 0:
            # no events -> return zero histogram
            nbins = int(np.ceil((2*max_tau) / bin_width)) if symmetric else int(np.ceil(max_tau / bin_width))
            centers = (np.arange(nbins) + 0.5) * bin_width - (max_tau if symmetric else 0.0)
            return centers, np.zeros(nbins, dtype=np.int64)

        # histogram bin edges
        if symmetric:
            edges = np.arange(-max_tau, max_tau + bin_width, bin_width)
        else:
            edges = np.arange(0, max_tau + bin_width, bin_width)

        hist = np.zeros(len(edges) - 1, dtype=np.int64)

        # Use two-pointer / searchsorted approach for efficiency
        # For each time in A, consider B in [t_A - max_tau, t_A + max_tau] (or [t_A, t_A + max_tau] if not symmetric).
        for ta in times_a:
            if symmetric:
                left = ta - max_tau
                right = ta + max_tau
            else:
                left = ta
                right = ta + max_tau

            i0 = np.searchsorted(times_b, left, side='left')
            i1 = np.searchsorted(times_b, right, side='right')

            if i1 <= i0:
                continue

            # compute differences and bin them
            diffs = times_b[i0:i1] - ta  # could be negative if symmetric
            # np.histogram is efficient for small-to-moderate arrays
            h, _ = np.histogram(diffs, bins=edges)
            hist += h

        # bin centers
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, hist

    def auto_correlation(self,
                         times: np.ndarray,
                         channels: np.ndarray,
                         chan: int,
                         max_tau: float = 1e-6,
                         bin_width: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute autocorrelation for a single channel (avoid counting zero-delay self-pairs).
        Uses split-into-two-halves technique (A vs B) by splitting the event list to avoid self-pair double-counting,
        or compute pairwise differences with same two-pointer technique but skip zero-lag pairs (i==j).
        For performance, we split times into two halves.
        """
        times_chan = times[channels == chan]
        if times_chan.size < 2:
            nbins = int(np.ceil((2*max_tau) / bin_width))
            centers = (np.arange(nbins) + 0.5) * bin_width - max_tau
            return centers, np.zeros(nbins, dtype=np.int64)

        # Split into two non-overlapping halves for cross-correlation to get auto-correlation estimate
        half = len(times_chan) // 2
        times_a = times_chan[:half]
        times_b = times_chan[half:]
        centers, hist = self.cross_correlation(np.concatenate([times_a, times_b]),
                                               np.concatenate([np.full_like(times_a, chan), np.full_like(times_b, chan)]),
                                               chan_a=chan, chan_b=chan, max_tau=max_tau, bin_width=bin_width, symmetric=True)
        return centers, hist
