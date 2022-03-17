import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.signal import argrelmin, savgol_filter

from lib.opus_converter import convert_opus
from lib.preprocessing import BaselineCorrector

# Starting values for the slider
sg_window_init = 20
threshold_init = 0.5


def find_peaks(signal, wns, sg_window, threshold):
    """Finds peaks in the signal, and returns the corresponding wavenumbers and
    the Savitzky-Golay smoothed signal."""
    signal_sg = savgol_filter(
        signal, window_length=2*sg_window+1, polyorder=3, deriv=2)

    peaks = argrelmin(signal_sg)[0]
    peaks = [peak for peak in peaks if signal_sg[peak]
             < -threshold]  # Remove peaks below threshold

    # Combine peaks w/o positive 2nd derivative between them
    peak_condensing = []
    peaks_condensed = []
    for j in range(len(signal_sg)):
        if j in peaks:
            peak_condensing.append(j)
        if signal_sg[j] > 0 and len(peak_condensing) > 0:
            peaks_condensed.append(int(np.mean(peak_condensing)))
            peak_condensing = []
    if len(peak_condensing) > 0:
        peaks_condensed.append(int(np.mean(peak_condensing)))

    peaks_wns = wns[peaks_condensed]

    return peaks_wns, signal_sg


# Import data (OPUS binary file)
data = convert_opus("data/AgNP.0")
wns = data[:, 0]
signal = data[:, 1]

# Subtract baseline
signal_bl = BaselineCorrector(method="mormol").fit_transform(
    signal.reshape((1, -1))).ravel()

# Make figure with extra space at the bottom for sliders
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 9),
                               gridspec_kw={"bottom": 0.25, "hspace": 0},
                               sharex=True)

# First plot: Spectrum with peaks

ax1.axhline(0, linestyle="--", linewidth=1, color="black")
ax1.plot(wns, signal_bl)

y_min = 0
y_max = np.max(signal_bl)*1.1

# Detect peaks in the spectrum
peaks, signal_sg = find_peaks(signal, wns, sg_window_init, threshold_init)

# Mark peaks with a red line
lines = ax1.vlines(peaks, y_min, y_max, colors="red", linewidth=1)

ax1.set_xlim(wns[0], wns[-1])
ax1.grid()
ax1.set_ylabel("Intensity (-)", fontdict={"weight": "bold", "size": 12})

# Second plot: 2nd Derivative (from Savitzky-Golay-Filter) with threshold

ax2.axhline(0, linestyle="--", linewidth=1, color="black")
sg_plot, = ax2.plot(wns, signal_sg)
threshold_line = ax2.axhline(-threshold_init, linewidth=1, color="red")


ax2.set_xlim(wns[0], wns[-1])
ax2.set_ylim(-11, 1)
ax2.grid()
ax2.set_xlabel("Raman Shift ($\mathregular{cm^{-1}}$)",
               fontdict={"weight": "bold", "size": 12})
ax2.set_ylabel("2nd Derivative",
               fontdict={"weight": "bold", "size": 12})

# Add axes for sliders
sl1 = fig.add_axes(rect=[0.125, 0.1, 0.775, 0.05])
sl2 = fig.add_axes(rect=[0.125, 0.05, 0.775, 0.05])

# Slider for Savitzky-Golay window half-width
sl_window = Slider(sl1, "Window", 2, 30, valinit=sg_window_init, valstep=1)

# Slider for 2nd Derivative threshold
sl_threshold = Slider(sl2, "Threshold", 0, 10, valinit=threshold_init)


def update(val):
    """Function to update the plots when a slider is moved"""
    new_peaks, new_sg = find_peaks(
        signal, wns, sl_window.val, sl_threshold.val)
    segments = [np.array([[peak, y_min], [peak, y_max]]) for peak in new_peaks]
    lines.set_segments(segments)
    sg_plot.set_ydata(new_sg)
    threshold_line.set_ydata([-sl_threshold.val, -sl_threshold.val])
    fig.canvas.draw_idle()


sl_window.on_changed(update)
sl_threshold.on_changed(update)

plt.show()
