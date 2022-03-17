import matplotlib.pyplot as plt

from lib.opus_converter import convert_opus
from lib.preprocessing import BaselineCorrector

data = convert_opus("data/AgNP.0")
wns = data[:, 0]
signal = data[:, 1]

signal_bl = BaselineCorrector(method="mormol").fit_transform(
    signal.reshape((1, -1))).ravel()
baseline = signal - signal_bl

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(wns, signal)
ax.plot(wns, baseline)
ax.set_xlim(wns[0], wns[-1])
ax.set_ylim(0, None)
ax.grid()

ax.set_xlabel("Raman Shift ($\mathregular{cm^{-1}}$)",
              fontdict={"weight": "bold", "size": 12})
ax.set_ylabel("Intensity (-)",
              fontdict={"weight": "bold", "size": 12})

plt.show()
