"""# **Practical No 2**

Aim: Write program to perform the following on signal
1.	Create a triangle signal and plot a 3-period segment.
2.	For a given signal, plot the segment and compute the correlation between them.
"""

import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import numpy as np
from scipy.signal import correlate

"""Function to create a triangle wave"""

def triangle_wave(periods, sampling_rate):
    t = np.linspace(0, periods, int(periods * sampling_rate), endpoint=False)
    return 2 * np.abs((t % 1) - 0.5) - 1

"""Sampling rate in Hz
Number of periods to plot Generate triangle wave signal

"""

sampling_rate = 1000
periods = 3

triangle_signal = triangle_wave(periods, sampling_rate)

"""Plotling the triangle wave"""

plt.figure(figsize=(10, 4))
plt.plot(np.arange(0, periods, 1/sampling_rate), triangle_signal[:int(periods * sampling_rate)])
plt.title('Triangle Wave - 3 Periods')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

"""Create another signal (e.g., a shifted version of the original signal)"""

shifted_triangle_signal = np.roll(triangle_signal, int(sampling_rate / 2))
plt.figure(figsize=(12, 4))

plt.subplot(2, 1, 1)
plt.plot(np.arange(0, periods, 1/sampling_rate), triangle_signal[:int(periods * sampling_rate)])
plt.title('Original Triangle Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, periods, 1/sampling_rate), shifted_triangle_signal[:int(periods * sampling_rate)])
plt.title('Shifted Triangle Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

"""Compute the correlation between the original and shifted signals"""

correlation_result = correlate(triangle_signal, shifted_triangle_signal, mode='full')

"""Plotling the correlation result plt.figure(figsize=(12, 4))"""

plt.plot(np.arange(-len(triangle_signal)+1, len(triangle_signal)), correlation_result)
plt.title('Cross-correlation between Original and Shifted Triangle Signals')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.grid(True)
plt.show()