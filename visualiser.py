import numpy as np
import time
import pyaudio
import pyqtgraph as pg
from scipy.ndimage.filters import gaussian_filter1d
from pyqtgraph.Qt import QtGui, QtCore
# Create GUI window
import mel
from mel import ExpFilter

app = QtGui.QApplication([])
view = pg.GraphicsView()
layout = pg.GraphicsLayout(border=(100, 100, 100))
view.setCentralItem(layout)
view.show()
view.setWindowTitle('Visualization')
view.resize(800, 600)
# Visualization plot
layout.nextRow()
led_plot = layout.addPlot(title='Visualization Output', colspan=3)
led_plot.setRange(yRange=[-5, 260])
led_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
r_pen = pg.mkPen((255, 30, 30, 200), width=4)
g_pen = pg.mkPen((30, 255, 30, 200), width=4)
b_pen = pg.mkPen((30, 30, 255, 200), width=4)
r_curve = pg.PlotCurveItem(pen=r_pen)
g_curve = pg.PlotCurveItem(pen=g_pen)
b_curve = pg.PlotCurveItem(pen=b_pen)
x_data = np.e * np.array(range(1, 31))
led_plot.addItem(r_curve)
led_plot.addItem(g_curve)
led_plot.addItem(b_curve)
NUM_LEDS = 60
history = np.random.rand(2, 735) / 1e16
fft_window = np.hamming(735 * 2)
filter_bank = mel.construct_filter()
gain = ExpFilter(np.tile(1e-1, 12),
                 alpha_decay=0.01, alpha_rise=0.99)
y_gain = ExpFilter(np.tile(0.01, 12),
                   alpha_decay=0.001, alpha_rise=0.99)
mel_smoother = ExpFilter(np.tile(1e-1, 12),
                         alpha_decay=0.5, alpha_rise=0.99)
p_smoother = ExpFilter(np.tile(1, (3, NUM_LEDS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)
p = np.tile(1.0, (3, NUM_LEDS // 2))


def interpolate(y, new_length):
    if len(y) == new_length:
        return y
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, new_length)
    z = np.interp(x_new, x_old, y)
    return z


def main(audio_samples):
    global history, app, filter_bank, p
    norm = audio_samples / 2.0 ** 15

    history[:-1] = history[1:]
    history[-1, :] = np.copy(norm)
    data = np.concatenate(history, axis=0).astype(np.float32)
    vol = np.max(np.abs(data))
    if vol > 1e-7:
        n = len(data)
        n_zeros = 2 ** int(np.ceil(np.log2(n))) - n

        data *= fft_window
        data = np.pad(data, (0, n_zeros), mode='constant')
        ffted = np.abs(np.fft.rfft(data, 512)[:n // 2])
        filtered = np.atleast_2d(ffted).T * filter_bank.T

        filtered = np.sum(filtered, axis=0)
        filtered **= 2.0

        f_gain = gain.update(np.max(gaussian_filter1d(filtered, sigma=1.0)))
        filtered /= f_gain

        filtered = mel_smoother.update(filtered)

        y = np.copy(filtered)
        y *= float((NUM_LEDS // 2) - 1)

        scale = 0.9
        r = int(np.mean(y[:len(y) // 3] ** scale))
        g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3] ** scale))
        b = int(np.mean(y[2 * len(y) // 3:] ** scale))
        p[0, :r] = 255.0
        p[0, r:] = 0.0
        p[1, :g] = 255.0
        p[1, g:] = 0.0
        p[2, :b] = 255.0
        p[2, b:] = 0.0

        p = np.round(p_smoother.update(p))

        p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)

        out = np.concatenate([p[:, ::-1], p], axis=1)

        r_curve.setData(y=out[0])
        g_curve.setData(y=out[1])
        b_curve.setData(y=out[2])

    app.processEvents()


def start():
    py = pyaudio.PyAudio()

    stream = py.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=44100,
                     input=True,
                     frames_per_buffer=735,
                     input_device_index=2)

    overflows = 0
    prev_ovf_time = time.time()
    while 1:
        try:
            y = np.fromstring(stream.read(735), dtype=np.int16)
            y = y.astype(np.float32)
            main(y)
        except IOError:
            print("overflow")
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
    stream.stop_stream()
    stream.close()
    py.terminate()


start()
