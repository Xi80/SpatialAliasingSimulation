import time

import numpy as np

from utils import Utils
from control_point import ControlPoint
from simulator import Simulator


def get_speaker_positions() -> list[tuple[float, float, float]]:
    """
    スピーカの座標を返す
    :return: スピーカの座標
    """

    # スピーカの個数
    SPEAKER_NUM: int = 16

    speaker_positions = []

    for i in range(SPEAKER_NUM):
        speaker_positions.append((round(i * 0.08 + 0.9, 2), 0.0, 1.5))

    return speaker_positions


def get_control_points() -> list[ControlPoint]:
    """
    制御点の座標と所望の特性(応答制御点/抑圧制御点)を返す
    :return: 制御点の座標と所望の特性(is_reproduction_point = Trueで応答制御点)
    """

    positions = [
        ControlPoint(coordinates=(0.5, 1.0, 1.5), is_reproduction_point=True)]

    for i in range(13):
        y = 1.95 - i * 0.1
        positions.append(
            ControlPoint(coordinates=(1.5, round(y, 2), 1.5),
                         is_reproduction_point=False))

    for x in np.arange(1.6, 2.8, 0.1):
        positions.append(
            ControlPoint(coordinates=(float(round(x, 2)), 0.75, 1.5),
                         is_reproduction_point=False))

    return positions


def main():

    rate, signal = Utils.load_wav_file("1kHz_0db.wav")

    sm = Simulator(
        [3.0, 3.0, 3.0],
        get_speaker_positions(),
        get_control_points(),
        signal,
        rate
    )

    start = time.perf_counter()
    sm.show_pressure_map()
    # sm.save_waveforms("wf",[(0.5, 1.0, 1.5), (2.0, 1.0, 1.5)])
    end = time.perf_counter()

    time_diff = end - start

    print(f"Done! Took {round(time_diff, 2)} [sec]")


if __name__ == '__main__':
    main()
