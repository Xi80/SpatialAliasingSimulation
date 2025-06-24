from concurrent.futures import ThreadPoolExecutor

import pyroomacoustics as pra

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import Utils

from control_point import ControlPoint

"""
    エリア再生のシミュレータ
    2025.06.24 Itsuki Hashimoto
"""


class Simulator:
    # インパルス応答の長さ
    IR_LEN: int = 1024

    # 正則化パラメータδ
    DELTA: float = 0.003

    # 音圧マップの分解能[m]
    RESOLUTION: float = 0.1

    def __init__(self,
                 room_dimensions: list[float],
                 speaker_positions: list[tuple[float, float, float]],
                 control_points: list[ControlPoint],
                 input_signal: np.array,
                 fs: int):

        self.room_dimensions = room_dimensions
        self.grid_num = int(self.room_dimensions[0] / self.RESOLUTION)

        self.s = []
        self.fs = fs

        self.speaker_positions = speaker_positions
        self.speaker_num = len(self.speaker_positions)
        self.control_points = control_points

        self.input_signal = input_signal

        ir = []

        for i in range(self.speaker_num):
            ir.append(self._get_ir_array(
                self.speaker_positions[i], self.control_points))

        self.w = Utils.calc_filter(
            self.fs, self.DELTA, ir, self.speaker_num, self.control_points)
        for i in range(self.speaker_num):
            self.s.append(Utils.apply_filter(self.w[i], self.input_signal))

    def _get_ir_array(self, speaker_position: tuple[float, float, float],
                      control_points: list[ControlPoint]) -> np.array:
        """
        与えられた座標のスピーカから各制御点までのインパルス応答を取得する
        :param speaker_position: スピーカの座標
        :param control_points: 制御点
        :return: インパルス応答
        """
        room = pra.ShoeBox(
            p=self.room_dimensions,
            fs=self.fs,
            air_absorption=True,
            max_order=1,
            absorption=1.0,
        )

        # 音源を追加
        room.add_source(speaker_position)

        # 制御点の位置にマイクを追加
        microphone_positions = []
        for i in range(len(control_points)):
            microphone_positions.append(control_points[i].coordinates)

        room.add_microphone_array(np.array(microphone_positions).T)

        # シミュレーションを実行
        room.compute_rir()

        ir_array = []

        for idx in range(len(room.rir)):
            ir_array.append(Utils.pad(room.rir[idx][0], self.IR_LEN))

        return ir_array

    def _get_rms_line(self, y):
        """
        グリッド状にマイクを配置してRMSを取得する(1ライン分)
        :param y: 取得するy座標
        :return: 音圧マップ(1ライン分)
        """
        ret = []

        room = pra.ShoeBox(
            p=self.room_dimensions,
            fs=self.fs,
            air_absorption=True,
            max_order=1,
            absorption=1.0,
        )

        for i in range(self.speaker_num):
            room.add_source(self.speaker_positions[i], self.s[i])

        k = []
        for x in range(self.grid_num):
            k.append([round(x * self.RESOLUTION, 2),
                      round(y * self.RESOLUTION, 2), 1.5])

        room.add_microphone_array(np.array(k).T)

        room.simulate()
        simulation_data = room.mic_array.signals

        for x in range(self.grid_num):
            ret.append(Utils.calc_rms(simulation_data[x, :]))

        return ret

    def save_waveforms(self, filename_prefix: str,
                       positions: list[tuple[float, float, float]]) -> None:
        """
        ある点の音を取得する
        :param filename_prefix: ファイル名のプレフィックス
        :param positions: マイクの位置
        :return: None
        """

        room = pra.ShoeBox(
            p=self.room_dimensions,
            fs=self.fs,
            air_absorption=True,
            max_order=1,
            absorption=1.0,
        )

        for i in range(self.speaker_num):
            room.add_source(self.speaker_positions[i], self.s[i])

        room.add_microphone_array(np.array(positions).T)

        room.simulate()
        simulation_data = room.mic_array.signals

        for x in range(len(positions)):
            Utils.save_wav_file(
                f"{filename_prefix}_{x}.wav", self.fs, simulation_data[x, :])

    def show_pressure_map(self, filename: str = "map.png",
                          save: bool = False) -> None:
        """
        音圧マップを取得する
        :return: 音圧マップ
        """
        rms = [[0] * self.grid_num for _ in range(self.grid_num)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(self._get_rms_line,
                                             range(self.grid_num)),
                                total=self.grid_num,
                                desc="Simulating..."))

        r = list(results)

        for y in range(self.grid_num):
            rms[y] = r[y]

        fig, ax = plt.subplots()
        _ = ax.imshow(rms)
        ax.invert_yaxis()

        if save:
            plt.savefig(filename)
        else:
            plt.show()
