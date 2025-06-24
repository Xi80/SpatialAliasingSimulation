import os

import scipy.io.wavfile as wavfile

import numpy as np

from scipy import fft

from control_point import ControlPoint


"""
    シミュレータとは関係ない関数類
        フィルタの作成とかファイル入出力とか
    2025.06.24 Itsuki Hashimoto
"""


class Utils:
    @staticmethod
    def pad(src: np.array, length: int) -> np.array:
        """
        配列の長さを一定にする
        長い場合は切り取り、短い場合は0でパディングする
        :param src: 入力
        :param length: 指定の長さ
        :return: 所望の長さの配列
        """
        if len(src) < length:
            return np.pad(src, (0, length - len(src)), 'constant')
        else:
            return src[:length]

    @staticmethod
    def calc_rms(src: np.array) -> float:
        """
        RMSを計算する(クリッピングあり)
        :param src: 入力
        :return: RMS
        """
        rms = np.sqrt(np.mean(src ** 2))
        if rms >= 1.0:
            rms = 1.0
        return rms

    @staticmethod
    def load_wav_file(filename: str) -> tuple[int, np.array]:
        """
        wavファイルを読み込む(-1.0~1.0)
        :param filename: ファイル名
        :return: サンプリング周波数と読み込んだ波形
        """
        assert os.path.isfile(filename)

        rate, data = wavfile.read(filename)

        if data.ndim > 1:
            data = data[:, 0]

        return rate, data / 32767.0

    @staticmethod
    def save_wav_file(filename: str, fs: int, data: np.array) -> None:
        """
        wavファイルを作成し書き込む
        :param filename: ファイル名
        :param fs: サンプリング周波数[Hz]
        :param data: 書き込む波形データ
        :return: なし
        """
        wavfile.write(filename, fs, data)

    @staticmethod
    def calc_filter(fs, delta, ir_array, speaker_num,
                    control_points: list[ControlPoint]):
        """
        フィルタ特性W_m(ω)を計算する
        :param fs: サンプリング周波数[Hz]
        :param delta: 正規化パラメータδ
        :param ir_array: インパルス応答列
        :param speaker_num: スピーカの個数
        :param control_points: 制御点
        :return: 得られたフィルタ特性(スピーカごと)
        """
        cp_num = len(control_points)

        omega = np.linspace(0, 2.0 * np.pi * fs, 1024)
        g = np.zeros((cp_num, speaker_num, len(omega)), dtype=np.complex128)
        p = np.zeros((1, cp_num), dtype=np.complex128)
        for i in range(cp_num):
            if control_points[i].is_reproduction_point:
                p[0][i] = 1
            else:
                p[0][i] = 0
        w = np.zeros((speaker_num, 1, len(omega)), dtype=np.complex128)

        for i in range(speaker_num):
            ir = ir_array[i]

            for j in range(cp_num):
                g[j, i, :] = fft.fft(ir[j])

        for i in range(len(omega)):
            w[:, :, i] = (np.linalg.inv(
                (np.conj(g[:, :, i]).T @ g[:, :, i])
                + np.eye(speaker_num) * delta) @
                          np.conj(g[:, :, i]).T @ p.T)

        ret = []

        for i in range(speaker_num):
            ret.append(np.real(fft.ifft(w[i, 0, :])))

        return ret

    @staticmethod
    def apply_filter(w, src) -> np.array:
        """
        フィルタを適用する
        :param w: フィルタ
        :param src: 入力信号
        :return: 出力信号
        """
        return np.convolve(src, w, 'full')
