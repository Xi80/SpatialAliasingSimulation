from dataclasses import dataclass


"""
    制御点
    2025.06.24 Itsuki Hashimoto
"""


@dataclass
class ControlPoint:
    coordinates: tuple[float, float, float]  # 座標
    is_reproduction_point: bool  # 応答制御点か？
