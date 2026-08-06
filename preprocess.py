"""
preprocess.py
=============
iPad描画キャプチャアプリ (index.html) が出力するJSONを読み込み、
RNN学習用の numpy 配列 (T, 2) 群に変換する前処理スクリプト。

設計方針（詳細はプロジェクトの design-notes.md を参照）:
  - x, y はアプリ側で既にキャンバス範囲固定の一次変換により [-0.9, 0.9] に
    正規化済みなので、ここでは追加の正規化は行わない。
  - t は生のタイムスタンプ（未リサンプリング）で記録されているため、
    ここで固定サンプリングレート(Hz)へリサンプリングする。
    Hzは要検討・要相談の値なので、決め打ちにせず引数で指定できるようにしてある。
  - RNNの重み行列自体は時系列長Tに依存しないため、可変長のまま
    (npzにragged配列として)保存することもできるし、
    バッチ学習用に固定長へパディング+マスクを作ることもできる。
    このスクリプトは両方に対応する。

使い方:
    python preprocess.py family_2026-08-06.json --hz 20 --out dataset.npz
    python preprocess.py data_dir/ --hz 20 --out dataset.npz   # ディレクトリ内の*.jsonをまとめて処理

対応する入力フォーマット:
    - "family" キー（v2、アプリのFamily機能で書き出したもの）
    - "strokes" キー（v1、旧フォーマット）にも後方互換で対応

出力 (npz) の中身:
    strokes_xy        : object配列。strokes_xy[i] は shape (T_i, 2) のfloat32配列（各シーケンス自然長のまま）
    strokes_meta      : object配列。各シーケンスの label / pointerType などのdict
    padded_xy         : shape (Nseq, Tmax, 2) のfloat32配列（0パディング済み、batch-first）
    mask              : shape (Nseq, Tmax) のfloat32配列（1=実データ, 0=パディング、batch-first）
    padded_xy_time_first : shape (Tmax, Nseq, 2)。PyTorchのRNN/GRU/LSTMのデフォルト
                           (batch_first=False, つまり (seq_len, batch, input_size)) にそのまま渡せる並び。
    mask_time_first   : shape (Tmax, Nseq)。padded_xy_time_firstに対応するmask。
    hz                : リサンプリングに使ったレート(Hz)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_source_files(path_arg: str):
    p = Path(path_arg)
    if p.is_dir():
        yield from sorted(p.glob("*.json"))
    else:
        yield p


def resample_stroke_to_fixed_hz(points: list[dict], hz: float) -> np.ndarray:
    """1本のストロークの生の(t, x, y)を、固定レートhzの等間隔グリッドに
    線形補間でリサンプリングする。

    - t は元データのミリ秒タイムスタンプ（PointerEvent.timeStamp由来、単調増加）。
    - 出力は shape (T, 2) の float32 配列。Tはストロークの継続時間とhzで決まる
      （固定値ではない: ストロークごとに長さが変わるのは意図通りで、
        バッチ化する際にパディング+マスクで対応する）。
    - ストロークの点数が1点しかない場合はそのまま1点の配列を返す。
    """
    if len(points) < 2:
        p = points[0] if points else {"x": 0.0, "y": 0.0}
        return np.array([[p["x"], p["y"]]], dtype=np.float32)

    t_raw = np.array([p["t"] for p in points], dtype=np.float64)
    x_raw = np.array([p["x"] for p in points], dtype=np.float64)
    y_raw = np.array([p["y"] for p in points], dtype=np.float64)

    duration_ms = t_raw[-1] - t_raw[0]
    step_ms = 1000.0 / hz
    n_steps = max(1, int(round(duration_ms / step_ms)) + 1)
    t_grid = t_raw[0] + np.arange(n_steps) * step_ms
    t_grid = np.clip(t_grid, t_raw[0], t_raw[-1])  # 外挿を避ける

    x_interp = np.interp(t_grid, t_raw, x_raw)
    y_interp = np.interp(t_grid, t_raw, y_raw)

    return np.stack([x_interp, y_interp], axis=1).astype(np.float32)


def load_dataset(path_arg: str, hz: float):
    strokes_xy = []
    strokes_meta = []

    files = list(iter_source_files(path_arg))
    if not files:
        raise FileNotFoundError(f"JSONファイルが見つかりません: {path_arg}")

    for fp in files:
        data = load_json_file(fp)
        meta = data.get("meta", {})
        # v2: "family" キー、v1: "strokes" キー（後方互換）
        sequences = data.get("family")
        if sequences is None:
            sequences = data.get("strokes", [])

        for seq_item in sequences:
            xy = resample_stroke_to_fixed_hz(seq_item["points"], hz)
            strokes_xy.append(xy)
            strokes_meta.append({
                "source_file": str(fp.name),
                "label": meta.get("label"),
                "pointerType": seq_item.get("pointerType"),
                "n_points_raw": len(seq_item["points"]),
                "n_points_resampled": int(xy.shape[0]),
            })

    return strokes_xy, strokes_meta


def pad_and_mask(strokes_xy: list[np.ndarray]):
    """可変長のストローク群を、最大長Tmaxに0パディングし、
    対応するmask (1=実データ, 0=パディング) を作る。
    RNNをバッチ学習させる際、損失計算でこのmaskを掛けてパディング部分を無視する。
    """
    n = len(strokes_xy)
    t_max = max(s.shape[0] for s in strokes_xy)

    padded = np.zeros((n, t_max, 2), dtype=np.float32)
    mask = np.zeros((n, t_max), dtype=np.float32)

    for i, s in enumerate(strokes_xy):
        t_i = s.shape[0]
        padded[i, :t_i, :] = s
        mask[i, :t_i] = 1.0

    return padded, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="キャプチャアプリが出力したJSONファイル、またはそれらを含むディレクトリ")
    parser.add_argument("--hz", type=float, default=20.0,
                         help="固定リサンプリングレート(Hz)。デフォルト20Hzは仮値なので、"
                              "実データで形状が崩れていないか確認の上、必要に応じて調整すること。")
    parser.add_argument("--out", default="dataset.npz", help="出力するnpzファイル名")
    args = parser.parse_args()

    strokes_xy, strokes_meta = load_dataset(args.input, args.hz)
    print(f"読み込んだストローク数: {len(strokes_xy)}")

    lengths = [s.shape[0] for s in strokes_xy]
    print(f"リサンプリング後の長さ T: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}  (hz={args.hz})")

    padded, mask = pad_and_mask(strokes_xy)
    # (Nseq, Tmax, 2) -> (Tmax, Nseq, 2)。PyTorchのbatch_first=Falseにそのまま渡せる並び。
    padded_time_first = np.transpose(padded, (1, 0, 2))
    mask_time_first = np.transpose(mask, (1, 0))

    np.savez_compressed(
        args.out,
        strokes_xy=np.array(strokes_xy, dtype=object),
        strokes_meta=np.array(strokes_meta, dtype=object),
        padded_xy=padded,
        mask=mask,
        padded_xy_time_first=padded_time_first,
        mask_time_first=mask_time_first,
        hz=args.hz,
    )
    print(f"保存しました: {args.out}  "
          f"(padded_xy shape={padded.shape} [Nseq,Tmax,2], "
          f"padded_xy_time_first shape={padded_time_first.shape} [Tmax,Nseq,2])")


if __name__ == "__main__":
    sys.exit(main())
