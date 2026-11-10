"""Загрузка данных: все скважины в один parquet-кэш + доступ к typewell."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("ROGII_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/competition/raw"
CACHE = f"{ROOT}/competition/derived"
TEST_IDS = {"000d7d20", "00bbac68", "00e12e8b"}

TOPS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

def load_train_hw(force: bool = False) -> pd.DataFrame:
    """Все train horizontal wells одним фреймом. Колонки: wid, row, MD..., is_eval."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/train_hw.parquet"
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    parts = []
    for f in sorted(glob.glob(f"{RAW}/train/*__horizontal_well.csv")):
        wid = os.path.basename(f).split("__")[0]
        df = pd.read_csv(f)
        df["wid"] = wid
        df["row"] = np.arange(len(df))
        df["is_eval"] = df["TVT_input"].isna()
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path)
    return out

def load_typewell(wid: str, train: bool = True) -> pd.DataFrame:
    split = "train" if train else "test"
    return pd.read_csv(f"{RAW}/{split}/{wid}__typewell.csv")

def load_test_hw() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(f"{RAW}/test/*__horizontal_well.csv")):
        wid = os.path.basename(f).split("__")[0]
        df = pd.read_csv(f)
        df["wid"] = wid
        df["row"] = np.arange(len(df))
        df["is_eval"] = df["TVT_input"].isna()
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
