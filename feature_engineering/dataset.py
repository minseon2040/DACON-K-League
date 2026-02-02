import pandas as pd
from feature_engineering.features import FEATURE_CANDIDATES, CAT_CANDIDATES, SORT_KEYS, DROP_COLS

def add_episode_len(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["episode_len"] = df.groupby("game_episode")["action_id"].transform("count")
    return df

def split_train_test(df: pd.DataFrame, flag_col="is_train"):
    df_train = df[df[flag_col] == 1].copy()
    df_test  = df[df[flag_col] == 0].copy()
    return df_train, df_test

def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(SORT_KEYS).reset_index(drop=True)

def get_last_event_per_episode(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(SORT_KEYS)
          .groupby("game_episode", as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )

def filter_train_targets(df: pd.DataFrame) -> pd.DataFrame:
    cond = df["end_x"].notnull() & df["end_y"].notnull()
    return df[cond].reset_index(drop=True)

def build_feature_cols(df: pd.DataFrame) -> list:
    # 존재하는 컬럼만 + 드랍 대상 제거 + 중복 제거
    cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    cols = [c for c in cols if c not in DROP_COLS]
    return list(dict.fromkeys(cols))

def build_cat_features(feature_cols: list) -> list:
    return [c for c in CAT_CANDIDATES if c in feature_cols]

def fill_missing(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_features: list):
    X_train = X_train.copy()
    X_test  = X_test.copy()

    for c in cat_features:
        X_train[c] = X_train[c].fillna("Missing").astype(str)
        X_test[c]  = X_test[c].fillna("Missing").astype(str)

    for c in ["dx", "dy", "legal_speed"]:
        if c in X_train.columns:
            X_train[c] = X_train[c].fillna(0.0)
            X_test[c]  = X_test[c].fillna(0.0)

    return X_train, X_test


def build_dataset(df_copy: pd.DataFrame):
    print(f"[Data] 데이터셋 구축 중... (사용 피처 수: {len(FEATURE_CANDIDATES)})")

    df_copy = df_copy.copy()

    if "player_role_pass" not in df_copy.columns and "player_role_pass_id" in df_copy.columns:
        df_copy["player_role_pass"] = df_copy["player_role_pass_id"]

    df = add_episode_len(df_copy)
    df_train, df_test = split_train_test(df, "is_train")
    df_train, df_test = sort_df(df_train), sort_df(df_test)

    train_epi = filter_train_targets(get_last_event_per_episode(df_train))
    test_epi  = get_last_event_per_episode(df_test)

    feature_used = [c for c in FEATURE_CANDIDATES if c in train_epi.columns]
    missing = [c for c in FEATURE_CANDIDATES if c not in train_epi.columns]
    if missing:
        print("[WARN] train_epi에 없는 피처 제외:", missing)

    X_train = train_epi[feature_used].copy()
    X_test  = test_epi[feature_used].copy()

    actual_cat_features = [c for c in CAT_CANDIDATES if c in X_train.columns]
    X_train, X_test = fill_missing(X_train, X_test, actual_cat_features)

    y_dx = train_epi["end_x"].values - train_epi["start_x"].values
    y_dy = train_epi["end_y"].values - train_epi["start_y"].values
    groups = train_epi["game_id"].values

    return X_train, X_test, y_dx, y_dy, groups, test_epi, actual_cat_features