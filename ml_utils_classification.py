def split_before_year(X, y, year):
    is_train = X.index.year < year
    X_train = X.loc[is_train]
    y_train = y.loc[is_train]
    X_val = X.loc[~is_train]
    y_val = y.loc[~is_train]
    return X_train, X_val, y_train, y_val

def direction_momentum(df, horizons=[5, 10, 20, 60]):
    new_df = df.copy()
    for horizon in horizons:
        name = f"direction_mom_{horizon}"
        new_df[name] = new_df['direction'].rolling(horizon).sum() - (horizon/2)
    return new_df