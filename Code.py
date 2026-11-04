import json
import time
import gc
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.signal import lfilter
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DATA_PATH = Path("measures_v2.csv")
OUTPUT_ROOT = Path("results")
CV_DIR = OUTPUT_ROOT / "group_cv"
OOF_DIR = CV_DIR / "oof_predictions"
FINAL_DIR = OUTPUT_ROOT / "final_model"

SEED = 42
DT = 0.5
THERMAL_TAUS = [30.0, 300.0, 1800.0]
TRANSITION_TAUS = [30.0, 300.0, 1800.0]
LANDMARK_COUNTS = [48, 128, 256]
WIDTHS = [0.5, 1.0, 2.0]
LAMBDAS = [1e-7, 1e-5, 1e-3, 1e-2]
REDUCTIONS = ["full", "pca"]
PCA_VARIANCE = 0.95
INNER_TRAIN_SAMPLE = 30000
INNER_VALIDATION_SAMPLE = 12000
FINAL_TRANSFORM_SAMPLE = 30000
THERMAL_STARTUP_WEIGHT = 20.0
BATCH_SIZE = 10000
THREADS = 2

SCENARIOS = {
    "operational": [
        "u_d",
        "u_q",
        "i_d",
        "i_q",
        "motor_speed",
        "torque",
        "coolant",
        "ambient"
    ],
    "sensor_assisted": [
        "u_d",
        "u_q",
        "i_d",
        "i_q",
        "motor_speed",
        "torque",
        "coolant",
        "ambient",
        "stator_winding",
        "stator_tooth",
        "stator_yoke"
    ]
}

METHODS = ["RBF_Base", "Thermal_Ridge", "Hybrid_Ridge"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_sample(indices, maximum, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= maximum:
        return indices.copy()
    return np.random.default_rng(seed).choice(
        indices,
        size=maximum,
        replace=False
    )


def profile_boundaries(groups):
    groups = np.asarray(groups)

    if len(groups) == 0:
        raise ValueError("No rows provided.")

    if pd.isna(groups).any():
        raise ValueError("Missing profile identifier.")

    boundaries = np.r_[
        0,
        np.flatnonzero(groups[1:] != groups[:-1]) + 1,
        len(groups)
    ]

    blocks = groups[boundaries[:-1]]

    if len(np.unique(blocks)) != len(blocks):
        raise ValueError(
            "Each profile must occupy exactly one contiguous block."
        )

    return boundaries


def make_age(groups):
    boundaries = profile_boundaries(groups)
    age = np.empty(len(groups), dtype=np.float32)

    for block in range(len(boundaries) - 1):
        start = boundaries[block]
        end = boundaries[block + 1]
        age[start:end] = (
            np.arange(end - start, dtype=np.float32) * DT
        )

    return age


def build_features(df, columns, groups, age):
    raw = df[columns].to_numpy(dtype=np.float32, copy=True)

    i_d = df["i_d"].to_numpy(dtype=np.float32)
    i_q = df["i_q"].to_numpy(dtype=np.float32)
    speed = df["motor_speed"].to_numpy(dtype=np.float32)
    torque = df["torque"].to_numpy(dtype=np.float32)
    coolant = df["coolant"].to_numpy(dtype=np.float32)
    ambient = df["ambient"].to_numpy(dtype=np.float32)

    current_proxy = i_d * i_d + i_q * i_q
    speed_torque_proxy = np.abs(speed * torque)

    history_sources = [
        current_proxy,
        speed_torque_proxy,
        coolant,
        ambient
    ]

    if "stator_tooth" in columns:
        history_sources.append(
            df["stator_tooth"].to_numpy(dtype=np.float32)
        )

    initial_names = ["coolant", "ambient"]

    if "stator_winding" in columns:
        initial_names += [
            "stator_winding",
            "stator_tooth",
            "stator_yoke"
        ]

    initial_source = df[initial_names].to_numpy(dtype=np.float32)
    boundaries = profile_boundaries(groups)

    history = np.empty(
        (len(df), len(history_sources) * len(THERMAL_TAUS)),
        dtype=np.float32
    )

    response = np.empty(
        (len(df), 2 * len(THERMAL_TAUS)),
        dtype=np.float32
    )

    initial = np.empty(
        (len(df), len(initial_names)),
        dtype=np.float32
    )

    for block in range(len(boundaries) - 1):
        start = boundaries[block]
        end = boundaries[block + 1]
        initial[start:end] = initial_source[start]

        column = 0

        for source in history_sources:
            x = source[start:end].astype(np.float64)

            for tau in THERMAL_TAUS:
                decay = np.exp(-DT / tau)
                alpha = 1.0 - decay

                filtered, _ = lfilter(
                    [alpha],
                    [1.0, -decay],
                    x,
                    zi=[decay * x[0]]
                )

                history[start:end, column] = filtered.astype(np.float32)
                column += 1

        column = 0

        for source in [current_proxy, speed_torque_proxy]:
            x = source[start:end].astype(np.float64)
            shifted = np.empty_like(x)
            shifted[0] = 0.0
            shifted[1:] = x[:-1]

            for tau in THERMAL_TAUS:
                decay = np.exp(-DT / tau)
                alpha = 1.0 - decay

                filtered = lfilter(
                    [alpha],
                    [1.0, -decay],
                    shifted
                )

                response[start:end, column] = filtered.astype(np.float32)
                column += 1

    base = np.column_stack([
        raw,
        current_proxy,
        speed_torque_proxy,
        history
    ]).astype(np.float32)

    thermal_parts = [response, initial]

    for tau in THERMAL_TAUS:
        decay_age = np.exp(-age / tau).astype(np.float32)
        thermal_parts.append(decay_age[:, None])
        thermal_parts.append(initial * decay_age[:, None])

    thermal = np.column_stack(thermal_parts).astype(np.float32)

    return base, thermal


def fit_feature_map(base, thermal, fit_idx, params, seed):
    base_scaler = StandardScaler()
    base_scaler.fit(base[fit_idx])

    standardized = base_scaler.transform(
        base[fit_idx]
    ).astype(np.float64)

    pca = None

    if params["reduction"] == "pca":
        pca = PCA(
            n_components=PCA_VARIANCE,
            svd_solver="full"
        )
        standardized = pca.fit_transform(standardized)

    rng = np.random.default_rng(seed)
    count = int(params["landmarks"])

    landmark_indices = rng.choice(
        len(standardized),
        size=count,
        replace=False
    )

    landmarks = standardized[landmark_indices].copy()

    distance_indices = rng.choice(
        len(standardized),
        size=min(2000, len(standardized)),
        replace=False
    )

    landmark_distance_indices = rng.choice(
        len(landmarks),
        size=min(128, len(landmarks)),
        replace=False
    )

    distances = cdist(
        standardized[distance_indices],
        landmarks[landmark_distance_indices],
        metric="sqeuclidean"
    )

    positive = distances[distances > 1e-12]

    if len(positive):
        median_distance = float(np.sqrt(np.median(positive)))
    else:
        median_distance = 1.0

    sigma = max(
        median_distance * float(params["width_multiplier"]),
        1e-8
    )

    rbf = np.exp(
        -cdist(standardized, landmarks, metric="sqeuclidean")
        / (2.0 * sigma * sigma)
    )

    rbf_scaler = StandardScaler()
    rbf_scaler.fit(rbf)

    thermal_scaler = None

    if params["method"] == "Thermal_Ridge":
        thermal_scaler = StandardScaler()
        thermal_scaler.fit(thermal[fit_idx])

    return {
        "base_scaler": base_scaler,
        "pca": pca,
        "landmarks": landmarks,
        "sigma": sigma,
        "rbf_scaler": rbf_scaler,
        "thermal_scaler": thermal_scaler,
        "method": params["method"],
        "pca_components": (
            int(pca.n_components_)
            if pca is not None
            else int(standardized.shape[1])
        )
    }


def transform_batch(base, thermal, indices, feature_map):
    standardized = feature_map["base_scaler"].transform(base[indices])

    if feature_map["pca"] is not None:
        standardized = feature_map["pca"].transform(standardized)

    distances = cdist(
        standardized,
        feature_map["landmarks"],
        metric="sqeuclidean"
    )

    rbf = np.exp(
        -distances / (2.0 * feature_map["sigma"] ** 2)
    )

    rbf = feature_map["rbf_scaler"].transform(rbf)

    parts = [
        np.ones((len(indices), 1), dtype=np.float64),
        standardized,
        rbf
    ]

    if feature_map["method"] == "Thermal_Ridge":
        parts.append(
            feature_map["thermal_scaler"].transform(thermal[indices])
        )

    return np.column_stack(parts).astype(np.float64, copy=False)


def ridge_from_statistics(gram, rhs, sum_weights, lam):
    matrix = gram.copy()
    diagonal = np.arange(1, matrix.shape[0])
    matrix[diagonal, diagonal] += float(lam) * float(sum_weights)

    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def get_weights(age_values, method):
    if method == "Thermal_Ridge":
        return np.where(
            age_values <= 30.0,
            THERMAL_STARTUP_WEIGHT,
            1.0
        ).astype(np.float64)

    return np.ones(len(age_values), dtype=np.float64)


def fit_full_model(base, thermal, y, age, train_idx, params, seed):
    transform_idx = uniform_sample(
        train_idx,
        FINAL_TRANSFORM_SAMPLE,
        seed
    )

    feature_map = fit_feature_map(
        base,
        thermal,
        transform_idx,
        params,
        seed + 17
    )

    n_features = transform_batch(
        base,
        thermal,
        transform_idx[:1],
        feature_map
    ).shape[1]

    gram = np.zeros((n_features, n_features), dtype=np.float64)
    rhs = np.zeros(n_features, dtype=np.float64)
    total_weight = 0.0

    for start in range(0, len(train_idx), BATCH_SIZE):
        indices = train_idx[start:start + BATCH_SIZE]
        X = transform_batch(base, thermal, indices, feature_map)
        weights = get_weights(age[indices], params["method"])
        square_root = np.sqrt(weights)

        weighted_X = X * square_root[:, None]
        weighted_y = y[indices] * square_root

        gram += weighted_X.T @ weighted_X
        rhs += weighted_X.T @ weighted_y
        total_weight += float(weights.sum())

    feature_map["coef"] = ridge_from_statistics(
        gram,
        rhs,
        total_weight,
        params["lambda"]
    )

    return feature_map


def predict_batches(base, thermal, indices, feature_map):
    prediction = np.empty(len(indices), dtype=np.float64)

    for start in range(0, len(indices), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(indices))
        batch_idx = indices[start:end]

        X = transform_batch(
            base,
            thermal,
            batch_idx,
            feature_map
        )

        prediction[start:end] = X @ feature_map["coef"]

    return prediction


def calculate_metrics(y_true, y_pred):
    error = (
        np.asarray(y_pred, dtype=np.float64)
        - np.asarray(y_true, dtype=np.float64)
    )

    return {
        "rows": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias_pred_minus_actual": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error)))
    }


def select_models(
    base,
    thermal,
    y,
    age,
    groups,
    available_idx,
    scenario_seed,
    include_thermal
):
    splitter = GroupShuffleSplit(
        n_splits=2,
        test_size=0.25,
        random_state=SEED
    )

    splits = []
    split_records = []
    local_groups = groups[available_idx]

    for split_number, (train_pos, validation_pos) in enumerate(
        splitter.split(
            np.zeros(len(available_idx)),
            groups=local_groups
        ),
        start=1
    ):
        train_idx = available_idx[train_pos]
        validation_idx = available_idx[validation_pos]

        train_sample = uniform_sample(
            train_idx,
            INNER_TRAIN_SAMPLE,
            scenario_seed + 1000 * split_number
        )

        validation_sample = uniform_sample(
            validation_idx,
            INNER_VALIDATION_SAMPLE,
            scenario_seed + 1000 * split_number + 1
        )

        startup_validation = validation_idx[
            age[validation_idx] <= 30.0
        ]

        if len(startup_validation) == 0:
            raise RuntimeError("No startup validation rows found.")

        splits.append({
            "train": train_sample,
            "validation": validation_sample,
            "startup": startup_validation
        })

        for role, indices in [
            ("train", train_idx),
            ("validation", validation_idx)
        ]:
            for profile_id in np.unique(groups[indices]):
                split_records.append({
                    "inner_split": split_number,
                    "profile_id": profile_id,
                    "role": role
                })

        print(
            f"    Inner {split_number}: "
            f"{len(np.unique(groups[train_idx]))} train profiles, "
            f"{len(np.unique(groups[validation_idx]))} validation profiles"
        )

    methods = ["RBF_Base"]

    if include_thermal:
        methods.append("Thermal_Ridge")

    selected = {}
    selected_predictions = {}
    all_records = []

    for method in methods:
        print(f"    Tuning {method}...")

        best_score = np.inf
        best_params = None
        best_predictions = None

        for reduction in REDUCTIONS:
            for count in LANDMARK_COUNTS:
                for width in WIDTHS:
                    cached = []

                    for split_id, split in enumerate(splits):
                        params = {
                            "method": method,
                            "reduction": reduction,
                            "landmarks": count,
                            "width_multiplier": width
                        }

                        feature_map = fit_feature_map(
                            base,
                            thermal,
                            split["train"],
                            params,
                            scenario_seed + 10000 * split_id + count
                        )

                        X_train = transform_batch(
                            base,
                            thermal,
                            split["train"],
                            feature_map
                        )

                        X_validation = transform_batch(
                            base,
                            thermal,
                            split["validation"],
                            feature_map
                        )

                        X_startup = transform_batch(
                            base,
                            thermal,
                            split["startup"],
                            feature_map
                        )

                        weights = get_weights(
                            age[split["train"]],
                            method
                        )

                        square_root = np.sqrt(weights)
                        weighted_X = X_train * square_root[:, None]
                        weighted_y = y[split["train"]] * square_root

                        cached.append({
                            "gram": weighted_X.T @ weighted_X,
                            "rhs": weighted_X.T @ weighted_y,
                            "sum_weights": float(weights.sum()),
                            "X_validation": X_validation,
                            "X_startup": X_startup,
                            "validation_idx": split["validation"],
                            "startup_idx": split["startup"],
                            "components": feature_map["pca_components"]
                        })

                        del X_train, weighted_X, weighted_y, feature_map

                    for lam in LAMBDAS:
                        overall_sse = 0.0
                        overall_count = 0
                        startup_sse = 0.0
                        startup_count = 0
                        predictions = []

                        for item in cached:
                            coef = ridge_from_statistics(
                                item["gram"],
                                item["rhs"],
                                item["sum_weights"],
                                lam
                            )

                            pred_validation = item["X_validation"] @ coef
                            pred_startup = item["X_startup"] @ coef

                            error = (
                                y[item["validation_idx"]]
                                - pred_validation
                            )

                            error_startup = (
                                y[item["startup_idx"]]
                                - pred_startup
                            )

                            overall_sse += float(error @ error)
                            overall_count += len(error)

                            startup_sse += float(
                                error_startup @ error_startup
                            )
                            startup_count += len(error_startup)

                            predictions.append({
                                "validation": pred_validation,
                                "startup": pred_startup
                            })

                        overall_mse = overall_sse / overall_count
                        startup_mse = startup_sse / startup_count

                        if method == "RBF_Base":
                            score = np.sqrt(overall_mse)
                        else:
                            score = np.sqrt(
                                0.5 * overall_mse
                                + 0.5 * startup_mse
                            )

                        record = {
                            "method": method,
                            "reduction": reduction,
                            "landmarks": count,
                            "width_multiplier": width,
                            "lambda": lam,
                            "inner_overall_rmse": float(
                                np.sqrt(overall_mse)
                            ),
                            "inner_startup_rmse": float(
                                np.sqrt(startup_mse)
                            ),
                            "inner_selection_score": float(score),
                            "mean_pca_components": float(
                                np.mean([
                                    item["components"]
                                    for item in cached
                                ])
                            )
                        }

                        all_records.append(record)

                        if score < best_score:
                            best_score = score
                            best_params = record.copy()
                            best_predictions = predictions

                    del cached
                    gc.collect()

        selected[method] = best_params
        selected_predictions[method] = best_predictions

        print(
            f"      {method}: "
            f"{best_params['reduction']}, "
            f"landmarks={best_params['landmarks']}, "
            f"width={best_params['width_multiplier']}, "
            f"lambda={best_params['lambda']:.1e}"
        )

    transition_records = []
    best_transition = None

    if include_thermal:
        best_transition_score = np.inf

        for tau in TRANSITION_TAUS:
            overall_sse = 0.0
            overall_count = 0
            startup_sse = 0.0
            startup_count = 0

            for split_id, split in enumerate(splits):
                base_pred = selected_predictions["RBF_Base"][split_id]
                thermal_pred = selected_predictions["Thermal_Ridge"][split_id]

                weight_validation = np.exp(
                    -age[split["validation"]] / tau
                )

                weight_startup = np.exp(
                    -age[split["startup"]] / tau
                )

                hybrid_validation = (
                    weight_validation * thermal_pred["validation"]
                    + (1.0 - weight_validation) * base_pred["validation"]
                )

                hybrid_startup = (
                    weight_startup * thermal_pred["startup"]
                    + (1.0 - weight_startup) * base_pred["startup"]
                )

                error = y[split["validation"]] - hybrid_validation
                error_startup = y[split["startup"]] - hybrid_startup

                overall_sse += float(error @ error)
                overall_count += len(error)

                startup_sse += float(error_startup @ error_startup)
                startup_count += len(error_startup)

            overall_mse = overall_sse / overall_count
            startup_mse = startup_sse / startup_count

            score = np.sqrt(
                0.8 * overall_mse + 0.2 * startup_mse
            )

            record = {
                "transition_tau_seconds": float(tau),
                "inner_overall_rmse": float(np.sqrt(overall_mse)),
                "inner_startup_rmse": float(np.sqrt(startup_mse)),
                "inner_selection_score": float(score)
            }

            transition_records.append(record)

            if score < best_transition_score:
                best_transition_score = score
                best_transition = record.copy()

        print(
            f"      Transition tau: "
            f"{best_transition['transition_tau_seconds']} s"
        )

    return (
        selected,
        best_transition,
        all_records,
        transition_records,
        split_records
    )


def predict_dual_profiles(
    model_bundle,
    input_df,
    scenario,
    profile_column=None
):
    if scenario not in model_bundle["scenarios"]:
        raise ValueError(f"Unknown scenario: {scenario}")

    if model_bundle["dt_seconds"] != DT:
        raise ValueError("Model DT does not match current feature code.")

    if model_bundle["thermal_taus_seconds"] != THERMAL_TAUS:
        raise ValueError(
            "Model thermal taus do not match current feature code."
        )

    columns = model_bundle["scenarios"][scenario]["columns"]
    missing = sorted(set(columns) - set(input_df.columns))

    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    if len(input_df) == 0:
        raise ValueError("Input is empty.")

    values = input_df[columns].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite input values.")

    if profile_column is None:
        groups = np.zeros(len(input_df), dtype=np.int64)
    else:
        if profile_column not in input_df.columns:
            raise ValueError(f"Missing profile column: {profile_column}")

        groups = input_df[profile_column].to_numpy()

    age = make_age(groups)
    base, thermal = build_features(input_df, columns, groups, age)
    indices = np.arange(len(input_df), dtype=np.int64)

    scenario_model = model_bundle["scenarios"][scenario]

    base_prediction = predict_batches(
        base,
        thermal,
        indices,
        scenario_model["base_model"]
    )

    if scenario == "operational":
        thermal_prediction = predict_batches(
            base,
            thermal,
            indices,
            scenario_model["thermal_model"]
        )

        tau = scenario_model["transition_tau_seconds"]
        weight = np.exp(-age / tau)

        prediction = (
            weight * thermal_prediction
            + (1.0 - weight) * base_prediction
        )
    else:
        prediction = base_prediction

    return pd.DataFrame({
        "age_seconds": age,
        "predicted_pm": prediction
    }, index=input_df.index)


start_time = time.time()

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

CV_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

with threadpool_limits(limits=THREADS):
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    required = set(
        ["profile_id", "pm"] + SCENARIOS["sensor_assisted"]
    )

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values in required columns.")

    for column in ["pm"] + SCENARIOS["sensor_assisted"]:
        if not np.isfinite(df[column].to_numpy()).all():
            raise ValueError(f"Non-finite values in {column}")

    groups = df["profile_id"].to_numpy()
    y = df["pm"].to_numpy(dtype=np.float64)
    age = make_age(groups)

    if len(np.unique(groups)) < 4:
        raise ValueError("Too few profiles for group validation.")

    data_hash = sha256_file(DATA_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Profiles: {len(np.unique(groups))}")
    print(f"Dataset SHA-256: {data_hash}")

    folds = list(
        GroupKFold(n_splits=3).split(
            np.zeros(len(df)),
            y,
            groups
        )
    )

    fold_of_row = np.zeros(len(df), dtype=np.int8)
    outer_split_records = []

    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_of_row[test_idx] = fold_number

        for role, indices in [("train", train_idx), ("test", test_idx)]:
            for profile_id in np.unique(groups[indices]):
                outer_split_records.append({
                    "outer_fold": fold_number,
                    "profile_id": profile_id,
                    "role": role
                })

    pooled_records = []
    fold_records = []
    profile_records = []
    band_records = []
    selected_records = []
    selection_records = []
    transition_records = []
    inner_split_records = []

    bands = np.select(
        [age <= 30.0, age <= 300.0, age <= 1800.0],
        [
            "0_to_30_s",
            "over_30_to_300_s",
            "over_300_to_1800_s"
        ],
        default="over_1800_s"
    )

    print("\n" + "=" * 80)
    print("THREE-FOLD GROUP CROSS-VALIDATION")

    for scenario_number, (scenario, columns) in enumerate(SCENARIOS.items()):
        print("\n" + "=" * 80)
        print(f"SCENARIO: {scenario}")

        base, thermal = build_features(df, columns, groups, age)

        oof = {
            method: np.full(len(df), np.nan, dtype=np.float64)
            for method in METHODS
        }

        for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
            print(f"\nOuter fold {fold_number}/3")

            scenario_seed = (
                SEED
                + 100000 * fold_number
                + 10000 * scenario_number
            )

            (
                selected,
                transition,
                all_selection,
                all_transitions,
                inner_splits
            ) = select_models(
                base,
                thermal,
                y,
                age,
                groups,
                train_idx,
                scenario_seed,
                include_thermal=True
            )

            for record in selected.values():
                selected_records.append({
                    **record,
                    "scenario": scenario,
                    "outer_fold": fold_number
                })

            for record in all_selection:
                selection_records.append({
                    **record,
                    "scenario": scenario,
                    "outer_fold": fold_number
                })

            for record in all_transitions:
                transition_records.append({
                    **record,
                    "scenario": scenario,
                    "outer_fold": fold_number
                })

            for record in inner_splits:
                inner_split_records.append({
                    **record,
                    "scenario": scenario,
                    "outer_fold": fold_number
                })

            print("    Fitting RBF_Base...")

            base_model = fit_full_model(
                base,
                thermal,
                y,
                age,
                train_idx,
                selected["RBF_Base"],
                scenario_seed + 500
            )

            base_pred = predict_batches(
                base,
                thermal,
                test_idx,
                base_model
            )

            del base_model
            gc.collect()

            print("    Fitting Thermal_Ridge...")

            thermal_model = fit_full_model(
                base,
                thermal,
                y,
                age,
                train_idx,
                selected["Thermal_Ridge"],
                scenario_seed + 500
            )

            thermal_pred = predict_batches(
                base,
                thermal,
                test_idx,
                thermal_model
            )

            del thermal_model
            gc.collect()

            tau = transition["transition_tau_seconds"]
            weight = np.exp(-age[test_idx] / tau)

            hybrid_pred = (
                weight * thermal_pred
                + (1.0 - weight) * base_pred
            )

            predictions = {
                "RBF_Base": base_pred,
                "Thermal_Ridge": thermal_pred,
                "Hybrid_Ridge": hybrid_pred
            }

            for method, prediction in predictions.items():
                oof[method][test_idx] = prediction

                record = calculate_metrics(y[test_idx], prediction)
                record.update({
                    "scenario": scenario,
                    "outer_fold": fold_number,
                    "method": method,
                    "transition_tau_seconds": tau
                })

                fold_records.append(record)

                print(
                    f"      {method}: "
                    f"RMSE={record['rmse']:.6f}, "
                    f"MAE={record['mae']:.6f}"
                )

            del base_pred, thermal_pred, hybrid_pred, predictions, weight
            gc.collect()

        for method in METHODS:
            prediction = oof[method]

            if not np.isfinite(prediction).all():
                raise RuntimeError(
                    f"Missing OOF predictions: {scenario}, {method}"
                )

            record = calculate_metrics(y, prediction)
            record.update({
                "scenario": scenario,
                "method": method
            })
            pooled_records.append(record)

            for profile_id in np.unique(groups):
                mask = groups == profile_id
                record = calculate_metrics(y[mask], prediction[mask])
                record.update({
                    "scenario": scenario,
                    "method": method,
                    "profile_id": profile_id
                })
                profile_records.append(record)

            for band in [
                "0_to_30_s",
                "over_30_to_300_s",
                "over_300_to_1800_s",
                "over_1800_s"
            ]:
                mask = bands == band

                if mask.any():
                    record = calculate_metrics(y[mask], prediction[mask])
                    record.update({
                        "scenario": scenario,
                        "method": method,
                        "age_band": band
                    })
                    band_records.append(record)

        output = pd.DataFrame({
            "row_index": np.arange(len(df), dtype=np.int64),
            "profile_id": groups,
            "outer_fold": fold_of_row,
            "age_seconds": age,
            "actual_pm": y,
            "RBF_Base": oof["RBF_Base"],
            "Thermal_Ridge": oof["Thermal_Ridge"],
            "Hybrid_Ridge": oof["Hybrid_Ridge"]
        })

        output_path = OOF_DIR / f"oof_predictions_{scenario}.csv.gz"

        output.to_csv(
            output_path,
            index=False,
            compression="gzip",
            float_format="%.10g"
        )

        print(f"Saved OOF predictions: {output_path}")

        del base, thermal, oof, output
        gc.collect()

    pooled_df = pd.DataFrame(pooled_records)
    fold_df = pd.DataFrame(fold_records)
    profile_df = pd.DataFrame(profile_records)
    band_df = pd.DataFrame(band_records)

    pooled_df.to_csv(CV_DIR / "pooled_outer_metrics.csv", index=False)
    fold_df.to_csv(CV_DIR / "outer_fold_metrics.csv", index=False)
    profile_df.to_csv(CV_DIR / "profile_metrics.csv", index=False)
    band_df.to_csv(CV_DIR / "age_band_metrics.csv", index=False)

    pd.DataFrame(selected_records).to_csv(
        CV_DIR / "inner_selected_parameters.csv",
        index=False
    )

    pd.DataFrame(selection_records).to_csv(
        CV_DIR / "inner_selection_all.csv",
        index=False
    )

    pd.DataFrame(transition_records).to_csv(
        CV_DIR / "inner_transition_scores.csv",
        index=False
    )

    pd.DataFrame(outer_split_records).to_csv(
        CV_DIR / "outer_profile_splits.csv",
        index=False
    )

    pd.DataFrame(inner_split_records).to_csv(
        CV_DIR / "inner_profile_splits.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("POOLED OUTER METRICS")
    print(pooled_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TRAINING FINAL DUAL-SCENARIO MODEL")

    all_indices = np.arange(len(df), dtype=np.int64)

    bundle = {
        "model_name": "Dual_Hybrid_RBF_Ridge",
        "dt_seconds": DT,
        "thermal_taus_seconds": THERMAL_TAUS,
        "data_sha256": data_hash,
        "scenarios": {}
    }

    final_selected_records = []
    final_selection_records = []
    final_transition_records = []
    final_split_records = []

    for scenario_number, (scenario, columns) in enumerate(SCENARIOS.items()):
        print("\n" + "=" * 80)
        print(f"FINAL MODEL: {scenario}")

        scenario_seed = SEED + 200000 + 10000 * scenario_number
        base, thermal = build_features(df, columns, groups, age)

        (
            selected,
            transition,
            all_selection,
            all_transitions,
            inner_splits
        ) = select_models(
            base,
            thermal,
            y,
            age,
            groups,
            all_indices,
            scenario_seed,
            include_thermal=(scenario == "operational")
        )

        for record in selected.values():
            final_selected_records.append({
                **record,
                "scenario": scenario
            })

        for record in all_selection:
            final_selection_records.append({
                **record,
                "scenario": scenario
            })

        for record in all_transitions:
            final_transition_records.append({
                **record,
                "scenario": scenario
            })

        for record in inner_splits:
            final_split_records.append({
                **record,
                "scenario": scenario
            })

        print("    Fitting final RBF_Base...")

        base_model = fit_full_model(
            base,
            thermal,
            y,
            age,
            all_indices,
            selected["RBF_Base"],
            scenario_seed + 500
        )

        scenario_bundle = {
            "columns": columns,
            "base_model": base_model,
            "base_parameters": selected["RBF_Base"]
        }

        if scenario == "operational":
            print("    Fitting final Thermal_Ridge...")

            thermal_model = fit_full_model(
                base,
                thermal,
                y,
                age,
                all_indices,
                selected["Thermal_Ridge"],
                scenario_seed + 500
            )

            scenario_bundle["thermal_model"] = thermal_model
            scenario_bundle["thermal_parameters"] = selected["Thermal_Ridge"]
            scenario_bundle["transition_tau_seconds"] = transition[
                "transition_tau_seconds"
            ]

        bundle["scenarios"][scenario] = scenario_bundle

        del base, thermal
        gc.collect()

    model_path = FINAL_DIR / "dual_ridge_model.joblib"
    joblib.dump(bundle, model_path, compress=3)

    pd.DataFrame(final_selected_records).to_csv(
        FINAL_DIR / "selected_parameters.csv",
        index=False
    )

    pd.DataFrame(final_selection_records).to_csv(
        FINAL_DIR / "inner_selection_all.csv",
        index=False
    )

    pd.DataFrame(final_transition_records).to_csv(
        FINAL_DIR / "transition_scores.csv",
        index=False
    )

    pd.DataFrame(final_split_records).to_csv(
        FINAL_DIR / "inner_profile_splits.csv",
        index=False
    )

    metadata = {
        "model_name": bundle["model_name"],
        "data_path": str(DATA_PATH),
        "data_sha256": data_hash,
        "development_rows": int(len(df)),
        "development_profiles": int(len(np.unique(groups))),
        "nominal_dt_seconds": DT,
        "thermal_taus_seconds": THERMAL_TAUS,
        "transition_taus_seconds": TRANSITION_TAUS,
        "landmark_counts": LANDMARK_COUNTS,
        "width_multipliers": WIDTHS,
        "normalized_lambdas": LAMBDAS,
        "reductions": REDUCTIONS,
        "pca_variance": PCA_VARIANCE,
        "outer_validation": "3-fold GroupKFold",
        "inner_validation": "2 GroupShuffleSplit splits",
        "inner_train_sample": INNER_TRAIN_SAMPLE,
        "inner_validation_sample": INNER_VALIDATION_SAMPLE,
        "final_transform_sample": FINAL_TRANSFORM_SAMPLE,
        "thermal_startup_weight": THERMAL_STARTUP_WEIGHT,
        "startup_window": "age <= 30 seconds",
        "base_selection": "inner overall RMSE",
        "thermal_selection": "sqrt(0.5 overall MSE + 0.5 startup MSE)",
        "hybrid_selection": "sqrt(0.8 overall MSE + 0.2 startup MSE)",
        "operational_policy": "soft hybrid of RBF_Base and Thermal_Ridge",
        "sensor_assisted_policy": "RBF_Base",
        "final_ridge_training": "all development rows",
        "target_as_input": False,
        "profile_id_as_input": False,
        "oof_generation": "directly during outer cross-validation",
        "independent_test_performed": False,
        "elapsed_minutes": (time.time() - start_time) / 60.0
    }

    with open(OUTPUT_ROOT / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SELECTED PARAMETERS")
    print(pd.DataFrame(final_selected_records).to_string(index=False))

    print(f"\nModel saved: {model_path}")
    print(f"All outputs: {OUTPUT_ROOT.resolve()}")
    print(f"Completed in {(time.time() - start_time) / 60.0:.2f} minutes.")
    print("No independent test was performed.")
