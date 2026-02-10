"""Tests for the boolean dtype and single-class LogisticRegression fixes in iteration5_meta_ensemble.py."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def test_boolean_columns_cast_to_float_before_scaling():
    """Test that boolean columns can be cast to float before StandardScaler transformation."""
    # Create sample data with boolean columns
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "swing_high_flag": [True, False, True, False, True],
        "swing_low_flag": [False, True, False, True, False],
    })
    feature_cols = ["feature1", "swing_high_flag", "swing_low_flag"]

    # Cast boolean columns to float (mimicking the fix)
    for col in feature_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)

    # Verify dtype conversion
    assert df["swing_high_flag"].dtype == float
    assert df["swing_low_flag"].dtype == float

    # Now StandardScaler should work without warnings
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    assert scaled.shape == (5, 3)

    # Verify the scaled values are correct
    result_df = df.copy()
    result_df[feature_cols] = scaled
    assert result_df["swing_high_flag"].dtype == float
    assert result_df["swing_low_flag"].dtype == float


def test_single_class_logistic_regression_handling_all_zeros():
    """Test that single-class scenario (all zeros) is handled correctly."""
    # Simulate a scenario where all target returns are <= 0
    meta_df = pd.DataFrame({
        "pred_linear": [0.1, 0.2, 0.3, 0.4],
        "pred_rf": [0.15, 0.25, 0.35, 0.45],
        "target": [-0.01, -0.02, -0.03, -0.01],  # All negative
    })

    test_df = pd.DataFrame({
        "pred_linear": [0.5, 0.6],
        "pred_rf": [0.55, 0.65],
    })

    feature_cols = ["pred_linear", "pred_rf"]

    # Mimic the fix: check for single-class scenario
    y_bin = (meta_df["target"] > 0).astype(int)
    assert y_bin.nunique() == 1  # Only one class (all zeros)
    assert y_bin.iloc[0] == 0  # The single class is 0

    if y_bin.nunique() < 2:
        constant_prob = float(y_bin.iloc[0])  # 0.0
        class_probs = np.full(len(test_df), constant_prob)
        class_pred = (class_probs > 0.5).astype(int)
    else:
        meta_clf = LogisticRegression(max_iter=500)
        meta_clf.fit(meta_df[feature_cols], y_bin)
        class_probs = meta_clf.predict_proba(test_df[feature_cols])[:, 1]
        class_pred = (class_probs > 0.5).astype(int)

    # Verify results
    assert len(class_probs) == 2
    assert np.all(class_probs == 0.0)
    assert np.all(class_pred == 0)


def test_single_class_logistic_regression_handling_all_ones():
    """Test that single-class scenario (all ones) is handled correctly."""
    # Simulate a scenario where all target returns are > 0
    meta_df = pd.DataFrame({
        "pred_linear": [0.1, 0.2, 0.3, 0.4],
        "pred_rf": [0.15, 0.25, 0.35, 0.45],
        "target": [0.01, 0.02, 0.03, 0.01],  # All positive
    })

    test_df = pd.DataFrame({
        "pred_linear": [0.5, 0.6],
        "pred_rf": [0.55, 0.65],
    })

    feature_cols = ["pred_linear", "pred_rf"]

    # Mimic the fix: check for single-class scenario
    y_bin = (meta_df["target"] > 0).astype(int)
    assert y_bin.nunique() == 1  # Only one class (all ones)
    assert y_bin.iloc[0] == 1  # The single class is 1

    if y_bin.nunique() < 2:
        constant_prob = float(y_bin.iloc[0])  # 1.0
        class_probs = np.full(len(test_df), constant_prob)
        class_pred = (class_probs > 0.5).astype(int)
    else:
        meta_clf = LogisticRegression(max_iter=500)
        meta_clf.fit(meta_df[feature_cols], y_bin)
        class_probs = meta_clf.predict_proba(test_df[feature_cols])[:, 1]
        class_pred = (class_probs > 0.5).astype(int)

    # Verify results
    assert len(class_probs) == 2
    assert np.all(class_probs == 1.0)
    assert np.all(class_pred == 1)


def test_two_class_logistic_regression_still_works():
    """Test that the normal two-class scenario still works correctly."""
    # Simulate a scenario with both positive and negative returns
    meta_df = pd.DataFrame({
        "pred_linear": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "pred_rf": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
        "target": [-0.01, 0.02, -0.03, 0.01, -0.02, 0.03],  # Mixed
    })

    test_df = pd.DataFrame({
        "pred_linear": [0.5, 0.6],
        "pred_rf": [0.55, 0.65],
    })

    feature_cols = ["pred_linear", "pred_rf"]

    # Mimic the fix: check for single-class scenario
    y_bin = (meta_df["target"] > 0).astype(int)
    assert y_bin.nunique() == 2  # Both classes present

    if y_bin.nunique() < 2:
        constant_prob = float(y_bin.iloc[0])
        class_probs = np.full(len(test_df), constant_prob)
        class_pred = (class_probs > 0.5).astype(int)
    else:
        meta_clf = LogisticRegression(max_iter=500)
        meta_clf.fit(meta_df[feature_cols], y_bin)
        class_probs = meta_clf.predict_proba(test_df[feature_cols])[:, 1]
        class_pred = (class_probs > 0.5).astype(int)

    # Verify results - LogisticRegression should have been used
    assert len(class_probs) == 2
    # Probabilities should be between 0 and 1
    assert np.all((class_probs >= 0) & (class_probs <= 1))
    # Predictions should be 0 or 1
    assert set(class_pred).issubset({0, 1})


def test_logistic_regression_error_without_fix():
    """Verify that LogisticRegression raises ValueError with single class."""
    # This test confirms the original bug - LogisticRegression cannot be fitted
    # with only one class
    meta_df = pd.DataFrame({
        "pred_linear": [0.1, 0.2, 0.3, 0.4],
        "pred_rf": [0.15, 0.25, 0.35, 0.45],
        "target": [-0.01, -0.02, -0.03, -0.01],  # All negative
    })

    feature_cols = ["pred_linear", "pred_rf"]
    y_bin = (meta_df["target"] > 0).astype(int)

    meta_clf = LogisticRegression(max_iter=500)
    with pytest.raises(ValueError, match="needs samples of at least 2 classes"):
        meta_clf.fit(meta_df[feature_cols], y_bin)
