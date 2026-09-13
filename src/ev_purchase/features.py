"""Feature policy definition and registry for EV purchase prediction."""

from collections.abc import Callable

import pandas as pd

FeaturePolicy = Callable[[pd.DataFrame], pd.DataFrame]


def base_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Base feature policy: returns an unmutated copy of valid raw model columns."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    return frame.copy()


BUILTIN_FEATURE_POLICIES: set[str] = {"base"}

_FEATURE_POLICIES: dict[str, FeaturePolicy] = {
    "base": base_features,
}


def register_feature_policy(name: str, policy: FeaturePolicy, overwrite: bool = False) -> None:
    """Register a new feature policy.

    Args:
        name: Unique string identifier for the policy.
        policy: Callable that accepts a DataFrame and returns a DataFrame.
        overwrite: If True, allow replacing an existing policy with the same name.

    Raises:
        TypeError: If policy is not callable.
        ValueError: If name is not a non-empty string or policy is already registered
            without overwrite=True.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be a non-empty string")
    if not callable(policy):
        raise TypeError("policy must be a callable accepting and returning a DataFrame")

    clean_name = name.strip()
    if clean_name in _FEATURE_POLICIES and not overwrite:
        raise ValueError(
            f"Policy '{clean_name}' is already registered. Set overwrite=True to replace it."
        )

    _FEATURE_POLICIES[clean_name] = policy


def unregister_feature_policy(name: str) -> None:
    """Unregister a non-builtin feature policy by name.

    Args:
        name: Policy name to unregister.

    Raises:
        ValueError: If name is not a non-empty string, is a built-in policy, or is not registered.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be a non-empty string")

    clean_name = name.strip()
    if clean_name in BUILTIN_FEATURE_POLICIES:
        raise ValueError(f"Cannot unregister built-in policy '{clean_name}'")

    if clean_name not in _FEATURE_POLICIES:
        raise ValueError(f"Policy '{clean_name}' is not registered")

    del _FEATURE_POLICIES[clean_name]


def reset_feature_policies() -> None:
    """Reset feature policy registry back to built-in defaults."""
    _FEATURE_POLICIES.clear()
    _FEATURE_POLICIES["base"] = base_features


def get_feature_policy(name: str) -> FeaturePolicy:
    """Look up a feature policy by name.

    Args:
        name: Name of the registered feature policy.

    Returns:
        The policy callable.

    Raises:
        ValueError: If name is not a non-empty string or policy is not found.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be a non-empty string")

    clean_name = name.strip()
    if clean_name not in _FEATURE_POLICIES:
        available = ", ".join(sorted(_FEATURE_POLICIES.keys()))
        raise ValueError(
            f"Unknown feature policy '{clean_name}'. Available policies: [{available}]"
        )
    return _FEATURE_POLICIES[clean_name]


def list_feature_policies() -> list[str]:
    """Return a sorted list of registered feature policy names."""
    return sorted(_FEATURE_POLICIES.keys())


def apply_feature_policy(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a named feature policy to a DataFrame without mutating the input.

    Args:
        name: Policy name.
        frame: Input DataFrame.

    Returns:
        Transformed DataFrame.

    Raises:
        TypeError: If frame is not a DataFrame or policy result is not a DataFrame.
        RuntimeError: If policy mutates the input DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    policy_fn = get_feature_policy(name)
    original_copy = frame.copy(deep=True)
    result = policy_fn(frame)

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Feature policy '{name}' must return a pandas DataFrame, got {type(result).__name__}"
        )

    if not frame.equals(original_copy):
        raise RuntimeError(f"Feature policy '{name}' mutated its input DataFrame")

    return result
