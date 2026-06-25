"""Second evaluator for the two generic 5x5 matrix invariant problem.

This evaluator is aimed at the stronger target:

* g is non-central and numerically generates a degree-5 algebra over the center.
* f is the returned monic quintic and is close to the minimal polynomial of g.
* h is close to a cyclic order-5 automorphism permuting the roots of f.

Unlike evaluator.py, this file does not replace the returned f by charpoly(g).
Instead it rewards f for matching charpoly(g), while separately checking that
I, g, ..., g^4 are independent so the characteristic polynomial is plausibly
also the minimal polynomial.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult


_BASE_EVALUATOR_PATH = Path(__file__).with_name("evaluator.py")
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_two_matrix_invariants_base_evaluator",
    _BASE_EVALUATOR_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"Could not load base evaluator from {_BASE_EVALUATOR_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)


MATRIX_SIZE = base.MATRIX_SIZE
F_DEGREE = base.F_DEGREE
H_DEGREE = base.H_DEGREE
STAGE1_SAMPLES = base.STAGE1_SAMPLES
STAGE2_SAMPLES = base.STAGE2_SAMPLES
MAX_LOSS = base.MAX_LOSS
MONIC_TOLERANCE = base.MONIC_TOLERANCE
DENOMINATOR_EPS = base.DENOMINATOR_EPS

TOTAL_EVALUATION_2_PARTS = 13
MEASURE_EPS = 1e-6
G_MEMBERSHIP_SAMPLE_CAP = 8
DEFAULT_MIN_RELATIVE_G_SCALE = 0.05
DEFAULT_MAX_RELATIVE_G_SCALE = 25.0

DEFAULT_EVAL2_WEIGHTS: Dict[str, float] = {
    "LAMBDA_G_IN_G": 1.0,
    "LAMBDA_EVAL_TO_ZERO": 1.0,
    "LAMBDA_CHARPOLY_MATCH": 2.0,
    "LAMBDA_ORBIT_ZEROES_F": 1.0,
    "LAMBDA_NEWTON_SUMS": 1.0,
    "LAMBDA_QUOTIENT_AUTOMORPHISM": 1.0,
    "LAMBDA_MATRIX_CYCLE": 1.0,
    "LAMBDA_SEPARABLE": 0.05,
    "LAMBDA_DEGREE5": 0.10,
    "LAMBDA_CYCLE_DISTINCT": 0.05,
    "LAMBDA_NOT_CENTRAL": 0.10,
    "LAMBDA_XY_DEPENDENCE": 0.05,
    "LAMBDA_G_SCALE": 0.10,
}


def _env_float(name: str, default: float) -> float:
    """Read either EVAL2_NAME or NAME as a finite non-negative float."""
    for env_name in (f"EVAL2_{name}", name):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            return max(0.0, value)
    return default


def _weight(name: str) -> float:
    return _env_float(name, DEFAULT_EVAL2_WEIGHTS[name])


def _measure_barrier(measure: float, epsilon: float = MEASURE_EPS) -> float:
    """A zero-near-one, very large-near-zero barrier for [0, 1] measures."""
    if not math.isfinite(measure) or measure <= 0.0:
        return MAX_LOSS
    value = 1.0 / (measure + epsilon) - 1.0 / (1.0 + epsilon)
    return float(min(MAX_LOSS, max(0.0, value)))


def _safe_rms(values: Sequence[float]) -> float:
    if not values:
        return MAX_LOSS
    clipped = [min(MAX_LOSS, max(0.0, float(value))) for value in values]
    return float(math.sqrt(sum(value * value for value in clipped) / len(clipped)))


def _identity() -> np.ndarray:
    return np.eye(MATRIX_SIZE, dtype=float)


def _zero() -> np.ndarray:
    return np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=float)


def _frobenius_norm(matrix: np.ndarray) -> float:
    if not np.all(np.isfinite(matrix)):
        return MAX_LOSS
    return float(np.linalg.norm(matrix, ord="fro"))


def _normalized_matrix_delta(left: np.ndarray, right: np.ndarray) -> float:
    numerator = base._matrix_norm(left - right)
    denominator = base._matrix_norm(left) + base._matrix_norm(right) + DENOMINATOR_EPS
    return float(min(MAX_LOSS, numerator / denominator))


def _specializations(
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return list(samples) if samples is not None else base._random_specializations(sample_count, seed)


def _eval_g_without_mutating_inputs(
    g: Any,
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    extra_context: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, float]:
    """Evaluate g and return its value plus normalized X/Y input mutation."""
    x_copy = np.array(x_matrix, dtype=float, copy=True)
    y_copy = np.array(y_matrix, dtype=float, copy=True)
    context: Dict[str, Any] = {"x_matrix": x_copy, "y_matrix": y_copy}
    if extra_context:
        context.update(extra_context)

    value = base._eval_element(g, context)
    x_mutation = float(np.linalg.norm(x_copy - x_matrix)) / (
        float(np.linalg.norm(x_matrix)) + DENOMINATOR_EPS
    )
    y_mutation = float(np.linalg.norm(y_copy - y_matrix)) / (
        float(np.linalg.norm(y_matrix)) + DENOMINATOR_EPS
    )
    return value, max(x_mutation, y_mutation)


def _g_membership_metrics(
    g: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Dict[str, float]:
    """Numerically guard that g behaves like a real rational function of X,Y.

    A Python callable cannot be certified symbolically here, so this check
    rejects common non-G loopholes: randomness, dependence on evaluator context
    variables such as base_value/current_value, mutation of X/Y, non-real or
    non-finite output, and discontinuous branchy behavior on generic samples.
    """
    specializations = _specializations(sample_count, seed, samples)[:G_MEMBERSHIP_SAMPLE_CAP]
    if not specializations:
        return {
            "g_in_g_loss": MAX_LOSS,
            "g_in_g_measure": 0.0,
            "g_determinism_error": MAX_LOSS,
            "g_rng_dependence_error": MAX_LOSS,
            "g_context_dependence_error": MAX_LOSS,
            "g_input_mutation_error": MAX_LOSS,
            "g_local_continuity_error": MAX_LOSS,
        }

    rng = np.random.default_rng(seed + 404_001)
    determinism_errors = []
    rng_errors = []
    context_errors = []
    mutation_errors = []
    continuity_errors = []

    py_random_state = random.getstate()
    np_random_state = np.random.get_state()
    try:
        for index, (x_matrix, y_matrix) in enumerate(specializations):
            base_value, mutation = _eval_g_without_mutating_inputs(g, x_matrix, y_matrix)
            repeated_value, repeated_mutation = _eval_g_without_mutating_inputs(
                g,
                x_matrix,
                y_matrix,
            )
            determinism_errors.append(_normalized_matrix_delta(base_value, repeated_value))
            mutation_errors.append(max(mutation, repeated_mutation))

            extra_context = {
                "base_value": rng.normal(0.0, 0.25, size=(MATRIX_SIZE, MATRIX_SIZE)),
                "current_value": rng.normal(0.0, 0.25, size=(MATRIX_SIZE, MATRIX_SIZE)),
            }
            context_value, context_mutation = _eval_g_without_mutating_inputs(
                g,
                x_matrix,
                y_matrix,
                extra_context,
            )
            context_errors.append(_normalized_matrix_delta(base_value, context_value))
            mutation_errors.append(context_mutation)

            random.seed(seed + 10_000 + index)
            np.random.seed((seed + 20_000 + index) % (2**32 - 1))
            seeded_value_1, seeded_mutation_1 = _eval_g_without_mutating_inputs(
                g,
                x_matrix,
                y_matrix,
            )
            random.seed(seed + 30_000 + index)
            np.random.seed((seed + 40_000 + index) % (2**32 - 1))
            seeded_value_2, seeded_mutation_2 = _eval_g_without_mutating_inputs(
                g,
                x_matrix,
                y_matrix,
            )
            rng_errors.append(_normalized_matrix_delta(seeded_value_1, seeded_value_2))
            mutation_errors.append(max(seeded_mutation_1, seeded_mutation_2))

            perturbation_scale = 1e-6
            x_perturbed = x_matrix + perturbation_scale * (
                1.0 + float(np.linalg.norm(x_matrix)) / MATRIX_SIZE
            ) * rng.normal(size=x_matrix.shape)
            y_perturbed = y_matrix + perturbation_scale * (
                1.0 + float(np.linalg.norm(y_matrix)) / MATRIX_SIZE
            ) * rng.normal(size=y_matrix.shape)
            perturbed_value, perturbed_mutation = _eval_g_without_mutating_inputs(
                g,
                x_perturbed,
                y_perturbed,
            )
            continuity_ratio = _normalized_matrix_delta(base_value, perturbed_value)
            continuity_errors.append(max(0.0, continuity_ratio - 1e-3) / 1e-3)
            mutation_errors.append(perturbed_mutation)
    finally:
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)

    determinism_error = _safe_rms(determinism_errors)
    rng_error = _safe_rms(rng_errors)
    context_error = _safe_rms(context_errors)
    mutation_error = _safe_rms(mutation_errors)
    continuity_error = _safe_rms(continuity_errors)
    g_in_g_loss = min(
        MAX_LOSS,
        1000.0 * determinism_error
        + 1000.0 * rng_error
        + 1000.0 * context_error
        + 1000.0 * mutation_error
        + 2.0 * continuity_error,
    )

    return {
        "g_in_g_loss": float(g_in_g_loss),
        "g_in_g_measure": float(1.0 / (1.0 + g_in_g_loss)),
        "g_determinism_error": float(determinism_error),
        "g_rng_dependence_error": float(rng_error),
        "g_context_dependence_error": float(context_error),
        "g_input_mutation_error": float(mutation_error),
        "g_local_continuity_error": float(continuity_error),
    }


def _charpoly_coefficients(matrix: np.ndarray) -> List[float]:
    coeffs = np.poly(matrix)
    coeffs = np.real_if_close(coeffs, tol=1000)
    return [float(np.real(coefficient)) for coefficient in coeffs]


def _apply_h(h_coefficients: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    return sum(base._matrix_polynomial_terms(h_coefficients, matrix), _zero())


def _h_orbit_for_context(
    h: Any,
    context: Dict[str, Any],
    g_value: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray, List[float]]:
    context["base_value"] = g_value
    context["current_value"] = g_value
    h_coefficients = base._coefficients_from_spec(h, H_DEGREE, context, "h")

    orbit = [g_value]
    value = g_value
    for step in range(5):
        value = _apply_h(h_coefficients, value)
        if not np.all(np.isfinite(value)):
            raise ValueError("h-iterate produced a non-finite matrix.")
        context["current_value"] = value
        if step < 4:
            orbit.append(value)
    return orbit, value, h_coefficients


def _charpoly_match_loss(
    g: Any,
    f: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    ratios = []
    for x_matrix, y_matrix in _specializations(sample_count, seed, samples):
        context: Dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
        g_value = base._eval_element(g, context)
        context["base_value"] = g_value
        context["current_value"] = g_value
        f_coefficients = np.asarray(
            base._coefficients_from_spec(f, F_DEGREE, context, "f"),
            dtype=float,
        )
        charpoly_coefficients = np.asarray(_charpoly_coefficients(g_value), dtype=float)
        numerator = float(np.linalg.norm(f_coefficients - charpoly_coefficients))
        denominator = (
            float(np.linalg.norm(f_coefficients))
            + float(np.linalg.norm(charpoly_coefficients))
            + DENOMINATOR_EPS
        )
        ratios.append(min(MAX_LOSS, numerator / denominator))
    return _safe_rms(ratios)


def _degree5_power_span_measure(
    g: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    """Return mean smallest singular value of normalized columns vec(g^i)."""
    measures = []
    for x_matrix, y_matrix in _specializations(sample_count, seed, samples):
        context: Dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
        g_value = base._eval_element(g, context)
        powers = [_identity()]
        for _ in range(4):
            powers.append(powers[-1] @ g_value)

        columns = []
        failed = False
        for power in powers:
            vector = np.asarray(power, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= DENOMINATOR_EPS:
                failed = True
                break
            columns.append(vector / norm)
        if failed:
            measures.append(0.0)
            continue

        singular_values = np.linalg.svd(np.column_stack(columns), compute_uv=False)
        smallest = float(np.min(singular_values))
        measures.append(max(0.0, min(1.0, smallest)))
    if not measures:
        return 0.0
    return float(sum(measures) / len(measures))


def _xy_dependence_measure(
    g: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    """Measure variation of the trace-free direction of g across samples."""
    directions = []
    for x_matrix, y_matrix in _specializations(sample_count, seed, samples):
        context: Dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
        g_value = base._eval_element(g, context)
        trace_free = g_value - float(np.trace(g_value)) / MATRIX_SIZE * _identity()
        norm = _frobenius_norm(trace_free)
        if not math.isfinite(norm) or norm <= DENOMINATOR_EPS:
            continue
        directions.append(trace_free.reshape(-1) / norm)

    if len(directions) < 2:
        return 0.0

    distances = []
    for i, left in enumerate(directions):
        for right in directions[i + 1 :]:
            inner = float(np.dot(left, right))
            distances.append(math.sqrt(max(0.0, 1.0 - min(1.0, inner * inner))))
    if not distances:
        return 0.0
    return float(sum(distances) / len(distances))


def _g_scale_metrics(
    g: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Tuple[float, float]:
    relative_scales = []
    for x_matrix, y_matrix in _specializations(sample_count, seed, samples):
        context: Dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
        g_value = base._eval_element(g, context)
        trace_free = g_value - float(np.trace(g_value)) / MATRIX_SIZE * _identity()
        reference = 0.5 * (base._matrix_norm(x_matrix) + base._matrix_norm(y_matrix))
        relative_scales.append(base._matrix_norm(trace_free) / (reference + DENOMINATOR_EPS))

    if not relative_scales:
        return 0.0, MAX_LOSS

    mean_scale = float(sum(relative_scales) / len(relative_scales))
    min_scale = _env_float("MIN_RELATIVE_G_SCALE", DEFAULT_MIN_RELATIVE_G_SCALE)
    max_scale = _env_float("MAX_RELATIVE_G_SCALE", DEFAULT_MAX_RELATIVE_G_SCALE)
    low_penalty = max(0.0, min_scale - mean_scale) / max(min_scale, DENOMINATOR_EPS)
    high_penalty = max(0.0, mean_scale - max_scale) / max(max_scale, DENOMINATOR_EPS)
    return mean_scale, float(low_penalty * low_penalty + high_penalty * high_penalty)


def _matrix_cycle_metrics(
    g: Any,
    h: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Tuple[float, float]:
    closure_ratios = []
    distinct_measures = []
    for x_matrix, y_matrix in _specializations(sample_count, seed, samples):
        context: Dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
        g_value = base._eval_element(g, context)
        orbit, h5_value, _ = _h_orbit_for_context(h, context, g_value)

        closure_numerator = base._matrix_norm(h5_value - g_value)
        closure_denominator = base._matrix_norm(h5_value) + base._matrix_norm(g_value) + DENOMINATOR_EPS
        closure_ratios.append(closure_numerator / closure_denominator)

        pairwise = []
        for i, left in enumerate(orbit):
            for right in orbit[i + 1 :]:
                numerator = base._matrix_norm(left - right)
                denominator = base._matrix_norm(left) + base._matrix_norm(right) + DENOMINATOR_EPS
                pairwise.append(numerator / denominator)
        distinct_measures.append(min(pairwise) if pairwise else 0.0)

    distinct = float(sum(distinct_measures) / len(distinct_measures)) if distinct_measures else 0.0
    return _safe_rms(closure_ratios), max(0.0, min(1.0, distinct))


def _newton_sums_from_monic_high(coefficients: Sequence[float]) -> List[float]:
    leading = float(coefficients[0])
    if abs(leading) <= DENOMINATOR_EPS:
        raise ValueError("f must have nonzero leading coefficient.")
    tail = [float(value) / leading for value in coefficients[1:]]
    sums = [0.0]
    for k in range(1, F_DEGREE + 1):
        total = 0.0
        for j in range(1, k):
            total += tail[j - 1] * sums[k - j]
        total += k * tail[k - 1]
        sums.append(-total)
    return sums[1:]


def _newton_power_sum_loss(
    g: Any,
    f: Any,
    h: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    ratios = []
    for x_matrix, y_matrix in _specializations(sample_count, seed, samples):
        context: Dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
        g_value = base._eval_element(g, context)
        context["base_value"] = g_value
        context["current_value"] = g_value
        f_coefficients = base._coefficients_from_spec(f, F_DEGREE, context, "f")
        orbit, _, _ = _h_orbit_for_context(h, context, g_value)
        target_sums = _newton_sums_from_monic_high(f_coefficients)

        for power, target_scalar in enumerate(target_sums, start=1):
            orbit_powers = [np.linalg.matrix_power(root, power) for root in orbit]
            observed = sum(orbit_powers, _zero())
            target = float(target_scalar) * _identity()
            numerator = base._matrix_norm(observed - target)
            denominator = (
                sum(base._matrix_norm(term) for term in orbit_powers)
                + base._matrix_norm(target)
                + DENOMINATOR_EPS
            )
            ratios.append(numerator / denominator)
    return _safe_rms(ratios)


def _trim_poly(poly: Sequence[float], tolerance: float = 1e-14) -> List[float]:
    result = [float(value) for value in poly]
    while len(result) > 1 and abs(result[-1]) <= tolerance:
        result.pop()
    return result


def _poly_add(left: Sequence[float], right: Sequence[float]) -> List[float]:
    size = max(len(left), len(right))
    result = [0.0] * size
    for index in range(size):
        if index < len(left):
            result[index] += float(left[index])
        if index < len(right):
            result[index] += float(right[index])
    return _trim_poly(result)


def _poly_scale(poly: Sequence[float], scalar: float) -> List[float]:
    return _trim_poly([float(scalar) * float(value) for value in poly])


def _poly_sub(left: Sequence[float], right: Sequence[float]) -> List[float]:
    return _poly_add(left, _poly_scale(right, -1.0))


def _poly_reduce_mod(poly_low: Sequence[float], f_low_monic: Sequence[float]) -> List[float]:
    result = [float(value) for value in poly_low]
    while len(result) > F_DEGREE:
        top = result[-1]
        shift = len(result) - (F_DEGREE + 1)
        if top:
            for index in range(F_DEGREE):
                result[shift + index] -= top * float(f_low_monic[index])
        result.pop()
    while len(result) < F_DEGREE:
        result.append(0.0)
    return result


def _poly_mul_mod(
    left: Sequence[float],
    right: Sequence[float],
    f_low_monic: Sequence[float],
) -> List[float]:
    raw = [0.0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            raw[i + j] += float(left_value) * float(right_value)
    return _poly_reduce_mod(raw, f_low_monic)


def _poly_compose_mod(
    outer_low: Sequence[float],
    inner_low: Sequence[float],
    f_low_monic: Sequence[float],
) -> List[float]:
    result = [0.0] * F_DEGREE
    power = [1.0] + [0.0] * (F_DEGREE - 1)
    for coefficient in outer_low:
        result = _poly_add(result, _poly_scale(power, coefficient))
        result = _poly_reduce_mod(result, f_low_monic)
        power = _poly_mul_mod(power, inner_low, f_low_monic)
    return _poly_reduce_mod(result, f_low_monic)


def _poly_norm(poly: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(poly, dtype=float)))


def _quotient_automorphism_metrics(
    g: Any,
    f: Any,
    h: Any,
    sample_count: int,
    seed: int,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Tuple[float, float]:
    relation_ratios = []
    distinct_measures = []
    for x_matrix, y_matrix in _specializations(sample_count, seed, samples):
        context: Dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
        g_value = base._eval_element(g, context)
        context["base_value"] = g_value
        context["current_value"] = g_value
        f_high = base._coefficients_from_spec(f, F_DEGREE, context, "f")
        h_high = base._coefficients_from_spec(h, H_DEGREE, context, "h")

        leading = float(f_high[0])
        if abs(leading) <= DENOMINATOR_EPS:
            raise ValueError("f must have nonzero leading coefficient.")
        f_high_monic = [float(value) / leading for value in f_high]
        f_low_monic = list(reversed(f_high_monic))
        h_low = list(reversed([float(value) for value in h_high]))

        t_poly = [0.0, 1.0, 0.0, 0.0, 0.0]
        f_of_h = _poly_compose_mod(f_low_monic, h_low, f_low_monic)

        iterates = [t_poly]
        current = t_poly
        for step in range(5):
            current = _poly_compose_mod(h_low, current, f_low_monic)
            if step < 4:
                iterates.append(current)

        cycle_residual = _poly_sub(current, t_poly)
        relation_numerator = _poly_norm(f_of_h) + _poly_norm(cycle_residual)
        relation_denominator = (
            1.0
            + _poly_norm(f_low_monic)
            + _poly_norm(h_low)
            + _poly_norm(current)
        )
        relation_ratios.append(relation_numerator / relation_denominator)

        pairwise = []
        for i, left in enumerate(iterates):
            for right in iterates[i + 1 :]:
                numerator = _poly_norm(_poly_sub(left, right))
                denominator = _poly_norm(left) + _poly_norm(right) + DENOMINATOR_EPS
                pairwise.append(numerator / denominator)
        distinct_measures.append(min(pairwise) if pairwise else 0.0)

    distinct = float(sum(distinct_measures) / len(distinct_measures)) if distinct_measures else 0.0
    return _safe_rms(relation_ratios), max(0.0, min(1.0, distinct))


def _evaluate(program_path: str, sample_count: int, seed: int, stage_name: str) -> EvaluationResult:
    started = time.time()
    tools = base._available_symbolic_tools()
    try:
        g, f, h = base._candidate_triple_from_program(program_path)
        samples = base._random_specializations(sample_count, seed)

        g_membership = _g_membership_metrics(
            g,
            sample_count,
            seed + 10_000,
            samples=samples,
        )
        validation = base._polynomial_validation_metrics(g, f, h)
        first_part_loss, first_part_terms = base._first_part_loss(
            g,
            f,
            h,
            sample_count,
            seed,
        )
        charpoly_match_loss = _charpoly_match_loss(
            g,
            f,
            sample_count,
            seed + 20_000,
            samples=samples,
        )
        separation_measure = base.sep_measure(
            f,
            sample_count=sample_count,
            seed=seed + 50_000,
        )
        orbit_zeroes_measure = base.extended_eval_f_of_elem(
            base._orbit_zeroes_relation(f, h),
            g,
            sample_count=sample_count,
            seed=seed + 90_000,
        )
        not_central_measure = base.eval_f_of_elem(
            base._not_central_polynomial(g),
            g,
            sample_count=sample_count,
            seed=seed + 130_000,
            base_element=g,
        )
        degree5_measure = _degree5_power_span_measure(
            g,
            sample_count,
            seed + 140_000,
            samples=samples,
        )
        xy_dependence_measure = _xy_dependence_measure(
            g,
            sample_count,
            seed + 150_000,
            samples=samples,
        )
        g_relative_scale, g_scale_barrier = _g_scale_metrics(
            g,
            sample_count,
            seed + 160_000,
            samples=samples,
        )
        matrix_cycle_closure_loss, matrix_cycle_distinct_measure = _matrix_cycle_metrics(
            g,
            h,
            sample_count,
            seed + 170_000,
            samples=samples,
        )
        newton_power_sum_loss = _newton_power_sum_loss(
            g,
            f,
            h,
            sample_count,
            seed + 180_000,
            samples=samples,
        )
        quotient_relation_loss, quotient_distinct_measure = _quotient_automorphism_metrics(
            g,
            f,
            h,
            sample_count,
            seed + 190_000,
            samples=samples,
        )

        weights = {name: _weight(name) for name in DEFAULT_EVAL2_WEIGHTS}
        g_in_g_loss = weights["LAMBDA_G_IN_G"] * g_membership["g_in_g_loss"]
        eval_to_zero_loss = weights["LAMBDA_EVAL_TO_ZERO"] * first_part_loss
        charpoly_loss = weights["LAMBDA_CHARPOLY_MATCH"] * charpoly_match_loss
        orbit_zeroes_f_loss = weights["LAMBDA_ORBIT_ZEROES_F"] * orbit_zeroes_measure
        newton_loss = weights["LAMBDA_NEWTON_SUMS"] * newton_power_sum_loss
        quotient_loss = weights["LAMBDA_QUOTIENT_AUTOMORPHISM"] * quotient_relation_loss
        matrix_cycle_loss = weights["LAMBDA_MATRIX_CYCLE"] * matrix_cycle_closure_loss
        separable_loss = weights["LAMBDA_SEPARABLE"] * _measure_barrier(separation_measure)
        degree5_loss = weights["LAMBDA_DEGREE5"] * _measure_barrier(degree5_measure)
        cycle_distinct_measure = min(matrix_cycle_distinct_measure, quotient_distinct_measure)
        cycle_distinct_loss = weights["LAMBDA_CYCLE_DISTINCT"] * _measure_barrier(
            cycle_distinct_measure
        )
        not_central_loss = weights["LAMBDA_NOT_CENTRAL"] * _measure_barrier(not_central_measure)
        xy_dependence_loss = weights["LAMBDA_XY_DEPENDENCE"] * _measure_barrier(
            xy_dependence_measure
        )
        g_scale_loss = weights["LAMBDA_G_SCALE"] * g_scale_barrier

        objective_loss = (
            g_in_g_loss
            + eval_to_zero_loss
            + charpoly_loss
            + orbit_zeroes_f_loss
            + newton_loss
            + quotient_loss
            + matrix_cycle_loss
            + separable_loss
            + degree5_loss
            + cycle_distinct_loss
            + not_central_loss
            + xy_dependence_loss
            + g_scale_loss
        )

        elapsed = time.time() - started
        complexity = base._candidate_complexity(g, f, h)
        combined_score = base._score_from_objective_loss(
            objective_loss,
            validation["monic_error"],
            complexity,
        )

        metrics = {
            "combined_score": float(combined_score),
            "objective_loss": float(objective_loss),
            "g_in_g_loss": float(g_in_g_loss),
            "g_in_g_raw_loss": float(g_membership["g_in_g_loss"]),
            "g_in_g_measure": float(g_membership["g_in_g_measure"]),
            "g_determinism_error": float(g_membership["g_determinism_error"]),
            "g_rng_dependence_error": float(g_membership["g_rng_dependence_error"]),
            "g_context_dependence_error": float(g_membership["g_context_dependence_error"]),
            "g_input_mutation_error": float(g_membership["g_input_mutation_error"]),
            "g_local_continuity_error": float(g_membership["g_local_continuity_error"]),
            "first_part_loss": float(first_part_loss),
            "eval_to_zero_loss": float(eval_to_zero_loss),
            "charpoly_match_loss": float(charpoly_match_loss),
            "charpoly_loss": float(charpoly_loss),
            "sep_measure": float(separation_measure),
            "separable_loss": float(separable_loss),
            "orbit_zeroes_measure": float(orbit_zeroes_measure),
            "orbit_zeroes_f_loss": float(orbit_zeroes_f_loss),
            "newton_power_sum_loss": float(newton_power_sum_loss),
            "newton_loss": float(newton_loss),
            "quotient_relation_loss": float(quotient_relation_loss),
            "quotient_loss": float(quotient_loss),
            "matrix_cycle_closure_loss": float(matrix_cycle_closure_loss),
            "matrix_cycle_loss": float(matrix_cycle_loss),
            "matrix_cycle_distinct_measure": float(matrix_cycle_distinct_measure),
            "quotient_distinct_measure": float(quotient_distinct_measure),
            "cycle_distinct_measure": float(cycle_distinct_measure),
            "cycle_distinct_loss": float(cycle_distinct_loss),
            "not_central_measure": float(not_central_measure),
            "not_central_loss": float(not_central_loss),
            "degree5_measure": float(degree5_measure),
            "degree5_loss": float(degree5_loss),
            "xy_dependence_measure": float(xy_dependence_measure),
            "xy_dependence_loss": float(xy_dependence_loss),
            "g_relative_scale": float(g_relative_scale),
            "g_scale_loss": float(g_scale_loss),
            "implemented_parts": float(TOTAL_EVALUATION_2_PARTS),
            "total_parts": float(TOTAL_EVALUATION_2_PARTS),
            "monic_error": float(validation["monic_error"]),
            "h_leading_abs": float(validation["h_leading_abs"]),
            "candidate_complexity": float(complexity),
            "eval_time": float(elapsed),
        }
        metrics.update({f"eval2_{key.lower()}": float(value) for key, value in weights.items()})

        artifacts = {
            "stage": stage_name,
            "purpose": (
                "Force g toward a non-central degree-5 generator, f toward the "
                "minimal/characteristic polynomial of g, and h toward a cyclic "
                "order-5 root automorphism."
            ),
            "first_part_terms": ", ".join(f"{value:.6e}" for value in first_part_terms),
            "formula": (
                "loss = g_in_G_guard + f(h^i(g)) + charpoly_match(f,g) "
                "+ orbit_polynomial + Newton power sums + quotient automorphism "
                "+ h^5(g)-g + inverse barriers for separability, degree5, "
                "distinct cycle, noncentrality, and X/Y dependence"
            ),
            "g_in_g_guard": (
                "Numerical guard that g behaves like a real rational function of "
                "X,Y: real finite 5x5 output, deterministic repeated calls, no "
                "dependence on evaluator base/current context, no RNG dependence, "
                "no input mutation, and local continuity on generic samples."
            ),
            "g_in_g_loss": f"{g_in_g_loss:.6e}",
            "g_in_g_measure": f"{g_membership['g_in_g_measure']:.6e}",
            "important_change_from_evaluator_py": (
                "evaluation_2 uses the candidate's returned f instead of replacing "
                "it by charpoly(g); charpoly_match_loss rewards f for becoming "
                "the degree-5 minimal polynomial once degree5_measure is high."
            ),
            "degree5_measure": f"{degree5_measure:.6e}",
            "degree5_hint": (
                "Mean smallest singular value of normalized vec(I), vec(g), ..., "
                "vec(g^4). Values near zero indicate a lower-degree minimal "
                "polynomial after specialization."
            ),
            "separable_loss_hint": (
                "Unlike evaluator.py, this evaluator minimizes an inverse barrier "
                "of the discriminant separation measure, so repeated roots are bad."
            ),
            "cycle_distinct_measure": f"{cycle_distinct_measure:.6e}",
            "xy_dependence_measure": f"{xy_dependence_measure:.6e}",
            "g_relative_scale": f"{g_relative_scale:.6e}",
            "sample_count": str(sample_count),
            "available_symbolic_tools": str(tools),
        }
        if validation["monic_error"] > MONIC_TOLERANCE:
            artifacts["monic_hint"] = (
                "f(t) should be monic of degree 5; the leading coefficient should "
                "evaluate to 1 at random specializations."
            )

        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "objective_loss": MAX_LOSS,
                "first_part_loss": MAX_LOSS,
                "eval_to_zero_loss": MAX_LOSS,
                "charpoly_match_loss": MAX_LOSS,
                "charpoly_loss": MAX_LOSS,
                "sep_measure": 0.0,
                "separable_loss": MAX_LOSS,
                "orbit_zeroes_measure": MAX_LOSS,
                "orbit_zeroes_f_loss": MAX_LOSS,
                "newton_power_sum_loss": MAX_LOSS,
                "newton_loss": MAX_LOSS,
                "quotient_relation_loss": MAX_LOSS,
                "quotient_loss": MAX_LOSS,
                "matrix_cycle_closure_loss": MAX_LOSS,
                "matrix_cycle_loss": MAX_LOSS,
                "cycle_distinct_measure": 0.0,
                "cycle_distinct_loss": MAX_LOSS,
                "not_central_measure": 0.0,
                "not_central_loss": MAX_LOSS,
                "degree5_measure": 0.0,
                "degree5_loss": MAX_LOSS,
                "xy_dependence_measure": 0.0,
                "xy_dependence_loss": MAX_LOSS,
                "g_relative_scale": 0.0,
                "g_scale_loss": MAX_LOSS,
                "g_in_g_loss": MAX_LOSS,
                "g_in_g_raw_loss": MAX_LOSS,
                "g_in_g_measure": 0.0,
                "g_determinism_error": MAX_LOSS,
                "g_rng_dependence_error": MAX_LOSS,
                "g_context_dependence_error": MAX_LOSS,
                "g_input_mutation_error": MAX_LOSS,
                "g_local_continuity_error": MAX_LOSS,
                "implemented_parts": float(TOTAL_EVALUATION_2_PARTS),
                "total_parts": float(TOTAL_EVALUATION_2_PARTS),
                "monic_error": MAX_LOSS,
                "candidate_complexity": 0.0,
                "error": 0.0,
            },
            artifacts={
                "failure_stage": stage_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "available_symbolic_tools": str(tools),
            },
        )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    return _evaluate(program_path, STAGE1_SAMPLES, 12_345, "stage1")


def evaluate_stage2(program_path: str) -> EvaluationResult:
    return _evaluate(program_path, STAGE2_SAMPLES, 67_890, "stage2")


def evaluate(program_path: str) -> EvaluationResult:
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluation_2.py path/to/program.py")
        raise SystemExit(2)
    result = evaluate(sys.argv[1])
    print(result.metrics)
    print(result.artifacts)
