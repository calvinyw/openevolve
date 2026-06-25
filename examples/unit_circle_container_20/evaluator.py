"""Evaluator for packing 20 unit circles in the smallest enclosing circle."""

import importlib.util
import math


N_CIRCLES = 20
UNIT_RADIUS = 1.0
MIN_DISTANCE = 2.0 * UNIT_RADIUS


def _load_program(program_path):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


def _as_points(raw_centers):
    centers = list(raw_centers)
    if len(centers) != N_CIRCLES:
        raise ValueError(f"Expected {N_CIRCLES} centers, got {len(centers)}")

    points = []
    for i, center in enumerate(centers):
        if len(center) != 2:
            raise ValueError(f"Center {i} is not a 2D point: {center!r}")
        x = float(center[0])
        y = float(center[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Center {i} has non-finite coordinates: {center!r}")
        points.append((x, y))
    return points


def evaluate(program_path):
    try:
        program = _load_program(program_path)
        centers = _as_points(program.run_packing())

        overlap_penalty = 0.0
        min_pair_distance = float("inf")
        for i in range(N_CIRCLES):
            xi, yi = centers[i]
            for j in range(i + 1, N_CIRCLES):
                xj, yj = centers[j]
                distance = math.hypot(xi - xj, yi - yj)
                min_pair_distance = min(min_pair_distance, distance)
                shortfall = max(0.0, MIN_DISTANCE - distance)
                overlap_penalty += shortfall * shortfall

        enclosing_radius = max(math.hypot(x, y) + UNIT_RADIUS for x, y in centers)
        effective_radius = enclosing_radius + 25.0 * overlap_penalty
        score = 1.0 / effective_radius

        return {
            "combined_score": score,
            "radius_score": score,
            "valid": 1.0 if overlap_penalty == 0.0 else 0.0,
            "enclosing_radius": enclosing_radius,
            "effective_radius": effective_radius,
            "overlap_penalty": overlap_penalty,
            "min_pair_distance": min_pair_distance,
        }
    except Exception as e:
        return {
            "combined_score": 0.0,
            "radius_score": 0.0,
            "valid": 0.0,
            "enclosing_radius": 1e9,
            "effective_radius": 1e9,
            "overlap_penalty": 1e9,
            "error": str(e),
        }
