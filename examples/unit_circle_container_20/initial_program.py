# EVOLVE-BLOCK-START
"""Place 20 unit circles inside the smallest possible enclosing circle."""

import math


def construct_centers():
    """
    Return 20 (x, y) centers for unit circles.

    The evaluator computes the radius of the smallest circle centered at the origin
    that contains every unit circle: max(sqrt(x*x + y*y) + 1).
    Centers must be at least distance 2 apart to avoid overlaps.
    """
    centers = []
    spacing = 2.1

    for row in range(4):
        for col in range(5):
            x = (col - 2) * spacing
            y = (row - 1.5) * spacing
            centers.append((x, y))

    return centers


# EVOLVE-BLOCK-END


def run_packing():
    """Return centers for 20 unit circles."""
    return construct_centers()


if __name__ == "__main__":
    centers = run_packing()
    radius = max(math.hypot(x, y) + 1.0 for x, y in centers)
    print(f"enclosing_radius={radius:.6f}")
    for i, center in enumerate(centers):
        print(i, center)
