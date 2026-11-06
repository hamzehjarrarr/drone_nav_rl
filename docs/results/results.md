# Evaluation results

50 episodes per scenario and mode, fixed test seeds.

**Success** = every waypoint passed and a 2 s hover held at the last one, judged by the
drone's onboard position estimate. In ideal mode that estimate is exact. In realistic
mode it drifts during a flight (like visual-inertial odometry), which no policy can see
from onboard sensors, so **Final error** gives the true distance from the target at
the end of the hover.

## policy · ideal

| Scenario | Success | Collision | Avg speed (m/s) | Route eff. | Final error (m) | Min clearance (m) |
| --- | --- | --- | --- | --- | --- | --- |
| open_field | 100% | 0% | 4.17 | 0.99 | 0.10 | 1.23 |
| forest_low | 98% | 0% | 3.95 | 0.91 | 0.12 | 0.92 |
| forest_medium | 88% | 4% | 3.93 | 0.88 | 0.14 | 0.77 |
| forest_high | 64% | 10% | 3.91 | 0.85 | 0.14 | 0.57 |
| walls_with_gaps | 72% | 22% | 3.96 | 0.83 | 0.12 | 0.74 |
| corridor | 0% | 100% | – | – | – | – |
| wind_and_gusts | 78% | 6% | 3.22 | 0.87 | 0.21 | 0.58 |
| push_recovery | 100% | 0% | 4.00 | 0.98 | 0.10 | 1.29 |

## policy · realistic

| Scenario | Success | Collision | Avg speed (m/s) | Route eff. | Final error (m) | Min clearance (m) |
| --- | --- | --- | --- | --- | --- | --- |
| open_field | 100% | 0% | 4.04 | 0.95 | 0.29 | 1.30 |
| forest_low | 100% | 0% | 3.98 | 0.92 | 0.28 | 0.78 |
| forest_medium | 100% | 0% | 4.04 | 0.91 | 0.30 | 0.55 |
| forest_high | 98% | 0% | 4.05 | 0.90 | 0.30 | 0.45 |
| walls_with_gaps | 74% | 14% | 4.06 | 0.87 | 0.30 | 0.68 |
| corridor | 98% | 2% | 3.25 | 0.96 | 0.26 | 0.58 |
| wind_and_gusts | 98% | 2% | 3.98 | 0.91 | 0.30 | 0.58 |
| push_recovery | 100% | 0% | 3.92 | 0.93 | 0.27 | 1.29 |

## reference · ideal

| Scenario | Success | Collision | Avg speed (m/s) | Route eff. | Final error (m) | Min clearance (m) |
| --- | --- | --- | --- | --- | --- | --- |
| open_field | 100% | 0% | 2.22 | 0.96 | 0.02 | 1.24 |
| forest_low | 44% | 56% | 2.21 | 0.97 | 0.03 | 0.79 |
| forest_medium | 34% | 66% | 2.17 | 0.96 | 0.03 | 0.54 |
| forest_high | 12% | 88% | 2.16 | 0.97 | 0.02 | 0.31 |
| walls_with_gaps | 18% | 82% | 2.16 | 0.96 | 0.04 | 0.52 |
| corridor | 60% | 38% | 2.07 | 0.87 | 0.03 | 0.14 |
| wind_and_gusts | 22% | 76% | 1.67 | 0.96 | 0.21 | 0.57 |
| push_recovery | 100% | 0% | 2.17 | 0.96 | 0.03 | 1.30 |

## reference · realistic

| Scenario | Success | Collision | Avg speed (m/s) | Route eff. | Final error (m) | Min clearance (m) |
| --- | --- | --- | --- | --- | --- | --- |
| open_field | 80% | 8% | 2.15 | 0.96 | 0.17 | 1.24 |
| forest_low | 46% | 52% | 2.13 | 0.96 | 0.19 | 0.67 |
| forest_medium | 22% | 66% | 2.18 | 0.95 | 0.14 | 0.47 |
| forest_high | 8% | 90% | 2.13 | 0.97 | 0.12 | 0.53 |
| walls_with_gaps | 14% | 86% | 2.16 | 0.95 | 0.16 | 0.58 |
| corridor | 42% | 46% | 1.95 | 0.90 | 0.19 | 0.21 |
| wind_and_gusts | 10% | 76% | 1.76 | 0.95 | 0.33 | 0.75 |
| push_recovery | 84% | 4% | 2.07 | 0.95 | 0.19 | 1.25 |
