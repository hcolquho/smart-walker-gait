import numpy as np
from walker_gait.core.types import GaitEvent, GaitEventType, Side
from walker_gait.gait.metrics import GaitMetricsCalculator


def _ev(etype, side, t):
    return GaitEvent(type=etype, side=side, timestamp=t)


def test_perfectly_symmetric_gait_yields_near_zero_asymmetry():
    """1 step/side per second alternating, perfectly even -> asymmetry ~ 0."""
    events = []
    for i in range(10):
        events.append(_ev(GaitEventType.HEEL_STRIKE, Side.LEFT, i * 1.0))
        events.append(_ev(GaitEventType.HEEL_STRIKE, Side.RIGHT, i * 1.0 + 0.5))
        events.append(_ev(GaitEventType.TOE_OFF, Side.LEFT, i * 1.0 + 0.3))
        events.append(_ev(GaitEventType.TOE_OFF, Side.RIGHT, i * 1.0 + 0.8))

    calc = GaitMetricsCalculator()
    metrics = calc.compute(events)

    assert abs(metrics.step_time_asymmetry) < 0.05
    assert metrics.cadence_steps_per_min > 0


def test_known_asymmetric_gait_yields_expected_sign_and_magnitude():
    """Right side steps twice as fast as left -> step_time_asymmetry should
    be negative (right step time < left step time) and clearly nonzero."""
    events = []
    t = 0.0
    for i in range(10):
        events.append(_ev(GaitEventType.HEEL_STRIKE, Side.LEFT, t))
        t += 1.0  # left: 1.0s between steps
    t = 0.0
    for i in range(20):
        events.append(_ev(GaitEventType.HEEL_STRIKE, Side.RIGHT, t))
        t += 0.5  # right: 0.5s between steps (twice as fast)

    calc = GaitMetricsCalculator()
    metrics = calc.compute(events)

    assert metrics.step_time_asymmetry < -0.3, (
        f"expected clearly negative asymmetry (right faster), got {metrics.step_time_asymmetry}"
    )


def test_loading_asymmetry_uses_fsr_when_available():
    left_load = np.full(100, 50.0)
    right_load = np.full(100, 30.0)  # right leg bearing less load -> limp signature
    calc = GaitMetricsCalculator(fsr_load_left=left_load, fsr_load_right=right_load)
    events = [_ev(GaitEventType.HEEL_STRIKE, Side.LEFT, 0.0),
              _ev(GaitEventType.HEEL_STRIKE, Side.RIGHT, 0.5)]
    metrics = calc.compute(events)
    assert metrics.loading_asymmetry < 0, "right leg under-loaded should give negative asymmetry"


def test_double_support_time_is_nonzero_for_overlapping_stance():
    events = [
        _ev(GaitEventType.HEEL_STRIKE, Side.LEFT, 0.0),
        _ev(GaitEventType.TOE_OFF, Side.RIGHT, 0.15),   # right pushes off shortly after left lands
        _ev(GaitEventType.HEEL_STRIKE, Side.RIGHT, 0.5),
        _ev(GaitEventType.TOE_OFF, Side.LEFT, 0.65),
    ]
    calc = GaitMetricsCalculator()
    metrics = calc.compute(events)
    assert metrics.double_support_time_s > 0
    assert abs(metrics.double_support_time_s - 0.15) < 1e-6
