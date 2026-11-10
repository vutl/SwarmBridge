"""Exact scalar/Bernoulli thought experiments for the research review; no training."""
import itertools
import json
from pathlib import Path

from scripts.audit_sb_assumptions import exact_terminal_variance


def gaussian_reverse_variance(q, nfe, p=0.5):
    """Exact reverse mixture for a Gaussian target conditional on a fixed endpoint y.

    Var(X0|Xt,y) = p*q*t / ((1-t)*p + q*t).
    A plug-in mean discards (1-s/t)^2 Var(X0|Xt,y) from each transition.
    """
    variance = 0.0
    for k in range(nfe):
        t, s = 1 - k / nfe, 1 - (k + 1) / nfe
        denominator = (1 - t) * p + q * t
        gain = p / denominator
        conditional_variance = p * q * t / denominator
        a = s / t + (1 - s / t) * gain
        variance = (
            a * a * variance
            + q * s * (t - s) / t
            + (1 - s / t) ** 2 * conditional_variance
        )
    return variance


def bit_risk(states):
    """Minimum expected number of incorrectly classified bits under a known belief."""
    return sum(
        min(sum(prob for z, prob in states if z[i]), sum(prob for z, prob in states if not z[i]))
        for i in range(len(states[0][0]))
    )


def reveal_first(states):
    after = 0.0
    for observed in (0, 1):
        mass = sum(prob for z, prob in states if z[0] == observed)
        if mass:
            conditional = [(z, prob / mass) for z, prob in states if z[0] == observed]
            after += mass * bit_risk(conditional)
    return bit_risk(states) - after


def main():
    gaussian = [
        dict(q=q, nfe=nfe, target_variance=0.5,
             plugin_variance=exact_terminal_variance(q, nfe),
             full_gaussian_mixture_variance=gaussian_reverse_variance(q, nfe))
        for q in (1.0, 10.0) for nfe in (1, 4, 16, 64, 256)
    ]
    independent = [(z, 0.25) for z in itertools.product((0, 1), repeat=2)]
    correlated = [((0, 0), 0.5), ((1, 1), 0.5)]
    result = {
        "scope": "Exact toy calculations, not learned-model performance or a general-purpose sampler fix.",
        "gaussian": gaussian,
        "same_marginal_variance_different_message_value": [
            dict(belief="independent", per_bit_variance=[0.25, 0.25],
                 initial_risk=bit_risk(independent), value_reveal_first=reveal_first(independent)),
            dict(belief="perfectly correlated", per_bit_variance=[0.25, 0.25],
                 initial_risk=bit_risk(correlated), value_reveal_first=reveal_first(correlated)),
        ],
        "uncertainty_ranking_can_fail": [
            dict(region="A", positive_probability=0.5, variance=0.25,
                 informative_observation_probability=0.05, expected_risk_reduction=0.05 * 0.5),
            dict(region="B", positive_probability=0.1, variance=0.09,
                 informative_observation_probability=1.0, expected_risk_reduction=1.0 * 0.1),
        ],
    }
    path = Path("reports/metadata/sb-decision-mechanism-2026-10-30.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
