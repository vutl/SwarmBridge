r"""Exact finite-state counterexamples for the SB design discussion; CPU, no models.

Run: .venv\Scripts\python.exe -m scripts.analyze_sb_mathematical_limits
These examples establish mathematical possibilities, not Town3 performance.
"""

import json
import math
from pathlib import Path


def kl(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q) if x)


def mse_examples():
    rows = []
    for p_plus in (0.5, 0.8):
        posterior = {-1: 1 - p_plus, 1: p_plus}
        mean = sum(x * p for x, p in posterior.items())
        variance = sum(p * (x - mean) ** 2 for x, p in posterior.items())
        point_risk = sum(p * (x - mean) ** 2 for x, p in posterior.items())
        sample_risk = sum(p * q * (x - draw) ** 2
                          for x, p in posterior.items() for draw, q in posterior.items())
        assert math.isclose(point_risk, variance)
        assert math.isclose(sample_risk, 2 * variance)
        candidate_errors = []
        for estimate in (-1.0, 0.0, mean, 1.0, 2.0):
            actual = sum(p * (x - estimate) ** 2 for x, p in posterior.items())
            identity = variance + (estimate - mean) ** 2
            candidate_errors.append(abs(actual - identity))
        rows.append(dict(p_plus=p_plus, posterior_mean=mean, bayes_mse=point_risk,
                         independent_posterior_sample_mse=sample_risk,
                         pythagorean_max_error=max(candidate_errors)))
    return rows


def marginal_counterexample():
    # Order: (-1,-1), (-1,+1), (+1,-1), (+1,+1).
    paths = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    reference = [0.45, 0.05, 0.05, 0.45]
    correct_pairing = [0.0, 0.5, 0.5, 0.0]
    marginal = lambda q, t: [sum(p for xy, p in zip(paths, q) if xy[t] == x) for x in (-1, 1)]
    assert all(marginal(q, t) == [0.5, 0.5]
               for q in (reference, correct_pairing) for t in (0, 1))
    # R itself is feasible and KL(R||R)=0, hence is the exact SB optimum.
    risk = lambda q: sum(p * (end - (-start)) ** 2 for (start, end), p in zip(paths, q))
    return dict(reference_is_feasible_exact_sb=True, both_marginals=[0.5, 0.5],
                sb_kl_to_reference=kl(reference, reference),
                true_pairing_kl_to_reference=kl(correct_pairing, reference),
                sb_pairwise_task_mse=risk(reference),
                true_pairing_task_mse=risk(correct_pairing))


def gibbs_update_example():
    prior = [0.7, 0.3]
    likelihood = [0.2, 0.9]
    evidence = sum(p * l for p, l in zip(prior, likelihood))
    posterior = [p * l / evidence for p, l in zip(prior, likelihood)]
    candidates = [prior, posterior, [0.5, 0.5], [0.1, 0.9]]
    errors = []
    for q in candidates:
        objective = kl(q, prior) - sum(p * math.log(l) for p, l in zip(q, likelihood))
        equivalent = kl(q, posterior) - math.log(evidence)
        errors.append(abs(objective - equivalent))
    assert max(errors) < 1e-12
    return dict(prior=prior, likelihood=likelihood, posterior=posterior,
                variational_identity_max_error=max(errors),
                scope="Unconstrained Bayes tilt; not a two-fixed-marginal SB theorem.")


def task_value_example():
    # S is the task bit, N an independent nuisance variable with four states.
    states = [(s, n) for s in (0, 1) for n in range(4)]
    before = 0.5
    rows = []
    for name, coordinate in (("reveal_task_bit", 0), ("reveal_nuisance", 1)):
        outcomes = sorted({x[coordinate] for x in states})
        after = 0.0
        for outcome in outcomes:
            subset = [x for x in states if x[coordinate] == outcome]
            p_one = sum(x[0] == 1 for x in subset) / len(subset)
            after += len(subset) / len(states) * min(p_one, 1 - p_one)
        rows.append(dict(query=name, information_gain_bits=math.log2(len(outcomes)),
                         bayes_classification_error_before=before,
                         expected_error_after=after, task_value_of_information=before - after))
    return rows


def main():
    result = dict(trained_any_model=False, used_gpu=False,
                  scope="Exact toy enumeration only; not evidence of Town3 gains.",
                  mse=mse_examples(), marginal_counterexample=marginal_counterexample(),
                  gibbs_update=gibbs_update_example(), task_value=task_value_example())
    out = Path("runs/sb_math_2026-11-02/analytic_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
