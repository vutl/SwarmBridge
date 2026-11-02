"""Independent mathematical and resumption checks for the isolated SB audit."""
import math
import tempfile
import unittest
from pathlib import Path

import torch

from swarmbridge.bridges.audit import AuditDenoiser, VarianceClock


class BridgeClockTests(unittest.TestCase):
    def test_ve_matches_closed_form_and_endpoints(self):
        clock = VarianceClock()
        t = torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)
        integral = 0.4 * (2.6 ** (2 * t) - 1) / (2 * math.log(2.6))
        self.assertTrue(torch.allclose(clock.cumulative(t), integral, atol=1e-12))
        u, w, v = clock.coefficients(t)
        self.assertTrue(torch.allclose(u + w, torch.ones_like(t)))
        self.assertEqual((u[0].item(), w[0].item(), v[0].item()), (1, 0, 0))
        self.assertEqual((u[-1].item(), w[-1].item(), v[-1].item()), (0, 1, 0))
        maximum = clock.cumulative(torch.ones(1, dtype=torch.float64)).item() / 4
        self.assertAlmostEqual(maximum, 0.301, delta=.001)
        self.assertGreater(abs(w[2].item() - .5), .1)  # mean must change with VE as well

    def test_reverse_matches_gaussian_product(self):
        clock = VarianceClock()
        t, s = torch.tensor(.8, dtype=torch.float64), torch.tensor(.3, dtype=torch.float64)
        eta_t, eta_s = clock.cumulative(t), clock.cumulative(s)
        # Independent multiplication of N(x0,eta_s) and N(xt,eta_t-eta_s).
        precision = 1 / eta_s + 1 / (eta_t - eta_s)
        x0, xt = torch.tensor(-2.), torch.tensor(3.)
        independent_mean = (x0 / eta_s + xt / (eta_t - eta_s)) / precision
        self.assertTrue(torch.allclose(clock.reverse(xt, x0, t, s), independent_mean))
        expected_var = 1 / precision
        delta = clock.reverse(xt, x0, t, s, noise=torch.tensor(1.)) - independent_mean
        self.assertAlmostEqual(delta.square().item(), expected_var.item(), places=10)

    def test_pinned_forward_marginals_for_ve(self):
        torch.manual_seed(17)
        clock = VarianceClock()
        n = 100_000
        clean, source = torch.full((n,), -.4), torch.full((n,), .7)
        state = source.clone()
        for t, s in ((1., .8), (.8, .4), (.4, .1)):
            state = clock.reverse(state, clean, torch.tensor(t), torch.tensor(s), noise=torch.randn(n))
            u, w, var = clock.coefficients(torch.tensor(s))
            self.assertAlmostEqual(state.mean().item(), (u * -.4 + w * .7).item(), delta=.006)
            self.assertAlmostEqual(state.var().item(), var.item(), delta=.006)

    def test_endpoint_bridge_is_exactly_mapping_training(self):
        torch.set_num_threads(2)
        mapping = AuditDenoiser(2, bridge=False, width=8)
        bridge = AuditDenoiser(2, bridge=True, endpoint_fraction=1, width=8)
        bridge.load_state_dict(mapping.state_dict())
        source, target = torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8)
        r = torch.rand(3, 1, 8, 8)
        a = mapping.loss(source, r, target, generator=torch.Generator().manual_seed(4))
        b = bridge.loss(source, r, target, generator=torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(a, b))
        a.backward()
        b.backward()
        for pa, pb in zip(mapping.parameters(), bridge.parameters()):
            self.assertTrue(torch.equal(pa.grad, pb.grad))

    def test_noise_parameterization_round_trip_and_projection(self):
        clock = VarianceClock(kind="brownian")
        net = AuditDenoiser(2, bridge=True, condition_source=True, csi=True,
                            clock=clock, prediction="noise", width=8)
        target, source = torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)
        r = torch.rand(2, 1, 8, 8)

        class Oracle(torch.nn.Module):
            def forward(self, inputs, t):
                return (inputs[:, :2] - target) / clock.cumulative(t).sqrt().view(-1, 1, 1, 1)

        net.net = Oracle()
        predicted = net.sample(source, r, nfe=4, stochastic=True,
                               generator=torch.Generator().manual_seed(2), known=r > .5)
        self.assertTrue(torch.allclose(predicted[r.expand_as(source) <= .5],
                                      target[r.expand_as(source) <= .5], atol=2e-6))
        self.assertTrue(torch.equal(predicted[r.expand_as(source) > .5], source[r.expand_as(source) > .5]))

    def test_checkpoint_restores_optimizer_ema_and_random_streams(self):
        from swarmbridge.experiments.sb_audit.run import atomic_checkpoint, capture_rng, restore_rng

        def initialize():
            torch.manual_seed(91)
            model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.2))
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return model, opt, ema, torch.Generator().manual_seed(17), torch.Generator().manual_seed(23)

        def update(model, opt, ema, dg, bg):
            x = torch.randn(4, 3, generator=dg)
            y = x + .1 * torch.randn(4, 3, generator=bg)
            opt.zero_grad(set_to_none=True)
            (model(x) - y).square().mean().backward()
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].lerp_(v, .1)

        full = initialize()
        for _ in range(3):
            update(*full)
        partial = initialize()
        update(*partial)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            atomic_checkpoint(path, dict(model=partial[0].state_dict(), optimizer=partial[1].state_dict(),
                ema=partial[2], rng=capture_rng(partial[3], partial[4], "cpu")))
            resumed = initialize()
            torch.randn(30)  # fresh process/model construction may consume randomness
            saved = torch.load(path, weights_only=False)
            resumed[0].load_state_dict(saved["model"])
            resumed[1].load_state_dict(saved["optimizer"])
            resumed = (resumed[0], resumed[1], saved["ema"], resumed[3], resumed[4])
            restore_rng(saved["rng"], resumed[3], resumed[4], "cpu")
            for _ in range(2):
                update(*resumed)
        for key, value in full[0].state_dict().items():
            self.assertTrue(torch.equal(value, resumed[0].state_dict()[key]))
            self.assertTrue(torch.equal(full[2][key], resumed[2][key]))


if __name__ == "__main__":
    unittest.main()
