import numpy as np
from feasibility import times,service


def test_mbps_units_and_deadline_boundary():
    total,tx=times([1_000_000],[.1],[.2],1,overhead_fraction=0,common_s=0)
    assert np.allclose(total,[1.3]) and np.allclose(tx,[1])
    on=service([.8],[1_000_000],[.1],[.2],1,1.3,overhead_fraction=0,common_s=0)
    late=service([.8],[1_000_000],[.1],[.2],1,1.299,overhead_fraction=0,common_s=0)
    assert on['mean_F1']==.8 and late['mean_F1']==0


def test_compute_budget_and_energy_reject_an_otherwise_timely_packet():
    args=([.9],[100_000],[.2],[0],10,2)
    assert service(*args,compute_budget_s=.1)['mean_F1']==0
    assert service(*args,compute_to_radio_power=10,energy_budget_radio_s=1)['mean_F1']==0
    assert service(*args,compute_to_radio_power=1,energy_budget_radio_s=1)['mean_F1']==.9


def test_faster_link_cannot_reduce_fixed_mode_delivery():
    args=([.7,.8],[100_000,2_000_000],[.1,.1],[.02,.02])
    slow=service(*args,.1,1)
    fast=service(*args,10,1)
    assert fast['delivery_fraction']>=slow['delivery_fraction']
    assert fast['mean_F1']>=slow['mean_F1']


def test_fixed_mission_baseline_energy_does_not_vanish_with_early_result():
    args=([.8],[1000],[.01],[.01],100,2)
    until=service(*args,idle_to_radio_power=10)
    fixed=service(*args,idle_to_radio_power=10,platform_horizon_s=2)
    assert fixed['mean_normalized_energy_radio_s']>20
    assert until['mean_normalized_energy_radio_s']<1
