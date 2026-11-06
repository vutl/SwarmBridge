from delivered_objective import winners,delivered


def test_numerical_quality_ties_are_not_claimed_as_wins():
    assert winners({'cpu':.5607715,'gpu':.5607819})==['cpu','gpu']
    assert winners({'cpu':.56,'gpu':.58})==['gpu']


def test_late_high_quality_does_not_beat_timely_lower_quality():
    a,ta,_=delivered([.9],[1_000_000],[1],[0],1,.5)
    b,tb,_=delivered([.7],[10_000],[.1],[0],1,.5)
    assert a[0]==0 and b[0]==.7 and ta[0]>.5 and tb[0]<.5
