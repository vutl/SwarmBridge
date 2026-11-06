import numpy as np
from analyze_paired import nondominated,bootstrap


def test_frontier_retains_quality_latency_tradeoff():
    assert nondominated([.4,.7,.6],[.1,1,.5]).tolist()==[True,True,True]
    assert nondominated([.4,.7,.6],[.1,.3,.5]).tolist()==[True,True,False]
    assert nondominated([.5,.5],[.2,.2]).tolist()==[True,True]


def test_cluster_bootstrap_constant_delta_is_constant():
    r=bootstrap([.2,.2,.2,.2],['a','a','b','c'])
    assert np.allclose(r['pointwise_cluster_bootstrap95'],[.2,.2])
    assert r['groups']==3
