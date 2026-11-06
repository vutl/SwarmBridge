import numpy as np
from core import *


def test_box_wire_roundtrip_and_real_size():
    b=[[10.1,20.2,100.3,150.4,.73,4],[0,0,640,480,.9,9]]
    packet=encode_boxes(b,640,480,17)
    assert len(packet)==HEADER.size+2*RECORD.size
    d=decode_boxes(packet)
    assert np.abs(np.array(d)-np.array(b)).max()<.01
    assert len(decode_boxes(encode_boxes([],640,480,18)))==0


def test_duplicate_and_class_mismatch_are_false_positives():
    gt=[[10,10,30,30,4]]
    r=image_f1(gt,[[10,10,30,30,.9,4],[10,10,30,30,.8,4],[10,10,30,30,.7,9]])
    assert (r['tp'],r['fp'],r['fn'])==(1,2,0)


def test_ignore_union_and_half_area_boundary():
    a=np.array([[1,1,5,10,0,0,0,0],[6,1,5,10,0,0,0,0]],float)
    integ=ignored_integral(a,30,30)
    keep=outside_ignored([[1,1,11,11],[15,15,25,25]],integ)
    assert keep.tolist()==[False,True]
    integ=np.zeros((10,10),int);integ[:, :5]=1;integ=integ.cumsum(0).cumsum(1)
    assert not outside_ignored([[1,1,9,9]],integ)[0]


def test_ignored_object_does_not_create_fp():
    r=image_f1([],[[2,2,5,5,.9,4]],[[0,0,10,10,4]])
    assert r['fp']==0
