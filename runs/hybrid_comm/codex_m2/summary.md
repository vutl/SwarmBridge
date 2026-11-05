# Matched receiver pilot (Codex)

All rows use dt=0.2 and cap=500. Development evidence only.

| Method | mAP | AMOTA | IDS | scene_11 | scene_13 | scene_15 |
|---|---:|---:|---:|---:|---:|---:|
| box_none | 0.5117 | 0.5421 | 69.0 | 0.5886 | 0.4962 | 0.5332 |
| box_m1 | 0.5541 | 0.5960 | 41.0 | 0.6405 | 0.5539 | 0.5884 |
| box_m1_age1 | 0.5564 | 0.6096 | 66.0 | 0.6570 | 0.5615 | 0.6027 |
| joint_none | 0.5132 | 0.5376 | 59.0 | 0.5965 | 0.4888 | 0.5218 |
| joint_m1 | 0.5536 | 0.5971 | 48.0 | 0.6469 | 0.5476 | 0.5974 |
| joint_m1_age1 | 0.5528 | 0.6037 | 56.0 | 0.6469 | 0.5557 | 0.6021 |
| joint_m2 | 0.5734 | 0.6338 | 60.0 | 0.6849 | 0.5667 | 0.6395 |
| joint_m3 | 0.5888 | 0.6613 | 52.0 | 0.6890 | 0.6258 | 0.6679 |
| joint_m2_age1 | 0.5703 | 0.6355 | 68.0 | 0.6750 | 0.5955 | 0.6295 |
| kf_baseline | 0.6484 | 0.6916 | 31.0 | 0.7269 | 0.7040 | 0.6303 |
| calibrated_m2 | 0.6376 | 0.6915 | 53.0 | 0.7251 | 0.6736 | 0.6667 |
| calibrated_m3 | 0.6503 | 0.6735 | 48.0 | 0.7021 | 0.6542 | 0.6574 |
| m1_same_nms | 0.6483 | 0.7011 | 44.0 | 0.7255 | 0.7096 | 0.6592 |
| feat_dense_q8 | 0.6886 | 0.7629 | 49.0 | 0.7831 | 0.7578 | 0.7397 |
| feat_dense_float | 0.6884 | 0.7608 | 43.0 | 0.7832 | 0.7601 | 0.7278 |

Gate baseline: **m1_same_nms**. Wireless gate: **False**.

{
  "baseline": "m1_same_nms",
  "threshold": 0.01,
  "required_positive_scenes": 2,
  "candidates": {
    "joint_m2": {
      "delta_amota": -0.06729637328665905,
      "per_scene_delta": {
        "scene_11": -0.04062093080818663,
        "scene_13": -0.14288214423116097,
        "scene_15": -0.019729950090530135
      },
      "ids_delta": 16.0,
      "pass_gate": false
    },
    "joint_m3": {
      "delta_amota": -0.039751279270873296,
      "per_scene_delta": {
        "scene_11": -0.036472815659778446,
        "scene_13": -0.08386611458987181,
        "scene_15": 0.008710860080860527
      },
      "ids_delta": 8.0,
      "pass_gate": false
    },
    "calibrated_m2": {
      "delta_amota": -0.009585618389425843,
      "per_scene_delta": {
        "scene_11": -0.0004330836372723468,
        "scene_13": -0.03604484483589454,
        "scene_15": 0.007493693023523096
      },
      "ids_delta": 9.0,
      "pass_gate": false
    },
    "calibrated_m3": {
      "delta_amota": -0.027570080425055155,
      "per_scene_delta": {
        "scene_11": -0.023389370876145232,
        "scene_13": -0.05538646291830829,
        "scene_15": -0.0017766757338140149
      },
      "ids_delta": 4.0,
      "pass_gate": false
    }
  },
  "open_wireless": false
}

Successful GPU phases: 21.4 min; paired adaptation: 14.3 min.
