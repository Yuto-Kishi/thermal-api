import numpy as np

from thermal_api.api.abnormal import AbnormalAlarm


def test_abnormal_masks():
    temp = 22.0 * np.ones((12, 12), np.float32)
    temp[1:3, 1:3] = 5.0 # low
    temp[4:6, 4:6] = 80.0 # high
    temp[8:9, 8:9] = 160.0 # fire


    abn = AbnormalAlarm()
    out = abn.detect(temp)
    assert len(out["low"]) >= 1
    assert len(out["high"]) >= 1
    assert len(out["fire"]) >= 1