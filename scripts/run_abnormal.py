import cv2 as cv
from _common import imwrite_seq, load_config, make_source

from thermal_api.api.abnormal import AbnormalAlarm
from thermal_api.config import AppConfig
from thermal_api.utils.timers import Rate
from thermal_api.viz.overlay import draw_abnormal


def main():
    cfg = load_config()
    src = make_source(cfg)
    abn = AbnormalAlarm(cfg.thresholds.abnormal_low_c_max,
                        cfg.thresholds.abnormal_high_c_min,
                        cfg.thresholds.abnormal_high_c_max,
                        cfg.thresholds.abnormal_fire_c_min,
                        cfg.processing.morph_kernel,
                        cfg.processing.min_region_area)


    rate = Rate(cfg.io.fps)
    for i, frame in enumerate(src.frames()):
        result = abn.detect(frame.temp_c)
        vis = cv.cvtColor(frame.gray8, cv.COLOR_GRAY2BGR)
        draw_abnormal(vis, result)
        if cfg.viz.show:
            cv.imshow("abnormal", vis)
            if cv.waitKey(1) & 0xFF == 27:
                break
        imwrite_seq(cfg.viz.save_dir, "abnormal", i, vis)
        rate.sleep()


if __name__ == "__main__":
    main()