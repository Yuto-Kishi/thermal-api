
import cv2 as cv
from thermal_api.api.footprint import FootPrint
from thermal_api.viz.overlay import draw_footprints
from thermal_api.utils.timers import Rate
from _common import load_config, make_source, imwrite_seq

def main():
    cfg = load_config()
    src = make_source(cfg)
    fp = FootPrint(cfg.thresholds.footprint_c_min, cfg.thresholds.footprint_c_max,
                   cfg.processing.morph_kernel, cfg.processing.min_region_area)
    rate = Rate(cfg.io.fps)
    for i, frame in enumerate(src.frames()):
        result = fp.detect(frame.temp_c)
        vis = cv.cvtColor(frame.gray8, cv.COLOR_GRAY2BGR)
        draw_footprints(vis, result)
        if cfg.viz.show:
            cv.imshow("footprint", vis)
            if cv.waitKey(1) & 0xFF == 27:
                break
        imwrite_seq(cfg.viz.save_dir, "footprint", i, vis)
        rate.sleep()

if __name__ == "__main__":
    main()
