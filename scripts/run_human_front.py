
import cv2 as cv
from thermal_api.processing.background import ExpMovingBG
from thermal_api.api.human_mark import HumanMark
from thermal_api.viz.overlay import draw_human
from thermal_api.utils.timers import Rate
from _common import load_config, make_source, imwrite_seq

def main():
    cfg = load_config()
    src = make_source(cfg)
    bg = ExpMovingBG((cfg.io.height, cfg.io.width), cfg.processing.bg_alpha)
    human = HumanMark(cfg.thresholds.human_body_c_min, cfg.thresholds.human_body_c_max,
                      cfg.thresholds.human_head_c_min, cfg.thresholds.human_head_c_max,
                      cfg.processing.morph_kernel, cfg.processing.min_region_area)
    rate = Rate(cfg.io.fps)
    for i, frame in enumerate(src.frames()):
        motion = bg.apply(frame.temp_c)
        result = human.detect(frame.temp_c, motion)
        vis = cv.cvtColor(frame.gray8, cv.COLOR_GRAY2BGR)
        draw_human(vis, result)
        if cfg.viz.show:
            cv.imshow("human_mark", vis)
            if cv.waitKey(1) & 0xFF == 27:
                break
        imwrite_seq(cfg.viz.save_dir, "human", i, vis)
        rate.sleep()

if __name__ == "__main__":
    main()
