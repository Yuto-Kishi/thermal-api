
import cv2 as cv
from thermal_api.api.mrt import MRT
from thermal_api.utils.timers import Rate
from _common import load_config, make_source

def main():
    cfg = load_config()
    src = make_source(cfg)
    mrt = MRT()
    rate = Rate(cfg.io.fps)
    for _, frame in enumerate(src.frames()):
        stats = mrt.compute(frame.temp_c, ambient_c=22.0)
        vis = cv.cvtColor(frame.gray8, cv.COLOR_GRAY2BGR)
        txt = f"mean:{stats['mean_c']:.1f}C min:{stats['min_c']:.1f}C max:{stats['max_c']:.1f}C OT:{(stats['operative_temp_c'] or 0):.1f}C"
        cv.putText(vis, txt, (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv.LINE_AA)
        if cfg.viz.show:
            cv.imshow("mrt", vis)
            if cv.waitKey(1) & 0xFF == 27:
                break
        rate.sleep()

if __name__ == "__main__":
    main()
