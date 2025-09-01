
import numpy as np

class MRT:
    def compute(self, temp_c, ambient_c: float | None = None):
        mean = float(np.mean(temp_c))
        tmin = float(np.min(temp_c))
        tmax = float(np.max(temp_c))
        ot = None
        if ambient_c is not None:
            ot = 0.5 * (ambient_c + mean)
        return {"mean_c": mean, "min_c": tmin, "max_c": tmax, "operative_temp_c": ot}
