import time


class Rate:
    def __init__(self, fps: float):
        self.period = 1.0 / float(fps)
        self.t_prev = time.perf_counter()


    def sleep(self):
        t = time.perf_counter()
        dt = self.period - (t - self.t_prev)
        if dt > 0:
            time.sleep(dt)
        self.t_prev = time.perf_counter()