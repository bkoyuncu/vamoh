

class ThresholdTransform(object):
  def __init__(self, thr_1: float):
    self.thr = thr_1 / 1.  # input threshold for [0..1] gray level

  def __call__(self, x):

    if self.thr == 0.0:
        return x
    else: 
        return (x > self.thr).to(x.dtype)  # do not change the data type
