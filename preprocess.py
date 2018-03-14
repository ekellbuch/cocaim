import scipy

import numpy as np
import scipy.interpolate as interp
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt


def remove_outliers(mov,
                    stim,
                    sample_pixel,
                    thresh_stdv=4,
                    buff=10,
                    verbose=False):

    # s-score to locate outliers
    keep_idx = abs(sample_pixel - np.mean(sample_pixel)) < thresh_stdv * \
        np.std(sample_pixel)

    # minimum filter removes pixels within butter distance of outliers
    keep_idx = filt.minimum_filter(keep_idx, size=2 * buff + 1)

    # Optionally plot flagged outliers
    if verbose:
        fig = plt.figure(figsize=(16, 4))
        plt.plot(np.argwhere(keep_idx),
                 sample_pixel[keep_idx], 'b')
        plt.plot(np.argwhere(~keep_idx),
                 sample_pixel[~keep_idx], 'r')
        plt.show()

    # list of indices where samples were cutout (possible discontinuities)
    disc_idx = np.argwhere(filt.convolve1d(
        keep_idx, np.array([1, -1]))[keep_idx])

    return mov[:, :, keep_idx], stim[keep_idx], disc_idx


def get_knots(stim,
              k=3,
              followup=100,
              spacing=250):

    # Locate transition indices
    trans_idx = np.argwhere(filt.convolve1d(stim > 0, np.array([1, -1])))
    # Repeat knots and add transition extras
    knots = np.append(np.append(np.zeros(k + 1),
                                np.sort(np.append(np.repeat(trans_idx, k),
                                                  trans_idx + followup))),
                      np.ones(k + 1) * len(stim)).astype('int')

    # add regularly spaced extra knots between transitions
    extras = np.empty(0)

    for idx in np.linspace(k + 1,
                           len(knots),
                           int(np.ceil(len(knots) / (k + 1))), dtype='int')[:-1]:
        extras = np.append(
            extras,
            np.linspace(knots[idx - 1], knots[idx],
                        int(np.round(
                            (knots[idx] - knots[idx - 1]) / spacing)) + 2,
                        dtype='int')[1:-1]
        )

    # Locate beginning/end of transition zones as knots
    return np.sort(np.append(knots, extras)).astype('int')


def spline_detrend(data,
                   stim,
                   order=3,
                   disc=None,
                   followup=100,
                   spacing=200,
                   axis=-1):

    # get knots from stim
    knots = get_knots(stim, k=order, followup=100, spacing=250)
    x = np.arange(len(stim))

    if disc is not None:
        knots = np.sort(np.append(knots, np.repeat(disc, order + 1)))

    def spline_fit(y):
        bspl = interp.make_lsq_spline(x=x, y=y, t=knots, k=order)
        return bspl(x)

    # Subtract Spline Fit From Input
    data_detr = np.subtract(data, np.apply_along_axis(spline_fit, axis, data))

    return data_detr
