
import logging as logger
import os
import numpy as np
from glob import glob
from astropy.io import fits
from scipy.special import wofz

from scipy import integrate, interpolate, ndimage, optimize as op, stats
from skimage.measure import block_reduce
from tqdm import tqdm

data_path = "../gemini-test/raw_data/"

paths = glob(os.path.join(data_path, "*.fits"))

obstypes = dict()

use_image = lambda image: (image[0].header["FILTER1"] == "open1-6") \
                        * (image[0].header["FILTER2"] == "open2-8") \
                        * (image[1].header["CCDSUM"] == "1 2")

for path in paths:
    if path.endswith("_bias.fits"): continue
    with fits.open(path) as image:
        if use_image(image):
            obstype = image[0].header["OBSTYPE"]
            obstypes.setdefault(obstype, [])
            obstypes[obstype].append(path)


def _voigt(x, *parameters):
    """
    Evaluate a Voigt profile at x, given the profile parameters.

    :param x:
        The x-values to evaluate the Voigt profile at.

    :param parameters:
        The position, fwhm, amplitude, and shape of the Voigt profile.
    """
    try:
        n = len(x)
    except TypeError:
        n = 1

    position, fwhm, amplitude, shape = parameters
    
    profile = 1 / wofz(np.zeros((n)) + 1j * np.sqrt(np.log(2.0)) * shape).real
    profile *= amplitude * wofz(2*np.sqrt(np.log(2.0)) * (x - position)/fwhm \
        + 1j * np.sqrt(np.log(2.0))*shape).real
    return profile


def _lorentzian(x, *parameters):
    """
    Evaluate a Lorentzian profile at x, given the profile parameters:

        y = (amplitude/PI) * (width/((x - positions)**2 + width**2))

    :param x:
        The x-values to evaluate the Lorentzian profile at.

    :param parameters:
        The position, width, and amplitude of the Lorentzian profile.
    """
    position, width, amplitude = parameters
    return (amplitude/np.pi) * (width/((x - position)**2 + width**2))


def parse_section_header(value):
    value = value.strip("[]")
    ys, xs = value.split(",")
    xs = np.array(xs.split(":")).astype(int) - [1, 0]
    ys = np.array(ys.split(":")).astype(int) - [1, 0]

    return (xs, ys)


def section_to_mask(header, parent_header=None):
    xs, ys = parse_section_header(header)

    if parent_header is not None:
        # create a mask from the parent header
        (_, hx), (__, hy) = parse_section_header(parent_header)
        mask = np.zeros((hx, hy), dtype=bool)

    else:
        mask = np.zeros((xs[1], ys[1]), dtype=bool)

    mask[slice(*xs), slice(*ys)] = True

    return mask


def get_mask_extent(headers):

    xs = []
    ys = []
    for header in headers:
        (_, x), (_, y) = parse_section_header(header)
        xs.append(x)
        ys.append(y)

    return np.max(np.vstack([xs, ys]), axis=1)


def trim_section(hdu, header_key, section_header_keys=None):

    header = hdu.header[header_key]

    if section_header_keys is None:
        section_headers = (header, )
    else:
        section_headers = [hdu.header[shk] for shk in section_header_keys]

    mask = np.zeros(get_mask_extent(section_headers), dtype=bool)

    xs, ys = parse_section_header(header)
    mask[slice(*xs), slice(*ys)] = True

    return hdu.data[mask].reshape((xs[1] - xs[0], ys[1] - ys[0]))


def overscan_correct(paths):
    """
    Perform overscan correction and trim.
    """

    if isinstance(paths, (str, unicode)):
        paths = [paths]

    # TODO: update with ROIs and detector info
    N = len(paths)
    shape = (N, 512 * 12, 512)

    data = np.zeros(shape, dtype=float)
    section_header_keys = ("DATASEC", "BIASSEC")

    for i, path in enumerate(paths):
        with fits.open(path) as image:

            for j, hdu in enumerate(image[1:]):
                data_section = trim_section(hdu, "DATASEC", section_header_keys)
                bias_section = trim_section(hdu, "BIASSEC", section_header_keys)

                # Overscan correction.
                overscan = np.atleast_2d(np.median(bias_section, axis=1))
                data_section = data_section.astype(float) - overscan.T
                
                data[i, 512*j:512*(j + 1), :] = data_section.T

    return data


def create_master_bias(paths):
    return np.median(overscan_correct(paths), axis=0)


def __new_and_wrong_create_cosmic_ray_masks(paths, threshold=4, niter=1):

    data = overscan_correct(paths)
    median_data = np.median(data, axis=0)
    cosmic_ray_masks = np.zeros(data.shape, dtype=bool)

    for i in range(data.shape[0]):
        crm = np.zeros(data.shape[1:], dtype=bool)
        scale = (data[i]/median_data)
        
        for k in range(niter):
            is_cosmic_ray = (scale >= (np.median(scale[~crm]) + threshold * np.std(scale[~crm])))
            crm[is_cosmic_ray] = True
            scale = (data[i]/median_data)

        cosmic_ray_masks[i] = crm

    return cosmic_ray_masks

def create_cosmic_ray_masks(paths, threshold=4, niter=1):

    N = len(paths)
    section_header_keys = ("DATASEC", "BIASSEC")

    with fits.open(paths[0]) as image:
        # assumes all sections will have the same shape
        data_section = trim_section(image[1], "DATASEC", section_header_keys)
        shape = (N, len(image) - 1, *data_section.shape)

    data = np.zeros(shape, dtype=float)

    for i, path in enumerate(paths):
        with fits.open(path) as image:
            for j, hdu in enumerate(image[1:]):
                data[i, j] = trim_section(hdu, "DATASEC", section_header_keys)

                row_scale = np.median(data[i, j], axis=1) / np.median(data[i, j])
                data[i, j] /= np.atleast_2d(row_scale).T

    median_data = np.median(data, axis=0)
    cosmic_ray_masks = np.zeros(data.shape, dtype=bool)

    for i in range(N):
        for j in range(shape[1]):

            scale = data[i, j]/median_data[j]
            crm = np.zeros(scale.shape, dtype=bool)
            
            for k in range(niter):
                is_cosmic_ray = scale >= (np.median(scale[~crm]) + threshold * np.std(scale[~crm]))
                crm[is_cosmic_ray] = True

            cosmic_ray_masks[i, j] = crm

            print(f"{i}, {j}, mean:{np.mean(scale):.2f}, std:{np.std(scale):.2f}, min:{np.min(scale):.2f}, max:{np.max(scale):.2f}")

            '''
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            axes[0].imshow(data[i, j])
            axes[1].imshow(scale)

            
            axes[2].imshow(crm)

            axes[2].set_title(f"{np.sum(crm):.0f}")
            if np.sum(crm) == 33:

                raise a
            '''

    return cosmic_ray_masks



def create_bad_pixel_mask(paths, blur_sigma=5, threshold=0.5):

    data = overscan_correct(paths)

    median_data = np.median(data, axis=0)
    v = threshold * np.median(median_data, axis=0)

    bpm = np.zeros_like(median_data)
    for i, t in enumerate(v):
        bpm[i] = (median_data[i] <= t)
        if blur_sigma > 0:
            blurred_bpm = ndimage.gaussian_filter1d(bpm[i].astype(float), blur_sigma)
            bpm[i] = blurred_bpm > 0

    return bpm.astype(bool)




def normalize_flat_field(path, master_bias=None, mask=None, fix_rows=True,
                         function="spline", function_kwds=None):

    data = overscan_correct(path)[0]

    available_functions = ("spline", "chebyshev")
    function = f"{function}".strip().lower()
    if function not in available_functions:
        raise ValueError(f"unknown function '{function}'")

    function_kwds = function_kwds or dict()
    if function == "spline":
        function_kwds.setdefault("k", 3)

    elif function == "chebyshev":
        function_kwds.setdefault("deg", 15)


    if master_bias is not None:
        data -= master_bias

    # TODO: don't assume structure of CCD
    shape = (12, 512, 512)
    data = data.reshape(shape)
    mask = mask.reshape(shape)

    print("assuming old cr_mask format; transposing")
    mask = mask.transpose(0, 2, 1)

    data_copy = np.array(data, copy=True)

    logger.info(f"Correcting cosmic rays in flat field {path}")
    for i, (amplifier, amplifier_mask) in enumerate(zip(data, mask)):
        if not np.any(amplifier_mask):
            continue

        x = np.arange(amplifier.shape[0])
        y = np.arange(amplifier.shape[1])

        xx, yy = np.meshgrid(x, y)
        xp, yp = (xx[~amplifier_mask], yy[~amplifier_mask])

        zp = amplifier[~amplifier_mask]

        data[i] = interpolate.griddata((xp, yp), zp.ravel(), (xx, yy))

        '''
        if np.sum(amplifier_mask) > 30:

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
            axes[0].imshow(amplifier)
            axes[1].imshow(foo)
            axes[2].imshow(amplifier_mask)
            axes[0].set_title(i)
        '''

    logger.info(f"Creating normalized flat field from {path}")

    model = np.zeros(shape, dtype=float)

    for i, (amplifier, amplifier_model) in enumerate(zip(data, model)):

        # fit with low-order chebyshev
        y = np.median(amplifier, axis=1)
        x = np.arange(y.size)

        function_kwds.update(x=x, y=y)

        if function == "spline":
            tck = interpolate.splrep(**function_kwds)
            z = interpolate.splev(function_kwds["x"], tck)

        elif function == "chebyshev":
            f = np.polynomial.chebyshev.Chebyshev.fit(**function_kwds)
            z = f(function_kwds["x"])

        amplifier_model = np.tile(z, shape[-1]).reshape(amplifier_model.shape).T

        if fix_rows:
            row_scale = np.atleast_2d(np.mean(amplifier/amplifier_model, axis=0))
            amplifier_model *= row_scale

        model[i] = amplifier_model

        '''
        if np.sum(mask[i]) >= 10:

            vmin, vmax = np.percentile(amplifier_model, [16, 84])

            fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)

            axes[0].imshow(data_copy[i], vmin=vmin, vmax=vmax)
            axes[1].imshow(amplifier, vmin=vmin, vmax=vmax)
            axes[2].imshow(amplifier_model, vmin=vmin, vmax=vmax)
            axes[3].imshow(amplifier/amplifier_model, vmin=0.99, vmax=1.01)
            axes[0].set_title(i)

            fig, ax = plt.subplots()
            ax.imshow(mask[i])
            ax.set_title(i)



            fig, ax = plt.subplots()
            ax.plot(x, y, c="k")
            ax.plot(x, z, c="tab:red")
            ax.set_title(i)
    
        #fig, ax = plt.subplots()
        #ax.plot(x, y, c='k')
        #ax.plot(x, z, c='tab:red')

        '''

    # If we are correcting per amplifier.
    if False:
        denominator = np.mean(model, axis=(1, 2)).reshape((-1, 1, 1))

        normalized_model = model / denominator
        normalized_model = normalized_model.reshape((-1, 512))

    else:
        normalized_model = model / np.median(model)
        normalized_model = normalized_model.reshape((-1, 512))

    return (normalized_model, model)



def long_reduce(path, master_bias=None, master_flat=None, cr_niter=3,
           cr_blur_sigma=0):

    data = overscan_correct(path)
    if master_bias is not None:
        data -= master_bias

    if master_flat is not None:
        data /= master_flat

    shape = data.shape

    # a "clearly background" mask
    background_mask = np.zeros(shape, dtype=bool)
    background_mask[0, :, :225] = True
    background_mask[0, :, 275:] = True

    #data_copy = np.copy(data)

    # generate a CR mask
    cr_mask = np.zeros(shape, dtype=bool)
    for i in range(cr_niter):
        #data_copy[~background_mask + cr_mask] = np.nan
        #cr_threshold = np.nanmedian(data_copy, axis=2).reshape((1, -1, 1))

        background_data = data[background_mask * ~cr_mask]
        cr_threshold = np.median(background_data) + 5 * np.std(background_data)

        print(i, cr_threshold)
        cr_mask += (data >= cr_threshold)
        cr_mask[~background_mask] = False

    # Blur to grow
    if cr_blur_sigma > 0:
        blurred_cr_mask = ndimage.gaussian_filter1d(cr_mask.astype(float), cr_blur_sigma)
        blurred_cr_mask = blurred_cr_mask > 0
        blurred_cr_mask[~background_mask] = False

    # interpolate per column
    for i, (column, mask) in enumerate(zip(data[0], cr_mask[0])):
        if not any(mask): continue

        xi = np.arange(column.size)
        data[0, i, mask] = np.interp(xi[mask], xi[~mask], column[~mask], left=0, right=0)



    '''
    fig, ax = plt.subplots()
    ax.imshow(data[0].T, vmin=0, vmax=10)
    fig, ax = plt.subplots()
    ax.plot(data[0, 1485])
    plt.show()
    '''
    f = lambda x, mu, sigma, amp: amp * stats.norm.pdf(x, mu, sigma)
    #f = lambda x, mu, fwhm, amp, shape: _voigt(x, mu, fwhm, amp, shape)
    #f = lambda x, mu, width, amp: _lorentzian(x, mu, width, amp)
    g = lambda x, *coeffs: np.polynomial.chebyshev.Chebyshev(coeffs, domain=[0, 511])(x)

    h = lambda x, mu, sigma, amp, *coeffs: f(x, mu, sigma, amp) + g(x, *coeffs)


    # Now fit everything.
    background = np.zeros(shape)
    mus = 250 * np.ones(data.shape[1])
    sigmas = 2.25 * np.ones(data.shape[1])

    for i, y in enumerate(data[0]):

        x = np.arange(y.size)
        mask = np.ones(y.size, dtype=bool)
        mask[225:275] = False

        kz = max(i - 1, 0)
            
        dm, ds = 0.05, 0.05
        p0 = [mus[kz], sigmas[kz], np.max(y[~mask])]
        kwds = dict(p0=p0)

        if i > 0:
            bounds=[
                [mus[kz] - dm, sigmas[kz] - ds, 0],
                [mus[kz] + 2 * dm, sigmas[kz] + 2*ds, np.inf]
            ]
            kwds.update(bounds=bounds)

        (mu, sigma, amp), p_cov = op.curve_fit(f, x[~mask], y[~mask], **kwds)


        get_slice = lambda scale: slice(int(mu - scale * sigma), int(mu + scale * sigma))

        foo = np.ones(y.size, dtype=bool)
        foo[get_slice(5)] = False

        tck = interpolate.splrep(x[foo], y[foo], k=1)
        '''

        fit = np.polynomial.chebyshev.Chebyshev.fit(x[mask], y[mask], 2)
        coeffs = fit.convert().coef

        p0 = [250, 2.25, np.max(y[~mask])]
        p_opt, p_cov = op.curve_fit(f, x[~mask], y[~mask], p0=p0)

        #p_opt[2] /= fit(x).max()

        # fit together
        p0 = np.hstack([p_opt, coeffs])
        p_opt2, p_cov2 = op.curve_fit(h, x, y, p0=p0)

        def cost(theta):
            return np.sum((y - h(x, *theta))**2)

        result = op.minimize(cost, p0)
        print(result)
        '''


        mus[i] = mu
        sigmas[i] = sigma

        ya = interpolate.splev(x, tck) 
        yb = f(x, mu, sigma, amp)
        yi = ya + yb

        '''
        fig, ax = plt.subplots()
        ax.plot(x, y, c="k")
        #ax.scatter(x[mask], y[mask], c="#666666")
        #ax.plot(x, h(x, *p_opt2), c="r")
        #ax.plot(x, h(x, *result.x), c="tab:blue")
        ax.plot(x, ya, c="tab:blue")
        ax.plot(x, yb, c="tab:blue")
        ax.plot(x, yi, c="tab:red")

        raise a
        '''
        background[0, i] = ya


    mus_model = np.zeros_like(mus)
    sigmas_model = np.zeros_like(sigmas)

    for k in range(3):
        si, ei = (int(k * 2048), int((k + 1) * 2048))

        x = np.arange(2048)
        y = mus[si:ei]
        z = sigmas[si:ei]

        mask = np.zeros(x.size, dtype=bool)
        for j in range(10):

            theta1 = np.polyfit(x[~mask], y[~mask], 2)
            theta2 = np.polyfit(x[~mask], z[~mask], 2)

            yi = np.polyval(theta1, x)
            zi = np.polyval(theta2, x)

            new_mask = (np.abs(yi - y)/np.std(yi) > 3) \
                     + (np.abs(zi - z)/np.std(zi) > 3)

            if not any(new_mask): continue

            mask += new_mask

        mus_model[si:ei] = yi
        sigmas_model[si:ei] = zi


        fig, ax = plt.subplots(1, 2)
        ax[0].plot(x, y)
        ax[0].plot(x, yi, zorder=10)
        ax[1].plot(x, z)
        ax[1].plot(x, zi, zorder=10)




    data2 = data - background
    '''

    sky_mask[:, :, :225] = True
    sky_mask[:, :, 275:] = True

    sky = data[sky_mask].reshape((6144, -1))

    is_outlier = (sky >= (np.median(sky) + 5 * np.std(sky)))
    sky[is_outlier] = np.nan
    median_sky = np.nanmedian(sky, axis=1)



    sky_model = np.zeros_like(data)
    x = np.arange(data.shape[2])
    xm = np.ones(x.size, dtype=bool)
    xm[225:275] = False

    chebyshev_deg = 2

    for j, y in tqdm(enumerate(sky)):

        keep = ~is_outlier[j]
        g = np.polynomial.chebyshev.Chebyshev.fit(x[xm][keep], y[keep], chebyshev_deg)
        sky_model[0, j] = g(x)


    vmin, vmax = np.nanpercentile(sky, [16, 84])

    median_sky = np.atleast_2d(median_sky)

    scale = np.median(sky) / np.median(sky, axis=1)


    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    axes[0].imshow(sky.T, vmin=vmin, vmax=vmax)
    axes[1].imshow(sky_model[0].T, vmin=vmin, vmax=vmax)
    axes[2].imshow(sky.T - median_sky, vmin=vmin, vmax=vmax)
    # fit a line-by-line function

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    foo = data[0].T - median_sky
    vmin, vmax = np.nanpercentile(foo, [16, 84])
    axes[0].imshow(data[0].T, vmin=vmin, vmax=vmax)
    axes[1].imshow(foo, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()

    ax.plot(np.median(foo[225:275], axis=0))

    intensity = data[0].T #- median_sky

    #blocked_data = block_reduce(intensity, block_size=(36, 1), func=np.median)
    #blocked_data = blocked_data[:, 200:300] # MAGIC HACK
    blocked_data = data[0, :, 200:300]

    mus = 250 * np.ones(blocked_data.shape[0])
    sigmas = 2.25 * np.ones(blocked_data.shape[0])

    for k, y in enumerate(blocked_data):

        x = np.arange(y.size) + 200
        kz = max(k - 1, 0)
        p0 = [mus[kz], sigmas[kz], np.max(y)]

        try:
            p_opt, p_cov = op.curve_fit(f, x, y, p0=p0)
        except:
            mus[k] = mus[kz]
            sigmas[k] = sigmas[kz]

        else:
            mus[k] = p_opt[0]
            sigmas[k] = p_opt[1]

    
    #xi = np.arange(intensity.shape[1])
    #mus = np.interp(xi, xi[18::36], mus)
    #sigmas = np.interp(xi, xi[18::36], sigmas)

    #ex = np.tile(np.arange(intensity.shape[1]), intensity.shape[0]).reshape(intensity.shape)
    '''

    #intensity = (data - background)[0]
    bg_per_col = np.median(background[0], axis=1)

    from scipy.signal import find_peaks

    idx, meta = find_peaks(bg_per_col, width=4)

    mask = np.zeros(bg_per_col.size, dtype=bool)
    for ea in idx:
        mask[ea - 5:ea+6] = True


    xi = np.arange(bg_per_col.size)

    for i in range(2):

        x = xi[~mask]
        y = bg_per_col[~mask]
        tck = interpolate.splrep(x, y, t=np.linspace(0, bg_per_col.size, 24 + 2)[1:-1])

        yi = interpolate.splev(xi, tck)

        diff = (yi - bg_per_col)/np.std(yi)

        mask += (np.abs(diff) > 3)

    fig, ax = plt.subplots()
    ax.plot(bg_per_col)
    ax.plot(xi, yi, c="tab:red")



    #intensity = (data / bg_per_col.reshape((1, -1, 1)))[0]
    intensity = data[0]

    intensity2 = (data - background)[0]

    integrated_flux = np.zeros(intensity.shape[0])
    for k in range(integrated_flux.size):
        integrated_flux[k] = integrate.trapz(intensity2[k])


    extraction_mask = np.zeros(intensity.shape, dtype=float)
    
    for k in range(intensity.shape[0]):
        extraction_mask[k] = stats.norm.pdf(np.arange(intensity.shape[1]), mus_model[k], sigmas_model[k])
        #extraction_mask[k] = stats.norm.pdf(np.arange(intensity.shape[1]), np.median(mus), np.median(sigmas))
    
    flux = np.sum(extraction_mask * intensity, axis=1)

    fig, ax = plt.subplots()
    ax.plot(mus)
    fig, ax = plt.subplots()
    ax.plot(sigmas)

    fig, ax = plt.subplots()
    ax.plot(flux)

    scale = 1/ (yi / np.mean(yi))
    flux2 = flux * scale
    fig, ax = plt.subplots()
    ax.plot(flux2[::-1], c="tab:blue")

    ax.plot((integrated_flux * scale)[::-1], c="tab:red")


    raise a


def reduce(path, master_bias=None, master_flat=None, cr_niter=3,
           cr_blur_sigma=0, **kwargs):

    data = overscan_correct(path)
    if master_bias is not None:
        data -= master_bias


    if master_flat is not None:
        data = data / master_flat

    shape = data.shape

    vmin, vmax = np.percentile(data, [16, 84])

    fig, axes = plt.subplots()
    ax.imshow(data[0].T, vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(data[0, :, 250], c="tab:blue")
    axes[1].plot(np.median(master_flat, axis=1))

    continuum(data[0, :, 250], **kwargs)
    raise a



try:
    master_bias
except NameError:

    # Create master bias.
    master_bias = create_master_bias(obstypes["BIAS"])


    flat_frame_paths = obstypes["FLAT"]
    bp_mask = create_bad_pixel_mask(flat_frame_paths)
    cr_mask = create_cosmic_ray_masks(flat_frame_paths)


    #emask = bp_mask + cr_mask
    mask = cr_mask

    # normalize flat fields
    master_flat, flat = normalize_flat_field(obstypes["FLAT"][0], 
                                       mask=mask[0], master_bias=master_bias)
else:
    print("Using previous biases etc")


def padded_running_mean(x, N):
    m = np.convolve(x, np.ones((N,))/N, mode='valid')
    f = np.ones(int(N/2))
    return np.hstack([m[0] * f, m, m[-1] * f])[:-1]


def continuum(y, x=None, k=3, sigma_clip=None, num_iter=3, num_knots=36, window=100, **kwargs):
    """ fit the continuum using a spline and iterative sigma clipping """

    if x is None:
        x = np.arange(y.size)

    t = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]

    mask = np.zeros(y.size, dtype=bool)

    fig, ax = plt.subplots()
    ax.plot(x, y, c='k')

    for i in range(num_iter):
        y_masked = np.copy(y)
        y_masked[mask] = np.interp(x[mask], x[~mask], y[~mask])

        mu = padded_running_mean(y_masked, window)
        diff = (y - mu)
        diff_masked = np.copy(diff)
        diff_masked[mask] = np.interp(x[mask], x[~mask], diff_masked[~mask])
        sigma = np.abs(padded_running_mean(diff_masked, window))

        add_to_mask = np.zeros(y.size, dtype=bool)
        if sigma_clip is not None:
            lower_clip, upper_clip = sigma_clip
            if upper_clip is not None:
                add_to_mask += diff > (upper_clip * sigma)
            if lower_clip is not None:
                add_to_mask += diff < (-lower_clip * sigma)

        if not any(add_to_mask & ~mask): break

        mask += add_to_mask
        ax.plot(x, mu, label=i)

    ax.plot(x, sigma)
    ax.legend()


    # fit a spline to the good stuff    
    for i in range(2):
        tck = interpolate.splrep(x[~mask], y[~mask], k=k, t=t)
        model = interpolate.splev(x, tck)

        sigma = np.abs(np.mean(padded_running_mean(y - model, window)))
        diff = (y - model)/sigma

        if sigma_clip is not None:
            lower_clip, upper_clip = sigma_clip
            if upper_clip is not None:
                mask += diff > (upper_clip * sigma)
            if lower_clip is not None:
                mask += diff < (-lower_clip * sigma)



    model = interpolate.splev(x, tck)

    # reverse
    y, model = (y[::-1], model[::-1])


    fig, ax = plt.subplots()
    ax.plot(x, y, c='k')
    ax.plot(x, model, c='r')

    fig, ax = plt.subplots()
    ax.plot(x, y/model)

    raise a






foo = reduce(obstypes["ARC"][0], master_bias=master_bias, master_flat=master_flat,
             sigma_clip=(None, 3))

foo = reduce(obstypes["OBJECT"][1], master_bias=master_bias, master_flat=master_flat,
             sigma_clip=(3, None))
raise a
#science_data = overscan_correct(obstypes["OBJECT"][1])
#science_data -= master_bias

# do extraction and sky subtraction at once.



fig, ax = plt.subplots()
ax.imshow(master_bias.T, vmin=0, vmax=1)

fig, ax = plt.subplots()
ax.imshow(master_flat.T)

fig, ax = plt.subplots()
ax.imshow(master_flat[:512, :].T)



fig, ax = plt.subplots()
ax.plot(image[0, :, 250].T)



# Create cosmic ray masks.
