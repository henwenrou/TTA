import numpy as np
import random
from scipy.special import comb


class ClassConditionalAffineCLP(object):
    """Class-conditional affine local perturbation without Bezier curves."""

    def __init__(self, alpha_range=(0.75, 1.25), beta_range=(-0.15, 0.15),
                 perturb_background=True, vrange=(0., 1.), seed=None):
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.perturb_background = perturb_background
        self.vrange = vrange
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def _uniform(self, low, high):
        rng = self.rng if self.rng is not None else np.random
        return rng.uniform(low, high)

    def __call__(self, image, mask):
        input_dtype = image.dtype
        image_f = image.astype(np.float32, copy=False)
        mask_i = np.asarray(mask).astype(np.int32)
        if mask_i.ndim == image_f.ndim and mask_i.shape[-1] == 1:
            mask_i = mask_i[..., 0]

        output = np.zeros_like(image_f, dtype=np.float32)
        for c in np.unique(mask_i):
            if c == 0 and not self.perturb_background:
                alpha, beta = 1.0, 0.0
            else:
                alpha = self._uniform(*self.alpha_range)
                beta = self._uniform(*self.beta_range)
            region = mask_i == c
            if image_f.ndim == mask_i.ndim + 1:
                region = np.expand_dims(region, axis=-1)
            output = np.where(region, image_f * alpha + beta, output)

        output = np.clip(output, self.vrange[0], self.vrange[1])
        return output.astype(input_dtype, copy=False)


class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point,end_point=inputs.min(),inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image=self.non_linear_transformation(image, inverse=False)
        image=self.location_scale_transformation(image).astype(np.float32)
        return image

    def Local_Location_Scale_Augmentation(self,image, mask):
        output_image = np.zeros_like(image)

        mask = mask.astype(np.int32)

        output_image[mask == 0] = self.location_scale_transformation(self.non_linear_transformation(image[mask==0], inverse=True, inverse_prop=1))

        for c in range(1,np.max(mask)+1):
            if (mask==c).sum()==0:continue
            output_image[mask == c] = self.location_scale_transformation(self.non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5))

        if self.background_threshold>=self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image
