class CompareRectangles:
    """
    #############################################################################################
    # True Negative = tn_pixels                                                                 #
    #                                                                                           #
    #                           +-------------------------------+                               #
    #                           |                               |                               #
    #                           | False Negative = fn_pixels    |                               #
    #                           |                               |                               #
    #                           |                               |                               #
    #                           |                               |                               #
    #           +---------------+--------------+                |                               #
    #           |               | True         |                |                               #
    #           |               | Positive     |                |                               #
    #           |               | = tp_pixels  |                |                               #
    #           |               |              |                |                               #
    #           |               |              |                |                               #
    #           |               |              |                |                               #
    #           |               |              |                |                               #
    #           |               |              |                |                               #
    #           |               |              |                |                               #
    #           |               |              |                |                               #
    #           |               |              |                |                               #
    #           |               |              |   GROUND TRUTH |                               #
    #           |               +--------------+----------------+                               #
    #           |                              |                                                #
    #           | False Positive = fp_pixels   |                                                #
    #           |                              |                                                #
    #           |                              |                                                #
    #           |                              |                                                #
    #           |                    DETECTION |                                                #
    #           +------------------------------+                                                #
    #                                                                           ENCLOSING IMAGE #
    #############################################################################################
    """

    def __init__(self, gt, det, image_height=0, image_width=0):

        self.gt_max_x = max(gt['x1'], gt['x0'])
        self.gt_max_y = max(gt['y1'], gt['y0'])
        self.gt_min_x = min(gt['x1'], gt['x0'])
        self.gt_min_y = min(gt['y1'], gt['y0'])

        self.det_max_x = det['x1']
        self.det_max_y = det['y1']
        self.det_min_x = det['x0']
        self.det_min_y = det['y0']

        # prevent negative areas
        assert self.det_max_x >= self.det_min_x, 'Invalid detection: mincol > maxcol'
        assert self.det_max_y >= self.det_min_y, 'Invalid detection: minrow > maxrow'
        assert self.gt_max_x >= self.gt_min_x, 'Invalid ground truth: mincol > maxcol'
        assert self.gt_max_y >= self.gt_min_y, 'Invalid ground truth: minrow > maxrow'

        self.gt_ignore = gt.get('ignore', False)

        self.image_height = image_height
        self.image_width = image_width
        self.img_size = self.image_height * self.image_width

        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = self.img_size

        self._calculated = {
            'tp': False,
            'fp': False,
            'fn': False,
            'tn': False,
        }

    @property
    def tp_pixels(self):
        if not self._calculated['tp']:
            self._calc_tp()
            self._calculated['tp'] = True
        return self._tp

    @property
    def fp_pixels(self):
        if not self._calculated['fp']:
            self._calc_fp()
            self._calculated['fp'] = True
        return self._fp

    @property
    def fn_pixels(self):
        if not self._calculated['fn']:
            self._calc_fn()
            self._calculated['fn'] = True
        return self._fn

    @property
    def tn_pixels(self):
        if not self._calculated['tn']:
            self._calc_tn()
            self._calculated['tn'] = True
        return self._tn

    def _calc_tp(self):
        w_inter = min(self.gt_max_x, self.det_max_x) - max(self.gt_min_x, self.det_min_x)
        h_inter = min(self.gt_max_y, self.det_max_y) - max(self.gt_min_y, self.det_min_y)
        if w_inter <= 0 or h_inter <= 0:
            self._tp = 0
        else:
            self._tp = w_inter * h_inter

    def _calc_fp(self):
        det_h = self.det_max_y - self.det_min_y
        det_w = self.det_max_x - self.det_min_x
        self._fp = det_h * det_w - self.tp_pixels

    def _calc_fn(self):
        gt_h = self.gt_max_y - self.gt_min_y
        gt_w = self.gt_max_x - self.gt_min_x
        self._fn = gt_h * gt_w - self.tp_pixels

    def _calc_tn(self):
        assert self.img_size > 0
        # make sure entities are not labeled beyond the edges of the image, otherwise this formula
        # would not be correct
        assert self.gt_min_x >= 0
        assert self.gt_min_y >= 0
        assert self.gt_max_x <= self.image_width
        assert self.gt_max_y <= self.image_height
        assert self.det_min_x >= 0
        assert self.det_min_y >= 0
        assert self.det_max_x <= self.image_width
        assert self.det_max_y <= self.image_height
        self._tn = self.img_size - self.tp_pixels - self.fp_pixels - self.fn_pixels
