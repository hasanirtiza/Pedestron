from .utils import CompareRectangles


class ParamsFactory:
    def __init__(self,
                 detections_type,
                 difficulty,
                 ignore_other_vru,
                 tolerated_other_classes=[],
                 dont_care_classes=[],
                 ignore_type_for_skipped_gts=1,
                 size_limits={'reasonable': 40, 'small': 30, 'occluded': 40, 'all': 20},
                 occ_limits={'reasonable': 40, 'small': 40, 'occluded': 80, 'all': 80},
                 size_upper_limits={'small': 60},
                 occ_lower_limits={'occluded': 40},
                 rider_boxes_including_vehicles=False,
                 discard_depictions=False,
                 clipping_boxes=False,
                 transform_det_to_xy_coordinates=False
                 ):
        """
        :param detections_type: list of strings - Selected types on which the evaluation is
                                performed (i.e: detector of this type is evaluated)
        :param difficulty: the selected difficulty to be evaluated
        :param ignore_other_vru: - If true, other VRU ground truth boxes are used during the
                                   matching process, therefore other classes (e.g. riders)
                                   which are detected and classified as the primary detection class
                                   (e.g. pedestrians) do not cause a false positive.
                                 - if true other VRUs (see tolerated_other_classes) are marked with
                                   the ignored flag, otherwise they are discarded
        :param tolerated_other_classes: list of strings - Other classes which are tolerated,
                                        if the ignore_other_vru flag is set.
        :param dont_care_classes: list of strings - don't care region class name
        """
        self.difficulty = difficulty
        self.ignore_other_vru = ignore_other_vru
        self.tolerated_other_classes = tolerated_other_classes
        self.dont_care_classes = dont_care_classes
        self.ignore_type_for_skipped_gts = ignore_type_for_skipped_gts
        self.detections_type = detections_type
        self.size_limits = size_limits
        self.occ_limits = occ_limits
        self.size_upper_limits = size_upper_limits
        self.occ_lower_limits = occ_lower_limits
        self.rider_boxes_including_vehicles = rider_boxes_including_vehicles
        self.discard_depictions = discard_depictions
        self.clipping_boxes = clipping_boxes
        self.transform_det_to_xy_coordinates = transform_det_to_xy_coordinates

    def ignore_gt(self, gt):
        h = gt['y1'] - gt['y0']

        if gt['identity'] in self.detections_type:
            pass
        elif self.ignore_other_vru and gt['identity'] in self.tolerated_other_classes:
            return 1
        elif gt['identity'] in self.dont_care_classes:
            if self.discard_depictions and gt['identity'] == 'person-group-far-away' and \
                    'depiction' in gt['tags']:
                return None
            else:
                return 2
        else:
            # None means don't use this annotation in skip_gt
            return None
        if gt['identity'] == 'pedestrian':
            for tag in gt['tags']:
                if tag in ['sitting-lying', 'behind-glass']:
                    return 1

        import re
        truncation = 0
        occlusion = 0
        for t in gt['tags']:
            if 'occluded' in t:
                matches = re.findall(r'\d+', t)
                if len(matches) == 1:
                    occlusion = int(matches[0])
            elif 'truncated' in t:
                matches = re.findall(r'\d+', t)
                if len(matches) == 1:
                    truncation = int(matches[0])

        if h < self.size_limits[self.difficulty] or \
                occlusion >= self.occ_limits[self.difficulty] or \
                truncation >= self.occ_limits[self.difficulty]:

            return self.ignore_type_for_skipped_gts

        if self.difficulty in self.size_upper_limits:
            if h > self.size_upper_limits[self.difficulty]:
                return self.ignore_type_for_skipped_gts

        if self.difficulty in self.occ_lower_limits:
            if occlusion < self.occ_lower_limits[self.difficulty]:
                return self.ignore_type_for_skipped_gts

        return 0

    def preprocess_gt(self, gt):
        if str(gt['identity']).lower() == 'rider' and self.rider_boxes_including_vehicles:
            for subent in gt['children']:
                gt['x0'] = min(gt['x0'], float(subent['x0']))
                gt['y0'] = min(gt['y0'], float(subent['y0']))
                gt['x1'] = max(gt['x1'], float(subent['x1']))
                gt['y1'] = max(gt['y1'], float(subent['y1']))

        if self.clipping_boxes:
            gt['y0'] = max(0, gt['y0'])
            gt['y1'] = min(1024, gt['y1'])
            gt['x0'] = max(0, gt['x0'])
            gt['x1'] = min(1920, gt['x1'])

    def skip_gt(self, gt):
        if self.ignore_gt(gt) is None:
            return True
        return False

    def preprocess_det(self, det):
        assert self.transform_det_to_xy_coordinates
        if self.transform_det_to_xy_coordinates and 'maxrow' in det:
            det['x0'] = det['mincol']
            det['y0'] = det['minrow']
            det['x1'] = det['maxcol'] + 1
            det['y1'] = det['maxrow'] + 1
            det.pop('maxrow')
            det.pop('minrow')
            det.pop('maxcol')
            det.pop('mincol')
            assert 'maxrow' not in det

        if self.clipping_boxes:
            det['y0'] = max(0, det['y0'])
            det['y1'] = min(1024, det['y1'])
            det['x0'] = max(0, det['x0'])
            det['x1'] = min(1920, det['x1'])

        # We only accept detections of type detectionsType. All other detections are ignored.
        if det['identity'] == 'cyclist':
            det['identity'] = 'rider'

    def skip_det(self, det):
        assert 'y1' in det

        wrong_type = det['identity'] not in self.detections_type

        expFilter = 1.25

        # corresponds to Dollar expanded filtering (minSize/r with r=1.25)
        height = det['y1'] - det['y0']

        too_small = height <= self.size_limits[self.difficulty] / expFilter
        too_high = False if self.difficulty not in self.size_upper_limits else \
            height >= self.size_upper_limits[self.difficulty] * expFilter

        return wrong_type or too_small or too_high


def iou_ecp(gt, det):
    comp = CompareRectangles(gt=gt, det=det)

    area_intersect = comp.tp_pixels
    if area_intersect == 0:
        return 0.0

    assert gt['ignore'] < 4
    if gt['ignore'] == 2:
        # only use intersection of det with gt DontCare
        area_union = comp.fp_pixels + comp.tp_pixels  # = det['w'] * det['h']
    elif gt['ignore'] == 3:
        # only visible part of gt labeled
        area_union = comp.fn_pixels + comp.tp_pixels  # = gt['w'] * gt['h']
    else:
        # Use IoU for toleratedOtherClasses and same class
        area_union = comp.tp_pixels + comp.fp_pixels + comp.fn_pixels

    iou = area_intersect / float(area_union)

    assert 0 <= iou <= 1
    return iou


class IoU:
    def __init__(self, gt, det, *vargs, **kwargs):
        self.score = iou_ecp(gt, det)
        self.match = self.score >= 0.5

    def better_than(self, other_score):
        if other_score is None:
            return True
        return self.score >= other_score
