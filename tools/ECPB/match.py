"""
The heart of the evaluation toolchain.
"""

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import numpy as np


def do_nothing(entity):
    pass


def skip_none(entity):
    return False


def ignore_none(entity):
    return 0


def compare_identities_by_name(gt, det):
    return gt['identity'].lower() == det['identity'].lower()


def compare_all(gt, det):
    return True


class Result:
    def __init__(self, raw_result, skipped_gts=None, skipped_dets=None):
        self.raw_result = raw_result
        self.skipped_gts = skipped_gts
        self.skipped_dets = skipped_dets

        self.dets_including_ignored = []
        self.gts_including_ignored = []
        for frame in raw_result:
            self.dets_including_ignored.extend(frame['det']['children'])
            self.gts_including_ignored.extend(frame['gt']['children'])
        self.dets_including_ignored.sort(key=lambda k: k['score'], reverse=True)

        self.dets = [det for det in self.dets_including_ignored if det['matched'] != -1]
        self.gts = [gt for gt in self.gts_including_ignored if gt['ignore'] == 0]

        # Do not count detections which are matched with ignored gts
        tp = np.array([(1 if det['matched'] == 1 else 0) for det in self.dets])
        fp = 1 - tp
        self.tp = np.cumsum(tp)
        self.fp = np.cumsum(fp)

        self.nof_gts = len(self.gts)  # only count gts which are not ignored
        self.nof_imgs = len(self.raw_result)

    def save_to_disc(self, path):
        with open(path, 'wb') as f:
            self.save_to_stream(f)

    def save_to_stream(self, stream=None, seek=None):
        if stream is None:
            import cStringIO
            stream = cStringIO.StringIO()
        pkl.dump([self.raw_result, self.skipped_gts, self.skipped_dets], stream, protocol=-1)
        if seek is not None:
            stream.seek(seek)
        return stream

    @classmethod
    def load_from_disc(cls, path):
        with open(path, 'rb') as f:
            return cls.load_from_stream(f)

    @classmethod
    def load_from_stream(cls, stream):
        raw_result, skipped_gts, skipped_dets = pkl.load(stream)
        return cls(raw_result, skipped_gts, skipped_dets)


class Evaluator:
    """Judgment Day"""

    def __init__(self,
                 data,
                 metric,
                 comparable_identities=compare_all,
                 ignore_gt=ignore_none,
                 skip_gt=skip_none,
                 skip_det=skip_none,
                 preprocess_gt=do_nothing,
                 preprocess_det=do_nothing,
                 allow_multiple_matches=False,
                 ):

        self.data = data
        self.metric = metric
        self.ignore_gt = ignore_gt
        self.skip_gt = skip_gt
        self.skip_det = skip_det
        self.preprocess_gt = preprocess_gt
        self.preprocess_det = preprocess_det
        self.comparable_identities = comparable_identities

        self.allow_multiple_matches = allow_multiple_matches

        self.skipped_gts = {'count': 0, 'types': set()}  # also present in the Result class
        self.skipped_dets = {'count': 0, 'types': set()}  # also present in the Result class

        # ---------------------------------------
        # private attributes, used for evaluation
        # ---------------------------------------

        # the frame which is currently evaluated
        # dictionary with keys 'gt' and 'det'
        self._current_frame = None
        # don't start at 0, because "if gt['matched_by']" would return false if it was 0
        self.__det_and_gt_id = 1

        # frame based result, list of all "current_frames"
        self._raw_result = []

        # -------------------------------------------
        # Run the matching right after initialisation
        # -------------------------------------------
        self._run()
        self.result = Result(self._raw_result, skipped_gts=self.skipped_gts,
                             skipped_dets=self.skipped_dets)

    # starts the evaluation process, main loop
    def _run(self):
        # process each frame
        for frame in self.data:
            self._prepare_next_frame(frame)
            # process current frame, calculate matches
            self._evaluate_current_frame()

    def _prepare_next_frame(self, frame):
        gts = []
        dets = []

        # prepare ground truths
        for gt in frame['gt']['children']:
            self.preprocess_gt(gt)
            if self.skip_gt(gt):
                self.skipped_gts['count'] += 1
                self.skipped_gts['types'].add(gt['identity'])
                continue
            gt['ignore'] = self.ignore_gt(gt)
            gt['matched'] = 0  # used during the evaluation
            gt['__id__'] = self.__det_and_gt_id
            self.__det_and_gt_id += 1
            gts.append(gt)

        # prepare detections
        for det in frame['det']['children']:
            self.preprocess_det(det)
            if self.skip_det(det):
                self.skipped_dets['count'] += 1
                self.skipped_dets['types'].add(det['identity'])
                continue
            det['matched'] = 0
            det['__id__'] = self.__det_and_gt_id
            self.__det_and_gt_id += 1
            dets.append(det)

        frame['gt']['children'] = gts  # this step removes skipped gts (if any)
        frame['det']['children'] = dets  # this step removes skipped dets (if any)
        self._current_frame = frame

    def _evaluate_current_frame(self):
        gts = self._current_frame['gt']['children']
        dets = self._current_frame['det']['children']

        # sort det lists by detection score desc (highest score first)
        dets.sort(key=lambda k: k['score'], reverse=True)

        # sort gt_list by ignore flag (ignored gt at the end of the list)
        gts.sort(key=lambda k: k['ignore'])

        # iterate over all detections (sorted by score) and find best matching gt
        # (highest matching_score)
        for det in dets:
            current_score = None
            idx_best_gt = -1
            matched_with_ignore = False

            # iterate over all gt annotations
            for idx_gt, gt in enumerate(gts):

                # if det is matched with a non ignorable gt and current gt is ignorable:
                # next det (due to the sorting only ignore-gts follow)
                if idx_best_gt >= 0 and not matched_with_ignore and gt['ignore']:
                    break

                if gt['matched'] and not self.allow_multiple_matches:
                    continue

                if not self.comparable_identities(gt, det):
                    continue

                # compare gt and det
                metric = self.metric(gt=gt, det=det)

                if not metric.match:
                    # no match
                    continue

                if not metric.better_than(current_score):
                    # worse than previous match
                    continue

                current_score = metric.score  # new best match
                idx_best_gt = idx_gt

                if gt['ignore']:
                    matched_with_ignore = True

            if idx_best_gt >= 0:
                # Write match back to det
                det['metric_score'] = current_score
                if matched_with_ignore:
                    det['matched'] = -1
                else:
                    det['matched'] = 1

                det['matched_with'] = gts[idx_best_gt]['__id__']
                # Write match back to gt
                if not matched_with_ignore:
                    # only write back if gt is not ignored:
                    # ignored gts can be matched multiple times
                    gts[idx_best_gt]['matched'] = 1
                    gts[idx_best_gt]['matched_by'] = det['__id__']
                    gts[idx_best_gt]['matched_by_score'] = det['score']

        self._raw_result.append(self._current_frame)
