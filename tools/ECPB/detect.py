import json
import os
import glob
import dataconverter


# TODO adapt this method to get real detections on the given image.
# You have to stick to the given result format.
def mock_detector(image):
    mock_detections = [{'x0': 0.0,
                        'x1': 10.0,
                        'y0': 0.0,
                        'y1': 100.0,
                        'score': 0.8,
                        'identity': 'pedestrian',
                        'orient': 0.0},
                       {'x0': 10.0,
                        'x1': 20.0,
                        'y0': 0.0,
                        'y1': 1000.0,
                        'score': 0.7,
                        'identity': 'rider',
                        'orient': 1.0}]
    return mock_detections


def run_detector_on_dataset(time='day', mode='val'):
    assert mode in ['val', 'test']
    assert time in ['day', 'night']

    eval_imgs = glob.glob('./data/{}/img/{}/*/*'.format(time, mode))
    destdir = './data/mock_detections/{}/{}/'.format(time, mode)
    dataconverter.create_base_dir(destdir)

    for im in eval_imgs:
        detections = mock_detector(im)
        destfile = os.path.join(destdir, os.path.basename(im).replace('.png', '.json'))
        frame = {'identity': 'frame'}
        frame['children'] = detections
        json.dump(frame, open(destfile, 'w'), indent=1)


if __name__ == "__main__":
    run_detector_on_dataset(time='day', mode='val')
