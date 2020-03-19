import json
import os


def get_inverse_dict(d):
    inverse_dict = {}
    for key, val in d.iteritems():
        inverse_dict[val] = key
    return inverse_dict


def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


kitti_to_ecp_classes = {'Pedestrian': 'pedestrian', 'Cyclist': 'rider'}
ecp_to_kitti_classes = get_inverse_dict(kitti_to_ecp_classes)
# kitti occlusion levels not really relate to ecp occlusions
# lvl 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
kitti_occlusion_level_to_ecp_occlusion = {3: 'occluded>80', 2: 'occluded>40', 1: 'occluded>10'}
ecp_occlusion_to_kitti_occlusion_level = get_inverse_dict(kitti_occlusion_level_to_ecp_occlusion)


def kitti_to_ecp(source, dest='conversion_test/ecp.json'):
    objects = []
    for l in open(source, 'r').readlines():
        lstripped = l.strip()
        fields = lstripped.split(' ')
        classtype = fields[0]
        occlusion_lvl = int(fields[2])
        truncation = float(fields[1])

        tags = []
        if occlusion_lvl in kitti_occlusion_level_to_ecp_occlusion:
            tags.append(kitti_occlusion_level_to_ecp_occlusion[occlusion_lvl])
        for trunc in [0.8, 0.4, 0.1]:
            if truncation > trunc:
                tags.append('truncated>{:2d}'.format(int(trunc * 100)))
                break

        if classtype in kitti_to_ecp_classes:
            obj = {'identity': kitti_to_ecp_classes[classtype],
                   'x0': float(fields[4]),
                   'x1': float(fields[6]),
                   'y0': float(fields[5]),
                   'y1': float(fields[7]),
                   'tags': tags,
                   'orient': float(fields[3]),
                   'score': float(fields[15]) if len(fields) == 16 else 0.0}
            objects.append(obj)

    frame = {'identity': 'frame'}
    frame['children'] = objects
    json.dump(frame, open(dest, 'w'), indent=1)


def ecp_to_kitti(source, dest='conversion_test/kitti.txt'):
    frame = json.load(open(source, 'r'))
    destf = open(dest, 'w')
    for c in frame['children']:
        if c['identity'] in ecp_to_kitti_classes:
            kitti_occlusion = 0
            kitti_truncation = 0
            for tag in c['tags']:
                if tag in ecp_occlusion_to_kitti_occlusion_level:
                    kitti_occlusion = ecp_occlusion_to_kitti_occlusion_level[tag]
                if 'truncated' in tag:
                    for t_val in [10, 40, 80]:
                        if str(t_val) in tag:
                            kitti_truncation = t_val / 100.0
                            break

            destf.writelines('{} {} {} {} {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {}\n'.format(
                ecp_to_kitti_classes[c['identity']],
                kitti_truncation,
                kitti_occlusion,
                c['orient'],
                c['x0'],
                c['y0'],
                c['x1'],
                c['y1'],
                c['score']))


# as truncation levels are discretized in ecp the transformation from kitti to ecp can not be
# completely inverted
kitti_to_ecp('conversion_test/kitti_in.txt')
ecp_to_kitti('conversion_test/ecp.json')
