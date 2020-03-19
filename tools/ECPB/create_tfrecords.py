import glob
import hashlib
import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )

tf.app.flags.DEFINE_string('out_dir', './data/ecp/tfrecords', 'Place to search for the created files.')
tf.app.flags.DEFINE_string('dataset_name', 'ecp-day',
                           'Name of the dataset, used to create the tfrecord files.')
tf.app.flags.DEFINE_string('anno_path', './data/day/labels',
                           'Base directory which contains the ecp annotations.')
tf.app.flags.DEFINE_string('img_path', './data/day/img',
                           'Base directory which contains the ecp images.')

tf.app.flags.DEFINE_integer('train_shards', 20, 'Number of training shards.')
tf.app.flags.DEFINE_integer('val_shards', 4, 'Number of validation shards.')

tf.app.flags.DEFINE_integer('shuffle', 1, 'Shuffle the data before writing it to tfrecord files.')

FLAGS = tf.app.flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class ExampleCreator:
    def __init__(self, create_sha256_key=False):
        self.create_sha256_key = create_sha256_key

        # Create a single Session to run all image coding calls.

        self._sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}, gpu_options={'allow_growth': True}))

        # Initializes function that decodes RGB PNG data.
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decoded = tf.image.decode_png(self._decode_data, channels=3)

        self._encode_data = tf.placeholder(dtype=tf.uint8)
        self._encoded = tf.image.encode_png(self._encode_data)

        self.identity_to_label = {
            'pedestrian': 0,
            'rider': 1,
        }

    def decode_png(self, img_data):
        img = self._sess.run(self._decoded, feed_dict={self._decode_data: img_data})
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        return img

    def encode_png(self, img):
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        return self._sess.run(self._encoded, feed_dict={self._encode_data: img})

    def load_img(self, path):
        ext = os.path.splitext(path)[1]
        if path.endswith('.pgm'):
            raise NotImplementedError('pgm not supported')
        if path.endswith('.png'):
            with tf.gfile.FastGFile(path, 'rb') as f:
                img_data = f.read()
            # seems a little bit stupid to first decode and then encode the image, but so what...
            return self.decode_png(img_data), ext[1:]
        else:
            raise NotImplementedError('unknown file format: {}'.format(ext))

    def create_example(self, img_path, anno_path):
        assert os.path.splitext(os.path.basename(img_path))[0] == os.path.splitext(os.path.basename(anno_path))[0]

        img, format = self.load_img(img_path)
        with open(anno_path, 'r') as f:
            annotations = json.load(f)
        img_height, img_width = img.shape[:2]
        assert img_height == 1024
        assert img_width == 1920
        encoded = self.encode_png(img)

        if self.create_sha256_key:
            key = hashlib.sha256(encoded).hexdigest()
        else:
            key = '__no_key_generated__'

        ymin, xmin, ymax, xmax, label, text = [], [], [], [], [], []
        img_tags = [tag.encode('utf8') for tag in annotations['tags']]
        skipped_annotations = 0
        box_cnt = 0
        box_sizes = []
        for anno in annotations['children']:
            if anno['identity'] not in self.identity_to_label.keys():
                skipped_annotations += 1
                continue
                # TODO add loading of ignore regions if you want to use them
            box_cnt += 1

            if anno['identity'] == 'rider':
                pass
                # TODO consider bounding box of ridden vehicle that is stored in anno['children']

            cls_label = self.identity_to_label[anno['identity']]
            ymin.append(float(anno['y0']) / img_height)
            xmin.append(float(anno['x0']) / img_width)
            ymax.append(float(anno['y1']) / img_height)
            xmax.append(float(anno['x1']) / img_width)
            if xmax[-1] > 1:
                print('oh no...')
            label.append(cls_label)
            text.append(anno['identity'].encode('utf8'))

            h = ymax[-1] - ymin[-1]
            w = xmax[-1] - xmin[-1]
            box_sizes.append((h, w))

        if skipped_annotations > 0:
            logging.debug(
                'Skipped {}/{} annotations for img {}'.format(skipped_annotations, len(annotations), img_path))

        feature_dict = {
            'image/height': int64_feature(img_height),
            'image/width': int64_feature(img_width),
            'img/tags': bytes_list_feature(img_tags),
            'image/filename': bytes_feature(img_path.encode('utf8')),
            'image/source_id': bytes_feature(img_path.encode('utf8')),
            'image/key/sha256': bytes_feature(key.encode('utf8')),
            'image/encoded': bytes_feature(encoded),
            'image/format': bytes_feature('png'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/text': bytes_list_feature(text),
            'image/object/class/label': int64_list_feature(label),
            'image/object/cnt': int64_feature(box_cnt),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return example, skipped_annotations, box_sizes, (img_height, img_width)


def write_shard(args):
    shard, num_shards, type, data, example_creator = args

    out_fn = '{}-{}-{:05d}-of-{:05d}'.format(FLAGS.dataset_name, type, shard, num_shards)
    out_file = os.path.join(FLAGS.out_dir, out_fn)
    writer = tf.python_io.TFRecordWriter(out_file)
    logging.info('Creating shard {}-{}/{}'.format(type, shard, num_shards))

    skipped_annotations = 0
    box_sizes = []
    img_sizes = set()
    cnt = 0
    for cnt, datum in enumerate(data, start=1):

        img_path, anno_path = datum

        example, skipped, sizes, img_size = example_creator.create_example(img_path, anno_path)
        skipped_annotations += skipped
        box_sizes.extend(sizes)
        img_sizes.add(img_size)

        writer.write(example.SerializeToString())
        if cnt % 10 == 0:
            logging.info('Written {} examples for shard {}-{}/{}'.format(cnt, type, shard, num_shards))

    if skipped_annotations > 0:
        logging.info('Written {} examples for shard {}-{}/{}'.format(cnt, type, shard, num_shards))

    logging.info(
        'Finished shard {}-{}/{}: {} examples written and {} annotations skipped'.format(type, shard, num_shards, cnt,
                                                                                         skipped_annotations))
    return box_sizes, type, img_sizes


def create_jobs(type, data, num_shards, example_creator):
    if FLAGS.shuffle:
        np.random.shuffle(data)

    # split into roughly even sized pieces
    k, m = divmod(len(data), num_shards)
    shards = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_shards)]

    # check if we didn't f@#! it up
    total_length = 0
    for shard in shards:
        total_length += len(shard)
    assert total_length == len(data)

    # create and run jobs
    jobs = [(shard_id + 1, num_shards, type, data, example_creator) for shard_id, data in enumerate(shards)]
    return jobs


def get_files(path, ext):
    files = glob.glob(os.path.join(path, '*', '*.{}'.format(ext)))
    files = sorted(files)
    return files


def process_dataset():
    create_dirs([FLAGS.out_dir])

    if FLAGS.shuffle:
        with open(os.path.join(FLAGS.out_dir, FLAGS.dataset_name + '-np_random_state'), 'wb') as f:
            pickle.dump(np.random.get_state(), f)

    # prepare train and val splits
    train_img_path = os.path.join(FLAGS.img_path, 'train')
    val_img_path = os.path.join(FLAGS.img_path, 'val')
    train_imgs = get_files(train_img_path, 'png')
    val_imgs = get_files(val_img_path, 'png')

    train_anno_path = os.path.join(FLAGS.anno_path, 'train')
    val_anno_path = os.path.join(FLAGS.anno_path, 'val')
    train_annos = get_files(train_anno_path, 'json')
    val_annos = get_files(val_anno_path, 'json')

    train_data = list(zip(train_imgs, train_annos))
    val_data = list(zip(val_imgs, val_annos))

    # object which does all the hard work
    example_creator = ExampleCreator()

    # Process each split in a different thread
    train_jobs = create_jobs('train', train_data, FLAGS.train_shards, example_creator)
    val_jobs = create_jobs('val', val_data, FLAGS.val_shards, example_creator)

    jobs = train_jobs + val_jobs

    with ThreadPoolExecutor() as executor:
        result = executor.map(write_shard, jobs,
                              chunksize=1)  # chunksize=1 is important, since our jobs are long running

    box_sizes = []
    img_sizes = set()
    for sizes, type, img_sizes_ in result:
        img_sizes.update(img_sizes_)
        if type == 'train':
            box_sizes.extend(sizes)

    if len(img_sizes) > 1:
        logging.error('Different image sizes detected: {}'.format(img_sizes))

    box_sizes = np.array(box_sizes, np.float64)
    np.save(os.path.join(FLAGS.out_dir, FLAGS.dataset_name + '-box_sizes'), box_sizes)
    np.save(os.path.join(FLAGS.out_dir, FLAGS.dataset_name + '-img_size_height_width'), list(img_sizes)[0])


def create_dirs(dirs):
    for dir in dirs:
        try:
            os.makedirs(dir)
        except OSError:
            assert os.path.isdir(dir), '{} exists but is not a directory'.format(dir)


def main(args):
    assert FLAGS.out_dir
    assert FLAGS.dataset_name
    assert FLAGS.img_path
    assert FLAGS.anno_path
    assert FLAGS.train_shards
    assert FLAGS.val_shards

    logging.info('Saving results to {}'.format(FLAGS.out_dir))
    logging.info('----- START -----')
    start = time.time()

    process_dataset()

    end = time.time()
    elapsed = int(end - start)
    logging.info('----- FINISHED in {:02d}:{:02d}:{:02d} -----'.format(elapsed // 3600,
                                                                       (elapsed // 60) % 60,
                                                                       elapsed % 60))


if __name__ == '__main__':
    tf.app.run()
