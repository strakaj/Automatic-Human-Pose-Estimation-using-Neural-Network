from pycocotools.cocoeval import COCOeval
import json
import os
import pickle
import numpy as np
import logging
from collections import defaultdict
from collections import OrderedDict

logger = logging.getLogger(__name__)


def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def evaluate(cfg, preds, output_dir, all_boxes, img_path, res_name="", *args, **kwargs):
    res_folder = os.path.join(output_dir, 'results' + res_name)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    res_file = os.path.join(
        res_folder, 'keypoints_%s_results.json' % cfg['IMAGE_SET'])

    _kpts = []
    for idx, kpt in enumerate(preds):
        _kpts.append({
            'keypoints': kpt,
            'center': all_boxes[idx][0:2],
            'scale': all_boxes[idx][2:4],
            'area': all_boxes[idx][4],
            'score': all_boxes[idx][5],
            'image': int(os.path.basename(img_path[idx]).split('.')[0])
        })

    kpts = defaultdict(list)
    for kpt in _kpts:
        kpts[kpt['image']].append(kpt)

    # rescoring and oks nms
    num_joints = cfg['NUM_JOINTS']
    in_vis_thre = cfg['IN_VIS_THRE']
    oks_thre = cfg['OKS_THRE']
    classes = cfg['CLASSES']
    image_set = cfg['IMAGE_SET']
    coco = cfg['COCO']
    oks_nmsed_kpts = []
    for img in kpts.keys():
        img_kpts = kpts[img]
        for n_p in img_kpts:
            box_score = n_p['score']
            kpt_score = 0
            valid_num = 0
            for n_jt in range(0, num_joints):
                t_s = n_p['keypoints'][n_jt][2]
                if t_s > in_vis_thre:
                    kpt_score = kpt_score + t_s
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            # rescoring
            n_p['score'] = kpt_score * box_score
        keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                       oks_thre)
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    _write_coco_keypoint_results(
        oks_nmsed_kpts, res_file, classes, num_joints)
    if 'test' not in image_set:
        info_str = _do_python_keypoint_eval(
            res_file, res_folder, coco, image_set)
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']
    else:
        return {'Null': 0}, 0


def _write_coco_keypoint_results(keypoints, res_file, classes, num_joints):
    data_pack = [{'cat_id': 1,
                  'cls_ind': cls_ind,
                  'cls': cls,
                  'ann_type': 'keypoints',
                  'keypoints': keypoints
                  }
                 for cls_ind, cls in enumerate(classes) if not cls == '__background__']

    results = _coco_keypoint_results_one_category_kernel(data_pack[0], num_joints)
    logger.info('=> Writing results json to %s' % res_file)
    with open(res_file, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    try:
        json.load(open(res_file))
    except Exception:
        content = []
        with open(res_file, 'r') as f:
            for line in f:
                content.append(line)
        content[-1] = ']'
        with open(res_file, 'w') as f:
            for c in content:
                f.write(c)


def _coco_keypoint_results_one_category_kernel(data_pack, num_joints):
    cat_id = data_pack['cat_id']
    keypoints = data_pack['keypoints']
    cat_results = []

    for img_kpts in keypoints:
        if len(img_kpts) == 0:
            continue

        _key_points = np.array([img_kpts[k]['keypoints']
                                for k in range(len(img_kpts))])
        key_points = np.zeros(
            (_key_points.shape[0], num_joints * 3), dtype=np.float)

        for ipt in range(num_joints):
            key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
            key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
            key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

        result = [{'image_id': img_kpts[k]['image'],
                   'category_id': cat_id,
                   'keypoints': list(key_points[k]),
                   'score': img_kpts[k]['score'],
                   'center': list(img_kpts[k]['center']),
                   'scale': list(img_kpts[k]['scale'])
                   } for k in range(len(img_kpts))]
        cat_results.extend(result)

    return cat_results


def _do_python_keypoint_eval(res_file, res_folder, coco, image_set):
    coco_dt = coco.loadRes(res_file)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    eval_file = os.path.join(
        res_folder, 'keypoints_%s_results.pkl' % image_set)

    with open(eval_file, 'wb') as f:
        pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
    logger.info('=> coco eval results saved to %s' % eval_file)

    return info_str


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
