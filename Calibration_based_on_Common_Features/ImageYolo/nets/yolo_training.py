import math
from functools import partial

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utils_bbox import get_anchors_and_decode



def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def box_iou(b1, b2):

    b1          = K.expand_dims(b1, -2)
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half


    b2          = K.expand_dims(b2, 0)
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def box_iou_loss(b1, b2, iou_type='siou'):

    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half


    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / K.maximum(union_area, K.epsilon())


    center_wh       = b1_xy - b2_xy
        

    enclose_mins    = K.minimum(b1_mins, b2_mins)
    enclose_maxes   = K.maximum(b1_maxes, b2_maxes)
    enclose_wh      = K.maximum(enclose_maxes - enclose_mins, 0.0)
    
    if iou_type == 'ciou':

        center_distance = K.sum(K.square(center_wh), axis=-1)

        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
        ciou    = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())
        
        v       = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
        alpha   = v /  K.maximum((1.0 - iou + v), K.epsilon())
        out     = ciou - alpha * v

    elif iou_type == 'siou':

        sigma       = tf.pow(center_wh[..., 0] ** 2 + center_wh[..., 1] ** 2, 0.5)
        

        sin_alpha_1 = tf.abs(center_wh[..., 0]) / K.maximum(sigma, K.epsilon())
        sin_alpha_2 = tf.abs(center_wh[..., 1]) / K.maximum(sigma, K.epsilon())


        threshold   = pow(2, 0.5) / 2
        sin_alpha   = tf.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)


        angle_cost  = tf.cos(tf.asin(sin_alpha) * 2 - math.pi / 2)
        gamma       = 2 - angle_cost
        

        rho_x           = (center_wh[..., 0] / K.maximum(enclose_wh[..., 0], K.epsilon())) ** 2
        rho_y           = (center_wh[..., 1] / K.maximum(enclose_wh[..., 1], K.epsilon())) ** 2
        distance_cost   = 2 - tf.exp(-gamma * rho_x) - tf.exp(-gamma * rho_y)
        

        omiga_w     = tf.abs(b1_wh[..., 0] - b2_wh[..., 0]) / K.maximum(tf.maximum(b1_wh[..., 0], b2_wh[..., 0]), K.epsilon())
        omiga_h     = tf.abs(b1_wh[..., 1] - b2_wh[..., 1]) / K.maximum(tf.maximum(b1_wh[..., 1], b2_wh[..., 1]), K.epsilon())
        shape_cost  = tf.pow(1 - tf.exp(-1 * omiga_w), 4) + tf.pow(1 - tf.exp(-1 * omiga_h), 4)
        out         = iou - 0.5 * (distance_cost + shape_cost)

    return K.expand_dims(out, -1)


def yolo_loss(
    args, 
    input_shape, 
    anchors, 
    anchors_mask, 
    num_classes, 
    ignore_thresh   = 0.5,
    balance         = [0.4, 1.0, 4], 
    box_ratio       = 0.05, 
    obj_ratio       = 1, 
    cls_ratio       = 0.5 / 4, 
    label_smoothing = 0.1, 
    focal_loss      = False,
    focal_loss_ratio= 10,
    gamma           = 2,
    alpha           = 0.25, 
    print_loss      = False,
    iou_type        = 'ciou'
):
    num_layers = len(anchors_mask)

    y_true          = args[num_layers:]
    yolo_outputs    = args[:num_layers]


    input_shape = K.cast(input_shape, K.dtype(y_true[0]))


    m = K.shape(yolo_outputs[0])[0]

    loss    = 0

    for l in range(num_layers):

        object_mask         = y_true[l][..., 4:5]

        true_class_probs    = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)


        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
             anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)
        

        pred_box = K.concatenate([pred_xy, pred_wh])


        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        

        def loop_body(b, ignore_mask):  # b is num of imgs

            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])

            iou = box_iou(pred_box[b], true_box)


            best_iou = K.max(iou, axis=-1)


            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask


        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])


        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)


        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]


        raw_true_box    = y_true[l][...,0:4]
        ciou            = box_iou_loss(pred_box, raw_true_box, iou_type)
        ciou_loss       = object_mask * (1 - ciou)
        location_loss   = K.sum(ciou_loss)
        

        if focal_loss:
            confidence_loss = (object_mask * (tf.ones_like(raw_pred[...,4:5]) - tf.sigmoid(raw_pred[...,4:5])) ** gamma * alpha * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
                        (1 - object_mask) * ignore_mask * tf.sigmoid(raw_pred[...,4:5]) ** gamma * (1 - alpha) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)) * focal_loss_ratio
        else:
            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
                        (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        
        class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        

        num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)


        location_loss   = location_loss * box_ratio / num_pos
        confidence_loss = K.sum(confidence_loss) * balance[l] * obj_ratio / (num_pos + num_neg)
        class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        loss            += location_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, tf.shape(ignore_mask)], summarize=100, message='loss: ')
    return loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

