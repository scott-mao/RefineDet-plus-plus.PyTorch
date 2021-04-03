import torch
from torch.autograd import Function
import numpy as np
from ..box_utils import decode, center_size#, nms
from data import coco_refinedet as cfg
from utils.nms_wrapper import nms, soft_nms


class Detect_RefineDet(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, size, bkg_label, objectness_threshold=0.01, 
                confidence_threshold=0.01, nms_threshold=0.5, top_k=1000, keep_top_k=500):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.objectness_threshold = objectness_threshold
        self.variance = cfg[str(size)]['variance']
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_threshold = nms_threshold
        if nms_threshold <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.confidence_threshold = confidence_threshold

    def forward(self, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data, scale):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data = odm_loc_data
        conf_data = odm_conf_data

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_threshold
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes)

        self.boxes = torch.zeros(num, num_priors, 4)
        self.scores = torch.zeros(num, num_priors, self.num_classes)
        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()

        # Decode predictions into bboxes.
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data, self.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, self.variance)
            conf_scores = conf_preds[i].clone()
            
            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        # only support single image test now.
        boxes = self.boxes[0]
        scores= self.scores[0]
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        for j in range(1, self.num_classes):
            inds = np.where(scores[:, j] > self.confidence_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            # keep top-K before NMS
            order = c_scores.argsort()[::-1][:self.top_k]
            c_bboxes = c_bboxes[order]
            c_scores = c_scores[order]
            # do NMS
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(c_dets, self.nms_threshold, force_cpu=(not loc_data.is_cuda))
            # keep = soft_nms(c_dets, sigma=0.5, Nt=self.nms_threshold, threshold=self.confidence_threshold, method=1)  # higher performance
            c_dets = c_dets[keep, :]  
            # keep top-K after NMS
            c_dets = c_dets[:self.keep_top_k, :]
            # record labels
            labels = np.ones(c_dets[:, [1]].shape, dtype=np.float32) * j
            det_c = np.hstack((c_dets, labels))
            try:
                det = np.row_stack((det, det_c))
            except:
                det = det_c

        try:
            return det
        except:
            return np.zeros((1, 6), dtype=np.float32)

    def forward_torch_nms(self, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data):
        """
        Deprecated.
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data = odm_loc_data
        conf_data = odm_conf_data

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_threshold
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data, self.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.confidence_threshold)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_threshold, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
