import math
import torch
import torchvision
from itertools import product as product

from .faceboxes import FaceBoxes

class FaceBoxesDetector(torch.nn.Module):
    def __init__(self, model_weights):
        super().__init__()
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        state_dict = torch.load(model_weights, map_location='cpu', weights_only=True)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        # build prior data
        self.currect_image_size = None
        self.current_prior_data = None

    def build_prior_data(self, image_size, device):
        if self.currect_image_size == image_size:
            return self.current_prior_data
        else:
            priorbox = PriorBox(image_size=image_size)
            self.current_prior_data = priorbox.forward().to(device).data
            self.currect_image_size = image_size
            return self.current_prior_data

    @torch.inference_mode()
    def detect(self, image, thresh=0.7):
        assert image.dim() == 3, f"Image must be 3D, got {image.dim()} dimensions"
        assert image.shape[0] == 3, f"Image must be RGB, got {image.shape[0]} channels"
        _, height, width = image.shape
        im_scale = 600. / min(height, width) if min(height, width) > 600 else 1.
        image_scale = torchvision.transforms.functional.resize(image, 600, antialias=False).float()
        scale = torch.Tensor([image_scale.shape[2], image_scale.shape[1], image_scale.shape[2], image_scale.shape[1]]).to(image.device)
        image_scale = image_scale[[2, 1, 0], :, :] # rgb to bgr
        image_scale -= torch.IntTensor([104, 117, 123]).to(image.device)[:, None, None]
        image_scale = image_scale[None]
        loc, conf = self.net(image_scale)
        prior_data = self.build_prior_data(image_scale.size()[2:], image.device).clone()
        boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2]) * scale 
        scores = conf.data[:, 1]
        # ignore low scores
        inds = torch.where(scores > thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        # do NMS
        keep = torchvision.ops.nms(boxes, scores, 0.3)
        dets = torch.cat([boxes[keep], scores[keep][:, None]], dim=1)
        # bbox x1y1x2y2 to x1y1w1h1
        detections_scale = torch.cat([dets[:, :2]/im_scale, (dets[:, 2:4] - dets[:, :2])/im_scale, dets[:, 4:5]], dim=1)
        if detections_scale.shape[0] == 0:
            return None
        # max_detections_scale = get_max_bbox(detections_scale)
        return detections_scale

    def crop_image(self, image, bbox, bbox_scale=1.42):
        x1, y1, w, h, _ = bbox
        _, image_height, image_width = image.shape
        center_x, center_y = x1 + w // 2, y1 + h // 2
        size = int(max(w, h) * bbox_scale)
        new_x1_min, new_y1_min = int(center_x - size // 2), int(center_y - size // 2)
        new_x1_max, new_y1_max = int(center_x + size // 2), int(center_y + size // 2)
        if new_x1_min < 0 or new_y1_min < 0:
            min_overflow = min(new_x1_min, new_y1_min)
            new_x1_min += -min_overflow
            new_y1_min += -min_overflow
        if new_x1_max > image_width - 1 or new_y1_max > image_height - 1:
            max_overflow = max(new_x1_max - image_width - 1, new_y1_max - image_height - 1)
            new_x1_max -= max_overflow
            new_y1_max -= max_overflow
        croped_image = image[:, new_y1_min:new_y1_max, new_x1_min:new_x1_max].clone()
        return croped_image

    def draw_detect(self, image, bbox):
        image = image.clone()
        bbox = bbox.clone()
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
        image = torchvision.utils.draw_bounding_boxes(image, bbox[..., :4], colors='red', width=5)
        return image

def get_max_bbox(bboxes):
    boxes_size = bboxes[:, 2] * bboxes[:, 3]
    max_idx = boxes_size.argmax()
    return bboxes[max_idx]


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class PriorBox(object):
    def __init__(self, image_size=None, clip=False):
        super(PriorBox, self).__init__()
        self.clip = clip
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.image_size = image_size
        self.feature_maps = [[math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
