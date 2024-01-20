import torch
import torch.nn as nn

from models.head import ClassificationHead
from models.ir50 import Backbone


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]

        stage_num = (3, 4, 14, 3)
        if k.startswith('body.'):
            index = int(k.split('.')[1])
            if 0 <= index < stage_num[0]:
                k = k.replace('body.', 'body1.')
            elif stage_num[0] <= index < sum(stage_num[:2]):
                k = f"body2.{index - sum(stage_num[:1])}.{'.'.join(k.split('.')[2:])}"
            elif sum(stage_num[:2]) <= index < sum(stage_num[:3]):
                k = f"body3.{index - sum(stage_num[:2])}.{'.'.join(k.split('.')[2:])}"
            elif sum(stage_num[:3]) <= index < sum(stage_num[:4]):
                k = f"body4.{index - sum(stage_num[:3])}.{'.'.join(k.split('.')[2:])}"
            else:
                k = k

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


class Model(nn.Module):
    def __init__(self, num_classes=7):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.ir_back = Backbone(50, 0.0, 'ir', with_head=True)

        ir_checkpoint = torch.load(r'/home/panr/all_results/POSTER_V2/pretrained_other/backbone_ir50_ms1m_epoch120.pth',
                                   map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)  # (b,256,14,14)
        self.head = ClassificationHead(size_in=512, size_out=num_classes, size_hidden=[256], dropout=0.5,
                                       batch_norm=False)
        # self.head = ArcFace(512, num_classes)

    def forward(self, imgs):
        features, features4 = self.ir_back(imgs)
        output = self.head(features)
        return output

# from torchinfo import summary
#
# model = Model()
# summary(model, (12, 3, 112, 112))
