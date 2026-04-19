import torch
import torchvision.transforms.functional as TF

def tta_voting(models_list, images, device):
    shift_img = TF.affine(images, angle=0, translate=[2, 2], scale=1.0, shear=0)
    rot_img = TF.affine(images, angle=10, translate=[0, 0], scale=1.0, shear=0)

    final_prob = torch.zeros(images.size(0), 10).to(device)

    for net in models_list:
        net.eval()
        with torch.no_grad():
            out1 = net(images)
            out2 = net(shift_img)
            out3 = net(rot_img)
            
            p1 = torch.softmax(out1, dim=1)
            p2 = torch.softmax(out2, dim=1)
            p3 = torch.softmax(out3, dim=1)
            
            tta_prob = (p1 + p2 + p3) / 3.0
            final_prob += tta_prob
            
    final_prob = final_prob / len(models_list)
    return torch.argmax(final_prob, dim=1)
