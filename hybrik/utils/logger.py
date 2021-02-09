import torch


def board_writing(writer, loss, acc, iterations, dataset='Train'):
    writer.add_scalar(
        '{}/Loss'.format(dataset), loss, iterations)
    writer.add_scalar(
        '{}/acc'.format(dataset), acc, iterations)


def debug_writing(writer, outputs, labels, inputs, iterations):
    tmp_tar = torch.unsqueeze(labels.cpu().data[0], dim=1)
    # tmp_out = torch.unsqueeze(outputs.cpu().data[0], dim=1)

    tmp_inp = inputs.cpu().data[0]
    tmp_inp[0] += 0.406
    tmp_inp[1] += 0.457
    tmp_inp[2] += 0.480

    # tmp_inp[0] += torch.sum(F.interpolate(tmp_tar, scale_factor=4, mode='bilinear', align_corners=True), dim=0)[0]
    tmp_inp.clamp_(0, 1)
    writer.add_image('Data/input', tmp_inp, iterations)
    for i in range(18, 18 + 17):
        writer.add_image('Data/target', tmp_tar[i], iterations)
