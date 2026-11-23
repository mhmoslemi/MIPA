import torch
import torch.nn as nn

if __package__:
    from .common import load_encoder
else:
    from common import load_encoder


class NormalizedTeacher(nn.Module):
    def __init__(self, network, source, target):
        super().__init__()
        self.network = network
        for name, values in zip(('source_mean', 'source_std', 'teacher_mean', 'teacher_std'), (*source, *target)):
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1))

    def embed(self, images):
        return self.network((images * self.source_std + self.source_mean - self.teacher_mean) / self.teacher_std)

    def forward(self, images):
        return self.embed(images)


def load_teacher(path, dataset, normalization, size, device):
    network, _, saved = load_encoder(path, device=device)
    if (saved['model'], saved['source'], saved['input_size']) != ('ResNet18', dataset, size):
        raise ValueError('Use a ResNet-18 teacher trained on the source dataset at its native resolution')
    teacher = NormalizedTeacher(network, normalization, saved['normalization']).to(device).eval()
    return teacher, saved['seed'], saved['encoder_sha256']
