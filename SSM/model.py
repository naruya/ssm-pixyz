import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, CrossEntropy, IterativeLoss
from core import Prior_S, Decoder_S, Inference_S, EncoderRNN_S
from torch_utils import init_weights


class SSM(Model):
    def __init__(self, h_dim, s_dim, a_dim, T, device):
        self.device = device
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.prior_s = Prior_S(s_dim, a_dim).to(device)
        self.encoder_s = Inference_S(h_dim, s_dim, a_dim).to(device)
        self.decoder_s = Decoder_S(s_dim).to(device)
        self.rnn_s = EncoderRNN_S(h_dim, device).to(device)

        self.generate_from_prior_s = self.prior_s * self.decoder_s
        distributions = [self.rnn_s, self.encoder_s, self.decoder_s, self.prior_s]

        for distribution in distributions:
            init_weights(distribution)

        step_loss = CrossEntropy(self.encoder_s, self.decoder_s) + KullbackLeibler(
            self.encoder_s, self.prior_s
        )
        _loss = IterativeLoss(
            step_loss,
            max_iter=T,
            series_var=["x", "h", "a"],
            update_value={"s": "s_prev"},
        )
        loss = _loss.expectation(self.rnn_s).mean()

        super(SSM, self).__init__(
            loss,
            distributions=distributions,
            optimizer=optim.RMSprop,
            optimizer_params={"lr": 5e-4},
            clip_grad_value=10,
        )

    def sample_video_from_latent_s(self, loader):
        return _sample_video_from_latent_s(self, loader)


def _sample_video_from_latent_s(model, loader):
    device = model.device
    video = []
    x, a, end_epoch = next(loader)
    a = a.to(device).transpose(0, 1)
    _T, _B = a.size(0), a.size(1)
    s_prev = torch.zeros(_B, model.s_dim).to(device)
    for t in range(_T):
        samples = model.generate_from_prior_s.sample({"s_prev": s_prev, "a": a[t]})
        frame = model.decoder_s.sample_mean({"s": samples["s"]})
        s_prev = samples["s"]
        video.append(frame[None, :])
    video = torch.cat(video, dim=0).transpose(0, 1)
    return video