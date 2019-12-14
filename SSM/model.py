import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, CrossEntropy, IterativeLoss
from core import Prior_S, Decoder_S, Inference_S, EncoderRNN_S, Inference_S0
from torch_utils import init_weights


class SSM(Model):
    def __init__(self, h_dim, s_dim, a_dim, T, device):
        self.device = device
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.encoder_s0 = Inference_S0(h_dim, s_dim).to(device)
        self.prior_s = Prior_S(s_dim, a_dim).to(device)
        self.encoder_s = Inference_S(h_dim, s_dim, a_dim).to(device)
        self.decoder_s = Decoder_S(s_dim).to(device)
        self.rnn_s = EncoderRNN_S(h_dim).to(device)

        distributions = [
            self.encoder_s0,
            self.rnn_s,
            self.encoder_s,
            self.decoder_s,
            self.prior_s,
        ]

        for distribution in distributions:
            init_weights(distribution)

        step_loss = CrossEntropy(self.encoder_s, self.decoder_s) + KullbackLeibler(
            self.encoder_s, self.prior_s
        )
        _loss = IterativeLoss(
            step_loss,
            max_iter=T,
            series_var=["x", "h", "a"],  # x0を時間方向に複製して転地?
            update_value={"s": "s_prev"},
        )
        loss = _loss.expectation(self.encoder_s0).expectation(self.rnn_s).mean()

        super(SSM, self).__init__(
            loss,
            distributions=distributions,
            optimizer=optim.RMSprop,
            optimizer_params={"lr": 5e-4},
            clip_grad_value=10,
        )

        # あとでdecoder使ってるし、これ必要?
        self.generate_from_prior_s = self.prior_s * self.decoder_s

    def sample_video_from_latent_s(self, loader):
        return _sample_video_from_latent_s(self, loader)


def _sample_video_from_latent_s(model, batch):
    device = model.device
    video = []
    x, a, _ = batch
    a = a.to(device).transpose(0, 1)
    x = x.to(device).transpose(0, 1)
    _T, _B = a.size(0), a.size(1)

    # s_prev = torch.zeros(_B, model.s_dim).to(device)
    s_prev = model.encoder_s0.sample_mean({"x0": x[0]})

    for t in range(_T):
        samples = model.generate_from_prior_s.sample({"s_prev": s_prev, "a": a[t]})
        frame = model.decoder_s.sample_mean({"s": samples["s"]})
        s_prev = samples["s"]  # TODO 一行前にする
        video.append(frame[None, :])
    video = torch.cat(video, dim=0).transpose(0, 1)
    return video