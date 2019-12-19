# optim
# https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193

import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, CrossEntropy, IterativeLoss
from core import Prior_S, Decoder_S, Inference_S, EncoderRNN_S
from torch_utils import init_weights
import sys


# s0の推論用のencoderを用いる
class SSM1(Model):
    def __init__(self, args, device):
        from core import Inference_S0

        self.device = device
        h_dim = args.h_dim
        s_dim = args.s_dim
        a_dim = args.a_dim

        self.encoder_s0 = Inference_S0(h_dim, s_dim).to(device)
        self.prior_s = Prior_S(s_dim, a_dim).to(device)
        self.encoder_s = Inference_S(h_dim, s_dim, a_dim).to(device)
        self.decoder_s = Decoder_S(s_dim).to(device)
        self.rnn_s = EncoderRNN_S(h_dim).to(device)

        self.distributions = [
            self.encoder_s0,
            self.rnn_s,
            self.encoder_s,
            self.decoder_s,
            self.prior_s,
        ]

        for dist in self.distributions:
            init_weights(dist)

        step_loss = CrossEntropy(self.encoder_s, self.decoder_s) + KullbackLeibler(
            self.encoder_s, self.prior_s
        )
        _loss = IterativeLoss(
            step_loss,
            max_iter=args.T,
            series_var=["x", "h", "a"],
            update_value={"s": "s_prev"},
        )
        loss = _loss.expectation(self.encoder_s0).expectation(self.rnn_s).mean()

        super(SSM1, self).__init__(
            loss,
            distributions=self.distributions,
            optimizer=optim.Adam,
            optimizer_params={}, # use default param lr: 1e-3
            clip_grad_norm=1000.0,
        )

        self.loss_ce = (
            IterativeLoss(
                CrossEntropy(self.encoder_s, self.decoder_s),
                max_iter=args.T,
                series_var=["x", "h", "a"],
                update_value={"s": "s_prev"},
            )
            .expectation(self.encoder_s0)
            .expectation(self.rnn_s)
            .mean()
        )

        self.loss_kl = (
            IterativeLoss(
                KullbackLeibler(self.encoder_s, self.prior_s),
                max_iter=args.T,
                series_var=["x", "h", "a"],
                # update_value={"s": "s_prev"},
            )
            .expectation(self.encoder_s0)
            .expectation(self.rnn_s)
            .mean()
        )

        # あとでdecoder使ってるし、これ必要?
        self.generate_from_prior_s = self.prior_s * self.decoder_s

    def sample_video_from_latent_s(self, loader):
        return _sample_video_from_latent_s(self, loader)


# s0の推論には事前学習済みのVAEを用いる
# class SSM2(Model):
#     def __init__(self, args, device):
#         sys.path.append("../")
#         from VAE.core import Inference as Inference_S0

#         self.device = device
#         h_dim = args.h_dim
#         s_dim = args.s_dim
#         a_dim = args.a_dim

#         vae_path = "epoch0404-iter00337-7104.103515625.pt"
#         self.encoder_s0 = Inference_S0(s_dim).to(device)
#         self.encoder_s0.load_state_dict(torch.load("../VAE/logs/q/" + vae_path))
#         self.encoder_s0.eval()

#         self.prior_s = Prior_S(s_dim, a_dim).to(device)
#         self.encoder_s = Inference_S(h_dim, s_dim, a_dim).to(device)
#         self.decoder_s = Decoder_S(s_dim).to(device)
#         self.rnn_s = EncoderRNN_S(h_dim).to(device)

#         self.distributions = [
#             self.rnn_s,
#             self.encoder_s,
#             self.decoder_s,
#             self.prior_s,
#         ]

#         for dist in self.distributions:
#             init_weights(dist)

#         step_loss = CrossEntropy(self.encoder_s, self.decoder_s) + KullbackLeibler(
#             self.encoder_s, self.prior_s
#         )
#         _loss = IterativeLoss(
#             step_loss,
#             max_iter=args.T,
#             series_var=["x", "h", "a"],
#             update_value={"s": "s_prev"},
#         )
#         loss = _loss.expectation(self.rnn_s).mean()

#         super(SSM2, self).__init__(
#             loss,
#             distributions=self.distributions,
#             optimizer=optim.RMSprop,
#             optimizer_params={"lr": 5e-4},
#             clip_grad_value=10,
#         )

#         # あとでdecoder使ってるし、これ必要?
#         self.generate_from_prior_s = self.prior_s * self.decoder_s

#     def sample_s0(self, x, train):
#         if train:
#             # s = self.encoder_s0.sample({"x": x}, return_all=False)["z"]
#             NotImplemented
#         else:
#             s = self.encoder_s0.sample_mean({"x": x})
#         return s

#     def sample_video_from_latent_s(self, loader):
#         return _sample_video_from_latent_s(self, loader)


# inferenceを使ってs0を推論する
class SSM3(Model):
    def __init__(self, args, device):
        self.device = device
        self.h_dim = h_dim = args.h_dim
        self.s_dim = s_dim = args.s_dim
        self.a_dim = a_dim = args.a_dim

        self.prior_s = Prior_S(s_dim, a_dim).to(device)
        self.encoder_s = Inference_S(h_dim, s_dim, a_dim).to(device)
        self.decoder_s = Decoder_S(s_dim).to(device)
        self.rnn_s = EncoderRNN_S(h_dim).to(device)

        self.distributions = [
            self.rnn_s,
            self.encoder_s,
            self.decoder_s,
            self.prior_s,
        ]

        for dist in self.distributions:
            init_weights(dist)

        step_loss = CrossEntropy(self.encoder_s, self.decoder_s) + KullbackLeibler(
            self.encoder_s, self.prior_s
        )
        _loss = IterativeLoss(
            step_loss,
            max_iter=args.T,
            series_var=["x", "h", "a"],
            update_value={"s": "s_prev"},
        )
        loss = _loss.expectation(self.rnn_s).mean()

        super(SSM3, self).__init__(
            loss,
            distributions=self.distributions,
            optimizer=optim.Adam,
            optimizer_params={}, # use default param lr: 1e-3
            clip_grad_norm=1000.0,
        )

        self.loss_ce = (
            IterativeLoss(
                CrossEntropy(self.encoder_s, self.decoder_s),
                max_iter=args.T,
                series_var=["x", "h", "a"],
                update_value={"s": "s_prev"},
            )
            .expectation(self.rnn_s)
            .mean()
        )

        self.loss_kl = (
            IterativeLoss(
                KullbackLeibler(self.encoder_s, self.prior_s),
                max_iter=args.T,
                series_var=["x", "h", "a"],
                # update_value={"s": "s_prev"},
            )
            .expectation(self.rnn_s)
            .mean()
        )

        # あとでdecoder使ってるし、これ必要?
        self.generate_from_prior_s = self.prior_s * self.decoder_s

    def sample_s0(self, x):  # context forwarding
        _T, _B = x.size(0), x.size(1)
        if not x.size(1) == 1:
            NotImplemented
        s_prev = torch.zeros(_B, self.s_dim).to(self.device)
        a = torch.zeros(_B, self.a_dim).to(self.device)
        h = self.rnn_s.sample({"x": x}, return_all=False)["h"]
        h = torch.transpose(h, 0, 1)[:, 0]
        feed_dict = {"h": h, "s_prev": s_prev, "a": a}
        s = self.encoder_s.sample(feed_dict, return_all=False)["s"]
        return s

    def sample_video_from_latent_s(self, loader):
        return _sample_video_from_latent_s(self, loader)


def _sample_video_from_latent_s(model, batch):
    device = model.device
    video = []
    x, a, _ = batch
    a = a.to(device).transpose(0, 1)
    x = x.to(device).transpose(0, 1)
    _T, _B = a.size(0), a.size(1)

    name = model.__class__.__name__
    if name == "SSM1":
        s_prev = model.encoder_s0.sample_mean({"x0": x[0]})
    elif name == "SSM2":
        s_prev = model.sample_s0(x[0], train=False)  # TODO: train=False?
    elif name == "SSM3":
        s_prev = model.sample_s0(x[0:1])

    for t in range(_T):
        samples = model.generate_from_prior_s.sample({"s_prev": s_prev, "a": a[t]})
        frame = model.decoder_s.sample_mean({"s": samples["s"]})
        s_prev = samples["s"]  # TODO 一行前にする
        video.append(frame[None, :])
    video = torch.cat(video, dim=0).transpose(0, 1)
    x = x.transpose(0, 1)  # 2,B,T,C,H,W -> B,2,T,C,H,W
    video = torch.stack([video, x]).transpose(0, 1).reshape(_B * 2, _T, 3, 64, 64)
    return video