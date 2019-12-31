# optimizer: https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193

import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, CrossEntropy, IterativeLoss
from core import Prior_S, Decoder_S, Inference_S, EncoderRNN_S
from torch.nn.utils import clip_grad_norm_
from torch_utils import init_weights


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
            optimizer_params={},  # TODO grad norm clipping
        )
        
        init_weights(self.distributions)

        # あとでdecoder使ってるし、これ必要?
        self.generate_from_prior_s = self.prior_s * self.decoder_s

    def sample_s0(self, x):
        return _sample_s0(self, x)

    def sample_video_from_latent_s(self, loader):
        return _sample_video_from_latent_s(self, loader)


class SSM4(nn.Module):
    def __init__(self, args, device):
        super(SSM4, self).__init__()
        self.device = device
        self.h_dim = h_dim = args.h_dim
        self.s_dim = s_dim = args.s_dim
        self.a_dim = a_dim = args.a_dim

        self.prior_s = Prior_S(s_dim, a_dim).to(device)
        self.encoder_s = Inference_S(h_dim, s_dim, a_dim).to(device)
        self.decoder_s = Decoder_S(s_dim).to(device)
        self.rnn_s = EncoderRNN_S(h_dim).to(device)
        
        self.distributions = nn.ModuleList([
            self.prior_s,
            self.encoder_s,
            self.decoder_s,
            self.rnn_s,
        ])  # to ModuleList fo save_model
        init_weights(self.distributions)

        self.s_loss_cls = KullbackLeibler(self.encoder_s, self.prior_s)  # TODO: 自分で書いてみる
        # self.x_loss_cls = CrossEntropy(self.encoder_s, self.decoder_s)
        self.optimizer = optim.Adam(self.distributions.parameters())

        self.generate_from_prior_s = self.prior_s * self.decoder_s  # これ必要?

    def forward(self, feed_dict, train):
        # TODO: if train: ~~
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_loss, x_loss = 0., 0.
        s_prev = self.sample_s0(x0)
        _T, _B = x.size(0), x.size(1)
        h = self.rnn_s.sample({"x": x}, return_all=False)["h"]  # TxBxh_dim
        for t in range(_T):
            x_t = x[t]  # Bx3x64x64
            a_t = a[t]  # Bx?
            h_t = h[t]  # Bx?
            feed_dict = {"h": h_t, "s_prev": s_prev, "a": a_t}
            s_loss += self.s_loss_cls.eval(feed_dict).mean()
            s_t = self.encoder_s.dist.rsample()
            # feed_dict = {"h": h_t, "s_prev": s_prev, "a": a_t, "x": x_t}
            # x_loss += self.x_loss_cls.eval(feed_dict).mean()
            feed_dict = {"s": s_t, "x": x_t}
            x_loss += - self.decoder_s.log_prob().eval(feed_dict).mean()
            s_prev = s_t
        loss = s_loss + x_loss
        return loss, {"loss": loss.item(), "ce": x_loss.item(), "kl": s_loss.item()}

    def train_(self, feed_dict):
        return _train(self, feed_dict)

    def test_(self, feed_dict):
        return _test(self, feed_dict)

    def sample_s0(self, x):
        return _sample_s0(self, x)

    def sample_video_from_latent_s(self, loader):
        return _sample_video_from_latent_s(self, loader)


def _train(model, feed_dict):
    model.train()
    model.optimizer.zero_grad()
    loss, omake_dict = model.forward(feed_dict, True)
    clip_grad_norm_(model.distributions.parameters(), 1000)  # TODO: args
    loss.backward()
    model.optimizer.step()
    return loss, omake_dict


def _test(model, feed_dict):
    model.eval()
    with torch.no_grad():
        loss, omake_dict = model.forward(feed_dict, False)
    return loss, omake_dict


def _sample_s0(model, x):  # context forwarding # TODO: train/test
    _T, _B = x.size(0), x.size(1)
    if not _T == 1:
        raise NotImplementedError
    s_prev = torch.zeros(_B, model.s_dim).to(model.device)
    a = torch.zeros(_B, model.a_dim).to(model.device)
    h = model.rnn_s.sample({"x": x}, return_all=False)["h"]
    h = torch.transpose(h, 0, 1)[:, 0]
    feed_dict = {"h": h, "s_prev": s_prev, "a": a}
    s = model.encoder_s.sample(feed_dict, return_all=False)["s"]
    return s


def _sample_video_from_latent_s(model, batch):
    device = model.device
    video = []
    x, a, _ = batch
    a = a.to(device).transpose(0, 1)
    x = x.to(device).transpose(0, 1)
    _T, _B = a.size(0), a.size(1)
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