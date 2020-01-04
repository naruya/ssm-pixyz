# optimizer: https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193

import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, CrossEntropy
from core import Prior, Posterior, Encoder, Decoder, DecoderEnsemble
from torch.nn.utils import clip_grad_norm_
from torch_utils import init_weights


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def forward(self, feed_dict, train):
        raise NotImplementedError

    def sample_s0(self, x):
        raise NotImplementedError

    def sample_video(self, batch):
        raise NotImplementedError

    def train_(self, feed_dict):
        return _train(self, feed_dict)

    def test_(self, feed_dict):
        return _test(self, feed_dict)


class SSM5(Base):
    def __init__(self, args, device):
        super(SSM5, self).__init__()
        self.device = device
        self.s_dim = s_dim = args.s_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim

        self.prior_s = Prior(s_dim, a_dim).to(device)
        self.posterior_s = Posterior(s_dim, a_dim, h_dim).to(device)
        self.encoder_s = Encoder().to(device)
        self.decoder_s = Decoder(s_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior_s,
            self.posterior_s,
            self.encoder_s,
            self.decoder_s,
        ])

        init_weights(self.distributions)

        self.s_loss_cls = KullbackLeibler(self.posterior_s, self.prior_s)  # TODO: 自分で書いてみる
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, train):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_loss, x_loss = 0., 0.
        s_prev = self.sample_s0(x0)
        _T, _B = x.size(0), x.size(1)
        for t in range(_T):
            x_t = x[t]
            a_t = a[t]

            h_t = self.encoder_s.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_loss += self.s_loss_cls.eval(feed_dict).mean()

            s_t = self.posterior_s.dist.rsample()
            feed_dict = {"s": s_t, "x": x_t}
            x_loss += - self.decoder_s.log_prob().eval(feed_dict).mean()

            s_prev = s_t

        loss = s_loss + x_loss
        return loss, {"loss": loss.item(), "ce": x_loss.item(), "kl": s_loss.item()}

    def sample_s0(self, x0):
        return _sample_s0(self.encoder_s, self.posterior_s,
                          self.s_dim, self.a_dim, self.device, x0)

    def sample_video(self, batch):
        return _sample_video(self.sample_s0, self.prior_s, self.decoder_s,
                             self.device, self.__class__.__name__, batch)


# EncoderRNN
# class SSM6(Base):


# ensemble, 1 decoder
class SSM7(Base):
    def __init__(self, args, device):
        super(SSM7, self).__init__()
        self.device = device
        self.s_dim = s_dim = args.s_dim
        self.ss_dim = ss_dim = args.ss_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim

        self.prior_s = Prior(s_dim, a_dim).to(device)
        self.posterior_s = Posterior(s_dim, a_dim, h_dim).to(device)
        self.encoder_s = Encoder().to(device)
        # self.decoder_s = Decoder(s_dim).to(device)
        self.prior_ss = Prior(ss_dim, a_dim).to(device)
        self.posterior_ss = Posterior(ss_dim, a_dim, h_dim).to(device)
        self.encoder_ss = Encoder().to(device)
        # self.decoder_ss = Decoder(ss_dim).to(device)
        self.decoder_s_ss = DecoderEnsemble(s_dim, ss_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior_s,
            self.posterior_s,
            self.encoder_s,
            # self.decoder_s,
            self.prior_ss,
            self.posterior_ss,
            self.encoder_ss,
            # self.decoder_ss,
            self.decoder_s_ss,
        ])

        init_weights(self.distributions)

        self.s_loss_cls = KullbackLeibler(self.posterior_s, self.prior_s)
        self.ss_loss_cls = KullbackLeibler(self.posterior_ss, self.prior_ss)
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, train):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        # s_loss, x_loss, ss_loss, xx_loss, x_xx_loss = 0., 0., 0., 0., 0.
        s_loss, ss_loss, x_xx_loss = 0., 0., 0.
        s_prev = self.sample_s0(x0)
        ss_prev = self.sample_ss0(x0)
        _T, _B = x.size(0), x.size(1)

        for t in range(_T):
            x_t = x[t]
            a_t = a[t]

            h_t = self.encoder_s.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_loss += self.s_loss_cls.eval(feed_dict).mean()

            hh_t = self.encoder_ss.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": ss_prev, "a": a_t, "h": hh_t}
            ss_loss += self.ss_loss_cls.eval(feed_dict).mean()

            s_t = self.posterior_s.dist.rsample()
            feed_dict = {"s": s_t, "x": x_t}
            # x_loss += - self.decoder_s.log_prob().eval(feed_dict).mean()

            ss_t = self.posterior_ss.dist.rsample()
            feed_dict = {"s": ss_t, "x": x_t}
            # xx_loss += - self.decoder_ss.log_prob().eval(feed_dict).mean()

            feed_dict = {"s": s_t, "ss": ss_t, "x": x_t}
            x_xx_loss += - self.decoder_s_ss.log_prob().eval(feed_dict).mean()

            s_prev = s_t
            ss_prev = ss_t

        # loss = s_loss + x_loss + ss_loss + xx_loss + x_xx_loss
        loss = s_loss + ss_loss + x_xx_loss
        # return loss, {"loss": loss.item(),
        #               "s_loss": s_loss.item(), "x_loss": x_loss.item(),
        #               "ss_loss": ss_loss.item(), "xx_loss": xx_loss.item(),
        #               "x_xx_loss": x_xx_loss.item()}
        return loss, {"loss": loss.item(),
                      "s_loss": s_loss.item(),
                      "ss_loss": ss_loss.item(),
                      "x_xx_loss": x_xx_loss.item()}

    def sample_s0(self, x0):
        return _sample_s0(self.encoder_s, self.posterior_s,
                          self.s_dim, self.a_dim, self.device, x0)

    def sample_ss0(self, x0):
        return _sample_s0(self.encoder_ss, self.posterior_ss,
                          self.ss_dim, self.a_dim, self.device, x0)

    def sample_video(self, batch):
        return _sample_video([self.sample_s0, self.sample_ss0],
                             [self.prior_s, self.prior_ss],
                             self.decoder_s_ss, self.device,
                             self.__class__.__name__, batch)

# ensemble, 3 decoder
class SSM8(Base):
    def __init__(self, args, device):
        super(SSM8, self).__init__()
        self.device = device
        self.s_dim = s_dim = args.s_dim
        self.ss_dim = ss_dim = args.ss_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim

        self.prior_s = Prior(s_dim, a_dim).to(device)
        self.posterior_s = Posterior(s_dim, a_dim, h_dim).to(device)
        self.encoder_s = Encoder().to(device)
        self.decoder_s = Decoder(s_dim).to(device)
        self.prior_ss = Prior(ss_dim, a_dim).to(device)
        self.posterior_ss = Posterior(ss_dim, a_dim, h_dim).to(device)
        self.encoder_ss = Encoder().to(device)
        self.decoder_ss = Decoder(ss_dim).to(device)
        self.decoder_s_ss = DecoderEnsemble(s_dim, ss_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior_s,
            self.posterior_s,
            self.encoder_s,
            self.decoder_s,
            self.prior_ss,
            self.posterior_ss,
            self.encoder_ss,
            self.decoder_ss,
            self.decoder_s_ss,
        ])

        init_weights(self.distributions)

        self.s_loss_cls = KullbackLeibler(self.posterior_s, self.prior_s)
        self.ss_loss_cls = KullbackLeibler(self.posterior_ss, self.prior_ss)
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, train):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_loss, x_loss, ss_loss, xx_loss, x_xx_loss = 0., 0., 0., 0., 0.
        s_prev = self.sample_s0(x0)
        ss_prev = self.sample_ss0(x0)
        _T, _B = x.size(0), x.size(1)

        for t in range(_T):
            x_t = x[t]
            a_t = a[t]

            h_t = self.encoder_s.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_loss += self.s_loss_cls.eval(feed_dict).mean()

            hh_t = self.encoder_ss.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": ss_prev, "a": a_t, "h": hh_t}
            ss_loss += self.ss_loss_cls.eval(feed_dict).mean()

            s_t = self.posterior_s.dist.rsample()
            feed_dict = {"s": s_t, "x": x_t}
            x_loss += - self.decoder_s.log_prob().eval(feed_dict).mean()

            ss_t = self.posterior_ss.dist.rsample()
            feed_dict = {"s": ss_t, "x": x_t}
            xx_loss += - self.decoder_ss.log_prob().eval(feed_dict).mean()

            feed_dict = {"s": s_t, "ss": ss_t, "x": x_t}
            x_xx_loss += - self.decoder_s_ss.log_prob().eval(feed_dict).mean()

            s_prev = s_t
            ss_prev = ss_t

        loss = s_loss + x_loss + ss_loss + xx_loss + x_xx_loss
        # loss = s_loss + ss_loss + x_xx_loss
        return loss, {"loss": loss.item(),
                      "s_loss": s_loss.item(), "x_loss": x_loss.item(),
                      "ss_loss": ss_loss.item(), "xx_loss": xx_loss.item(),
                      "x_xx_loss": x_xx_loss.item()}
        # return loss, {"loss": loss.item(),
        #               "s_loss": s_loss.item(),
        #               "ss_loss": ss_loss.item(),
        #               "x_xx_loss": x_xx_loss.item()}

    def sample_s0(self, x0):
        return _sample_s0(self.encoder_s, self.posterior_s,
                          self.s_dim, self.a_dim, self.device, x0)

    def sample_ss0(self, x0):
        return _sample_s0(self.encoder_ss, self.posterior_ss,
                          self.ss_dim, self.a_dim, self.device, x0)

    def sample_video(self, batch):
        return _sample_video([self.sample_s0, self.sample_ss0],
                             [self.prior_s, self.prior_ss],
                             self.decoder_s_ss, self.device,
                             self.__class__.__name__, batch)


# ensemble, 2 decoder (s_loss, x_loss, x_xx_loss)
class SSM9(Base):
    def __init__(self, args, device):
        super(SSM9, self).__init__()
        self.device = device
        self.s_dim = s_dim = args.s_dim
        self.ss_dim = ss_dim = args.ss_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim

        self.prior_s = Prior(s_dim, a_dim).to(device)
        self.posterior_s = Posterior(s_dim, a_dim, h_dim).to(device)
        self.encoder_s = Encoder().to(device)
        self.decoder_s = Decoder(s_dim).to(device)
        self.prior_ss = Prior(ss_dim, a_dim).to(device)
        self.posterior_ss = Posterior(ss_dim, a_dim, h_dim).to(device)
        self.encoder_ss = Encoder().to(device)
        self.decoder_ss = Decoder(ss_dim).to(device)
        self.decoder_s_ss = DecoderEnsemble(s_dim, ss_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior_s,
            self.posterior_s,
            self.encoder_s,
            self.decoder_s,
            self.prior_ss,
            self.posterior_ss,
            self.encoder_ss,
            self.decoder_ss,
            self.decoder_s_ss,
        ])

        init_weights(self.distributions)

        self.s_loss_cls = KullbackLeibler(self.posterior_s, self.prior_s)
        self.ss_loss_cls = KullbackLeibler(self.posterior_ss, self.prior_ss)
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, train):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_loss, x_loss, ss_loss, xx_loss, x_xx_loss = 0., 0., 0., 0., 0.
        s_prev = self.sample_s0(x0)
        ss_prev = self.sample_ss0(x0)
        _T, _B = x.size(0), x.size(1)

        for t in range(_T):
            x_t = x[t]
            a_t = a[t]

            h_t = self.encoder_s.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_loss += self.s_loss_cls.eval(feed_dict).mean()

            hh_t = self.encoder_ss.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": ss_prev, "a": a_t, "h": hh_t}
            ss_loss += self.ss_loss_cls.eval(feed_dict).mean()

            s_t = self.posterior_s.dist.rsample()
            feed_dict = {"s": s_t, "x": x_t}
            x_loss += - self.decoder_s.log_prob().eval(feed_dict).mean()

            ss_t = self.posterior_ss.dist.rsample()
            feed_dict = {"s": ss_t, "x": x_t}
            xx_loss += - self.decoder_ss.log_prob().eval(feed_dict).mean()

            feed_dict = {"s": s_t, "ss": ss_t, "x": x_t}
            x_xx_loss += - self.decoder_s_ss.log_prob().eval(feed_dict).mean()

            s_prev = s_t
            ss_prev = ss_t

        loss = s_loss + x_loss + x_xx_loss  # x_xx_loss入れるだけでNanに飛ぶ???
        # loss = s_loss + x_loss + ss_loss + xx_loss + x_xx_loss
        # loss = s_loss + ss_loss + x_xx_loss
        return loss, {"loss": loss.item(),
                      "s_loss": s_loss.item(), "x_loss": x_loss.item(),
                      "ss_loss": ss_loss.item(), "xx_loss": xx_loss.item(),
                      "x_xx_loss": x_xx_loss.item()}
        # return loss, {"loss": loss.item(),
        #               "s_loss": s_loss.item(),
        #               "ss_loss": ss_loss.item(),
        #               "x_xx_loss": x_xx_loss.item()}

    def sample_s0(self, x0):
        return _sample_s0(self.encoder_s, self.posterior_s,
                          self.s_dim, self.a_dim, self.device, x0)

    def sample_ss0(self, x0):
        return _sample_s0(self.encoder_ss, self.posterior_ss,
                          self.ss_dim, self.a_dim, self.device, x0)

    def sample_video(self, batch):
        return _sample_video([self.sample_s0, self.sample_ss0],
                             [self.prior_s, self.prior_ss],
                             self.decoder_s_ss, self.device,
                             self.__class__.__name__, batch)



# ensemble, 3 decoder (s_loss, x_loss, xx_loss, x_xx_loss)
class SSM10(Base):
    def __init__(self, args, device):
        super(SSM10, self).__init__()
        self.device = device
        self.s_dim = s_dim = args.s_dim
        self.ss_dim = ss_dim = args.ss_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim

        self.prior_s = Prior(s_dim, a_dim).to(device)
        self.posterior_s = Posterior(s_dim, a_dim, h_dim).to(device)
        self.encoder_s = Encoder().to(device)
        self.decoder_s = Decoder(s_dim).to(device)
        self.prior_ss = Prior(ss_dim, a_dim).to(device)
        self.posterior_ss = Posterior(ss_dim, a_dim, h_dim).to(device)
        self.encoder_ss = Encoder().to(device)
        self.decoder_ss = Decoder(ss_dim).to(device)
        self.decoder_s_ss = DecoderEnsemble(s_dim, ss_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior_s,
            self.posterior_s,
            self.encoder_s,
            self.decoder_s,
            self.prior_ss,
            self.posterior_ss,
            self.encoder_ss,
            self.decoder_ss,
            self.decoder_s_ss,
        ])

        init_weights(self.distributions)

        self.s_loss_cls = KullbackLeibler(self.posterior_s, self.prior_s)
        self.ss_loss_cls = KullbackLeibler(self.posterior_ss, self.prior_ss)
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, train):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_loss, x_loss, ss_loss, xx_loss, x_xx_loss = 0., 0., 0., 0., 0.
        s_prev = self.sample_s0(x0)
        ss_prev = self.sample_ss0(x0)
        _T, _B = x.size(0), x.size(1)

        for t in range(_T):
            x_t = x[t]
            a_t = a[t]

            h_t = self.encoder_s.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_loss += self.s_loss_cls.eval(feed_dict).mean()

            hh_t = self.encoder_ss.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": ss_prev, "a": a_t, "h": hh_t}
            ss_loss += self.ss_loss_cls.eval(feed_dict).mean()

            s_t = self.posterior_s.dist.rsample()
            feed_dict = {"s": s_t, "x": x_t}
            x_loss += - self.decoder_s.log_prob().eval(feed_dict).mean()

            ss_t = self.posterior_ss.dist.rsample()
            feed_dict = {"s": ss_t, "x": x_t}
            xx_loss += - self.decoder_ss.log_prob().eval(feed_dict).mean()

            feed_dict = {"s": s_t, "ss": ss_t, "x": x_t}
            x_xx_loss += - self.decoder_s_ss.log_prob().eval(feed_dict).mean()

            s_prev = s_t
            ss_prev = ss_t

        loss = s_loss + x_loss + xx_loss + x_xx_loss
        # loss = s_loss + x_loss + x_xx_loss
        # loss = s_loss + x_loss + ss_loss + xx_loss + x_xx_loss
        # loss = s_loss + ss_loss + x_xx_loss
        return loss, {"loss": loss.item(),
                      "s_loss": s_loss.item(), "x_loss": x_loss.item(),
                      "ss_loss": ss_loss.item(), "xx_loss": xx_loss.item(),
                      "x_xx_loss": x_xx_loss.item()}
        # return loss, {"loss": loss.item(),
        #               "s_loss": s_loss.item(),
        #               "ss_loss": ss_loss.item(),
        #               "x_xx_loss": x_xx_loss.item()}

    def sample_s0(self, x0):
        return _sample_s0(self.encoder_s, self.posterior_s,
                          self.s_dim, self.a_dim, self.device, x0)

    def sample_ss0(self, x0):
        return _sample_s0(self.encoder_ss, self.posterior_ss,
                          self.ss_dim, self.a_dim, self.device, x0)

    def sample_video(self, batch):
        return _sample_video([self.sample_s0, self.sample_ss0],
                             [self.prior_s, self.prior_ss],
                             self.decoder_s_ss, self.device,
                             self.__class__.__name__, batch)


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


def _sample_s0(encoder, posterior, s_dim, a_dim, device, x0):
    _B = x0.size(0)
    s_prev = torch.zeros(_B, s_dim).to(device)
    a = torch.zeros(_B, a_dim).to(device)
    h = encoder.sample({"x": x0}, return_all=False)["h"]
    s = posterior.sample({"s_prev": s_prev, "a": a, "h": h}, return_all=False)["s"]
    return s


def _sample_video(sample_s0, prior, decoder, device, name, batch):
    # with torch.no_grad():
    x, a, _ = batch
    x = x.to(device).transpose(0, 1)  # TxB
    a = a.to(device).transpose(0, 1)  # TxB
    _T, _B = a.size(0), a.size(1)
    x0 = x[0].clone()
    if name in ["SSM7", "SSM8", "SSM9", "SSM10"]:  # ensemble
        s_prev = sample_s0[0](x0)
        ss_prev = sample_s0[1](x0)
    else:
        s_prev = sample_s0(x0)
    _x = []
    for t in range(_T):
        if name in ["SSM7", "SSM8", "SSM9", "SSM10"]:
            s = prior[0].sample_mean({"s_prev": s_prev, "a": a[t]})
            ss = prior[1].sample_mean({"s_prev": ss_prev, "a": a[t]})
            frame = decoder.sample_mean({"s": s, "ss": ss})
            s_prev = s
            ss_prev = ss
        else:
            s = prior.sample_mean({"s_prev": s_prev, "a": a[t]})
            frame = decoder.sample_mean({"s": s})
            s_prev = s
        _x.append(frame)
    _x = torch.stack(_x).transpose(0, 1)  # BxT
    _x = torch.clamp(_x, 0, 1)
    x = x.transpose(0, 1)  # BxT
    video = []
    for i in range(4):
        video.append(x[i*8:i*8+8])
        video.append(_x[i*8:i*8+8])
    video = torch.cat(video)  # B*2xT
    return video