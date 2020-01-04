# optimizer: https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193

import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, LogProb
from core import Prior, Posterior, Encoder, Decoder, DecoderEnsemble
from torch.nn.utils import clip_grad_norm_
from torch_utils import init_weights


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def forward(self, feed_dict, train):
        raise NotImplementedError

    def sample_s0(self, x0, train):
        return _sample_s0(self, x0, train)

    def sample_x(self, feed_dict):
        return _sample_x(self, feed_dict)

    def train_(self, feed_dict):
        return _train(self, feed_dict)

    def test_(self, feed_dict):
        return _test(self, feed_dict)


# class SimpleSSM(Base):
#     def __init__(self, args, device):
#         super(SimpleSSM, self).__init__()
#         self.device = device
#         self.s_dim = s_dim = args.s_dim
#         self.a_dim = a_dim = args.a_dim
#         self.h_dim = h_dim = args.h_dim

#         self.prior_s = Prior(s_dim, a_dim).to(device)
#         self.posterior_s = Posterior(s_dim, a_dim, h_dim).to(device)
#         self.encoder_s = Encoder().to(device)
#         self.decoder_s = Decoder(s_dim).to(device)

#         self.distributions = nn.ModuleList([
#             self.prior_s,
#             self.posterior_s,
#             self.encoder_s,
#             self.decoder_s,
#         ])

#         init_weights(self.distributions)

#         self.s_loss_cls = KullbackLeibler(self.posterior_s, self.prior_s)  # TODO: 自分で書いてみる
#         self.optimizer = optim.Adam(self.distributions.parameters())

#     def forward(self, feed_dict, train):
#         x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
#         s_loss, x_loss = 0., 0.
#         s_prev = self.sample_s0(x0)
#         _T, _B = x.size(0), x.size(1)
#         for t in range(_T):
#             x_t = x[t]
#             a_t = a[t]

#             h_t = self.encoder_s.sample({"x": x_t}, return_all=False)["h"]
#             feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
#             s_loss += self.s_loss_cls.eval(feed_dict).mean()

#             s_t = self.posterior_s.dist.rsample()
#             feed_dict = {"s": s_t, "x": x_t}
#             x_loss += - self.decoder_s.log_prob().eval(feed_dict).mean()

#             s_prev = s_t

#         loss = s_loss + x_loss
#         return loss, {"loss": loss.item(), "ce": x_loss.item(), "kl": s_loss.item()}


class SSM(Base):
    def __init__(self, args, device, query, extra=None):
        assert not (extra and len(args.s_dim) > 2), NotImplementedError
        assert query == "action", NotImplementedError
        super(SSM, self).__init__()

        self.device = device
        self.s_dims = args.s_dim  # list
        self.a_dim = args.a_dim
        self.h_dim = args.h_dim
        self.num_states = len(self.s_dims)
        self.extra = extra        
        self.query = query

        self.priors = []
        self.posteriors = []
        self.encoders = []
        self.decoders = []
        self.s_loss_clss = []
        self.x_loss_clss = []

        for i, s_dim in enumerate(self.s_dims):
            if query == "action":
                a_dim = self.a_dim
            elif query == "state":
                if i == 0:
                    a_dim = self.a_dim
                else:
                    a_dim = self.s_dims[i-1]
            self.priors.append(Prior(s_dim, a_dim).to(device))
            self.posteriors.append(Posterior(s_dim, a_dim, self.h_dim).to(device))
            self.encoders.append(Encoder().to(device))
            self.decoders.append(Decoder(s_dim).to(device))
            self.s_loss_clss.append(KullbackLeibler(self.posteriors[-1], self.priors[-1]))
            self.x_loss_clss.append(LogProb(self.decoders[-1]))

        distributions = []
        distributions.extend(self.priors)
        distributions.extend(self.posteriors)
        distributions.extend(self.encoders)
        distributions.extend(self.decoders)

        if extra:
            if extra == "ensemble":
                self.ex_core = DecoderEnsemble(*self.s_dims).to(device)
                self.ex_loss_cls = LogProb(self.ex_core)
            elif extra == "residual":
                self.ex_core = DecoderResidual(*self.s_dims).to(device)
                self.ex_loss_cls = LogProb(self.ex_core)
            distributions.append(self.ex_core)

        self.distributions = nn.ModuleList(distributions)
        init_weights(self.distributions)
        self.optimizer = optim.Adam(self.distributions.parameters())
        
        self.keys = ["loss"]
        for i in range(self.num_states):
            self.keys.append("s_loss[{}]".format(i))
            self.keys.append("x_loss[{}]".format(i))
        if self.extra:
            self.keys.append(self.extra + "_loss")

    def forward(self, feed_dict, train, sample=False):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_losss = [0.] * len(self.s_dims)
        x_losss = [0.] * len(self.s_dims)
        ex_loss = 0.
        _T, _B = x.size(0), x.size(1)
        _x = []

        s_prevs = self.sample_s0(x0, train)

        for t in range(_T):
            x_t = x[t]
            a_t = a[t]
            h_t = []
            s_t = []

            for i in range(self.num_states):
                h_t.append(self.encoders[i].sample({"x": x_t}, return_all=False)["h"])

                if self.query == "action":
                    feed_dict = {"s_prev": s_prevs[i], "a": a_t, "h": h_t[-1]}
                elif self.query == "state":
                    if i == 0:
                        feed_dict = {"s_prev": s_prevs[i], "a": a_t, "h": h_t[-1]}
                    else:
                        feed_dict = {"s_prev": s_prevs[i], "a": s_t[i-1], "h": h_t[-1]}
                else:
                    assert False, [query, i]
                s_losss[i] += self.s_loss_clss[i].eval(feed_dict).mean()

                if train:
                    s_t.append(self.posteriors[i].dist.rsample())
                else:
                    s_t.append(self.priors[i].dist.mean)

                feed_dict = {"s": s_t[-1], "x": x_t}
                x_losss[i] += - self.x_loss_clss[i].eval(feed_dict).mean()

            if self.extra == None:
                _x.append(self.decoders[-1].dist.mean)
            elif self.extra == "ensemble":
                feed_dict = {"s": s_t[0], "ss": s_t[1], "x": x_t}
                ex_loss += - self.ex_loss_cls.eval(feed_dict).mean()
                _x.append(self.ex_core.dist.mean)
            elif self.extra == "residual":
                raise NotImplementedError

            s_prevs = s_t

        loss = ex_loss
        for i in range(self.num_states):
            loss += s_losss[i] + x_losss[i]

        if sample:
            return _x
        else:
            return_dict = {"loss": loss.item()}
            for i in range(self.num_states):
                return_dict.update({"s_loss[{}]".format(i): s_losss[i].item()})
                return_dict.update({"x_loss[{}]".format(i): x_losss[i].item()})
            if self.extra:
                return_dict.update({self.extra + "_loss": ex_loss.item()})
            return loss, return_dict


def _sample_s0(model, x0, train):
    device = model.device
    _B = x0.size(0)
    s_prevs = [torch.zeros(_B, s_dim).to(device) for s_dim in model.s_dims]
    a_t = torch.zeros(_B, model.a_dim).to(device)
    h_t = []
    s_t = []
    for i in range(model.num_states):
        if train:
            h_t.append(model.encoders[i].sample({"x": x0}, return_all=False)["h"])
            feed_dict = {"s_prev": s_prevs[i], "a": a_t, "h": h_t[-1]}
            s_t.append(model.posteriors[i].sample(feed_dict, return_all=False)["s"])
        else:
            h_t.append(model.encoders[i].sample_mean({"x": x0}))
            feed_dict = {"s_prev": s_prevs[i], "a": a_t, "h": h_t[-1]}
            s_t.append(model.posteriors[i].sample_mean(feed_dict))
    return s_t


def _sample_x(model, feed_dict):
    with torch.no_grad():
        _x = model.forward(feed_dict, train=False, sample=True)
    x = feed_dict["x"].transpose(0, 1)  # BxT
    _x = torch.stack(_x).transpose(0, 1)  # BxT
    _x = torch.clamp(_x, 0, 1)
    video = []
    for i in range(4):
        video.append(x[i*8:i*8+8])
        video.append(_x[i*8:i*8+8])
    video = torch.cat(video)  # B*2xT
    return video


def _train(model, feed_dict):
    model.train()
    model.optimizer.zero_grad()
    loss, omake_dict = model.forward(feed_dict, True)
    clip_grad_norm_(model.distributions.parameters(), 1000)
    loss.backward()
    model.optimizer.step()
    return loss, omake_dict


def _test(model, feed_dict):
    model.eval()
    with torch.no_grad():
        loss, omake_dict = model.forward(feed_dict, False)
    return loss, omake_dict