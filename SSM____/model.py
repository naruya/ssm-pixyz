# optimizer: https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193

import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, LogProb
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from core import Prior, Posterior, Encoder, Decoder
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


class SimpleSSM(Base):
    def __init__(self, args, device):
        super(SimpleSSM, self).__init__()
        self.device = device
        self.s_dim = s_dim = args.s_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim
        self.keys = ["loss", "x_loss[0]", "s_loss[0]", "x_loss", "s_abs[0]", "s_std[0]", "s_aux_loss[0]"]

        self.prior = Prior(s_dim, a_dim).to(device)
        self.posterior = Posterior(self.prior, h_dim, s_dim, a_dim).to(device)
        self.encoder = Encoder().to(device)
        self.decoder = Decoder(s_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior,
            self.posterior,
            self.encoder,
            self.decoder,
        ])

        init_weights(self.distributions)

        self.prior01 = Normal(torch.tensor(0.), scale=torch.tensor(1.))
        self.s_loss_cls = KullbackLeibler(self.posterior, self.prior)
        self.x_loss_cls = LogProb(self.decoder)
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, train, sample=False):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_loss, x_loss, s_aux_loss = 0., 0., 0.
        s_abs, s_std = 0., 0.
        s_prev = self.sample_s0(x0, train)
        _T, _B = x.size(0), x.size(1)
        _x = []

        for t in range(_T):
            x_t, a_t = x[t], a[t]

            h_t = self.encoder.sample({"x": x_t}, return_all=False)["h"]
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_loss += self.s_loss_cls.eval(feed_dict).mean()
            s_aux_loss += kl_divergence(self.posterior.dist, self.prior01).mean()
            if train:
                s_t = self.posterior.dist.rsample()
                s_abs += self.posterior.dist.mean.abs().mean()
                s_std += self.posterior.dist.stddev.mean()
            else:
                s_t = self.prior.dist.mean
                s_abs += self.prior.dist.mean.abs().mean()
                s_std += self.prior.dist.stddev.mean()
            feed_dict = {"s": s_t, "x": x_t}
            x_loss += - self.decoder.log_prob().eval(feed_dict).mean()
            _x.append(self.decoder.dist.mean)
            s_prev = s_t

        # loss = s_loss + x_loss + s_aux_loss
        loss = s_loss + x_loss

        if sample:
            return _x
        else:
            return loss, {"loss": loss.item(), "x_loss": x_loss.item(),
                          "x_loss[0]": x_loss.item(), "s_loss[0]": s_loss.item(),
                          "s_aux_loss[0]": s_aux_loss.item(),
                          "s_abs[0]": s_abs.item(), "s_std[0]": s_std.item()}

    def sample_s0(self, x0, train):
        device = self.device
        _B = x0.size(0)
        s_prev = torch.zeros(_B, self.s_dim).to(device)
        a_t = torch.zeros(_B, self.a_dim).to(device)
        if train:
            h_t = self.encoder.sample({"x": x0}, return_all=False)["h"]
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_t = self.posterior.sample(feed_dict, return_all=False)["s"]
        else:
            h_t = self.encoder.sample_mean({"x": x0})
            feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
            s_t = self.posterior.sample_mean(feed_dict)
        return s_t


class SSM(Base):
    def __init__(self, args, device, query, extra=None):
        assert not (extra and len(args.s_dim) > 2), NotImplementedError
        assert extra in [None, "ensemble", "residual"], NotImplementedError
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
        
        self.keys = ["loss", "x_loss"]
        for i in range(self.num_states):
            self.keys.append("s_loss[{}]".format(i))
            self.keys.append("x_loss[{}]".format(i))
        if extra == "ensemble":
            self.keys.append(extra + "_loss")

        for i in range(self.num_states):
            kwargs = {"s_dim": self.s_dims[i]}
            if "a" in query:
                kwargs.update({"a_dim": self.a_dim})
            if "s" in query:
                if i == 0:
                    kwargs.update({"a_dim": self.a_dim})
                else:
                    kwargs.update({"aa_dim": self.s_dims[i-1]})
            print(kwargs)
            self.priors.append(Prior(**kwargs).to(device))
            self.posteriors.append(Posterior(self.priors[-1], self.h_dim, **kwargs).to(device))
            self.encoders.append(Encoder().to(device))
            self.decoders.append(Decoder(self.s_dims[i]).to(device))
            self.s_loss_clss.append(KullbackLeibler(self.posteriors[-1], self.priors[-1]))
            self.x_loss_clss.append(LogProb(self.decoders[-1]))
            # if not extra == "residual":

        distributions = []
        distributions.extend(self.priors)
        distributions.extend(self.posteriors)
        distributions.extend(self.encoders)
        distributions.extend(self.decoders)

        if extra:
            raise NotImplementedError
            # if extra == "ensemble":
            #     self.ex_core = DecoderEnsemble(*self.s_dims).to(device)
            #     self.ex_loss_cls = LogProb(self.ex_core)
            # elif extra == "residual":
            #     self.ex_core = DecoderResidual(*self.s_dims).to(device)
            #     self.ex_loss_cls = LogProb(self.ex_core)
            # distributions.append(self.ex_core)

        self.distributions = nn.ModuleList(distributions)
        init_weights(self.distributions)
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, train, sample=False):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_losss = [0.] * len(self.s_dims)
        x_losss = [0.] * len(self.s_dims)
        ex_loss = 0.
        _T, _B = x.size(0), x.size(1)
        _x = []

        s_prevs = self.sample_s0(x0, train)

        for t in range(_T):
            x_t, a_t = x[t], a[t]
            h_t, s_t = [], []

            for i in range(self.num_states):
                h_t.append(self.encoders[i].sample({"x": x_t}, return_all=False)["h"])

                feed_dict = {"s_prev": s_prevs[i], "h": h_t[-1]}
                if "a" in self.query:
                    feed_dict.update({"a": a_t})
                if "s" in self.query:
                    if i == 0:
                        feed_dict.update({"a": a_t})
                    else:
                        feed_dict.update({"aa": s_t[i-1]})
                s_losss[i] += self.s_loss_clss[i].eval(feed_dict).mean()

                if train:
                    s_t.append(self.posteriors[i].dist.rsample())
                else:
                    s_t.append(self.priors[i].dist.mean)

            for i in range(self.num_states):
                feed_dict = {"s": s_t[i], "x": x_t}
                x_losss[i] += - self.x_loss_clss[i].eval(feed_dict).mean()
                # if not self.extra == "residual":
                # else:
                    # feed_dict = {"s": s_t[0], "ss": s_t[1], "x": x_t}
                    # self.ex_core.mode = i
                    # x_losss[i] += - self.ex_loss_cls.eval(feed_dict).mean()

            if self.extra == None:
                _x.append(self.decoders[-1].dist.mean)
            else:
                raise NotImplementedError
            # elif self.extra == "ensemble":
            #     feed_dict = {"s": s_t[0], "ss": s_t[1], "x": x_t}
            #     ex_loss += - self.ex_loss_cls.eval(feed_dict).mean()
            #     _x.append(self.ex_core.dist.mean)
            # elif self.extra == "residual":
            #     _x.append(self.ex_core.dist.mean)

            s_prevs = s_t

        loss = 0.
        for i in range(self.num_states):
            loss += s_losss[i] + x_losss[i]
        loss += ex_loss

        if sample:
            return _x
        else:
            return_dict = {"loss": loss.item()}
            for i in range(self.num_states):
                return_dict.update({"s_loss[{}]".format(i): s_losss[i].item()})
                return_dict.update({"x_loss[{}]".format(i): x_losss[i].item()})
            # if self.extra == "ensemble":
            #     return_dict.update({self.extra + "_loss": ex_loss.item()})
            #     return_dict.update({"x_loss": ex_loss.item()})
            # elif self.extra == "residual":
            #     return_dict.update({"x_loss": x_losss[-1].item()})
            else:
                return_dict.update({"x_loss": x_losss[-1].item()})
            return loss, return_dict


# --------------------------------
# default
# --------------------------------

def _sample_s0(model, x0, train):
    device = model.device
    _B = x0.size(0)
    s_prevs = [torch.zeros(_B, s_dim).to(device) for s_dim in model.s_dims]

    a_t, aa_t = [], []
    if "a" in model.query:
        for i in range(model.num_states):
            a_t.append(torch.zeros(_B, model.a_dim).to(device))
    if "s" in model.query:
        aa_t.append(None)
        for i in range(1, model.num_states):
            aa_t.append(torch.zeros(_B, model.s_dims[i-1]).to(device))

    h_t, s_t = [], []
    for i in range(model.num_states):
        if train:
            h_t.append(model.encoders[i].sample({"x": x0}, return_all=False)["h"])
        else:
            h_t.append(model.encoders[i].sample_mean({"x": x0}))

        feed_dict = {"s_prev": s_prevs[i], "h": h_t[-1]}
        if "a" in model.query:
            feed_dict.update({"a": a_t[i]})
        if "s" in model.query:
            if i == 0 and not "a" in model.query:
                feed_dict.update({"aa": a_t[i]})
            else:
                feed_dict.update({"aa": aa_t[i]})

        if train:
            s_t.append(model.posteriors[i].sample(feed_dict, return_all=False)["s"])
        else:
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