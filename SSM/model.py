# optimizer: https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193

import torch
from torch import nn, optim
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, LogProb
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from core import Prior, Posterior, Encoder, Decoder
from torch.nn.utils import clip_grad_norm_
from utils import init_weights, flatten_dict, check_params
from copy import deepcopy
from logzero import logger


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.device = args.device
        self.debug = args.debug

    def forward(self):
        raise NotImplementedError

    def train_(self, feed_dict):
        self.train()
        self.optimizer.zero_grad()
        loss, info = self.forward(feed_dict, prior_sample=False)
        loss.backward()
        clip_grad_norm_(self.distributions.parameters(), 1000)
        self.optimizer.step()
        if self.debug:
            check_params(self)
        return loss, info

    def test_(self, feed_dict):
        self.eval()
        with torch.no_grad():
            loss, info = self.forward(feed_dict, prior_sample=True)
        return loss, info


class SSM(Base):
    def __init__(self, args):
        super(SSM, self).__init__(args)
        self.s_dims = args.s_dims  # list
        self.a_dim = args.a_dim
        self.h_dim = args.h_dim
        self.gamma = args.gamma
        self.min_stddev = args.min_stddev
        self.num_states = len(self.s_dims)
        self.static_hierarchy = args.static_hierarchy

        self.priors = []
        self.posteriors = []
        self.encoders = []
        self.decoders = []
        self.s_loss_clss = []
        self.x_loss_clss = []
        self.distributions = []  # for train, save_model
        self.all_distributions = []  # for load_model

        for i in range(self.num_states):
            s_dim = self.s_dims[i]
            a_dims = [self.a_dim] + self.s_dims[i-1:i]  # use s_dims[i-1]

            prior = Prior(i, s_dim, a_dims, self.min_stddev).to(self.device)
            posterior = Posterior(i, s_dim, self.h_dim, a_dims, self.min_stddev).to(self.device)
            encoder = Encoder(i).to(self.device)
            decoder = Decoder(i, s_dim).to(self.device)
            s_loss_cls = KullbackLeibler(posterior, prior)
            x_loss_cls = LogProb(decoder)

            self.priors.append(prior)
            self.posteriors.append(posterior)
            self.encoders.append(encoder)
            self.decoders.append(decoder)
            self.s_loss_clss.append(s_loss_cls)
            self.x_loss_clss.append(x_loss_cls)
            dists = [prior, posterior, encoder, decoder]
            self.all_distributions += dists
            if i not in self.static_hierarchy:  # 2とか
                self.distributions += dists  # trainable
            else:  # 1とか
                for dist in dists:
                    for param in dist.parameters():
                        param.requires_grad = False  # 早くなる?

        if self.debug:
            for dist in self.all_distributions:
                logger.debug(dist.name)
                for param in dist.parameters():
                    logger.debug(param.requires_grad)

        self.distributions = nn.ModuleList(self.distributions)
        init_weights(self.distributions)
        self.optimizer = optim.Adam(self.distributions.parameters())

    def forward(self, feed_dict, prior_sample=True, return_x=False):
        keys = set(locals().keys())
        loss = 0.
        x_loss = 0.  # final output. equal to x_losss[-1]
        s_losss = [0.] * self.num_states
        x_losss = [0.] * self.num_states
        if self.debug:
            s_aux_losss = [0.] * self.num_states
            s_abss = [0.] * self.num_states
            s_stds = [0.] * self.num_states
        keys = set(locals().keys()) - keys - {"keys"}

        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        _T, _B = x.size(0), x.size(1)
        _x = []

        s_prevs = self.sample_s0(x0)

        for t in range(_T):
            x_t, a_t = x[t], a[t]
            h_t, s_t = [], []

            for i in range(self.num_states):
                h_t.append(self.encoders[i].sample({"x": x_t}, return_all=False)["h"])
                feed_dict = {"s_prev": s_prevs[i], "h": h_t[-1], "a_list": [a_t] + s_t[-1:]}
                s_losss[i] += self.s_loss_clss[i].eval(feed_dict).mean()
                if self.debug:
                    prior01 = Normal(torch.tensor(0.), scale=torch.tensor(1.))
                    s_aux_losss[i] += kl_divergence(self.posteriors[i].dist, prior01).mean()
                if prior_sample:  # or i in self.static_hierarchy:  # test
                    s_t.append(self.priors[i].dist.mean)
                    if self.debug:
                        s_abss[i] += self.priors[i].dist.mean.abs().mean()
                        s_stds[i] += self.priors[i].dist.stddev.mean()
                else:  # train
                    s_t.append(self.posteriors[i].dist.rsample())  # rsample!
                    if self.debug:
                        s_abss[i] += self.posteriors[i].dist.mean.abs().mean()
                        s_stds[i] += self.posteriors[i].dist.stddev.mean()
                feed_dict = {"s": s_t[-1], "x": x_t}
                x_losss[i] += - self.x_loss_clss[i].eval(feed_dict).mean()

            _x.append(self.decoders[-1].dist.mean)
            s_prevs = s_t

        if return_x:
            return _x
        else:
            for i in range(self.num_states):
                if i not in self.static_hierarchy:
                    loss += s_losss[i] + x_losss[i]  # + self.gamma * s_aux_losss[i]
            x_loss = x_losss[-1]

            _locals = locals()
            info = flatten_dict({key:_locals[key] for key in keys})
            return loss, info

    def sample_s0(self, x0):
        device = self.device
        _B = x0.size(0)
        s_prevs = [torch.zeros(_B, s_dim).to(device) for s_dim in self.s_dims]
        h_t, s_t = [], []
        for i in range(self.num_states):
            h_t.append(self.encoders[i].sample_mean({"x": x0}))
            a_list = [torch.zeros(_B, d).to(device) for d in [self.a_dim] + self.s_dims[i-1:i]]
            feed_dict = {"s_prev": s_prevs[i], "h": h_t[-1], "a_list": a_list}
            s_t.append(self.posteriors[i].sample_mean(feed_dict))
        return s_t

    def sample_x(self, feed_dict):
        with torch.no_grad():
            _x = self.forward(feed_dict, False, sample=True)
        x = feed_dict["x"].transpose(0, 1)  # BxT
        _x = torch.stack(_x).transpose(0, 1)  # BxT
        _x = torch.clamp(_x, 0, 1)
        video = []
        for i in range(4):
            video.append(x[i*8:i*8+8])
            video.append(_x[i*8:i*8+8])
        video = torch.cat(video)  # B*2xT
        return video
