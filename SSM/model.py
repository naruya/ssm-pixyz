# optimizer: https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from core import Prior, Posterior, Encoder, Decoder, Posterior_s_0
from core_res_encdec import ResEncoder, ResDecoder
from core_res_transition import ResPrior, ResPosterior
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

    def forward_(self, x_0, x, a, train):
        if train:
            loss, info = self.train_(x_0, x, a)
        else:
            loss, info = self.test_(x_0, x, a)
        return loss, info

    def train_(self, x_0, x, a):
        self.train()
        self.optimizer.zero_grad()
        loss, info = self(x_0, x, a, prior_sample=False)
        loss.backward()
        # clip_grad_norm_(self.distributions.parameters(), 1000)
        self.optimizer.step()
        if self.debug:
            check_params(self)
        return loss, info

    def test_(self, x_0, x, a):
        self.eval()
        with torch.no_grad():
            loss, info = self(x_0, x, a, prior_sample=True)
        return loss, info


class SSM(Base):
    def __init__(self, args):
        super(SSM, self).__init__(args)
        self.s_dims = args.s_dims  # list
        self.a_dim = args.a_dim
        self.h_dim = args.h_dim
        self.num_states = len(self.s_dims)
        self.args = args

        self.priors = []
        self.posteriors = []
        self.encoders = []
        self.decoders = []
        if args.posterior_s_0:
            self.posterior_s_0s = []
        self.distributions = []  # for train, save_model
        self.all_distributions = []  # for load_model

        for i in range(self.num_states):
            s_dim = self.s_dims[i]
            a_dims = [self.a_dim] + self.s_dims[i-1:i]  # use s_dims[i-1]

            posterior = Posterior(i, s_dim, self.h_dim, a_dims,
                                  args.min_stddev).to(self.device)

            if args.res_transition:
                prior = ResPrior(i, s_dim, a_dims,
                              args.min_stddev).to(self.device)
            else:
                prior = Prior(i, s_dim, a_dims,
                              args.min_stddev).to(self.device)
            if args.res_encdec:
                encoder = ResEncoder(i).to(self.device)
                decoder = ResDecoder(i, s_dim, self.device).to(self.device)
            else:
                encoder = Encoder(i).to(self.device)
                decoder = Decoder(i, s_dim, self.device).to(self.device)

            if len(args.device_ids) > 0:
                prior = nn.DataParallel(prior,
                                        device_ids=args.device_ids)
                posterior = nn.DataParallel(posterior,
                                            device_ids=args.device_ids)
                encoder = nn.DataParallel(encoder,
                                          device_ids=args.device_ids)
                decoder = nn.DataParallel(decoder,
                                          device_ids=args.device_ids)

            self.priors.append(prior)
            self.posteriors.append(posterior)
            self.encoders.append(encoder)
            self.decoders.append(decoder)
            dists = [prior, posterior, encoder, decoder]

            if args.posterior_s_0:
                posterior_s_0 = Posterior_s_0(i, s_dim, self.h_dim,
                                  args.min_stddev).to(self.device)
                posterior_s_0 = nn.DataParallel(posterior_s_0,
                                                device_ids=args.device_ids)
                self.posterior_s_0s.append(posterior_s_0)
                dists.append(posterior_s_0)

            self.all_distributions += dists
            if i not in self.args.static_hierarchy:  # 2とか
                self.distributions += dists  # trainable
            else:  # 1とか
                for dist in dists:
                    for param in dist.parameters():
                        param.requires_grad = False  # 早くなる?結果変わらない?

        if self.debug:
            for dist in self.all_distributions:
                logger.debug(dist.module.name)
                for param in dist.parameters():
                    logger.debug(param.requires_grad)

        self.distributions = nn.ModuleList(self.distributions)
        init_weights(self.distributions)
        self.optimizer = optim.Adam(self.distributions.parameters(), lr=args.lr)

    # def forward(self, x_0, a, x=None, prior_sample=True, return_x=False):
    def forward(self, x_0, x, a, prior_sample=True, return_x=False):
        keys = set(locals().keys())

        loss = 0.  # for backprop
        s_losss = [0.] * self.num_states
        x_losss_p = [0.] * self.num_states
        x_losss_q = [0.] * self.num_states
        s01_losss = [0.] * self.num_states

        if self.args.posterior_s_0:
            s_0_s_loss = None
            s_0_x_loss = None

        if self.debug:
            s_abss_p = [0.] * self.num_states
            s_stds_p = [0.] * self.num_states
            s_abss_q = [0.] * self.num_states
            s_stds_q = [0.] * self.num_states

        keys = set(locals().keys()) - keys - {"keys"}

        x = x.transpose(0, 1).to(self.device)  # T,B,3,28,28
        a = a.transpose(0, 1).to(self.device)  # T,B,4
        x_0 = x_0.to(self.device)
        _T, _B = x.size(0), x.size(1)
        _x_p = []
        _x_q = []

        if self.args.posterior_s_0:
            s_prevs, s_0_s_loss, s_0_x_loss = self.sample_s_0(x_0)
        else:
            s_prevs = self.sample_s_0(x_0)

        for t in range(_T):
            x_t, a_t = x[t], a[t]
            s_ts = []  # store all hierarchy's state

            for i in range(self.num_states):
                h_t = self.encoders[i](x_t)
                s_prev = s_prevs[i]
                a_list = [a_t] + s_ts[-1:]  # use upper state

                q = Normal(*self.posteriors[i](s_prev, h_t, a_list))
                p = Normal(*self.priors[i](s_prev, a_list))
                s_losss[i] = torch.sum(kl_divergence(q, p), dim=[1]).mean()
                s_t_p = p.mean if prior_sample else p.rsample()
                s_t_q = q.rsample()
                s_t = s_t_p if prior_sample else s_t_q
                s_ts.append(s_t)
                s01_losss[i] += torch.sum(kl_divergence(q, Normal(0., 1.)), dim=[1]).mean()

                if self.debug:
                    s_abss_p[i] += p.mean.abs().mean()
                    s_stds_p[i] += p.stddev.mean()
                    s_abss_q[i] += q.mean.abs().mean()
                    s_stds_q[i] += q.stddev.mean()

                decoder_dist_p = Normal(*self.decoders[i](s_t_p))
                x_losss_p[i] += - torch.sum(decoder_dist_p.log_prob(x_t),
                                          dim=[1,2,3]).mean()
                decoder_dist_q = Normal(*self.decoders[i](s_t_q))
                x_losss_q[i] += - torch.sum(decoder_dist_q.log_prob(x_t),
                                          dim=[1,2,3]).mean()

            if return_x:
                _x_p.append(decoder_dist_p.mean)
                _x_q.append(decoder_dist_q.mean)

            s_prevs = s_ts

        if return_x:
            _x_p = torch.stack(_x_p).transpose(0, 1)
            _x_p = torch.clamp(_x_p, 0, 1)
            _x_q = torch.stack(_x_q).transpose(0, 1)
            _x_q = torch.clamp(_x_q, 0, 1)
            return _x_p, _x_q
        else:
            for i in range(self.num_states):
                if i not in self.args.static_hierarchy:
                    loss += s_losss[i] + x_losss_q[i]
                    if self.args.beta_x_p:
                        loss += self.args.beta_x_p * x_losss_p[i]
                    if self.args.beta_s01:
                        loss += self.args.beta_s01 * s01_losss[i]
                    if self.args.posterior_s_0:
                        loss += s_0_s_loss + s_0_x_loss

            _locals = locals()
            info = flatten_dict({key:_locals[key] for key in keys})
            info = dict(sorted(info.items()))
            return loss, info

    def sample_s_0(self, x_0):
        _B = x_0.size(0)
        s_prevs = [torch.zeros(_B, s_dim).to(self.device)
                   for s_dim in self.s_dims]
        s_ts = []
        for i in range(self.num_states):
            if self.args.posterior_s_0:
                logger.debug("posterior_s_0 sample")
                with torch.no_grad():  # TODO: no_grad or not !!!!!
                    h_t = self.encoders[i](x_0)
                q = Normal(*self.posterior_s_0s[i](h_t))  ## grad here!
                s_t_q = q.rsample()
                s_0_s_loss = torch.sum(kl_divergence(q, Normal(0., 1.)), dim=[1]).mean()
                with torch.no_grad():
                    decoder_dist_q = Normal(*self.decoders[i](s_t_q))
                s_0_x_loss = - torch.sum(decoder_dist_q.log_prob(x_0),
                                          dim=[1,2,3]).mean()
            else:
                # with torch.no_grad():  # TODO: no_grad or not !!!!!
                h_t = self.encoders[i](x_0)
                s_prev = s_prevs[i]
                a_list = [torch.zeros(_B, d).to(self.device)
                          for d in [self.a_dim] + self.s_dims[i-1:i]]
                # with torch.no_grad():  # TODO: no_grad or not !!!!!
                q = Normal(*self.posteriors[i](s_prev, h_t, a_list))
                s_t_q = q.mean  # rsample()

            s_ts.append(s_t_q)

        if self.args.posterior_s_0:
            return s_ts, s_0_s_loss, s_0_x_loss
        else:
            return s_ts

    def sample_x_temp(self, x_0, x, a):
        with torch.no_grad():
            _x_p, _x_q = self.forward(x_0, x, a, False, return_x=True)
        return _x_p, _x_q

    # def sample_x(self, x_0, a):
    #     return _x_p