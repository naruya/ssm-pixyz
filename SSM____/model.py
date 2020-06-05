# optimizer: https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L193
# optimizer(GAN) 1: https://github.com/alexlee-gk/video_prediction/blob/master/hparams/bair_action_free/ours_savp/model_hparams.json
# optimizer(GAN) 2: https://github.com/naruya/vgan-pytorch/blob/master/notebooks/vgan-celabA.ipynb

import torch
from torch import nn, optim
from torch.nn import functional as F
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, LogProb
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from core import Prior, Posterior, Encoder, Decoder, Discriminator
from torch.nn.utils import clip_grad_norm_
from torch_utils import init_weights
import numpy as np


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def forward(self, feed_dict, train):
        raise NotImplementedError

    def sample_s0(self, x0, train):
        return _sample_s0(self, x0, train)

    def sample_x(self, feed_dict):
        return _sample_x(self, feed_dict)

    # def sample_dx(self, feed_dict):
    #     return _sample_dx(self, feed_dict)

    def train_(self, feed_dict, epoch):
        return _train(self, feed_dict, epoch)

    def test_(self, feed_dict, epoch):
        return _test(self, feed_dict, epoch)


# class SimpleSSM(Base):
#     def __init__(self, args, device):
#         super(SimpleSSM, self).__init__()
#         self.device = device
#         self.s_dim = s_dim = args.s_dim
#         self.a_dim = a_dim = args.a_dim
#         self.h_dim = h_dim = args.h_dim
#         self.gamma = args.gamma
#         self.keys = ["loss", "x_loss[0]", "s_loss[0]", "x_loss", "s_abs[0]", "s_std[0]", "s_aux_loss[0]"]

#         self.prior = Prior(s_dim, a_dim).to(device)
#         self.posterior = Posterior(h_dim, s_dim, a_dim).to(device)
#         self.encoder = Encoder().to(device)
#         self.decoder = Decoder(s_dim).to(device)

#         self.distributions = nn.ModuleList([
#             self.prior,
#             self.posterior,
#             self.encoder,
#             self.decoder,
#         ])

#         init_weights(self.distributions)

#         self.prior01 = Normal(torch.tensor(0.), scale=torch.tensor(1.))
#         self.s_loss_cls = KullbackLeibler(self.posterior, self.prior)
#         self.x_loss_cls = LogProb(self.decoder)
#         self.optimizer = optim.Adam(self.distributions.parameters())

#     def forward(self, feed_dict, train, sample=False):
#         x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
#         s_loss, x_loss, s_aux_loss = 0., 0., 0.
#         s_abs, s_std = 0., 0.
#         s_prev = self.sample_s0(x0, train)
#         _T, _B = x.size(0), x.size(1)
#         _x = []

#         for t in range(_T):
#             x_t, a_t = x[t], a[t]

#             h_t = self.encoder.sample({"x": x_t}, return_all=False)["h"]
#             feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
#             s_loss += self.s_loss_cls.eval(feed_dict).mean()
#             s_aux_loss += kl_divergence(self.posterior.dist, self.prior01).mean()
#             if train:
#                 s_t = self.posterior.dist.rsample()
#                 s_abs += self.posterior.dist.mean.abs().mean()
#                 s_std += self.posterior.dist.stddev.mean()
#             else:
#                 s_t = self.prior.dist.mean
#                 s_abs += self.prior.dist.mean.abs().mean()
#                 s_std += self.prior.dist.stddev.mean()
#             feed_dict = {"s": s_t, "x": x_t}
#             x_loss += - self.decoder.log_prob().eval(feed_dict).mean()
#             _x.append(self.decoder.dist.mean)
#             s_prev = s_t

#         loss = s_loss + x_loss + self.gamma * s_aux_loss
#         # loss = s_loss + x_loss + s_aux_loss
#         # loss = s_loss + x_loss

#         if sample:
#             return _x
#         else:
#             return loss, {"loss": loss.item(), "x_loss": x_loss.item(),
#                           "x_loss[0]": x_loss.item(), "s_loss[0]": s_loss.item(),
#                           "s_aux_loss[0]": s_aux_loss.item(),
#                           "s_abs[0]": s_abs.item(), "s_std[0]": s_std.item()}

#     def sample_s0(self, x0, train):
#         device = self.device
#         _B = x0.size(0)
#         s_prev = torch.zeros(_B, self.s_dim).to(device)
#         a_t = torch.zeros(_B, self.a_dim).to(device)
#         if train:
#             h_t = self.encoder.sample({"x": x0}, return_all=False)["h"]
#             feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
#             s_t = self.posterior.sample(feed_dict, return_all=False)["s"]
#         else:
#             h_t = self.encoder.sample_mean({"x": x0})
#             feed_dict = {"s_prev": s_prev, "a": a_t, "h": h_t}
#             s_t = self.posterior.sample_mean(feed_dict)
#         return s_t


class SSM(Base):
    def __init__(self, args, device):
        super(SSM, self).__init__()

        self.device = device
        self.s_dims = args.s_dim  # list
        self.a_dim = args.a_dim
        self.h_dim = args.h_dim
        self.num_states = len(self.s_dims)
        self.gamma = args.gamma
        self.min_stddev = args.min_stddev
        self.B = args.B

        self.priors = []
        self.posteriors = []
        self.encoders = []
        self.decoders = []
        self.s_loss_clss = []
        self.x_loss_clss = []
        self.keys = ["loss", "x_loss", "beta", "vdb_kl"]

        for i in range(self.num_states):
            s_dim = self.s_dims[i]
            a_dims = [self.a_dim] + self.s_dims[:i]
            print(s_dim)
            print(a_dims)
            self.priors.append(Prior(s_dim, a_dims, self.min_stddev))
            self.posteriors.append(Posterior(self.priors[-1], s_dim, self.h_dim, a_dims, self.min_stddev))
            self.encoders.append(Encoder())
            self.decoders.append(Decoder(s_dim).to(device))
            self.s_loss_clss.append(KullbackLeibler(self.posteriors[-1], self.priors[-1]))
            self.x_loss_clss.append(LogProb(self.decoders[-1]))
            # kl loss
            self.keys.append("s_loss[{}]".format(i))
            self.keys.append("s_aux_loss[{}]".format(i))
            # x reconst loss
            self.keys.append("xq_loss[{}]".format(i))
            self.keys.append("xp_loss[{}]".format(i))
            # dx reconst loss
            # self.keys.append("dxq_loss[{}]".format(i))
            # self.keys.append("dxp_loss[{}]".format(i))
            # GAN loss
            self.keys.append("gq_loss[{}]".format(i))
            self.keys.append("gp_loss[{}]".format(i))
            self.keys.append("gn_loss[{}]".format(i))
            self.keys.append("dq_loss[{}]".format(i))
            self.keys.append("dp_loss[{}]".format(i))
            self.keys.append("dn_loss[{}]".format(i))

        distributions = self.priors + self.posteriors + self.encoders + self.decoders
        self.distributions = nn.ModuleList(distributions).to(device)
        init_weights(self.distributions)

        self.prior01 = Normal(torch.tensor(0.), scale=torch.tensor(1.))

        # GAN
        self.discriminator = Discriminator(device).to(device)
        init_weights(self.discriminator)

        # self.optimizer = optim.Adam(self.distributions.parameters())
        self.g_optimizer = optim.Adam(self.distributions.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.I_c = 0.2
        self.beta = 0.
        self.alpha = 1e-5
        self.g_criterion = nn.BCELoss()
        self.d_criterion = self.VDB_loss


    def VDB_loss(self, out, label, dist):
        normal_D_loss = torch.mean(F.binary_cross_entropy(out, label))

        kldiv_loss = kl_divergence(dist, self.prior01).mean()
        _kldiv_loss = kldiv_loss - self.I_c
        final_loss = normal_D_loss + self.beta * _kldiv_loss

        return final_loss, kldiv_loss.detach()


    def forward(self, feed_dict, train, epoch=None, return_x=False, return_dx=False):
        x0, x, a = feed_dict["x0"], feed_dict["x"], feed_dict["a"]
        s_losss = [0.] * self.num_states
        xq_losss = [0.] * self.num_states
        xp_losss = [0.] * self.num_states
        # dxq_losss = [0.] * self.num_states
        # dxp_losss = [0.] * self.num_states
        s_aux_losss = [0.] * self.num_states
        gq_losss = [0.] * self.num_states
        gp_losss = [0.] * self.num_states
        gn_losss = [0.] * self.num_states
        dq_losss = [0.] * self.num_states
        dp_losss = [0.] * self.num_states
        dn_losss = [0.] * self.num_states

        _T, _B = x.size(0), x.size(1)
        _xq = []
        _xp = []
        # dx = []
        # _dxq = []
        # _dxp = []
        vdb_kls = []

        s_prevs = self.sample_s0(x0, train)

        y_real = torch.ones(self.B, 1, device=self.device)
        y_fake = torch.zeros(self.B, 1, device=self.device)

        for t in range(_T):
            x_t, a_t = x[t], a[t]
            h_t, s_t = [], []

            for i in range(self.num_states):
                h_t.append(self.encoders[i].sample({"x": x_t}, return_all=False)["h"])

                # s loss
                s_losss[i] += self.s_loss_clss[i].eval(
                    {"s_prev": s_prevs[i], "h": h_t[-1], "a_list": [a_t] + s_t}).mean()
                s_aux_losss[i] += kl_divergence(
                    self.posteriors[i].dist, self.prior01).mean()
                sq_t = self.posteriors[i].dist.rsample()
                sp_t = self.priors[i].dist.mean
                s_t.append(sq_t if train else sp_t)

                # x loss
                xq_losss[i] += - self.x_loss_clss[i].eval(
                    {"s": sq_t, "x": x_t}).mean()
                _xq.append(self.decoders[-1].dist.mean)
                xp_losss[i] += - self.x_loss_clss[i].eval(
                    {"s": sp_t, "x": x_t}).mean()
                _xp.append(self.decoders[-1].dist.mean)

                # dx loss
                # if t == 0:
                #     continue
                # dx_t = x[t] - x[t-1]
                # _dxq_t = _xq[t] - _xq[t-1].detach()  # very important
                # _dxp_t = _xp[t] - _xp[t-1].detach()  # very important
                # dx.append(dx_t)
                # _dxq.append(_dxq_t)
                # _dxp.append(_dxp_t)

                # dxq_losss[i] += - torch.sum(Normal(_dxq_t, 1.).log_prob(dx_t),
                #                             dim=[1,2,3]).mean()
                # dxp_losss[i] += - torch.sum(Normal(_dxp_t, 1.).log_prob(dx_t),
                #                             dim=[1,2,3]).mean()

                # generator loss
                sn = (torch.rand(sq_t.shape) * 2. - 1.).to(self.device)  # -1~1
                _xn = self.decoders[-1].sample_mean({"s": sn})

                y_pred, dist = self.discriminator(_xq[t])
                gq_losss[i] += self.g_criterion(y_pred, y_real) / 3.

                y_pred, dist = self.discriminator(_xp[t])
                gp_losss[i] += self.g_criterion(y_pred, y_real) / 3.

                y_pred, dist = self.discriminator(_xn)
                gn_losss[i] += self.g_criterion(y_pred, y_real) / 3.

                # discriminator loss
                y_pred, dist = self.discriminator(x[t])
                d_real_loss, d_kl_real = self.d_criterion(
                    y_pred, y_real, dist)

                y_pred, dist = self.discriminator(_xq[t].detach())
                dq_fake_loss, dq_kl_fake = self.d_criterion(
                    y_pred, y_fake, dist)

                y_pred, dist = self.discriminator(_xp[t].detach())
                dp_fake_loss, dp_kl_fake = self.d_criterion(
                    y_pred, y_fake, dist)

                y_pred, dist = self.discriminator(_xn.detach())
                dn_fake_loss, dn_kl_fake = self.d_criterion(
                    y_pred, y_fake, dist)

                dq_losss[i] += d_real_loss / 3. + dq_fake_loss / 3.
                dp_losss[i] += d_real_loss / 3. + dp_fake_loss / 3.
                dn_losss[i] += d_real_loss / 3. + dn_fake_loss / 3.

                vdb_kls.append(d_kl_real.item() + dq_kl_fake.item())
                vdb_kls.append(d_kl_real.item() + dp_kl_fake.item())
                vdb_kls.append(d_kl_real.item() + dn_kl_fake.item())

            s_prevs = s_t

        # VDB
        self.vdb_kl = np.mean(vdb_kls)
        self.beta = max(0.0, self.beta + self.alpha * self.vdb_kl)

        if return_x:
            return x, _xp
        # if return_dx:
        #     return dx, _dxp
        else:
            g_loss = 0.
            d_loss = 0.

            for i in range(self.num_states):
                g_loss += s_losss[i] + xq_losss[i] + xp_losss[i] \
                        + gq_losss[i] + gp_losss[i] + gn_losss[i] \
                        + self.gamma * s_aux_losss[i]
                        # + dxq_losss[i] + dxp_losss[i] \
                d_loss += dq_losss[i] + dp_losss[i] + dn_losss[i]

            return_dict = {"loss": g_loss.item(),
                           "x_loss": xp_losss[-1].item(),
                           "beta": self.beta,
                           "vdb_kl": self.vdb_kl}
            for i in range(self.num_states):
                return_dict.update({"s_loss[{}]".format(i): s_losss[i].item()})
                return_dict.update({"s_aux_loss[{}]".format(i): s_aux_losss[i].item()})
                return_dict.update({"xq_loss[{}]".format(i): xq_losss[i].item()})
                return_dict.update({"xp_loss[{}]".format(i): xp_losss[i].item()})
                # return_dict.update({"dxq_loss[{}]".format(i): dxq_losss[i].item()})
                # return_dict.update({"dxp_loss[{}]".format(i): dxp_losss[i].item()})
                return_dict.update({"gq_loss[{}]".format(i): gq_losss[i].item()})
                return_dict.update({"gp_loss[{}]".format(i): gp_losss[i].item()})
                return_dict.update({"gn_loss[{}]".format(i): gp_losss[i].item()})
                return_dict.update({"dq_loss[{}]".format(i): dq_losss[i].item()})
                return_dict.update({"dp_loss[{}]".format(i): dp_losss[i].item()})
                return_dict.update({"dn_loss[{}]".format(i): dn_losss[i].item()})

            return g_loss, d_loss, return_dict


# --------------------------------
# default methods
# --------------------------------

def _sample_s0(model, x0, train):
    device = model.device
    _B = x0.size(0)
    s_prevs = [torch.zeros(_B, s_dim).to(device) for s_dim in model.s_dims]
    h_t, s_t = [], []
    for i in range(model.num_states):
        h_t.append(model.encoders[i].sample_mean({"x": x0}))
        a_list = [torch.zeros(_B, d).to(device) for d in [model.a_dim] + model.s_dims[:i]]
        feed_dict = {"s_prev": s_prevs[i], "h": h_t[-1], "a_list": a_list}
        s_t.append(model.posteriors[i].sample_mean(feed_dict))
    return s_t


def _sample_x(model, feed_dict):
    with torch.no_grad():
        x, _x = model.forward(feed_dict, False, return_x=True)
    x = x.transpose(0, 1)  # BxT
    _x = torch.stack(_x).transpose(0, 1)  # BxT
    _x = torch.clamp(_x, 0, 1)
    video = []
    for i in range(4):
        video.append(x[i*8:i*8+8])
        video.append(_x[i*8:i*8+8])
    video = torch.cat(video)  # B*2xT
    return video


# def _sample_dx(model, feed_dict):
#     with torch.no_grad():
#         x, _x = model.forward(feed_dict, False, return_dx=True)
#     x = torch.stack(x).transpose(0, 1)  # BxT
#     x = (x + 1.) / 2.  # (-1,1) -> (0,1)
#     _x = torch.stack(_x).transpose(0, 1)  # BxT
#     _x = (_x + 1.) / 2.  # (-1,1) -> (0,1)
#     video = []
#     for i in range(4):
#         video.append(x[i*8:i*8+8])
#         video.append(_x[i*8:i*8+8])
#     video = torch.cat(video)  # B*2xT
#     return video


def _train(model, feed_dict, epoch):
    model.train()
    g_loss, d_loss, omake_dict = model.forward(feed_dict, True, epoch=epoch)

    model.g_optimizer.zero_grad()
    g_loss.backward()
    model.g_optimizer.step()

    model.d_optimizer.zero_grad()
    d_loss.backward()
    model.d_optimizer.step()

    # for checkking total_norm
    # total_norm = clip_grad_norm_(model.distributions.parameters(), 1e+8)
    return g_loss, omake_dict


def _test(model, feed_dict, epoch):
    model.eval()
    with torch.no_grad():
        g_loss, d_loss, omake_dict = model.forward(feed_dict, False, epoch=epoch)
    return g_loss, omake_dict
