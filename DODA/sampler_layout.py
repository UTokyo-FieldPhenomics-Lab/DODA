# modified from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py

import torch
from tqdm.auto import trange


class eulerSampler(object):
    def __init__(self, model, sample_steps=50, **kwargs):
        super().__init__()
        self.model = model
        self.device = self.model.device
        self.ddpm_num_timesteps = model.num_timesteps
        self.make_schedule()
        self.step_sigma, self.timesteps = self.get_sigmas(sample_steps)
        #self.step_sigma = self.get_sigmas_karras(sample_steps)
        self.sigma_data = 0.5

        #timesteps = np.linspace(self.model.num_timesteps - 1, 0, sample_steps, dtype=self.dtype)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.device):
                attr = attr.to(torch.device(self.device))
        setattr(self, name, attr)

    def make_schedule(self):
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.register_buffer('log_sigmas', sigmas.log())
        self.register_buffer('sigmas', to_torch(sigmas))

    def get_sigmas(self, n=None):
        if n is None:
            return self.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return self.append_zero(self.t_to_sigma(t)), t
    
    def get_sigmas_karras(self, n, sigma_min=0.002, sigma_max=80, rho=7.):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()
    
    def sigma_to_t(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)
    
    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])
    
    def sample_euler(self, shape, c, layout, unconditional_layout, x=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., unconditional_guidance_scale=1., unconditional_conditioning=None):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
        
        if x is None:
            x = torch.randn(shape, device=self.device)
        x = x*self.step_sigma[0]

        print(f'Data shape for euler sampling is {shape}')

        for i in trange(len(self.step_sigma) - 1, disable=disable):
            gamma = min(s_churn / (len(self.step_sigma) - 1), 2 ** 0.5 - 1) if s_tmin <= self.step_sigma[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = self.step_sigma[i] * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - self.step_sigma[i] ** 2) ** 0.5

            ts = torch.round(self.timesteps[i])
            #ts = self.sigma_to_t(sigma_hat)
            ts = torch.full((shape[0],), ts, device=self.device, dtype=torch.long)
            c_in = 1 / (sigma_hat ** 2 + 1) ** 0.5
            layout_emb = self.model.layout_encoder(layout, ts)
            if unconditional_conditioning is None or unconditional_guidance_scale==1.:
                '''model_output = self.model.apply_model(x*c_in, ts, c)
                pred_original_sample = x - sigma_hat * model_output
                d = (x - pred_original_sample) / sigma_hat'''
                d = self.model.apply_model(x*c_in, ts, c, layout_emb)
            else:
                unconditional_layout_emb = self.model.layout_encoder(unconditional_layout, ts)
                d = self.model.apply_model(x*c_in, ts, c, layout_emb)
                un_d = self.model.apply_model(x*c_in, ts, unconditional_conditioning, unconditional_layout_emb)
                d = un_d + unconditional_guidance_scale * (d - un_d)
            dt = self.step_sigma[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt

        return x

    def sample_heun(self, shape, c, x=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., unconditional_guidance_scale=1., unconditional_conditioning=None):
        """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""

        if x is None:
            x = torch.randn(shape, device=self.device)
        x = x*self.step_sigma[0]

        print(f'Data shape for heun sampling is {shape}')
        for i in trange(len(self.step_sigma) - 1, disable=disable):
            gamma = min(s_churn / (len(self.step_sigma) - 1), 2 ** 0.5 - 1) if s_tmin <= self.step_sigma[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = self.step_sigma[i] * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - self.step_sigma[i] ** 2) ** 0.5

            ts = torch.round(self.timesteps[i])
            #ts = self.sigma_to_t(sigma_hat)
            ts = torch.full((shape[0],), ts, device=self.device, dtype=torch.long)

            c_in = 1 / (sigma_hat ** 2 + 1) ** 0.5
            if unconditional_conditioning is None or unconditional_guidance_scale==1.:
                d = self.model.apply_model(x*c_in, ts, c)
            else:
                d = self.model.apply_model(x*c_in, ts, c)
                un_d = self.model.apply_model(x*c_in, ts, unconditional_conditioning)
                d = un_d + unconditional_guidance_scale * (d - un_d)

            dt = self.step_sigma[i + 1] - sigma_hat
            if self.step_sigma[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt

                '''ts = torch.round(self.timesteps[i+1])
                #ts = self.sigma_to_t(self.step_sigma[i+1])
                ts = torch.full((shape[0],), ts, device=self.device, dtype=torch.long)'''

                c_in = 1 / (self.step_sigma[i+1] ** 2 + 1) ** 0.5
                if unconditional_conditioning is None or unconditional_guidance_scale==1.:
                    d_2 = self.model.apply_model(x_2*c_in, ts, c)
                else:
                    d_2 = self.model.apply_model(x_2*c_in, ts, c)
                    un_d_2 = self.model.apply_model(x_2*c_in, ts, unconditional_conditioning)
                    d_2 = un_d_2 + unconditional_guidance_scale * (d_2 - un_d_2)

                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x

    def sample_edm(self, shape, c, x=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., unconditional_guidance_scale=1., unconditional_conditioning=None):
        """Implements Algorithm 2 (edm steps) from Karras et al. (2022)."""

        if x is None:
            x = torch.randn(shape, device=self.device)
        x = x*self.step_sigma[0]

        print(f'Data shape for edm sampling is {shape}')
        for i in trange(len(self.step_sigma) - 1, disable=disable):
            gamma = min(s_churn / (len(self.step_sigma) - 1), 2 ** 0.5 - 1) if s_tmin <= self.step_sigma[i] <= s_tmax else 0.

            eps = torch.randn_like(x) * s_noise
            sigma_hat = self.step_sigma[i] * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - self.step_sigma[i] ** 2) ** 0.5



            c_skip = self.sigma_data ** 2 / (sigma_hat ** 2 + self.sigma_data ** 2)
            c_out = sigma_hat * self.sigma_data / (sigma_hat ** 2 + self.sigma_data ** 2).sqrt()
            c_in = 1 / (self.sigma_data ** 2 + sigma_hat ** 2).sqrt()
            c_noise = sigma_hat.log() / 4


            #ts = torch.round(self.timesteps[i])
            ts = self.sigma_to_t(sigma_hat)
            ts = torch.full((shape[0],), ts, device=self.device, dtype=torch.long)


            if unconditional_conditioning is None or unconditional_guidance_scale==1.:
                d = self.model.apply_model(x*c_in, ts, c)


            else:
                d = self.model.apply_model(x*c_in, ts, c)
                un_d = self.model.apply_model(x*c_in, ts, unconditional_conditioning)
                d = un_d + unconditional_guidance_scale * (d - un_d)

            dt = self.step_sigma[i + 1] - sigma_hat
            #if self.step_sigma[i + 1] == 0:
            if True:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                sigma_next = self.step_sigma[i+1]
                x_2 = x + d * dt

                #ts = torch.round(self.timesteps[i+1])
                ts = self.sigma_to_t(sigma_next)
                ts = torch.full((shape[0],), ts, device=self.device, dtype=torch.long)
                c_skip = self.sigma_data ** 2 / (sigma_next ** 2 + self.sigma_data ** 2)
                c_out = sigma_next * self.sigma_data / (sigma_next ** 2 + self.sigma_data ** 2).sqrt()
                c_in = 1 / (self.sigma_data ** 2 + sigma_next ** 2).sqrt()

                if unconditional_conditioning is None or unconditional_guidance_scale==1.:
                    e_2 = self.model.apply_model(x_2*c_in, ts, c)
                    denoised_2 = c_skip * x_2 + c_out * e_2
                    d_2 = (x_2 - denoised_2)/sigma_next
                else:
                    d_2 = self.model.apply_model(x_2*c_in, ts, c)
                    un_d_2 = self.model.apply_model(x_2*c_in, ts, unconditional_conditioning)
                    d_2 = un_d_2 + unconditional_guidance_scale * (d_2 - un_d_2)

                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x