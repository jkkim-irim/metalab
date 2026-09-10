from __future__ import annotations

from tensordict import TensorDict
import torch


class ObsPipeline:
    def __init__(self, spec, num_envs: int):
        self.spec = spec
        self.num_envs = int(num_envs)
        self.hist_len = dict(spec.obs_history_length)
        self.noise_groups = set(spec.obs_noise_groups)
        self.hist: dict = {}

    @staticmethod
    def add_noise(v: torch.Tensor, ns) -> torch.Tensor:
        if ns.std is not None:
            return v if ns.std == 0.0 else v + torch.randn_like(v) * ns.std
        w = v.shape[-1]
        assert w % 7 == 0, f"ObsNoise(pos=/rot=) needs a pos3+quat4 layout, term is {w} wide"
        b = v.reshape(v.shape[0], w // 7, 7)
        p, q = b[..., :3], b[..., 3:7]
        if ns.pos:
            p = p + torch.randn_like(p) * ns.pos
        if ns.rot:
            q = q + torch.randn_like(q) * (0.5 * ns.rot)
            q = q / q.norm(dim=-1, keepdim=True)
        return torch.cat([p, q], dim=-1).reshape(v.shape)

    def value(self, env, t, group: str) -> torch.Tensor:
        v = env.run_term("obs", t)
        if group in self.noise_groups and t.noise is not None:
            v = self.add_noise(v, t.noise)
        return v * t.scale

    def capture(self, env) -> dict:
        return {gname: torch.cat([self.value(env, t, gname) for t in terms], dim=-1)
                for gname, terms in self.spec.obs.items()}

    def stack(self, frames: dict) -> TensorDict:
        groups = {}
        for gname, f in frames.items():
            if self.hist_len.get(gname, 1) > 1:
                groups[gname] = self.hist[gname].reshape(self.num_envs, -1)
            else:
                groups[gname] = f
        return TensorDict(groups, batch_size=[self.num_envs])

    def advance_history(self, frames: dict, reset_ids) -> None:
        for gname, f in frames.items():
            h = self.hist_len.get(gname, 1)
            if h <= 1:
                continue
            buf = self.hist.get(gname)
            if buf is None:
                buf = f.unsqueeze(1).repeat(1, h, 1)
            else:
                buf = torch.roll(buf, shifts=-1, dims=1)
                buf[:, -1] = f
            if reset_ids is not None and reset_ids.numel() > 0:
                buf[reset_ids] = f[reset_ids].unsqueeze(1)
            self.hist[gname] = buf

    def observe(self, env) -> TensorDict:
        frames = self.capture(env)
        for gname, f in frames.items():
            h = self.hist_len.get(gname, 1)
            if h > 1 and gname not in self.hist:
                self.hist[gname] = f.unsqueeze(1).repeat(1, h, 1)
        return self.stack(frames)

    def reset(self) -> None:
        self.hist = {}
