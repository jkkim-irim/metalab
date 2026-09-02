from __future__ import annotations

import torch
import warp as wp

from sim.metalab.control.motor_coupling import MotorCoupledPDArm, MotorCoupledPDHand


class TorchCoupledPD:
    def __init__(self, groups: list[dict], num_envs: int, device):
        wp.init()
        d = len(groups[0]["joints"])
        k = sum(len(g["joints"]) for g in groups)
        idx = torch.arange(num_envs * k, dtype=torch.int32, device=device).reshape(num_envs, k)
        self._gc_buf = torch.zeros(num_envs, k, dtype=torch.float32, device=device)
        cls = MotorCoupledPDHand if d == 3 else MotorCoupledPDArm
        self._inner = cls(groups, idx, idx, num_envs, device,
                          gravcomp=wp.from_torch(self._gc_buf),
                          gc_dof=torch.arange(k, dtype=torch.int32, device=device))
        self._jf = torch.zeros(num_envs * k, dtype=torch.float32, device=device)
        self.joints = self._inner.joints
        self.n_groups = self._inner.n_groups
        self.tau_torch = self._inner.tau_torch
        self.tau_pd_torch = self._inner.tau_pd_torch
        self.tau_gc_torch = self._inner.tau_gc_torch
        self.set_fold_kd_zero = self._inner.set_fold_kd_zero
        self.reload_gains = self._inner.reload_gains
        self.gain_warnings = self._inner.gain_warnings
        self.kq = self._inner.kq
        self.tau_lim = self._inner.tau_lim

    def compute(self, q, qd, q_tgt, tau_g=None):
        if tau_g is not None:
            self._gc_buf.copy_(tau_g)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            self._inner.launch(wp.from_torch(q.reshape(-1).contiguous()),
                               wp.from_torch(qd.reshape(-1).contiguous()),
                               wp.from_torch(q_tgt.reshape(-1).contiguous()),
                               wp.from_torch(self._jf))
        return self.tau_torch


class CoupledPDMixin:
    _coupled_owners: list
    _coupled_col_cache: dict

    def _actuator_gravcomp(self, names) -> torch.Tensor:
        raise NotImplementedError

    def joint_torque_pd(self, names):
        return self._cached(("jtpd", tuple(names)), lambda: self._joint_torque_components(names)[0])

    def joint_torque_gravcomp(self, names):
        return self._cached(("jtgc", tuple(names)), lambda: self._joint_torque_components(names)[1])

    def _joint_torque_components(self, names):
        gc = self._actuator_gravcomp(names)
        pd = self.joint_torque(names) - gc
        for oi, o in enumerate(self._coupled_owners):
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                pd[:, cols] = o.tau_pd_torch[:, tcols]
                gc[:, cols] = o.tau_gc_torch[:, tcols]
        return pd, gc

    def _coupled_cols(self, oi, owner, names):
        key = (oi, tuple(names))
        r = self._coupled_col_cache.get(key)
        if r is None:
            jmap = {j: i for i, j in enumerate(owner.joints)}
            cols, tcols = [], []
            for i, n in enumerate(names):
                if n in jmap:
                    cols.append(i)
                    tcols.append(jmap[n])
            r = (cols, tcols)
            self._coupled_col_cache[key] = r
        return r

    def set_coupled_float_damping(self, off: bool):
        for o in self._coupled_owners:
            o.set_fold_kd_zero(off)

    def coupled_kq(self):
        return [e for o in self._coupled_owners
                for e in o.kq(self.joint_pos(o.joints)[0].detach().cpu().numpy())]

    def coupled_tau_lim(self):
        return [e for o in self._coupled_owners
                for e in o.tau_lim(self.joint_pos(o.joints)[0].detach().cpu().numpy(),
                                   self.joint_vel(o.joints)[0].detach().cpu().numpy())]

    def motor_gain_warnings(self):
        return [w for o in self._coupled_owners for w in o.gain_warnings()]

    def reload_motor_gains(self):
        return [n for o in self._coupled_owners for n in o.reload_gains()]
