from __future__ import annotations

import numpy as np
import torch
import warp as wp

_SHARED_MESH_FIELDS = ("mesh_vertnum", "mesh_faceadr", "mesh_octadr", "mesh_normaladr", "mesh_normalnum",
                       "mesh_graphadr", "mesh_polynum", "mesh_polyadr", "mesh_quat")
_WP_DTYPE = {"mesh_pos": wp.vec3, "mesh_quat": wp.quat}


class MjwObjectScale:
    def __init__(self, solver, obj_geoms: list[int], num_envs: int, device):
        m = solver.mjw_model
        self._m = m
        self._num_envs = int(num_envs)
        self._geoms = torch.tensor(obj_geoms, dtype=torch.long, device=device)
        n_mesh_copies = len(obj_geoms)

        dataid0 = wp.to_torch(m.geom_dataid)[0]
        src = [int(dataid0[g]) for g in obj_geoms]
        assert all(d >= 0 for d in src), f"object geoms {obj_geoms} are not all mesh geoms (dataid {src})"
        vertadr = wp.to_torch(m.mesh_vertadr)
        vertnum = wp.to_torch(m.mesh_vertnum)
        per_mesh = [int(vertnum[d]) for d in src]
        n_vert = int(sum(per_mesh))

        vert = wp.to_torch(m.mesh_vert)
        pmadr, pmnum = wp.to_torch(m.mesh_polymapadr), wp.to_torch(m.mesh_polymapnum)
        rows = [(int(vertadr[d]), int(vertadr[d]) + n) for d, n in zip(src, per_mesh)]
        base_vert = torch.cat([vert[a:b] for a, b in rows]).contiguous()
        base_pmadr = torch.cat([pmadr[a:b] for a, b in rows]).contiguous()
        base_pmnum = torch.cat([pmnum[a:b] for a, b in rows]).contiguous()
        vert_off = int(vert.shape[0])
        new_vert = torch.cat([vert, base_vert.repeat(self._num_envs, 1)]).contiguous()
        new_pmadr = torch.cat([pmadr, base_pmadr.repeat(self._num_envs)]).contiguous()
        new_pmnum = torch.cat([pmnum, base_pmnum.repeat(self._num_envs)]).contiguous()

        nmesh0 = int(m.nmesh)
        keep = [new_vert, new_pmadr, new_pmnum]
        added = sum(t.numel() * t.element_size() for t in (base_vert, base_pmadr, base_pmnum)) * self._num_envs
        for f in _SHARED_MESH_FIELDS + ("mesh_pos",):
            t = wp.to_torch(getattr(m, f))
            tile = t[src].repeat(self._num_envs, *([1] * (t.dim() - 1)))
            grown = torch.cat([t, tile]).contiguous()
            keep.append(grown)
            added += tile.numel() * tile.element_size()
            setattr(m, f, wp.from_torch(grown, dtype=_WP_DTYPE.get(f, wp.int32)))
        within = torch.tensor(np.tile(np.cumsum([0] + per_mesh)[:-1], self._num_envs),
                              dtype=vertadr.dtype, device=device)
        world_of = torch.arange(self._num_envs, dtype=vertadr.dtype,
                               device=device).repeat_interleave(n_mesh_copies)
        new_adr = torch.cat([vertadr, vert_off + world_of * n_vert + within]).to(vertadr.dtype).contiguous()
        keep.append(new_adr)
        added += within.numel() * within.element_size()
        m.mesh_vertadr = wp.from_torch(new_adr, dtype=wp.int32)
        m.mesh_vert = wp.from_torch(new_vert, dtype=wp.vec3)
        m.mesh_polymapadr = wp.from_torch(new_pmadr, dtype=wp.int32)
        m.mesh_polymapnum = wp.from_torch(new_pmnum, dtype=wp.int32)
        m.nmesh = nmesh0 + self._num_envs * n_mesh_copies
        m.nmeshvert = int(new_vert.shape[0])

        dataid = dataid0.repeat(self._num_envs, 1).contiguous()
        dataid[:, self._geoms] = (nmesh0 + torch.arange(self._num_envs * n_mesh_copies, device=device,
                                                        dtype=dataid.dtype).view(self._num_envs, -1))
        keep.append(dataid)
        added += dataid.numel() * dataid.element_size()
        m.geom_dataid = wp.from_torch(dataid, dtype=wp.int32)

        if m.geom_aabb.shape[0] != self._num_envs:
            aabb = wp.to_torch(m.geom_aabb)[0].unsqueeze(0).repeat(self._num_envs, 1, 1, 1).contiguous()
            keep.append(aabb)
            added += aabb.numel() * aabb.element_size()
            m.geom_aabb = wp.from_torch(aabb, dtype=wp.vec3)
        assert m.geom_rbound.shape[0] == self._num_envs, \
            f"geom_rbound has {m.geom_rbound.shape[0]} rows, expected one per world"

        self._keep = keep
        self._base_vert = base_vert
        self._base_mesh_pos = wp.to_torch(m.mesh_pos)[src].clone()
        self._vert_view = new_vert[vert_off:].view(self._num_envs, n_vert, 3)
        self._mesh_pos_view = wp.to_torch(m.mesh_pos)[nmesh0:].view(self._num_envs, n_mesh_copies, 3)
        self._rbound = wp.to_torch(m.geom_rbound)
        self._aabb = wp.to_torch(m.geom_aabb)
        self._base_rbound = self._rbound[0, self._geoms].clone()
        self._base_aabb = self._aabb[0, self._geoms].clone()
        self.bytes = added
        self.n_vert = n_vert

    def apply(self, env_idx: torch.Tensor, scale: torch.Tensor) -> None:
        s = scale.view(-1, 1, 1)
        self._vert_view[env_idx] = self._base_vert * s
        self._mesh_pos_view[env_idx] = self._base_mesh_pos * s
        rows = env_idx.view(-1, 1)
        self._rbound[rows, self._geoms] = self._base_rbound * scale.view(-1, 1)
        self._aabb[rows, self._geoms] = self._base_aabb * s.unsqueeze(-1)


def install(solver, obj_geoms: list[int], num_envs: int, device) -> MjwObjectScale:
    return MjwObjectScale(solver, obj_geoms, num_envs, device)
