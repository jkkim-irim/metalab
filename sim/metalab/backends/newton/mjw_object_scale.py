"""Per-env object SIZE under mjwarp contacts — give every world its own copy of the object's collision
vertices, then scale that copy on reset.

WHY THIS FILE EXISTS. ``shape_scale`` (newton's per-shape geometry scale) reaches the renderer and newton's
own narrowphase, but NOT mjwarp's: mjwarp collides a mesh geom against ``mesh_vert`` — ONE global array with
no world dimension — via MuJoCo's hull adjacency graph, and never multiplies by ``geom_size``
(``collision_gjk.py`` GeomType.MESH branch). newton's MuJoCo bridge compiles that array from the TEMPLATE
world only and emits a single-row ``geom_dataid``, so every world collides against the same frozen hull.

WHAT IS PER-WORLD IN MJWARP. ``geom_dataid`` — which mesh a geom reads — is declared ``("*", "ngeom")`` and
read as ``geom_dataid[worldid % rows, geom]``. That is mjwarp's own mechanism for per-world geometry (its
``io_test`` builds exactly this table for mesh variants, then fixes up the dependent per-world fields). So the
fix is not to fight the shared vertex array but to GROW it: append one copy of the object's collision
vertices per world, hand each world its own mesh entries, and rewrite that world's copy when the DR draws.

WHAT IS SHARED AND WHAT IS NOT. The hull graph, faces and polygons are indexed mesh-LOCALLY
(``mesh_vert[vertadr + local_idx]``, ``mesh_polyvert`` holds local indices) and polygon normals are
directions, so every copy points at the ORIGINAL topology — only the vertex block is duplicated. Vertex-
indexed side tables (``mesh_polymapadr/num``) must grow with the vertex block; their values are global
polygon addresses and copy verbatim.

WHAT MUST MOVE WITH THE SIZE (mjwarp's own list, io_test._populate_dependent_fields): ``geom_rbound`` and
``geom_aabb`` (broadphase culling — a 2x object with a 1x bound is filtered out before narrowphase ever
runs) and ``geom_pos``, because MuJoCo re-centres each mesh on its inertial frame and keeps the offset in
``mesh_pos``: a vertex block scaled about that frame needs its ``mesh_pos`` scaled too, or the hull collides
a few mm off. ``geom_pos`` itself is refreshed by newton from ``mesh_pos[geom_dataid[world]]`` on the
SHAPE_PROPERTIES notify the caller already issues.

Body mass/inertia are NOT touched here — ``set_object_mass`` owns them.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

# nmesh-indexed arrays a copy inherits VERBATIM from its source mesh: topology addresses (the graph, faces
# and polygons are shared) and the mesh frame's rotation (uniform scale leaves principal axes alone).
# ``mesh_vertadr`` is the one that must differ per copy, and ``mesh_pos`` the one that must scale.
_SHARED_MESH_FIELDS = ("mesh_vertnum", "mesh_faceadr", "mesh_octadr", "mesh_normaladr", "mesh_normalnum",
                       "mesh_graphadr", "mesh_polynum", "mesh_polyadr", "mesh_quat")
_WP_DTYPE = {"mesh_pos": wp.vec3, "mesh_quat": wp.quat}


class MjwObjectScale:
    """Owns the per-world vertex copies + the writes that resize one world's object."""

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

        # --- 1. the vertex block and its two vertex-indexed side tables grow by one copy per world -------
        vert = wp.to_torch(m.mesh_vert)
        pmadr, pmnum = wp.to_torch(m.mesh_polymapadr), wp.to_torch(m.mesh_polymapnum)
        rows = [(int(vertadr[d]), int(vertadr[d]) + n) for d, n in zip(src, per_mesh)]
        base_vert = torch.cat([vert[a:b] for a, b in rows]).contiguous()          # (n_vert, 3)
        base_pmadr = torch.cat([pmadr[a:b] for a, b in rows]).contiguous()
        base_pmnum = torch.cat([pmnum[a:b] for a, b in rows]).contiguous()
        vert_off = int(vert.shape[0])                                            # where world 0's copy starts
        new_vert = torch.cat([vert, base_vert.repeat(self._num_envs, 1)]).contiguous()
        new_pmadr = torch.cat([pmadr, base_pmadr.repeat(self._num_envs)]).contiguous()
        new_pmnum = torch.cat([pmnum, base_pmnum.repeat(self._num_envs)]).contiguous()

        # --- 2. one mesh ENTRY per (world, object mesh): shared topology, private vertex address ---------
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
                              dtype=vertadr.dtype, device=device)                # offset inside one copy
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

        # --- 3. point each world at its own copy (once — the assignment never changes again) -------------
        dataid = dataid0.repeat(self._num_envs, 1).contiguous()
        dataid[:, self._geoms] = (nmesh0 + torch.arange(self._num_envs * n_mesh_copies, device=device,
                                                        dtype=dataid.dtype).view(self._num_envs, -1))
        keep.append(dataid)
        added += dataid.numel() * dataid.element_size()
        m.geom_dataid = wp.from_torch(dataid, dtype=wp.int32)

        # --- 4. broadphase bounds: rbound is already per-world, aabb may still be the 1-row template -----
        if m.geom_aabb.shape[0] != self._num_envs:
            aabb = wp.to_torch(m.geom_aabb)[0].unsqueeze(0).repeat(self._num_envs, 1, 1, 1).contiguous()
            keep.append(aabb)
            added += aabb.numel() * aabb.element_size()
            m.geom_aabb = wp.from_torch(aabb, dtype=wp.vec3)
        assert m.geom_rbound.shape[0] == self._num_envs, \
            f"geom_rbound has {m.geom_rbound.shape[0]} rows, expected one per world"

        self._keep = keep            # the wp arrays alias this memory — dropping these frees it under warp
        self._base_vert = base_vert
        self._base_mesh_pos = wp.to_torch(m.mesh_pos)[src].clone()
        self._vert_view = new_vert[vert_off:].view(self._num_envs, n_vert, 3)
        self._mesh_pos_view = wp.to_torch(m.mesh_pos)[nmesh0:].view(self._num_envs, n_mesh_copies, 3)
        self._rbound = wp.to_torch(m.geom_rbound)
        self._aabb = wp.to_torch(m.geom_aabb)
        self._base_rbound = self._rbound[0, self._geoms].clone()
        self._base_aabb = self._aabb[0, self._geoms].clone()
        self.bytes = added        # GPU memory this scheme ADDS (not the size of the arrays it now owns)
        self.n_vert = n_vert

    def apply(self, env_idx: torch.Tensor, scale: torch.Tensor) -> None:
        """Resize ``env_idx``'s object to ``scale`` x the asset. Caller issues the SHAPE_PROPERTIES notify."""
        s = scale.view(-1, 1, 1)
        self._vert_view[env_idx] = self._base_vert * s
        self._mesh_pos_view[env_idx] = self._base_mesh_pos * s
        rows = env_idx.view(-1, 1)
        self._rbound[rows, self._geoms] = self._base_rbound * scale.view(-1, 1)
        self._aabb[rows, self._geoms] = self._base_aabb * s.unsqueeze(-1)


def install(solver, obj_geoms: list[int], num_envs: int, device) -> MjwObjectScale:
    """Grow the mjwarp mesh registry so each world owns the object's collision vertices. Call ONCE, before
    the first step — the physics loop is CUDA-graph captured and a captured graph holds the array pointers."""
    return MjwObjectScale(solver, obj_geoms, num_envs, device)
