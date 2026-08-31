"""Live-engine check: the contact friction the solver USES is sqrt(mu_a*mu_b), not MuJoCo/genesis' max().

Needs a GPU and one engine venv, so it is a script rather than a unit test (test_motor_coupling.py covers
what runs on CPU). It makes the two mixing rules give different answers on purpose — the hammer's mu is
lowered to 0.25 while everything else stays 1.0, so every hammer contact has mu_a != mu_b (max would report
1.0, the geometric mean 0.5) — then reads the solver's own contact buffer and compares EVERY live contact
against the shared reference in ``runtime/physics/friction.py``, using that contact's own geom ids.

Run both, expect PASS from each:
    <newton venv>  python sim/metalab/tests/check_contact_friction.py newton
    <genesis venv> python sim/metalab/tests/check_contact_friction.py genesis
"""
import sys

import numpy as np

from sim.metalab.runtime.physics.friction import geomean_friction

engine = sys.argv[1]
LOW = 0.25          # mu written onto one body; everything else keeps 1.0 -> sqrt = 0.5 vs max = 1.0

if engine == "newton":
    import mujoco
    import warp as wp

    from sim.metalab.backends.newton.server import build_env

    env = build_env(task="hammer_lift_student", recipe="only_ycb", num_envs=2, device="cuda:0",
                    viz=None, telemetry=False)
    b = env.backend
    solver = b.solver
    m, d = solver.mjw_model, solver.mjw_data
    mjm = solver.mj_model
    fr = wp.to_torch(m.geom_friction)                     # (nworld, ngeom, 3)
    obj = [g for g in range(mjm.ngeom)
           if (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[g])) or "").startswith("hammer")]
    assert obj, "no hammer geoms found"
    fr[:, obj, :] = LOW                                    # the hammer becomes the low-mu body
    print(f"wrote mu={LOW} on {len(obj)} hammer geoms (others stay 1.0)")
    for _ in range(40):
        b.step(render=False)
    n = int(wp.to_torch(d.nacon)[0])
    geom = wp.to_torch(d.contact.geom)[:n].cpu().numpy()
    world = wp.to_torch(d.contact.worldid)[:n].cpu().numpy()
    got = wp.to_torch(d.contact.friction)[:n].cpu().numpy()   # vec5: slide, slide, tors, roll, roll
    frn = fr.cpu().numpy()
    mixed, checked = [], 0
    for k in range(n):
        a, bb, w = int(geom[k][0]), int(geom[k][1]), int(world[k])
        want = geomean_friction(frn[w, a], frn[w, bb])
        want5 = np.array([want[0], want[0], want[1], want[2], want[2]])
        assert np.allclose(got[k], want5, rtol=1e-5, atol=1e-7), (k, got[k], want5)
        checked += 1
        if abs(frn[w, a][0] - frn[w, bb][0]) > 1e-9:
            mixed.append((a, bb, float(frn[w, a][0]), float(frn[w, bb][0]), float(got[k][0])))
else:
    import torch

    from sim.metalab.backends.genesis.server import build_env

    env = build_env(task="hammer_lift_student", recipe="only_ycb", num_envs=2, device="cuda:0",
                    viz=None, telemetry=False)
    b = env.backend
    rs = b.scene.sim.rigid_solver
    assert getattr(rs, "_metalab_geomean_friction", False), "geometric-mean hook not installed"
    b.set_object_friction("object", torch.arange(b.num_envs, device=b.device),
                          torch.full((b.num_envs,), LOW, device=b.device))   # our own DR write surface
    print(f"wrote mu={LOW} on the hammer via backend.set_object_friction (others stay 1.0)")
    for _ in range(40):
        b.step(render=False)
    cst = rs.collider._collider_state
    n_c = cst.n_contacts.to_numpy()
    ga, gb = cst.contact_data.geom_a.to_numpy(), cst.contact_data.geom_b.to_numpy()
    got = cst.contact_data.friction.to_numpy()
    mu = rs.geoms_info.friction.to_numpy()
    ratio = rs.geoms_state.friction_ratio.to_numpy()
    mixed, checked = [], 0
    for i_b in range(len(n_c)):
        for k in range(int(n_c[i_b])):
            a, bb = int(ga[k, i_b]), int(gb[k, i_b])
            ma, mb = float(mu[a] * ratio[a, i_b]), float(mu[bb] * ratio[bb, i_b])
            want = max(float(np.sqrt(ma * mb)), 1.0e-2)      # genesis keeps its own 1e-2 floor
            assert abs(float(got[k, i_b]) - want) < 1e-5, (i_b, k, got[k, i_b], want)
            checked += 1
            if abs(ma - mb) > 1e-9:
                mixed.append((a, bb, ma, mb, float(got[k, i_b])))

print(f"contacts checked against sqrt(mu_a*mu_b): {checked}")
print(f"mixed-mu contacts (where max() would differ): {len(mixed)}")
for e in mixed[:4]:
    print(f"   geoms({e[0]},{e[1]}) mu=({e[2]:.2f},{e[3]:.2f}) -> got {e[4]:.4f} "
          f"| sqrt={np.sqrt(e[2] * e[3]):.4f} max={max(e[2], e[3]):.4f}")
ok = bool(mixed) and checked > 0
print(f"\n{'PASS' if ok else 'INCONCLUSIVE (no mixed-mu contact was live)'}")
sys.exit(0 if ok else 2)
