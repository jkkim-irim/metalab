from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC_DIR = _HERE.parent / "firmware_sources_완료하면삭제하자"


class _Poly:
    def __init__(self, terms, coeffs):
        self.terms = np.asarray(terms, dtype=np.int64)
        self.coeffs = np.asarray(coeffs, dtype=np.float64)

    def grad(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        g = np.zeros(self.terms.shape[1])
        for e, c in zip(self.terms, self.coeffs):
            for j in range(len(e)):
                if e[j] > 0:
                    ej = e.copy(); ej[j] -= 1
                    g[j] += c * e[j] * np.prod(x ** ej)
        return g


def _fn_body(src: str, signature: str) -> str:
    i = src.index(signature)
    start = src.index("{", i)
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return re.sub(r"//[^\n]*", "", src[start + 1:j])
    raise ValueError(f"unbalanced braces after {signature!r}")


_WRIST_TERM_RE = re.compile(r'([+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?))((?:\*q[rp]\d*)*)')
_WRIST_VAR_RE = re.compile(r'q([rp])(\d*)')


def _parse_wrist_poly(expr: str) -> list[tuple[list[int], float]]:
    expr = re.sub(r"\s+", "", expr)
    out = []
    for coeff, vars_str in _WRIST_TERM_RE.findall(expr):
        if coeff in ("", "+", "-"):
            continue
        exps = [0, 0]
        for var, pw in _WRIST_VAR_RE.findall(vars_str):
            exps[0 if var == "r" else 1] += int(pw) if pw else 1
        out.append((exps, float(coeff)))
    return out


def _arrays(terms_coeffs: list[tuple[list[int], float]]) -> tuple[np.ndarray, np.ndarray]:
    terms = np.array([tc[0] for tc in terms_coeffs], dtype=np.int64)
    coeffs = np.array([tc[1] for tc in terms_coeffs], dtype=np.float64)
    return terms, coeffs


_THUMB_TERMS_RE = re.compile(
    r"constexpr\s+std::array<std::array<int,\s*(\d+)>,\s*(\d+)>\s+kM([123])Terms\{\{(.*?)\}\};",
    re.DOTALL)
_THUMB_COEF_RE = re.compile(
    r"constexpr\s+std::array<double,\s*(\d+)>\s+kM([123])Coef\{\{(.*?)\}\};", re.DOTALL)
_NUM_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _parse_thumb_tables(src: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    terms: dict[int, np.ndarray] = {}
    for d, n, idx, body in _THUMB_TERMS_RE.findall(src):
        rows = [[int(v) for v in r.split(",")] for r in re.findall(r"\{([^{}]*)\}", body)]
        assert len(rows) == int(n), f"kM{idx}Terms: declared {n} rows, found {len(rows)}"
        assert all(len(r) == int(d) for r in rows), f"kM{idx}Terms: a row is not {d} wide"
        terms[int(idx)] = np.array(rows, dtype=np.int64)
    out = {}
    for n, idx, body in _THUMB_COEF_RE.findall(src):
        vals = [float(v) for v in _NUM_RE.findall(body)]
        assert len(vals) == int(n), f"kM{idx}Coef: declared {n} values, found {len(vals)}"
        assert int(idx) in terms, f"kM{idx}Coef has no matching kM{idx}Terms"
        assert terms[int(idx)].shape[0] == len(vals), \
            f"kM{idx}: {terms[int(idx)].shape[0]} terms vs {len(vals)} coefficients"
        out[int(idx)] = (terms[int(idx)], np.array(vals, dtype=np.float64))
    assert set(out) == {1, 2, 3}, f"thumb: expected kM1/kM2/kM3 tables, got {sorted(out)}"
    return out


def _fw_eval_derivative(terms: np.ndarray, coeffs: np.ndarray, x, var: int) -> float:
    value = 0.0
    for i in range(len(coeffs)):
        e = int(terms[i, var])
        if e == 0:
            continue
        t = coeffs[i] * float(e)
        for j in range(terms.shape[1]):
            t *= x[j] ** (int(terms[i, j]) - (1 if j == var else 0))
        value += t
    return value


_THUMB = {
    "cpp": _SRC_DIR / "thumb_r_mj_map.cpp",
    "roles": ["Yaw", "CMC", "MCP"],
    "q_lo": [-2.617994, 0.0, 0.0], "q_hi": [2.617994, 1.570796, 1.570796],
    "term_counts": [2, 36, 120],
    "source": "firmware thumb_r_mj_map.cpp kM{1,2,3}Terms/kM{1,2,3}Coef, evaluated by "
              "cal_motorAngles_thumb (deployed J2M map). Shared by both thumbs — thumb_r/thumb_l are "
              "identical apart from the class name; Yaw range is the L/R-symmetric union. Source rev "
              "allex_control 56bf59f35bde353e3b697f49d1ea249dda0f03b6.",
}


def build_thumb() -> dict:
    cfg = _THUMB
    tables = _parse_thumb_tables(cfg["cpp"].read_text())

    counts = [tables[i][0].shape[0] for i in (1, 2, 3)]
    assert counts == cfg["term_counts"], f"thumb: unexpected term counts {counts} (expected {cfg['term_counts']})"
    for i in (1, 2, 3):
        assert tables[i][0].shape[1] == i, \
            f"thumb: kM{i} is {tables[i][0].shape[1]}-var, expected {i} (lower-triangular layout)"
    j2m = {f"m{i}": _Poly(*tables[i]) for i in (1, 2, 3)}

    lo, hi = np.asarray(cfg["q_lo"]), np.asarray(cfg["q_hi"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(lo, hi)
        for i in (1, 2, 3):
            terms, coeffs = tables[i]
            g = j2m[f"m{i}"].grad(q[:i])
            for var in range(i):
                expected = _fw_eval_derivative(terms, coeffs, q[:i], var)
                assert abs(g[var] - expected) <= 1e-11 * max(1.0, abs(expected)), \
                    f"thumb: grad mismatch dm{i}/dq{var + 1}: parsed {g[var]} vs firmware {expected}"

    node = lambda p: {"model_terms": p.terms.tolist(), "coefficients": p.coeffs.tolist()}  # noqa: E731
    return {
        "part": "thumb",
        "joint_roles": cfg["roles"],
        "joint_unit": "rad",
        "motor_unit": "rad",
        "source": cfg["source"],
        "q_fit_range_rad": {"lo": cfg["q_lo"], "hi": cfg["q_hi"]},
        "j2m": {k: node(v) for k, v in j2m.items()},
    }


_WRIST = {
    "cpp": _SRC_DIR / "arm_r_mj_map.cpp",
    "motor_fn": "calcWristJoint2MotorAngle(const Eigen::Vector2d",
    "jac_fn": "wristJac_dMdq(double qr",
    "roles": ["Roll", "Pitch"],
    "q_lo": [-0.872665, -1.396263], "q_hi": [0.872665, 1.396263],
    "term_counts": [66, 66],
    "source": "firmware arm_{r,l}_mj_map.cpp calcWristJoint2MotorAngle (ballscrew J2M, full 2x2). "
              "Shared by both wrists — arm_r/arm_l wrist maps are identical.",
}


def build_wrist() -> dict:
    cfg = _WRIST
    src = cfg["cpp"].read_text()

    ma: dict[int, list] = {0: [], 1: []}
    for idx, expr in re.findall(r"a\((\d)\)\s*\+=\s*([^;]+);", _fn_body(src, cfg["motor_fn"])):
        terms = _parse_wrist_poly(expr)
        assert len(terms) == 1, f"a({idx}) += expects one monomial per line, got {expr!r}"
        ma[int(idx)].append(terms[0])
    counts = [len(ma[0]), len(ma[1])]
    assert counts == cfg["term_counts"], f"wrist: unexpected term counts {counts} (expected {cfg['term_counts']})"
    j2m = {"m1": _Poly(*_arrays(ma[0])), "m2": _Poly(*_arrays(ma[1]))}

    jac: dict[tuple[int, int], list] = {}
    for i, j, expr in re.findall(r"dadq\((\d),(\d)\)\s*=\s*([^;]+);", _fn_body(src, cfg["jac_fn"])):
        jac[(int(i), int(j))] = _parse_wrist_poly(expr)
    assert set(jac) == {(0, 0), (0, 1), (1, 0), (1, 1)}, f"wrist: expected full 2×2 dadq, got {sorted(jac)}"

    lo, hi = np.asarray(cfg["q_lo"]), np.asarray(cfg["q_hi"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(lo, hi)
        g = [j2m["m1"].grad(q), j2m["m2"].grad(q)]
        for (i, j), tc in jac.items():
            expected = sum(c * np.prod(q ** np.array(e, dtype=float)) for e, c in tc)
            got = g[i][j]
            assert abs(got - expected) <= 1e-9 * max(1.0, abs(expected)), \
                f"wrist: grad mismatch dadq({i},{j}): parsed {got} vs firmware {expected}"

    node = lambda p: {"model_terms": p.terms.tolist(), "coefficients": p.coeffs.tolist()}  # noqa: E731
    return {
        "part": "wrist",
        "joint_roles": cfg["roles"],
        "joint_unit": "rad",
        "motor_unit": "rad",
        "source": cfg["source"],
        "q_fit_range_rad": {"lo": cfg["q_lo"], "hi": cfg["q_hi"]},
        "j2m": {k: node(v) for k, v in j2m.items()},
    }


_ELBOW = {
    "n1": 13.0 / 98.0,
    "n2": (30.2 + 1.8) / (80.0 + 1.8),
    "n3": (2.38 + 80.0) / (2.38 + 46.0),
    "roles": ["Elbow", "Yaw"],
    "q_lo": [-2.7925268, -4.799655], "q_hi": [0.0872665, 4.799655],
    "source": "firmware arm_{r,l}_mj_map.hpp elbowJoint2M_J_ (differential pulley, constant 2x2): "
              "G=[[a,b],[-a,b]], a=1/(n1*n2), b=1/(n1*n2*n3); hw_version v2 (motor_joint_helper.hpp "
              "elbow_pulley::v2) n1=13/98, n2=32/81.8, n3=82.38/48.38. "
              "Shared by both arms; Yaw fit range = symmetric L/R union.",
}


def build_elbow() -> dict:
    cfg = _ELBOW
    n1, n2, n3 = cfg["n1"], cfg["n2"], cfg["n3"]
    a = 1.0 / (n1 * n2)
    b = 1.0 / (n1 * n2 * n3)
    j2m = {"m1": _Poly([[1, 0], [0, 1]], [a, b]), "m2": _Poly([[1, 0], [0, 1]], [-a, b])}

    G = np.array([[a, b], [-a, b]])
    lo, hi = np.asarray(cfg["q_lo"]), np.asarray(cfg["q_hi"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(lo, hi)
        m = G @ q
        p = m * (n1 * n2)
        q_ep = (p[0] - p[1]) / 2.0
        q_ey = (p[0] + p[1]) * n3 / 2.0
        assert np.allclose([q_ep, q_ey], q, rtol=1e-12, atol=1e-12), \
            f"elbow: differential round-trip mismatch {[q_ep, q_ey]} vs {q}"

    node = lambda p: {"model_terms": p.terms.tolist(), "coefficients": p.coeffs.tolist()}  # noqa: E731
    return {
        "part": "elbow",
        "joint_roles": cfg["roles"],
        "joint_unit": "rad",
        "motor_unit": "rad",
        "source": cfg["source"],
        "q_fit_range_rad": {"lo": cfg["q_lo"], "hi": cfg["q_hi"]},
        "j2m": {k: node(v) for k, v in j2m.items()},
    }


_SHOULDER = {
    "ratio": 40.23846154,
    "roles": ["Pitch", "Roll", "Yaw"],
    "q_lo": [-3.316126, -3.752458, -3.316126], "q_hi": [1.745329, 3.752458, 3.316126],
    "source": "firmware arm_{r,l}_mj_map.hpp shoulder_actu_ratio_ (per-joint scalar reduction, "
              "diagonal): m_i = r*q_i, r = 40.23846154. Authored as a degenerate 3-DOF lower-"
              "triangular map (hand-kernel layout). Shared by both arms; Roll/Yaw fit ranges = "
              "symmetric L/R unions.",
}


def build_shoulder() -> dict:
    cfg = _SHOULDER
    r = cfg["ratio"]
    j2m = {"m1": _Poly([[1]], [r]), "m2": _Poly([[0, 1]], [r]), "m3": _Poly([[0, 0, 1]], [r])}

    G = np.diag([r, r, r])
    lo, hi = np.asarray(cfg["q_lo"]), np.asarray(cfg["q_hi"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(lo, hi)
        m = np.array([j2m["m1"].grad(q[:1])[0] * q[0],
                      j2m["m2"].grad(q[:2])[1] * q[1],
                      j2m["m3"].grad(q[:3])[2] * q[2]])
        assert np.allclose(m, r * q, rtol=1e-12), f"shoulder: J2M mismatch {m} vs {r * q}"
        tau_j = rng.uniform(-120.0, 120.0, 3)
        assert np.allclose(np.linalg.inv(G).T @ tau_j, tau_j / r, rtol=1e-12), \
            "shoulder: G^-T does not reproduce the cpp tau_j / ratio relation"

    node = lambda p: {"model_terms": p.terms.tolist(), "coefficients": p.coeffs.tolist()}  # noqa: E731
    return {
        "part": "shoulder",
        "joint_roles": cfg["roles"],
        "joint_unit": "rad",
        "motor_unit": "rad",
        "source": cfg["source"],
        "q_fit_range_rad": {"lo": cfg["q_lo"], "hi": cfg["q_hi"]},
        "j2m": {k: node(v) for k, v in j2m.items()},
    }


def _compact_int_rows(s: str) -> str:
    return re.sub(r'\[\s*((?:-?\d+\s*,\s*)*-?\d+)\s*\]',
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", s)


def main() -> None:
    for build, name, check in ((build_thumb, "thumb.json", "parsed J2M gradient == firmware evalPolynomialDerivative"),
                               (build_wrist, "wrist.json", "parsed J2M gradient == firmware wristJac_dMdq"),
                               (build_elbow, "elbow.json", "G inverse == cpp pulley differential round-trip"),
                               (build_shoulder, "shoulder.json", "G^-T == cpp tau_j / ratio relation")):
        out = build()
        path = _HERE / name
        with open(path, "w") as f:
            f.write(_compact_int_rows(json.dumps(out, indent=2)))
            f.write("\n")
        counts = [len(out["j2m"][k]["coefficients"]) for k in sorted(out["j2m"])]
        print(f"wrote {path}  (self-check: {check}, 200 configs OK; term counts {counts})")


if __name__ == "__main__":
    main()
