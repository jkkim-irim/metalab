"""Engine-agnostic primitive API (building blocks for obs/reward).

- :mod:`sim.metalab.api.state`     — StateAdapter Protocol (normalized read implemented by each engine; layer 1)
- :mod:`sim.metalab.api.frames`    — quat ops (wxyz) and frame transforms (layer 2, pure torch)
- :mod:`sim.metalab.api.keypoints` — keypoint cage and max-dist metric
- :mod:`sim.metalab.api.shaping`   — min-dist progress, exp kernel, lift gate
- :mod:`sim.metalab.api.contact`   — pad-vs-nail fingertip contact predicate

obs/reward terms (:mod:`sim.metalab.terms.obs` / :mod:`sim.metalab.terms.reward`) only **compose** these primitives.
This package does not import any engine (gs/isaaclab).
"""
