"""Robot part library — ``<family>/<name>.yaml`` (with physics props) → :class:`~sim.metalab.contract.spec.RobotSpec`."""
from sim.metalab.contract.loader import load_robot as load

__all__ = ["load"]
