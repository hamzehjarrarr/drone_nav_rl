"""PyTorch port of the simulator in `sim/`, for training on a GPU (CUDA or Apple MPS).

Same physics, sensor, rewards and curriculum as the NumPy version, but every
environment's state lives in GPU tensors, so thousands of drones step at once.
The NumPy simulator stays the reference: the app and evaluation use it, and
tests/test_gpu_equivalence.py checks that both versions agree.
"""

from simgpu.env import TorchDroneEnv

__all__ = ["TorchDroneEnv"]