"""Pure-PyTorch reimplementation of pytorch3d.transforms.

Covers the subset used by sam3d_objects: quaternion conversions,
Transform3d composable transforms, and convenience constructors.

Conventions match pytorch3d exactly:
  - Quaternions: wxyz (real part first)
  - Transform3d stores (N, 4, 4) matrices in row-major layout
  - Points are right-multiplied: p' = p @ M[:3, :3] + M[3, :3]
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Quaternion conversions
# ---------------------------------------------------------------------------


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert wxyz quaternions to 3x3 rotation matrices.

    Args:
        quaternions: (..., 4) tensor, real part first.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(-1)

    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = w * tx, w * ty, w * tz
    txx, txy, txz = x * tx, x * ty, x * tz
    tyy, tyz, tzz = y * ty, y * tz, z * tz

    matrix = torch.stack(
        [
            1 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1 - (txx + tyy),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)

    return matrix


def _sqrt_positive_part(x: Tensor) -> Tensor:
    """sqrt(max(0, x)) but with a gradient-safe implementation."""
    return torch.sqrt(torch.clamp(x, min=0.0))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert 3x3 rotation matrices to wxyz quaternions.

    Uses the numerically stable method from pytorch3d that selects
    the best-conditioned quaternion component to compute first.

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions in wxyz convention, real part non-negative.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) matrix, got {matrix.shape}")

    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # For each element, pick the component with largest absolute value
    quat_by_max = torch.stack(
        [
            torch.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[:, 1] ** 2, m01 + m10, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m01 + m10, q_abs[:, 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[:, 3] ** 2], dim=-1),
        ],
        dim=1,
    )

    max_idx = q_abs.argmax(dim=-1)
    quat = quat_by_max[torch.arange(m.shape[0], device=m.device), max_idx]
    quat = quat / (2.0 * q_abs[torch.arange(m.shape[0], device=m.device), max_idx].unsqueeze(-1))

    # Ensure non-negative real part
    quat = quat * quat[:, :1].sign()

    return quat.reshape(*batch_shape, 4)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product of two wxyz quaternions.

    Args:
        a: (..., 4) quaternions in wxyz.
        b: (..., 4) quaternions in wxyz.

    Returns:
        (..., 4) product quaternion in wxyz.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return torch.stack([ow, ox, oy, oz], dim=-1)


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle representation to wxyz quaternions.

    Args:
        axis_angle: (..., 3) vectors where direction is axis and
                    magnitude is angle in radians.

    Returns:
        (..., 4) quaternions in wxyz.
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6

    # For small angles, use Taylor expansion
    small = (angles < eps).squeeze(-1)
    sin_half = torch.sin(half_angles)
    cos_half = torch.cos(half_angles)

    # Normalize axis (avoid division by zero for near-zero angles)
    axis = axis_angle / angles.clamp(min=eps)

    w = cos_half.squeeze(-1)
    xyz = axis * sin_half

    quat = torch.cat([w.unsqueeze(-1), xyz], dim=-1)

    # For near-zero angles: quaternion ≈ [1, axis_angle/2]
    if small.any():
        quat[small] = torch.cat(
            [
                torch.ones(*axis_angle[small].shape[:-1], 1, device=axis_angle.device),
                axis_angle[small] * 0.5,
            ],
            dim=-1,
        )

    return quat


# ---------------------------------------------------------------------------
# Transform3d
# ---------------------------------------------------------------------------

Device = Union[str, torch.device]


class Transform3d:
    """Composable 3D transformation stored as (N, 4, 4) homogeneous matrices.

    Compatible with pytorch3d.transforms.Transform3d's public API.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        matrix: Optional[Tensor] = None,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)

        if matrix is not None:
            if matrix.dim() == 2:
                matrix = matrix.unsqueeze(0)
            if matrix.shape[-2:] != (4, 4):
                raise ValueError(f"Expected (N, 4, 4) matrix, got {matrix.shape}")
            self._matrix = matrix.to(device=device, dtype=dtype)
        else:
            self._matrix = torch.eye(4, dtype=dtype, device=device).unsqueeze(0)

        self.device = device
        self.dtype = dtype

    def get_matrix(self) -> Tensor:
        """Return the (N, 4, 4) transformation matrix."""
        return self._matrix

    def compose(self, *others: "Transform3d") -> "Transform3d":
        """Compose with other transforms: self then others (left to right)."""
        mat = self._matrix
        for other in others:
            mat = torch.bmm(mat, other._matrix)
        return Transform3d(dtype=self.dtype, device=self.device, matrix=mat)

    def transform_points(
        self, points: Tensor, eps: Optional[float] = None
    ) -> Tensor:
        """Transform points by the stored matrix.

        Args:
            points: (P, 3) or (N, P, 3) tensor.
            eps: optional epsilon for perspective division.

        Returns:
            Transformed points, same shape.
        """
        mat = self._matrix

        if points.dim() == 2:
            # (P, 3) -> (1, P, 3)
            points = points.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Homogeneous coordinates
        ones = torch.ones(
            *points.shape[:-1], 1, device=points.device, dtype=points.dtype
        )
        points_h = torch.cat([points, ones], dim=-1)  # (N, P, 4)

        # Right-multiply: points_h @ M gives (N, P, 4)
        # Use matmul instead of bmm to allow broadcasting (e.g. 512 points, 1 matrix)
        transformed = torch.matmul(points_h, mat)

        # Perspective division
        w = transformed[..., 3:]
        if eps is not None:
            w = w.clamp(min=eps)
        result = transformed[..., :3] / w

        if squeeze:
            result = result.squeeze(0)

        return result

    def transform_normals(self, normals: Tensor) -> Tensor:
        """Transform normals using the inverse transpose."""
        mat = self._matrix[:, :3, :3]  # (N, 3, 3)
        inv_t = torch.inverse(mat).transpose(-1, -2)

        if normals.dim() == 2:
            normals = normals.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        result = torch.bmm(normals, inv_t)

        if squeeze:
            result = result.squeeze(0)

        return result

    def inverse(self, invert_composed: bool = False) -> "Transform3d":
        """Return the inverse transform."""
        inv = torch.inverse(self._matrix)
        return Transform3d(dtype=self.dtype, device=self.device, matrix=inv)

    def translate(self, *args, **kwargs) -> "Transform3d":
        """Return a new transform with translation appended."""
        t = Translate(*args, device=self.device, dtype=self.dtype, **kwargs)
        return self.compose(t)

    def scale(self, *args, **kwargs) -> "Transform3d":
        """Return a new transform with scale appended."""
        s = Scale(*args, device=self.device, dtype=self.dtype, **kwargs)
        return self.compose(s)

    def rotate(self, *args, **kwargs) -> "Transform3d":
        """Return a new transform with rotation appended."""
        r = Rotate(*args, device=self.device, dtype=self.dtype, **kwargs)
        return self.compose(r)

    def clone(self) -> "Transform3d":
        return Transform3d(
            dtype=self.dtype, device=self.device, matrix=self._matrix.clone()
        )

    def to(
        self,
        device: Device = "cpu",
        copy: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Transform3d":
        new_dtype = dtype if dtype is not None else self.dtype
        if isinstance(device, str):
            device = torch.device(device)
        mat = self._matrix.to(device=device, dtype=new_dtype)
        if copy:
            mat = mat.clone()
        return Transform3d(dtype=new_dtype, device=device, matrix=mat)

    def cpu(self) -> "Transform3d":
        return self.to("cpu")

    def cuda(self) -> "Transform3d":
        return self.to("cuda")

    def __len__(self) -> int:
        return self._matrix.shape[0]

    def __getitem__(self, index) -> "Transform3d":
        mat = self._matrix[index]
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
        return Transform3d(dtype=self.dtype, device=self.device, matrix=mat)

    def __repr__(self) -> str:
        return f"Transform3d(N={len(self)}, device={self.device}, dtype={self.dtype})"


class Translate(Transform3d):
    """Create a translation transform."""

    def __init__(
        self,
        x: Union[Tensor, float, Sequence],
        y: Optional[Union[Tensor, float]] = None,
        z: Optional[Union[Tensor, float]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        if isinstance(x, Tensor) and x.dim() >= 1 and x.shape[-1] == 3 and y is None:
            xyz = x.to(dtype=dtype)
            if device is not None:
                xyz = xyz.to(device=device)
        else:
            if device is None:
                device = x.device if isinstance(x, Tensor) else "cpu"
            _x = torch.as_tensor(x, dtype=dtype, device=device).reshape(-1)
            _y = torch.as_tensor(y if y is not None else 0.0, dtype=dtype, device=device).reshape(-1)
            _z = torch.as_tensor(z if z is not None else 0.0, dtype=dtype, device=device).reshape(-1)
            N = max(len(_x), len(_y), len(_z))
            xyz = torch.stack(
                [_x.expand(N), _y.expand(N), _z.expand(N)], dim=-1
            )

        N = xyz.shape[0] if xyz.dim() > 1 else 1
        if xyz.dim() == 1:
            xyz = xyz.unsqueeze(0)

        dev = xyz.device
        mat = torch.eye(4, dtype=dtype, device=dev).unsqueeze(0).expand(N, -1, -1).clone()
        mat[:, 3, :3] = xyz

        super().__init__(dtype=dtype, device=dev, matrix=mat)


class Scale(Transform3d):
    """Create a scale transform."""

    def __init__(
        self,
        x: Union[Tensor, float, Sequence],
        y: Optional[Union[Tensor, float]] = None,
        z: Optional[Union[Tensor, float]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        if isinstance(x, Tensor) and x.dim() >= 1 and x.shape[-1] == 3 and y is None:
            xyz = x.to(dtype=dtype)
            if device is not None:
                xyz = xyz.to(device=device)
        elif y is None and z is None:
            # Uniform scale
            if device is None:
                device = x.device if isinstance(x, Tensor) else "cpu"
            _x = torch.as_tensor(x, dtype=dtype, device=device).reshape(-1)
            xyz = _x.unsqueeze(-1).expand(-1, 3)
        else:
            if device is None:
                device = x.device if isinstance(x, Tensor) else "cpu"
            _x = torch.as_tensor(x, dtype=dtype, device=device).reshape(-1)
            _y = torch.as_tensor(y if y is not None else 1.0, dtype=dtype, device=device).reshape(-1)
            _z = torch.as_tensor(z if z is not None else 1.0, dtype=dtype, device=device).reshape(-1)
            N = max(len(_x), len(_y), len(_z))
            xyz = torch.stack(
                [_x.expand(N), _y.expand(N), _z.expand(N)], dim=-1
            )

        N = xyz.shape[0] if xyz.dim() > 1 else 1
        if xyz.dim() == 1:
            xyz = xyz.unsqueeze(0)

        dev = xyz.device
        mat = torch.zeros(N, 4, 4, dtype=dtype, device=dev)
        mat[:, 0, 0] = xyz[:, 0]
        mat[:, 1, 1] = xyz[:, 1]
        mat[:, 2, 2] = xyz[:, 2]
        mat[:, 3, 3] = 1.0

        super().__init__(dtype=dtype, device=dev, matrix=mat)


class Rotate(Transform3d):
    """Create a rotation transform from a rotation matrix."""

    def __init__(
        self,
        R: Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        if R.dim() == 2:
            R = R.unsqueeze(0)

        R = R.to(dtype=dtype)
        if device is not None:
            R = R.to(device=device)

        N = R.shape[0]
        dev = R.device
        mat = torch.eye(4, dtype=dtype, device=dev).unsqueeze(0).expand(N, -1, -1).clone()
        mat[:, :3, :3] = R

        super().__init__(dtype=dtype, device=dev, matrix=mat)
