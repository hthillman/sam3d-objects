"""Stub renderer classes that raise when the full pytorch3d is needed.

The mesh renderer pipeline (rasteriser, shaders, cameras, blend
params, textures) requires pytorch3d's CUDA extensions, which can't
be built in Scope's uv-based installer.  These stubs let the rest of
the codebase *import* without error; they only raise when someone
actually tries to construct or call the renderer objects.

Functions/classes that can be implemented in pure-PyTorch (like
``look_at_view_transform``) are implemented for real.
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor


_MISSING = (
    "This feature requires the full pytorch3d package with CUDA extensions. "
    "Install pytorch3d to use mesh rendering / layout post-optimization."
)


# ---------------------------------------------------------------------------
# look_at_view_transform — pure PyTorch, no CUDA needed
# ---------------------------------------------------------------------------


def look_at_view_transform(
    dist: Union[float, Tensor] = 1.0,
    elev: Union[float, Tensor] = 0.0,
    azim: Union[float, Tensor] = 0.0,
    degrees: bool = True,
    eye: Optional[Union[Sequence, Tensor, np.ndarray]] = None,
    at: Union[Sequence, Tensor, np.ndarray] = ((0.0, 0.0, 0.0),),
    up: Union[Sequence, Tensor, np.ndarray] = ((0.0, 1.0, 0.0),),
    device: Union[str, torch.device] = "cpu",
):
    """Pure-PyTorch implementation of pytorch3d.renderer.look_at_view_transform."""
    if eye is not None:
        if isinstance(eye, np.ndarray):
            eye = torch.from_numpy(eye).float()
        elif not isinstance(eye, Tensor):
            eye = torch.tensor(eye, dtype=torch.float32)
        eye = eye.to(device)
    else:
        if degrees:
            elev = math.radians(elev) if isinstance(elev, (int, float)) else elev * (math.pi / 180.0)
            azim = math.radians(azim) if isinstance(azim, (int, float)) else azim * (math.pi / 180.0)
        if isinstance(dist, (int, float)):
            dist = torch.tensor([dist], dtype=torch.float32, device=device)
        if isinstance(elev, (int, float)):
            elev = torch.tensor([elev], dtype=torch.float32, device=device)
        if isinstance(azim, (int, float)):
            azim = torch.tensor([azim], dtype=torch.float32, device=device)
        x = dist * torch.cos(elev) * torch.sin(azim)
        y = dist * torch.sin(elev)
        z = dist * torch.cos(elev) * torch.cos(azim)
        eye = torch.stack([x, y, z], dim=-1)

    if isinstance(at, np.ndarray):
        at = torch.from_numpy(at).float()
    elif not isinstance(at, Tensor):
        at = torch.tensor(at, dtype=torch.float32)
    at = at.to(device)

    if isinstance(up, np.ndarray):
        up = torch.from_numpy(up).float()
    elif not isinstance(up, Tensor):
        up = torch.tensor(up, dtype=torch.float32)
    up = up.to(device)

    if eye.dim() == 1:
        eye = eye.unsqueeze(0)
    if at.dim() == 1:
        at = at.unsqueeze(0)
    if up.dim() == 1:
        up = up.unsqueeze(0)

    # pytorch3d convention: camera looks from eye toward at
    z_axis = at - eye
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)

    x_axis = torch.cross(z_axis, up, dim=-1)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)

    y_axis = torch.cross(x_axis, z_axis, dim=-1)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)

    # Rotation matrix (world → camera)
    R = torch.stack([x_axis, y_axis, -z_axis], dim=-1)  # (N, 3, 3)

    # Translation
    T = -torch.bmm(eye.unsqueeze(1), R).squeeze(1)  # (N, 3)

    return R, T


# ---------------------------------------------------------------------------
# Stub camera classes
# ---------------------------------------------------------------------------


class CamerasBase:
    """Stub base class so isinstance checks work."""

    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


class PerspectiveCameras(CamerasBase):
    """Stub PerspectiveCameras.

    Accepts ``R`` and ``T`` keyword arguments and stores them, which is
    enough for the visualization code in scene_visualizer.py. Methods
    that need the full pytorch3d will raise.
    """

    def __init__(self, R=None, T=None, **kwargs):
        # Bypass the parent __init__ that raises — we actually want this
        # to be constructible for the viz code path.
        self.R = R
        self.T = T
        self._kwargs = kwargs

    def get_world_to_view_transform(self):
        raise ImportError(_MISSING)


# ---------------------------------------------------------------------------
# Stub renderer / rasterizer / shader / settings
# ---------------------------------------------------------------------------


class RasterizationSettings:
    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


class MeshRasterizer:
    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


class MeshRenderer:
    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


class SoftSilhouetteShader:
    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


class BlendParams:
    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


# ---------------------------------------------------------------------------
# Stub texture classes
# ---------------------------------------------------------------------------


class TexturesVertex:
    """Minimal TexturesVertex that stores verts_features for viz code."""

    def __init__(self, verts_features=None, **kwargs):
        self._verts_features = verts_features

    def verts_features_packed(self):
        if self._verts_features is None:
            return None
        if isinstance(self._verts_features, list):
            return torch.cat(self._verts_features, dim=0)
        return self._verts_features

    def detach(self):
        if self._verts_features is None:
            return TexturesVertex(verts_features=None)
        if isinstance(self._verts_features, list):
            return TexturesVertex(
                verts_features=[v.detach() for v in self._verts_features]
            )
        return TexturesVertex(verts_features=self._verts_features.detach())

    def cpu(self):
        if self._verts_features is None:
            return TexturesVertex(verts_features=None)
        if isinstance(self._verts_features, list):
            return TexturesVertex(
                verts_features=[v.cpu() for v in self._verts_features]
            )
        return TexturesVertex(verts_features=self._verts_features.cpu())

    def to(self, device):
        if self._verts_features is None:
            return TexturesVertex(verts_features=None)
        if isinstance(self._verts_features, list):
            return TexturesVertex(
                verts_features=[v.to(device) for v in self._verts_features]
            )
        return TexturesVertex(verts_features=self._verts_features.to(device))


class TexturesAtlas:
    """Stub TexturesAtlas — raises on construction."""

    def __init__(self, **kwargs):
        raise ImportError(_MISSING)

    def atlas_packed(self):
        raise ImportError(_MISSING)


# ---------------------------------------------------------------------------
# Stub ray-bundle classes
# ---------------------------------------------------------------------------


class RayBundle:
    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


class HeterogeneousRayBundle:
    def __init__(self, **kwargs):
        raise ImportError(_MISSING)


def ray_bundle_to_ray_points(**kwargs):
    raise ImportError(_MISSING)


# ---------------------------------------------------------------------------
# Stub camera_utils
# ---------------------------------------------------------------------------


def camera_to_eye_at_up(world_to_view_transform):
    raise ImportError(_MISSING)


# ---------------------------------------------------------------------------
# Stub plotly_vis helpers
# ---------------------------------------------------------------------------


AxisArgs = namedtuple(
    "AxisArgs",
    [
        "showgrid",
        "zeroline",
        "showline",
        "ticks",
        "showticklabels",
        "backgroundcolor",
        "showaxeslabels",
    ],
    defaults=(True, False, False, "outside", False, "rgb(230, 230, 230)", False),
)

Lighting = namedtuple(
    "Lighting",
    [
        "ambient",
        "diffuse",
        "fresnel",
        "specular",
        "roughness",
        "facenormalsepsilon",
        "vertexnormalsepsilon",
    ],
    defaults=(0.5, 1.0, 0.2, 0.0, 0.5, 1e-6, 1e-12),
)


def _add_camera_trace(fig, cameras, trace_name, subplot_idx, ncols, camera_scale=0.3):
    """Stub — camera trace rendering requires full pytorch3d."""
    import warnings
    warnings.warn(
        f"Skipping camera trace '{trace_name}': pytorch3d renderer not available."
    )


def _add_pointcloud_trace(
    fig,
    pointclouds,
    trace_name,
    subplot_idx,
    ncols,
    max_points=20000,
    marker_size=1,
):
    """Render a Pointclouds object as a plotly Scatter3d trace."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    points = pointclouds.points_packed()
    features = pointclouds.features_packed()

    if points.shape[0] > max_points:
        idx = torch.randperm(points.shape[0])[:max_points]
        points = points[idx]
        if features is not None:
            features = features[idx]

    points = points.detach().cpu()
    color = None
    if features is not None:
        features = features.detach().cpu()
        if features.shape[-1] == 3:
            rgb = (features.clamp(0, 1) * 255).to(torch.uint8)
            color = [f"rgb({r},{g},{b})" for r, g, b in rgb.tolist()]

    row = subplot_idx // ncols + 1
    col = subplot_idx % ncols + 1
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color),
            name=trace_name,
        ),
        row=row,
        col=col,
    )

    # Update axis bounds
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]
    center = points.mean(0)
    max_expand = (points.max(0)[0] - points.min(0)[0]).max()
    _update_axes_bounds(center, max_expand, current_layout)


def _add_ray_bundle_trace(*args, **kwargs):
    """Stub — ray bundle rendering requires full pytorch3d."""
    import warnings
    warnings.warn("Skipping ray bundle trace: pytorch3d renderer not available.")


def _is_ray_bundle(obj):
    """Check if an object is a RayBundle or HeterogeneousRayBundle."""
    return isinstance(obj, (RayBundle, HeterogeneousRayBundle))


def _scale_camera_to_bounds(value, range_bounds, is_position=True):
    """Scale a camera coordinate to plotly bounds."""
    rmin, rmax = range_bounds
    if rmax - rmin < 1e-8:
        return 0.0
    if is_position:
        return (value - rmin) / (rmax - rmin) * 2 - 1
    else:
        return value / (rmax - rmin) * 2


def _update_axes_bounds(center, max_expand, current_layout):
    """Update plotly subplot axis ranges to fit the data."""
    if isinstance(center, Tensor):
        center = center.detach().cpu()
    half = float(max_expand) / 2.0
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    for axis_name, c in [("xaxis", cx), ("yaxis", cy), ("zaxis", cz)]:
        axis = current_layout[axis_name]
        cur_range = axis.get("range", [c - half, c + half])
        new_min = min(cur_range[0], c - half)
        new_max = max(cur_range[1], c + half)
        axis["range"] = [new_min, new_max]
