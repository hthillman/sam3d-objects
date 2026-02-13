"""Lightweight replacements for pytorch3d.structures.

Provides Meshes and Pointclouds containers that store the data
needed by sam3d_objects without depending on pytorch3d's CUDA
extensions.  The API surface is intentionally minimal — only methods
actually called in the codebase are implemented.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


class Meshes:
    """Minimal Meshes container compatible with sam3d_objects usage.

    Stores batched verts, faces, and optional textures.  Implements:
        verts_packed, faces_packed, detach, cpu, textures access.
    """

    def __init__(
        self,
        verts: Union[List[Tensor], Tensor],
        faces: Union[List[Tensor], Tensor],
        textures=None,
    ) -> None:
        # Normalise to lists of tensors
        if isinstance(verts, Tensor):
            if verts.dim() == 2:
                verts = [verts]
            else:
                verts = [verts[i] for i in range(verts.shape[0])]
        if isinstance(faces, Tensor):
            if faces.dim() == 2:
                faces = [faces]
            else:
                faces = [faces[i] for i in range(faces.shape[0])]

        self._verts_list = [v.float() for v in verts]
        self._faces_list = [f.long() for f in faces]
        self.textures = textures
        self._device = self._verts_list[0].device

    # -- packed representations (concatenate across batch) -------------------

    def verts_packed(self) -> Tensor:
        return torch.cat(self._verts_list, dim=0)

    def faces_packed(self) -> Tensor:
        """Return faces with vertex indices offset per mesh in the batch."""
        if len(self._faces_list) == 1:
            return self._faces_list[0]
        faces = []
        offset = 0
        for v, f in zip(self._verts_list, self._faces_list):
            faces.append(f + offset)
            offset += v.shape[0]
        return torch.cat(faces, dim=0)

    def verts_list(self) -> List[Tensor]:
        return self._verts_list

    def faces_list(self) -> List[Tensor]:
        return self._faces_list

    # -- utility -------------------------------------------------------------

    def detach(self) -> "Meshes":
        verts = [v.detach() for v in self._verts_list]
        faces = [f.detach() for f in self._faces_list]
        tex = self.textures
        if tex is not None and hasattr(tex, "detach"):
            tex = tex.detach()
        return Meshes(verts=verts, faces=faces, textures=tex)

    def cpu(self) -> "Meshes":
        verts = [v.cpu() for v in self._verts_list]
        faces = [f.cpu() for f in self._faces_list]
        tex = self.textures
        if tex is not None and hasattr(tex, "cpu"):
            tex = tex.cpu()
        return Meshes(verts=verts, faces=faces, textures=tex)

    def to(self, device) -> "Meshes":
        verts = [v.to(device) for v in self._verts_list]
        faces = [f.to(device) for f in self._faces_list]
        tex = self.textures
        if tex is not None and hasattr(tex, "to"):
            tex = tex.to(device)
        return Meshes(verts=verts, faces=faces, textures=tex)

    @property
    def device(self):
        return self._device

    def __len__(self) -> int:
        return len(self._verts_list)


def join_meshes_as_scene(meshes: Meshes) -> Meshes:
    """Join a batch of meshes into a single mesh (scene)."""
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    tex = meshes.textures
    return Meshes(verts=[verts], faces=[faces], textures=tex)


# ---------------------------------------------------------------------------
# Pointclouds
# ---------------------------------------------------------------------------


class Pointclouds:
    """Minimal Pointclouds container compatible with sam3d_objects usage.

    Stores batched points and optional per-point features (colours).
    """

    def __init__(
        self,
        points: Union[List[Tensor], Tensor],
        normals: Optional[Union[List[Tensor], Tensor]] = None,
        features: Optional[Union[List[Tensor], Tensor]] = None,
    ) -> None:
        if isinstance(points, Tensor):
            if points.dim() == 2:
                points = [points]
            elif points.dim() == 3:
                points = [points[i] for i in range(points.shape[0])]
        self._points_list = [p.float() for p in points]

        if features is not None:
            if isinstance(features, Tensor):
                if features.dim() == 2:
                    features = [features]
                elif features.dim() == 3:
                    features = [features[i] for i in range(features.shape[0])]
            self._features_list = [f.float() for f in features]
        else:
            self._features_list = None

        if normals is not None:
            if isinstance(normals, Tensor):
                if normals.dim() == 2:
                    normals = [normals]
                elif normals.dim() == 3:
                    normals = [normals[i] for i in range(normals.shape[0])]
            self._normals_list = [n.float() for n in normals]
        else:
            self._normals_list = None

    def points_packed(self) -> Tensor:
        return torch.cat(self._points_list, dim=0)

    def points_list(self) -> List[Tensor]:
        return self._points_list

    def features_packed(self) -> Optional[Tensor]:
        if self._features_list is None:
            return None
        return torch.cat(self._features_list, dim=0)

    def features_list(self) -> Optional[List[Tensor]]:
        return self._features_list

    def normals_packed(self) -> Optional[Tensor]:
        if self._normals_list is None:
            return None
        return torch.cat(self._normals_list, dim=0)

    def __len__(self) -> int:
        return len(self._points_list)
