"""Lightweight shim replacing pytorch3d for environments where it can't be built.

Provides pure-PyTorch implementations of the transform functions and
lightweight data containers used by sam3d_objects. The mesh renderer
is stubbed — it raises ImportError if actually invoked, since it
requires the full pytorch3d CUDA build.
"""
