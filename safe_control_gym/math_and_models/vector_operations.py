"""Vector operations with numpy 

unit_vector, projection_matrix from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/transformation.py
"""
import math
import numpy as np 



def unit_vector(data, axis=None, out=None):
  """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]
    """
  if out is None:
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
      data /= math.sqrt(np.dot(data, data))
      return data
  else:
    if out is not data:
      out[:] = np.array(data, copy=False)
    data = out
  length = np.atleast_1d(np.sum(data * data, axis))
  np.sqrt(length, length)
  if axis is not None:
    length = np.expand_dims(length, axis)
  data /= length
  if out is None:
    return data


def projection_matrix(point, normal, direction=None, perspective=None, pseudo=False):
  """Return matrix to project onto plane defined by point and normal.
    Using either perspective point, projection direction, or none of both.
    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).
    >>> P = projection_matrix([0, 0, 0], [1, 0, 0])
    >>> np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])
    True
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> P1 = projection_matrix(point, normal, direction=direct)
    >>> P2 = projection_matrix(point, normal, perspective=persp)
    >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, np.dot(P0, P3))
    True
    >>> P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
    >>> v0 = (np.random.rand(4, 5) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(P, v0)
    >>> np.allclose(v1[1], v0[1])
    True
    >>> np.allclose(v1[0], 3-v1[1])
    True
    """
  M = np.identity(4)
  point = np.array(point[:3], dtype=np.float64, copy=False)
  normal = unit_vector(normal[:3])
  if perspective is not None:
    # perspective projection
    perspective = np.array(perspective[:3], dtype=np.float64, copy=False)
    M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective - point, normal)
    M[:3, :3] -= np.outer(perspective, normal)
    if pseudo:
      # preserve relative depth
      M[:3, :3] -= np.outer(normal, normal)
      M[:3, 3] = np.dot(point, normal) * (perspective + normal)
    else:
      M[:3, 3] = np.dot(point, normal) * perspective
    M[3, :3] = -normal
    M[3, 3] = np.dot(perspective, normal)
  elif direction is not None:
    # parallel projection
    direction = np.array(direction[:3], dtype=np.float64, copy=False)
    scale = np.dot(direction, normal)
    M[:3, :3] -= np.outer(direction, normal) / scale
    M[:3, 3] = direction * (np.dot(point, normal) / scale)
  else:
    # orthogonal projection
    M[:3, :3] -= np.outer(normal, normal)
    M[:3, 3] = np.dot(point, normal) * normal
  return M