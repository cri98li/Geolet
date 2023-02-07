#--------------------------------------------------------------------- from https://github.com/nbro/aal
import numpy as np
from numpy.linalg import norm


def movement_vec(t, p):
    """Movement vector at time t of the sequence p."""
    assert 0 < t < len(p)
    return np.subtract(p[t], p[t - 1])


def to3d(a, b):
    """Assume that a and b are on the xy plane."""
    return (a[0], a[1], 0), (b[0], b[1], 0)


def sign(v_t, v_ref=(1, 0)):
    """It returns 1 if v_ref is oriented anti-clockwise with respect to v_t.
    If v_ref oriented clockwise with respect to v_t, returns -1.
    If v_t and v_ref are parallel, it returns 0."""
    a, b = to3d(v_t, v_ref)
    # See: https://math.stackexchange.com/a/285369/168764.
    zd = np.dot(np.cross(a, b), [0, 0, 1])
    if zd > 0:
        # b is anti-clockwise with respect to a.
        # See the right-hand rule: if b is oriented anti-clockwise
        # with respect to a, then axb is pointing upwards, that is,
        # it is positive.
        return 1
    if zd < 0:
        # b is clockwise w.r.t. a.
        return -1
    if zd == 0:
        # a and b are parallel
        return 0


def angle(v_t, v_ref=(1, 0)):
    """Angle between the movement vector v_t and the reference vector (1, 0),
    which means that the angles will be exact angles."""
    # The angle is between 0 and pi radians, given that the range of arc-cosine
    # is [0, 180], while its domain is [-1, 1].
    angle = np.arccos(np.dot(v_t, v_ref) / (norm(v_t) * norm(v_ref)))
    assert sign(v_t) == 0 if (
            np.isclose(angle, 0) or np.isclose(angle, np.pi)) else sign(
        v_t) in (-1, 1)
    return sign(v_t) * angle


def iterative_normalization(angles, k=5):
    """Iterative modulo-π normalization algorithm."""
    # Make sure that angles is a NumPy array.
    angles = np.asarray(angles)
    average_angle = np.average(angles)

    # Repeat until stability.
    for i in range(k):
        if average_angle != np.average(angles):
            break

        average_angle = np.average(angles)
        angles = angles - average_angle

        # Wrap angles in [-pi, pi].
        for j in range(0, len(angles)):
            if angles[j] < (-np.pi):
                angles[j] = 2 * np.pi + angles[j]
            if angles[j] > np.pi:
                angles[j] = (-2) * np.pi + angles[j]

    return list(angles)


def get(a, i=0):
    return [elem[i] for elem in a]


def normalise_scale(aals):
    """Scale-invariance is achieved by dividing the arc length of each
    movement vector by the total arc length of the trajectory."""
    new_aals = []
    for aal in aals:
        arc_length_sum = sum(get(aal, 1))
        new_aals.append(
            [(angle, arc_length / arc_length_sum) for angle, arc_length in
             aal])
    return new_aals


def normalise_rotation(aals):
    new_aals = []
    for aal in aals:
        new_aals.append(
            list(zip(iterative_normalization(get(aal)), get(aal, 1))))
    return new_aals


def to_aal(sequences, scale_invariant=True, rotation_invariant=True):
    """Returns a representation of the sequences in translation-invariant
    space, which is called the AAL (angle-arc-length) space.
    If scale_invariant and rotation_invariant are true, then it is also scale
    and translation-invariant, respectively."""
    for sequence in sequences:
        assert all(len(x) == 2 for x in sequence)

    # It will contain a list of the sequences in AAL space.
    aals = [[] for _ in range(len(sequences))]

    for i, sequence in enumerate(sequences):
        for t in range(1, len(sequence)):
            v_t = movement_vec(t, sequence)
            aals[i].append((angle(v_t), norm(v_t)))

    if scale_invariant:
        aals = normalise_scale(aals)

    if rotation_invariant:
        aals = normalise_rotation(aals)

    for aal, sequence in zip(aals, sequences):
        assert len(aal) == len(sequence) - 1

    return tuple(aals)


def d_warp(a, b):
    return min(norm(np.subtract(a, b)), 2 * np.pi - norm(np.subtract(a, b)))