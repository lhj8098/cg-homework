import taichi as ti
import math

@ti.func
def get_model_matrix(angle):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    x, y, z = eye_pos.x, eye_pos.y, eye_pos.z
    return ti.Matrix([
        [1.0, 0.0, 0.0, -x],
        [0.0, 1.0, 0.0, -y],
        [0.0, 0.0, 1.0, -z],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    fov_rad = eye_fov * math.pi / 180.0
    n = -zNear
    f = -zFar

    t = ti.tan(fov_rad / 2.0) * (-n)
    b = -t
    r = aspect_ratio * t
    l = -r

    persp2ortho = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    ortho = ti.Matrix([
        [2.0/(r-l), 0.0, 0.0, -(r+l)/(r-l)],
        [0.0, 2.0/(t-b), 0.0, -(t+b)/(t-b)],
        [0.0, 0.0, 2.0/(n-f), -(n+f)/(n-f)],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return ortho @ persp2ortho