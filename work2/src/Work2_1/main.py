import taichi as ti
from .config import *
from .physics import *

ti.init(arch=ti.gpu)

angle = ti.field(ti.f32, shape=())
angle[None] = 0.0

vertices = ti.Vector.field(4, ti.f32, shape=3)
screen_vertices = ti.Vector.field(2, ti.f32, shape=3)
eye_pos = ti.Vector([eye_pos_x, eye_pos_y, eye_pos_z])

@ti.kernel
def transform():
    vertices[0] = ti.Vector([2.0, 0.0, -2.0, 1.0])
    vertices[1] = ti.Vector([0.0, 2.0, -2.0, 1.0])
    vertices[2] = ti.Vector([-2.0, 0.0, -2.0, 1.0])

    model = get_model_matrix(angle[None])
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(fov, aspect_ratio, zNear, zFar)
    mvp = proj @ view @ model

    for i in ti.static(range(3)):
        v = mvp @ vertices[i]
        v /= v.w
        x = (v.x + 1.0) * 0.5
        y = (v.y + 1.0) * 0.5
        screen_vertices[i] = ti.Vector([x, y])

gui = ti.GUI("CG Experiment 2", res=(res, res))

while gui.running:
    # ========= 核心修复：必须写这一句刷新按键状态 =========
    gui.get_event()
    # ====================================================

    if gui.is_pressed('a'):
        angle[None] += 2.0
    if gui.is_pressed('d'):
        angle[None] -= 2.0

    transform()

    p = screen_vertices
    gui.triangle(p[0], p[1], p[2], color=0x333333)
    gui.line(p[0], p[1], color=0xff0000)
    gui.line(p[1], p[2], color=0x00ff00)
    gui.line(p[2], p[0], color=0x0000ff)

    gui.show()