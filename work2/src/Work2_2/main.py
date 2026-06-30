import taichi as ti
# 导入配置和数学工具模块
from .config import *
from .math_utils import *

# ==============================================
# 1. 初始化Taichi（使用GPU）
# ==============================================
ti.init(arch=ti.gpu)

# ==============================================
# 2. 定义全局数据场（Taichi字段）
# ==============================================
# 立方体8个3D顶点 (x,y,z,w)
vertices = ti.Vector.field(4, ti.f32, shape=8)
# 投影后的2D屏幕坐标
screen_vertices = ti.Vector.field(2, ti.f32, shape=8)
# 12条边的索引数组（24个元素）
edges = ti.field(ti.i32, shape=24)

# 旋转角度控制（当前角度 + 目标角度）
angle = ti.field(ti.f32, shape=())
target_angle = ti.field(ti.f32, shape=())
angle[None] = 0.0
target_angle[None] = 0.0

# 相机位置向量
eye = ti.Vector([eye_pos_x, eye_pos_y, eye_pos_z])

# ==============================================
# 3. 初始化立方体数据
# ==============================================
init_cube(vertices, edges)

# ==============================================
# 4. 创建GUI窗口
# ==============================================
gui = ti.GUI("3D立方体透视旋转", res=res)

# ==============================================
# 5. 主渲染循环
# ==============================================
while gui.running:
    # 处理输入事件（必须调用，否则按键不响应）
    gui.get_event()

    # A键：向左旋转（增加目标角度）
    if gui.is_pressed('a'):
        target_angle[None] += 1.0
    # D键：向右旋转（减少目标角度）
    if gui.is_pressed('d'):
        target_angle[None] -= 1.0

    # 平滑更新旋转角度（插值）
    update_rotation(angle, target_angle)

    # MVP变换：将3D顶点投影到2D屏幕
    transform_vertices(
        vertices, screen_vertices,
        angle[None], eye,
        fov, aspect_ratio, zNear, zFar
    )

    # 绘制立方体的12条线框边
    for i in range(0, 24, 2):
        start = screen_vertices[edges[i]]
        end = screen_vertices[edges[i + 1]]
        gui.line(start, end, color=0xffffff)  # 白色线条

    # 刷新窗口显示
    gui.show()