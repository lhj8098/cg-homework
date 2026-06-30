import taichi as ti
import math

# ==============================================
# @ti.func: Taichi函数（只能在kernel/func内部调用）
# 功能：生成绕Y轴的旋转矩阵（模型变换矩阵）
# 参数：angle - 旋转角度（度）
# 返回：4x4旋转矩阵
# ==============================================
@ti.func
def rotate_y(angle):
    rad = angle / 180.0 * math.pi  # 角度转弧度
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

# ==============================================
# 功能：生成视图矩阵（相机变换，将世界坐标转为相机坐标）
# 参数：eye - 相机位置向量
# 返回：4x4视图矩阵
# ==============================================
@ti.func
def view_matrix(eye):
    return ti.Matrix([
        [1, 0, 0, -eye.x],
        [0, 1, 0, -eye.y],
        [0, 0, 1, -eye.z],
        [0, 0, 0, 1]
    ])

# ==============================================
# 功能：生成透视投影矩阵（将3D相机坐标投影到2D裁剪空间）
# 参数：fov-视场角, aspect-宽高比, n-近裁剪面, f-远裁剪面
# 返回：4x4透视投影矩阵
# ==============================================
@ti.func
def proj_matrix(fov, aspect, n, f):
    rad = fov / 180.0 * math.pi
    t = ti.tan(rad / 2) * abs(n)
    r = t * aspect
    return ti.Matrix([
        [n / r, 0, 0, 0],
        [0, n / t, 0, 0],
        [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
        [0, 0, -1, 0]
    ])

# ==============================================
# 功能：初始化标准立方体（中心原点，边长2）
# 顶点：8个，坐标范围[-1,1]
# 边：12条（24个索引）
# ==============================================
@ti.kernel
def init_cube(vertices: ti.template(), edges: ti.template()):
    # 定义立方体8个顶点 (x,y,z,w)，w=1表示点坐标
    vertices[0] = ti.Vector([1, 1, 1, 1])
    vertices[1] = ti.Vector([1, 1, -1, 1])
    vertices[2] = ti.Vector([1, -1, 1, 1])
    vertices[3] = ti.Vector([1, -1, -1, 1])
    vertices[4] = ti.Vector([-1, 1, 1, 1])
    vertices[5] = ti.Vector([-1, 1, -1, 1])
    vertices[6] = ti.Vector([-1, -1, 1, 1])
    vertices[7] = ti.Vector([-1, -1, -1, 1])

    # 定义12条边（每两个数字为一条边）
    edge_indices = [
        0, 1, 1, 3, 3, 2, 2, 0,  # 前面
        4, 5, 5, 7, 7, 6, 6, 4,  # 后面
        0, 4, 1, 5, 2, 6, 3, 7   # 连接前后
    ]
    # 静态循环赋值
    for i in ti.static(range(24)):
        edges[i] = edge_indices[i]

# ==============================================
# 功能：MVP变换 + 透视除法 + 屏幕坐标映射
# ==============================================
@ti.kernel
def transform_vertices(
    vertices: ti.template(),
    screen_vertices: ti.template(),
    angle: ti.f32,
    eye: ti.template(),
    fov: ti.f32,
    aspect: ti.f32,
    z_near: ti.f32,
    z_far: ti.f32
):
    # 1. 模型矩阵（旋转）
    model_mat = rotate_y(angle)
    # 2. 视图矩阵（相机位置）
    view_mat = view_matrix(eye)
    # 3. 投影矩阵（透视）
    proj_mat = proj_matrix(fov, aspect, z_near, z_far)
    # 组合MVP矩阵
    mvp_mat = proj_mat @ view_mat @ model_mat

    # 对每个顶点进行变换
    for i in range(8):
        # MVP变换
        v_clip = mvp_mat @ vertices[i]
        # 透视除法（齐次坐标转标准化设备坐标NDC）
        v_ndc = v_clip / v_clip.w
        # NDC坐标[-1,1]映射到屏幕坐标[0,1]
        x_screen = (v_ndc.x + 1.0) * 0.5
        y_screen = (v_ndc.y + 1.0) * 0.5
        screen_vertices[i] = ti.Vector([x_screen, y_screen])

# ==============================================
# 功能：旋转角度插值（平滑动画）
# 原理：当前角度向目标角度缓慢靠近
# ==============================================
@ti.kernel
def update_rotation(angle: ti.template(), target_angle: ti.template()):
    angle_diff = target_angle[None] - angle[None]
    # 插值系数0.05，数值越大转动越快
    angle[None] += angle_diff * 0.05