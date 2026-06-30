import taichi as ti

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# ======================================================
# 物理与网格参数（新增剪切/弯曲弹簧系数、球体参数）
# ======================================================
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 10000.0      # 结构弹簧劲度系数（上下左右）
k_sh = 5000.0      # 剪切弹簧劲度系数（对角线方向）
k_bend = 1000.0    # 弯曲弹簧劲度系数（间隔两点，抗弯）
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限，防止数值爆炸

# 碰撞球体参数
sphere_radius = 0.3
sphere_pos = ti.Vector.field(3, dtype=float, shape=1)  # 球心位置场，用于渲染

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# ======================================================
# 弹簧数据场（扩容 + 新增单弹簧劲度系数字段）
# ======================================================
max_springs = N * N * 10  # 扩容，容纳结构+剪切+弯曲三类弹簧
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_k = ti.field(dtype=float, shape=max_springs)       # 每个弹簧独立的劲度系数
num_springs = ti.field(dtype=int, shape=())

# ============ 初始化 (拆分为多个 kernel 保证 GPU 同步) ============

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定第一排的两个角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    """初始化三类弹簧：结构弹簧 + 剪切弹簧 + 弯曲弹簧"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # ---------- 1. 结构弹簧（水平+垂直） ----------
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_k[c] = k_s
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_k[c] = k_s

        # ---------- 2. 剪切弹簧（两条对角线） ----------
        if i < N - 1 and j < N - 1:
            # 主对角线：(i,j) -> (i+1,j+1)
            idx_diag1 = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag1])
            spring_lengths[c] = (x[idx] - x[idx_diag1]).norm()
            spring_k[c] = k_sh

            # 副对角线：(i+1,j) -> (i,j+1)
            idx_diag2 = i * N + (j + 1)
            idx_diag2_other = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx_diag2_other, idx_diag2])
            spring_lengths[c] = (x[idx_diag2_other] - x[idx_diag2]).norm()
            spring_k[c] = k_sh

        # ---------- 3. 弯曲弹簧（间隔一个质点，水平+垂直） ----------
        if i < N - 2:
            idx_right2 = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right2])
            spring_lengths[c] = (x[idx] - x[idx_right2]).norm()
            spring_k[c] = k_bend
        if j < N - 2:
            idx_down2 = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down2])
            spring_lengths[c] = (x[idx] - x[idx_down2]).norm()
            spring_k[c] = k_bend

@ti.kernel
def init_spring_indices():
    """同步渲染索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    """从 Python 层按顺序调用各初始化 kernel，确保 GPU 同步"""
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()
    # 初始化球体位置：放在布料正下方，让布料自然垂落碰撞
    sphere_pos[0] = ti.Vector([0.0, 0.2, 0.0])

# ============ 力计算 + 碰撞处理 ============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力：重力 + 阻尼 + 三类弹簧力"""
    # 第一阶段：清空受力，施加重力与阻尼
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    # 第二阶段：累加所有弹簧力（每个弹簧使用自身劲度系数）
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -spring_k[i] * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def resolve_sphere_collision(pos: ti.template(), vel: ti.template()):
    """
    球体碰撞响应：
    1. 位置修正：将穿入球体的质点推回球面
    2. 速度修正：移除法向朝向球心的速度分量，保留切向速度（非弹性碰撞）
    """
    center = sphere_pos[0]
    for i in range(N * N):
        if is_fixed[i] == 1:
            continue

        delta = pos[i] - center
        dist = delta.norm()

        if dist < sphere_radius and dist > 1e-6:
            normal = delta / dist
            # 位置修正：强制拉到球面
            pos[i] = center + normal * sphere_radius
            # 速度修正：抵消法向入射速度，保留切向
            vel_normal = vel[i].dot(normal)
            if vel_normal < 0:  # 仅修正朝向球心的速度
                vel[i] -= vel_normal * normal

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

# ============ 积分器 ============

@ti.kernel
def step_explicit():
    """显式欧拉"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
    resolve_sphere_collision(x, v)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉（推荐使用，稳定性好）"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
    resolve_sphere_collision(x, v)

@ti.kernel
def step_implicit_iter():
    """隐式欧拉（定点迭代近似）"""
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]

    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt

    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
    resolve_sphere_collision(x, v)

# ============ 主函数 ============

def main():
    init_cloth()

    window = ti.ui.Window("Mass Spring System - Full Version", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1  # 0: 显式, 1: 半隐式, 2: 隐式
    paused = False

    while window.running:
        # =========== GUI 控制面板 ===========
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.4)

        window.GUI.text("Integration Method:")
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler"):
            current_method = 2
            init_cloth()

        window.GUI.text("")
        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.text("")
        window.GUI.text(f"Structural k: {k_s:.0f}")
        window.GUI.text(f"Shear k: {k_sh:.0f}")
        window.GUI.text(f"Bending k: {k_bend:.0f}")
        window.GUI.text(f"Sphere radius: {sphere_radius}")

        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # 渲染场景
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 绘制布料质点与弹簧线框
        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))
        # 绘制碰撞球体（红色）
        scene.particles(sphere_pos, radius=sphere_radius, color=(1.0, 0.3, 0.3))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()