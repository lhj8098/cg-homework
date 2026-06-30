import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# ========== 常量定义 ==========
WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
BEZIER_SEGMENTS = 1000       # 贝塞尔曲线总采样段数
BSPLINE_SEG_PER_SPAN = 200   # B样条单段曲线的采样数
MAX_CURVE_POINTS = 20000     # 曲线点缓冲区最大容量

# ========== 缓冲区定义 ==========
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)

# ========== 贝塞尔曲线计算 ==========
def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

# ========== B样条曲线计算 ==========
def uniform_cubic_bspline(control_points, seg_per_span=200):
    """均匀三次B样条曲线采样点计算（矩阵形式）"""
    n = len(control_points)
    if n < 4:
        return np.zeros((0, 2), dtype=np.float32)
    
    M = np.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0]
    ], dtype=np.float32) / 6.0
    
    num_spans = n - 3
    total_points = num_spans * seg_per_span + 1
    curve_points = np.zeros((total_points, 2), dtype=np.float32)
    
    for span_idx in range(num_spans):
        P = np.array(control_points[span_idx:span_idx+4], dtype=np.float32)
        for u_int in range(seg_per_span):
            u = u_int / seg_per_span
            U = np.array([u**3, u**2, u, 1.0], dtype=np.float32)
            curve_points[span_idx * seg_per_span + u_int] = U @ M @ P
    
    u_last = 1.0
    U_last = np.array([u_last**3, u_last**2, u_last, 1.0], dtype=np.float32)
    P_last = np.array(control_points[-4:], dtype=np.float32)
    curve_points[-1] = U_last @ M @ P_last
    
    return curve_points

# ========== GPU渲染内核 ==========
@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """带反走样的曲线渲染内核"""
    curve_color = ti.Vector([0.0, 1.0, 0.0])
    effect_radius = 1.5
    
    for i in range(n):
        pt = curve_points_field[i]
        fx = pt[0] * WIDTH
        fy = pt[1] * HEIGHT
        
        x0 = ti.cast(ti.floor(fx), ti.i32)
        y0 = ti.cast(ti.floor(fy), ti.i32)
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = x0 + dx
                y = y0 + dy
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    px = x + 0.5
                    py = y + 0.5
                    dist = ti.sqrt((fx - px)**2 + (fy - py)**2)
                    weight = ti.max(0.0, 1.0 - dist / effect_radius)
                    pixels[x, y] = ti.max(pixels[x, y], weight * curve_color)

# ========== 主循环 ==========
def main():
    window = ti.ui.Window("Bezier & B-spline with Anti-aliasing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    current_mode = 'bezier'
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c': 
                control_points = []
                print("Canvas cleared.")
            elif e.key == 'b':
                current_mode = 'bspline' if current_mode == 'bezier' else 'bezier'
                print(f"Switched to {current_mode} mode")
        
        clear_pixels()
        
        current_count = len(control_points)
        curve_points_np = None
        
        # 模式分支：计算曲线采样点
        if current_mode == 'bezier' and current_count >= 2:
            curve_points_np = np.zeros((BEZIER_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(BEZIER_SEGMENTS + 1):
                t = t_int / BEZIER_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
        elif current_mode == 'bspline' and current_count >= 4:
            curve_points_np = uniform_cubic_bspline(control_points, BSPLINE_SEG_PER_SPAN)
        
        # 发送GPU并渲染
        if curve_points_np is not None:
            point_count = len(curve_points_np)
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(point_count)
        
        canvas.set_image(pixels)
        
        # 绘制控制点和控制折线
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()