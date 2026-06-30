import taichi as ti

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2  # 玻璃材质

# 每像素采样数（MSAA）
SAMPLES_PER_PIXEL = 4

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """斯涅尔定律折射计算（Taichi 单出口规范）"""
    cosi = -I.dot(N)
    sin2_t = eta * eta * (1.0 - cosi * cosi)
    
    has_refraction = True
    refr_dir = ti.Vector([0.0, 0.0, 0.0])
    
    if sin2_t > 1.0:
        has_refraction = False
    else:
        cost = ti.sqrt(1.0 - sin2_t)
        refr_dir = normalize(eta * I + (eta * cosi - cost) * N)
    
    return has_refraction, refr_dir

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """遍历场景，寻找最近交点"""
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 红色玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([1.0, 0.85, 0.85])
        hit_mat = MAT_GLASS

    # 2. 银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 3. 棋盘格地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    ior_glass = 1.5  # 玻璃折射率

    for i, j in pixels:
        accumulated_color = ti.Vector([0.0, 0.0, 0.0])
        
        # MSAA 抗锯齿：每像素多次采样
        for s in range(SAMPLES_PER_PIXEL):
            offset_u = (ti.random() - 0.5) / res_y * 2.0
            offset_v = (ti.random() - 0.5) / res_y * 2.0
            
            u = (i - res_x / 2.0) / res_y * 2.0 + offset_u
            v = (j - res_y / 2.0) / res_y * 2.0 + offset_v
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            sample_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            
            # 迭代式光线追踪
            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                
                if t > 1e9:
                    sample_color += throughput * bg_color
                    break
                    
                p = ro + rd * t
                
                # 镜面材质
                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color
                
                # 玻璃材质（折射 + 全反射）
                elif mat_id == MAT_GLASS:
                    N_local = N
                    eta = 1.0 / ior_glass
                    
                    # 判断是否从玻璃内部射出
                    if rd.dot(N_local) > 0:
                        N_local = -N_local
                        eta = ior_glass / 1.0
                    
                    has_refraction, refr_dir = refract(rd, N_local, eta)
                    
                    if not has_refraction:
                        # 全反射
                        ro = p + N_local * 1e-4
                        rd = normalize(reflect(rd, N_local))
                        throughput *= obj_color
                    else:
                        # 正常折射，向介质内侧偏移避免自相交
                        ro = p - N_local * 1e-4
                        rd = refr_dir
                        throughput *= obj_color
                
                # 漫反射材质
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    
                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = shadow_t < dist_to_light
                    
                    ambient = 0.2 * obj_color
                    direct_light = ambient
                    if not in_shadow:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse
                    
                    sample_color += throughput * direct_light
                    break
            
            accumulated_color += sample_color
        
        final_color = accumulated_color / SAMPLES_PER_PIXEL
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing: Glass + MSAA", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 4  # 玻璃建议至少 4 次弹射

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.22):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 6)

        window.show()

if __name__ == '__main__':
    main()