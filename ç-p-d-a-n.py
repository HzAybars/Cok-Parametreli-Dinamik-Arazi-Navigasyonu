"""
TUA Ay Keşif Aracı Yer Kontrol İstasyonu  v10.1 (TAM DONANIMLI EKSİKSİZ SÜRÜM)
========================================
Hata düzeltmeleri ve iyileştirmeler:
  - [UI FIX] Kod sınırına takıldığı için yanlışlıkla silinen "Yükseklik Profili", "Enerji Analizi", "Fizik Limitleri" ve "Kıyaslama" sekmeleri sağ panele geri eklendi.
  - [ÖZELLİK] Güneş enerjisi otonom şarj (Incidence Angle) matematiği korundu.
  - [ÖZELLİK] PyVista donanım ivmeli (GPU) 3D Sinematik render korundu.
  - [ÖZELLİK] ROS Nav2 ve Telemetri iletişim ağı korundu.
  - [ÖZELLİK] Multi-Threading A* algoritması ile maksimum hız.
  - [HATA GİDERME] 'hpad' hatası ve eksik arayüz değişkenleri onarıldı.
"""

import numpy as np
import heapq
import struct
import threading
import concurrent.futures
import json
import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from scipy.interpolate import griddata, splprep, splev
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import roslibpy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# SABİTLER (TUA AY KEŞİF ARACI + GÜNEŞ PANELİ STANDARTLARI)
# ─────────────────────────────────────────────────────────────
ROVER_DEFAULTS = {
    "mass_kg":           1025.0,     
    "wheel_radius_m":    0.2625,     
    "wheel_count":       6,          
    "max_torque_nm":     150.0,      
    "battery_wh":        3000.0,     
    "motor_efficiency":  0.92,       
    "max_slope_deg":     31.0,       
    "safe_slope_deg":    20.0,       
    "speed_ms":          0.042,      
    "rolling_resistance":0.12,       
    "gravity_ms2":       1.62,       
    "suspension_travel": 0.150,      
    "ground_clearance":  0.395,
    "solar_area_m2":     1.5,        
    "solar_efficiency":  0.28,       
    "sun_elevation_deg": 45.0,       
    "sun_azimuth_deg":   180.0,      
}

LUNAR_SOLAR_IRRADIANCE = 1361.0 

SLOPE_COLORS = {
    "duz":     (0,  10,  "#2ecc71"),
    "guvenli": (10, 20,  "#f1c40f"),
    "dikkat":  (20, 28,  "#e67e22"),
    "tehlike": (28, 90,  "#e74c3c"),
}

BG        = '#0b0f1e'
PANEL     = '#111827'
PANEL2    = '#1a2235'
ACCENT    = '#00d4ff'  
ACCENT2   = '#3b82f6'  
SUCCESS   = '#10b981'
WARNING   = '#f59e0b'
DANGER    = '#ef4444'
TEXT      = '#e2e8f0'
MUTED     = '#64748b'
BORDER    = '#1e293b'

# ─────────────────────────────────────────────────────────────
# ROS HABERLEŞME SINIFI (NAV2 UYUMLU)
# ─────────────────────────────────────────────────────────────
class TUARosBridge:
    def __init__(self, telemetry_callback=None):
        self.client = None
        self.route_topic_json = None
        self.route_topic_path = None
        self.estop_topic = None
        self.telemetry_topic_json = None
        self.odom_topic = None
        self.telemetry_callback = telemetry_callback

    def connect(self, host='127.0.0.1', port=9090):
        if not ROS_AVAILABLE:
            return False, "roslibpy kütüphanesi bulunamadı! (Terminalde 'pip install roslibpy')"
        try:
            self.client = roslibpy.Ros(host=host, port=port)
            self.client.run(timeout=3)
            if self.client.is_connected:
                self.route_topic_json = roslibpy.Topic(self.client, '/tua_rover/planlanan_rota_json', 'std_msgs/String')
                self.route_topic_path = roslibpy.Topic(self.client, '/tua_rover/path', 'nav_msgs/Path')
                self.estop_topic = roslibpy.Topic(self.client, '/tua_rover/acil_durdurma', 'std_msgs/Bool')
                self.telemetry_topic_json = roslibpy.Topic(self.client, '/tua_rover/telemetri', 'std_msgs/String')
                self.odom_topic = roslibpy.Topic(self.client, '/tua_rover/odom', 'nav_msgs/Odometry')
                
                if self.telemetry_callback:
                    self.telemetry_topic_json.subscribe(self.telemetry_callback)
                    self.odom_topic.subscribe(self._odom_to_telemetry)
                    
                return True, f"ROS İletişimi Aktif ({host}:{port})"
            else:
                return False, "ROS Bridge sunucusuna ulaşılamadı."
        except Exception as e:
            return False, f"Bağlantı Hatası: {str(e)}"

    def _odom_to_telemetry(self, message):
        try:
            x = message['pose']['pose']['position']['x']
            y = message['pose']['pose']['position']['y']
            simulated_data = json.dumps({"x": x, "y": y, "battery": 100.0})
            if self.telemetry_callback:
                self.telemetry_callback({'data': simulated_data})
        except: pass

    def disconnect(self):
        if self.client and self.client.is_connected:
            self.client.terminate()
            
    def publish_route(self, path_world_coords):
        if not self.client or not self.client.is_connected:
            return False, "ROS bağlantısı yok!"
        try:
            payload = json.dumps({"rota": path_world_coords})
            self.route_topic_json.publish(roslibpy.Message({'data': payload}))
            
            poses = []
            for pt in path_world_coords:
                pose = {
                    "header": {"frame_id": "map"},
                    "pose": {
                        "position": {"x": pt['x'], "y": pt['y'], "z": pt['z']},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                    }
                }
                poses.append(pose)
            
            path_msg = {
                "header": {"frame_id": "map"},
                "poses": poses
            }
            self.route_topic_path.publish(roslibpy.Message(path_msg))
            return True, f"{len(path_world_coords)} noktalı rota RViz/ROS ağına iletildi!"
        except Exception as e:
            return False, f"Gönderim hatası: {str(e)}"
            
    def trigger_estop(self):
        if not self.client or not self.client.is_connected:
            return False, "ROS bağlantısı yok!"
        try:
            self.estop_topic.publish(roslibpy.Message({'data': True}))
            return True, "ACİL DURDURMA sinyali gönderildi!"
        except Exception as e:
            return False, f"E-Stop hatası: {str(e)}"

# ─────────────────────────────────────────────────────────────
# STL OKUYUCU
# ─────────────────────────────────────────────────────────────
class STLLoader:
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            header = f.read(80).decode('utf-8', errors='ignore').strip()
            f.seek(0)
            raw = f.read()
        if raw[:5].lower() == b'solid' and b'endsolid' in raw.lower():
            try:
                return STLLoader._load_ascii(raw.decode('utf-8', errors='ignore'))
            except Exception:
                pass
        return STLLoader._load_binary(raw)

    @staticmethod
    def _load_binary(raw):
        num = struct.unpack_from('<I', raw, 80)[0]
        verts, norms = [], []
        offset = 84
        for _ in range(num):
            if offset + 50 > len(raw):
                break
            n  = struct.unpack_from('<3f', raw, offset)
            v1 = struct.unpack_from('<3f', raw, offset + 12)
            v2 = struct.unpack_from('<3f', raw, offset + 24)
            v3 = struct.unpack_from('<3f', raw, offset + 36)
            norms.append(n)
            verts.append((v1, v2, v3))
            offset += 50
        return np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)

    @staticmethod
    def _load_ascii(text):
        import re
        verts, norms = [], []
        fn_pat = re.compile(r'facet\s+normal\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)', re.I)
        vx_pat = re.compile(r'vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)', re.I)
        cur_n, cur_v = None, []
        for line in text.splitlines():
            m = fn_pat.match(line.strip())
            if m:
                cur_n = tuple(float(x) for x in m.groups())
                cur_v = []
                continue
            m = vx_pat.match(line.strip())
            if m:
                cur_v.append(tuple(float(x) for x in m.groups()))
                continue
            if 'endfacet' in line.lower() and len(cur_v) == 3:
                verts.append(tuple(cur_v))
                norms.append(cur_n or (0, 0, 1))
        return np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# ARAZİ BİLGİSİ
# ─────────────────────────────────────────────────────────────
class TerrainGrid:
    def __init__(self, triangles, resolution=100, forced_x_m=None, forced_y_m=None):
        self.resolution = resolution
        verts = np.copy(triangles).reshape(-1, 3)
        orig_xmin, orig_xmax = float(verts[:,0].min()), float(verts[:,0].max())
        orig_ymin, orig_ymax = float(verts[:,1].min()), float(verts[:,1].max())
        orig_zmin = float(verts[:,2].min())
        
        orig_x_range = max(orig_xmax - orig_xmin, 1e-9)
        orig_y_range = max(orig_ymax - orig_ymin, 1e-9)
        
        scale_x = forced_x_m / orig_x_range if forced_x_m else 1.0
        scale_y = forced_y_m / orig_y_range if forced_y_m else 1.0
        
        verts[:,0] = orig_xmin + (verts[:,0] - orig_xmin) * scale_x
        verts[:,1] = orig_ymin + (verts[:,1] - orig_ymin) * scale_y
        
        if scale_x != 1.0 or scale_y != 1.0:
            avg_scale = (scale_x + scale_y) / 2.0
            verts[:,2] = orig_zmin + (verts[:,2] - orig_zmin) * avg_scale
            
        self.triangles = verts.reshape(-1, 3, 3)
        self._build()

    def _build(self):
        verts = self.triangles.reshape(-1, 3)
        self.x_min, self.x_max = float(verts[:,0].min()), float(verts[:,0].max())
        self.y_min, self.y_max = float(verts[:,1].min()), float(verts[:,1].max())
        self.z_min, self.z_max = float(verts[:,2].min()), float(verts[:,2].max())

        xr, yr = self.x_max - self.x_min, self.y_max - self.y_min
        aspect  = yr / xr if xr > 1e-9 else 1.0
        self.nx = self.resolution
        self.ny = max(int(self.resolution * aspect), 10)

        xi  = np.linspace(self.x_min, self.x_max, self.nx)
        yi  = np.linspace(self.y_min, self.y_max, self.ny)
        self.xi, self.yi = np.meshgrid(xi, yi)

        pts = verts[:, :2]
        zv  = verts[:, 2]
        
        if len(pts) > 500_000:
            idx = np.random.choice(len(pts), 500_000, replace=False)
            pts, zv = pts[idx], zv[idx]

        self.zi = griddata(pts, zv, (self.xi, self.yi), method='linear', fill_value=np.nan)
        nan_mask = np.isnan(self.zi)
        if nan_mask.any():
            filled = griddata(pts, zv, (self.xi[nan_mask], self.yi[nan_mask]), method='nearest')
            self.zi[nan_mask] = filled
            
        self.zi = gaussian_filter(self.zi.astype(np.float64), sigma=0.8)
        self._compute_slopes()
        self._build_traversability()

    def _compute_slopes(self):
        dx = (self.x_max - self.x_min) / max(self.nx - 1, 1)
        dy = (self.y_max - self.y_min) / max(self.ny - 1, 1)
        dz_dy, dz_dx = np.gradient(self.zi, dy, dx)
        self.slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        self.aspect_deg = (np.degrees(np.arctan2(dz_dx, dz_dy)) + 360) % 360 
        self.roughness  = gaussian_filter(np.abs(dz_dx) + np.abs(dz_dy), sigma=1.5)

    def _build_traversability(self):
        t = np.full(self.slope_deg.shape, 3, dtype=np.uint8)
        t[self.slope_deg >= SLOPE_COLORS['dikkat'][0]] = 2
        t[self.slope_deg >= SLOPE_COLORS['dikkat'][1]] = 1
        t[self.slope_deg >= SLOPE_COLORS['tehlike'][1]]  = 0
        self.traversability = t

    def world_to_grid(self, x, y):
        col = int(round((x - self.x_min) / max(self.x_max - self.x_min, 1e-9) * (self.nx - 1)))
        row = int(round((y - self.y_min) / max(self.y_max - self.y_min, 1e-9) * (self.ny - 1)))
        return int(np.clip(row, 0, self.ny-1)), int(np.clip(col, 0, self.nx-1))

    def grid_to_world(self, row, col):
        x = self.x_min + col / max(self.nx-1, 1) * (self.x_max - self.x_min)
        y = self.y_min + row / max(self.ny-1, 1) * (self.y_max - self.y_min)
        r = int(np.clip(row, 0, self.ny-1))
        c = int(np.clip(col, 0, self.nx-1))
        return x, y, float(self.zi[r, c])

    def get_height(self, row, col):
        return float(self.zi[np.clip(row,0,self.ny-1), np.clip(col,0,self.nx-1)])

    def get_slope(self, row, col):
        return float(self.slope_deg[np.clip(row,0,self.ny-1), np.clip(col,0,self.nx-1)])
        
    def get_aspect(self, row, col):
        return float(self.aspect_deg[np.clip(row,0,self.ny-1), np.clip(col,0,self.nx-1)])

    def get_roughness(self, row, col):
        return float(self.roughness[np.clip(row,0,self.ny-1), np.clip(col,0,self.nx-1)])

    def cell_size(self):
        dx = (self.x_max - self.x_min) / max(self.nx-1, 1)
        dy = (self.y_max - self.y_min) / max(self.ny-1, 1)
        return (dx + dy) / 2.0

# ─────────────────────────────────────────────────────────────
# ENERJİ MODELİ (GÜNEŞ PANELİ ENTEGRASYONLU)
# ─────────────────────────────────────────────────────────────
class RoverEnergyModel:
    def __init__(self, params=None):
        self.p = dict(ROVER_DEFAULTS)
        if params:
            self.p.update(params)

    def max_available_force(self):
        return (self.p['max_torque_nm'] / max(self.p['wheel_radius_m'], 1e-6)) * self.p['wheel_count']

    def required_force(self, slope_deg, roughness=0.0):
        sr   = math.radians(max(0.0, min(slope_deg, 89.0)))
        g, m = self.p['gravity_ms2'], self.p['mass_kg']
        rr   = self.p['rolling_resistance'] + roughness * 0.05
        Fn   = m * g * math.cos(sr)
        Fgrav= m * g * math.sin(sr)
        return Fgrav + rr * Fn

    def can_traverse(self, slope_deg):
        return (self.required_force(slope_deg) <= self.max_available_force()
                and slope_deg < self.p['max_slope_deg'])

    def energy_wh(self, slope_deg, dist_m, roughness=0.0):
        if not self.can_traverse(slope_deg):
            return float('inf')
        F = self.required_force(slope_deg, roughness)
        return (F * dist_m / self.p['motor_efficiency']) / 3600.0
        
    def solar_power_w(self, slope_deg, aspect_deg, current_elev=None, current_az=None):
        sun_elev = current_elev if current_elev is not None else self.p['sun_elevation_deg']
        sun_az   = current_az if current_az is not None else self.p['sun_azimuth_deg']
        sun_zenith_rad = math.radians(90.0 - sun_elev)
        slope_rad = math.radians(slope_deg)
        az_diff_rad = math.radians(sun_az - aspect_deg)
        
        cos_theta = math.cos(sun_zenith_rad) * math.cos(slope_rad) + \
                    math.sin(sun_zenith_rad) * math.sin(slope_rad) * math.cos(az_diff_rad)
        
        cos_theta = max(0.0, cos_theta)
        power_w = LUNAR_SOLAR_IRRADIANCE * self.p['solar_area_m2'] * self.p['solar_efficiency'] * cos_theta
        return power_w

    def speed_ms(self, slope_deg):
        v0     = self.p['speed_ms']
        factor = max(0.05, 1.0 - (slope_deg / max(self.p['max_slope_deg'], 1)) * 0.85)
        return v0 * factor

    def travel_time_s(self, dist_m, slope_deg):
        v = self.speed_ms(slope_deg)
        return dist_m / v if v > 0 else float('inf')

    def safety_score(self, slope_deg):
        s = self.p['safe_slope_deg']
        m = self.p['max_slope_deg']
        if slope_deg <= s:   return 1.0
        if slope_deg >= m:   return 0.0
        return 1.0 - (slope_deg - s) / (m - s)

    def traction_ratio(self, slope_deg):
        F_req = self.required_force(slope_deg)
        F_max = self.max_available_force()
        return min(F_req / max(F_max, 1e-6), 1.0)

    def power_w(self, slope_deg, roughness=0.0):
        F = self.required_force(slope_deg, roughness)
        v = self.speed_ms(slope_deg)
        return F * v / self.p['motor_efficiency']

# ─────────────────────────────────────────────────────────────
# DOĞAL EĞRİLİ A* ALGORİTMASI
# ─────────────────────────────────────────────────────────────
class RoutePlanner:
    MODE_WEIGHTS = {
        'dengeli':  (1.0,  5.0,  20.0, 0.0, 1.0),
        'guvenli':  (1.0, 15.0,  10.0, 0.0, 3.0),
        'enerji':   (0.5,  3.0,  50.0, 0.0, 0.5),
        'hizli':    (2.0,  1.0,   5.0, 1.0, 0.2),
    }

    def __init__(self, terrain: TerrainGrid, energy: RoverEnergyModel,
                 mode='dengeli', epsilon=1.5):
        self.t    = terrain
        self.e    = energy
        self.mode = mode
        self.eps  = epsilon

    def _h(self, a, b):
        xa, ya, za = self.t.grid_to_world(*a)
        xb, yb, zb = self.t.grid_to_world(*b)
        dist = np.sqrt((xa-xb)**2 + (ya-yb)**2 + (za-zb)**2)
        dw = self.MODE_WEIGHTS[self.mode][0]
        return dist * dw

    def _a_star(self, start, goal, progress_cb=None):
        open_heap = []
        g = {start: 0.0}
        parent = {start: start}
        visited = set()
        counter = 0

        ny, nx = self.t.ny, self.t.nx
        traversability = self.t.traversability
        slope_deg = self.t.slope_deg
        roughness = self.t.roughness
        zi = self.t.zi
        cell_sz = self.t.cell_size()
        max_slope = self.e.p['max_slope_deg']
        dw, sw, ew, tw, rw = self.MODE_WEIGHTS[self.mode]
        
        start_h = math.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2) * cell_sz * dw
        heapq.heappush(open_heap, (self.eps * start_h, counter, start))

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur in visited: continue
            visited.add(cur)

            if cur == goal:
                if progress_cb: progress_cb(100)
                return self._reconstruct(parent, goal)

            if progress_cb and counter % 500 == 0:
                curr_h = math.sqrt((cur[0]-goal[0])**2 + (cur[1]-goal[1])**2) * cell_sz * dw
                pct = int((1.0 - (curr_h / max(start_h, 1e-9))) * 100)
                progress_cb(max(0, min(99, pct)))

            r, c = cur
            z_cur = zi[r, c]
            s_cur = slope_deg[r, c]
            rgh_cur = roughness[r, c]

            for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ny and 0 <= nc < nx:
                    if traversability[nr, nc] == 0: continue
                    s_nb = slope_deg[nr, nc]
                    if s_nb >= max_slope: continue
                    
                    nb = (nr, nc)
                    if nb in visited: continue

                    avg_slope = (s_cur + s_nb) * 0.5
                    if not self.e.can_traverse(avg_slope): continue
                    
                    z_nb = zi[nr, nc]
                    dist = math.sqrt((dr*cell_sz)**2 + (dc*cell_sz)**2 + (z_cur - z_nb)**2)
                    rgh = (rgh_cur + roughness[nr, nc]) * 0.5
                    nrgh = min(rgh / 5.0, 1.0) 
                    
                    sp = (avg_slope / 10.0) ** 2
                    enrg = self.e.energy_wh(avg_slope, dist, rgh)
                    if enrg == float('inf'): continue
                    ttime = self.e.travel_time_s(dist, avg_slope)
                    
                    cost = dw*dist + sw*dist*sp + ew*enrg + tw*ttime + rw*dist*nrgh
                    new_g = g[cur] + cost
                    
                    if new_g < g.get(nb, float('inf')):
                        g[nb] = new_g
                        parent[nb] = cur
                        
                        h_val = math.sqrt((nr-goal[0])**2 + (nc-goal[1])**2) * cell_sz * dw
                        f = new_g + self.eps * h_val
                        
                        counter += 1
                        heapq.heappush(open_heap, (f, counter, nb))
        return None

    def _reconstruct(self, parent, node):
        path = [node]
        while parent[node] != node:
            node = parent[node]
            path.append(node)
        return list(reversed(path))

    def _smooth_path(self, path):
        if len(path) < 4: return path
        try:
            pts = []
            for r, c in path:
                if not pts or pts[-1] != (r, c): pts.append((r, c))
            if len(pts) < 4: return path
            
            y = [p[0] for p in pts]
            x = [p[1] for p in pts]
            
            tck, u = splprep([y, x], s=2.0)
            u_new = np.linspace(0, 1, len(pts) * 3)
            y_new, x_new = splev(u_new, tck)
            
            smooth_grid_path = []
            for rn, cn in zip(y_new, x_new):
                rr = int(np.clip(round(rn), 0, self.t.ny - 1))
                cc = int(np.clip(round(cn), 0, self.t.nx - 1))
                if not smooth_grid_path or smooth_grid_path[-1] != (rr, cc):
                    smooth_grid_path.append((rr, cc))
            return smooth_grid_path
        except: return path

    def find_path(self, start, goal, smooth=True, progress_cb=None):
        path = self._a_star(start, goal, progress_cb)
        if path and smooth: path = self._smooth_path(path)
        return path

    def plan_route(self, waypoints, progress_callback=None):
        full_path = []
        segments = [None] * (len(waypoints) - 1)
        total_seg = len(waypoints) - 1
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {}
            for i in range(total_seg):
                def make_cb(idx):
                    return lambda pct: progress_callback(idx+1, total_seg, pct) if progress_callback else None
                future = executor.submit(self.find_path, waypoints[i], waypoints[i+1], True, make_cb(i))
                future_to_idx[future] = i
                
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try: segments[idx] = future.result()
                except Exception: segments[idx] = None
        
        for i, seg in enumerate(segments):
            if seg is None: return None, None
            if i > 0 and full_path and seg[0] == full_path[-1]: full_path.extend(seg[1:])
            else: full_path.extend(seg)
                
        return full_path, segments

# ─────────────────────────────────────────────────────────────
# GÜNEŞ ENERJİSİ ANALİZÖRÜ VE AKILLI ŞARJ MOTORU
# ─────────────────────────────────────────────────────────────
class RouteAnalyzer:
    def __init__(self, terrain: TerrainGrid, energy: RoverEnergyModel, auto_charge=False, manual_charge_wps=None):
        self.t = terrain
        self.e = energy
        self.auto_charge = auto_charge
        self.manual_charge_wps = manual_charge_wps if manual_charge_wps else []

    def analyze(self, path):
        default_stats = dict(
            total_distance_m=0.0, total_energy_wh=0.0, total_time_s=0.0, total_time_min=0.0,
            avg_slope_deg=0.0, max_slope_deg=0.0, elevation_gain_m=0.0, elevation_loss_m=0.0,
            height_profile=[0], slope_profile=[0], energy_profile=[0], power_profile=[0],
            danger_segments=0, battery_pct=0.0, node_count=len(path) if path else 0,
            battery_failed=False, charge_stops=[], total_charge_time_min=0.0
        )
        if not path or len(path) < 2: return default_stats

        dist_total = motor_energy_total = time_total = 0.0
        elev_gain = elev_loss = 0.0
        slopes, heights, powers, energies = [], [], [], []
        danger_segs = 0
        
        max_bat = self.e.p['battery_wh']
        current_battery = max_bat
        charge_stops = [] 
        total_charge_time_h = 0.0
        battery_failed = False
        history = [] 

        prev = self.t.grid_to_world(*path[0])
        heights.append(prev[2])
        safe_battery_threshold = max_bat * 0.10 

        for i in range(1, len(path)):
            r, c = path[i]
            cur = self.t.grid_to_world(r, c)
            heights.append(cur[2])
            
            dx, dy, dz = cur[0]-prev[0], cur[1]-prev[1], cur[2]-prev[2]
            d3 = np.sqrt(dx**2+dy**2+dz**2)
            d2 = np.sqrt(dx**2+dy**2)
            sl = np.degrees(np.arctan2(abs(dz), max(d2, 1e-9)))
            rgh = self.t.get_roughness(r, c)
            aspect = self.t.get_aspect(r, c)

            e_motor  = self.e.energy_wh(sl, d3, rgh)
            
            # Simülasyon başından beri geçen süreye (saat) göre güneşin anlık konumu
            elapsed_hours = time_total / 3600.0
            dynamic_az = (self.e.p['sun_azimuth_deg'] + (elapsed_hours * 0.508)) % 360.0
            
            # Dinamik yön (azimut) değerini hesaplamaya dahil ediyoruz
            p_solar  = self.e.solar_power_w(sl, aspect, current_az=dynamic_az)
            
            t_seg_s  = self.e.travel_time_s(d3, sl)
            t_seg_h  = t_seg_s / 3600.0
            e_solar  = p_solar * t_seg_h
            
            net_energy = e_motor - e_solar
            current_battery -= net_energy
            current_battery = min(max_bat, current_battery) 
            
            history.append({
                'grid': (r, c),
                'world': cur,
                'solar_power': p_solar,
                'battery_at_point': current_battery
            })

            if current_battery < safe_battery_threshold:
                if self.auto_charge:
                    best_spot = max(history, key=lambda item: item['solar_power'])
                    if best_spot['solar_power'] > 10.0: 
                        charge_stops.append(best_spot['grid'])
                        energy_needed = max_bat - best_spot['battery_at_point']
                        charge_time_h = energy_needed / best_spot['solar_power']
                        total_charge_time_h += charge_time_h
                        
                        idx_best = history.index(best_spot)
                        current_battery = max_bat
                        for hist_item in history[idx_best+1:]:
                            current_battery -= (e_motor - e_solar) 
                        if current_battery <= 0: battery_failed = True
                    else:
                        battery_failed = True 
                else:
                    battery_failed = True

            if (r, c) in self.manual_charge_wps:
                energy_needed = max_bat - current_battery
                if p_solar > 0:
                    charge_time_h = energy_needed / p_solar
                    total_charge_time_h += charge_time_h
                current_battery = max_bat

            slopes.append(sl)
            energies.append(e_motor if e_motor != float('inf') else 0.0)
            powers.append(self.e.power_w(sl, rgh))
            dist_total += d3
            motor_energy_total += (e_motor if e_motor != float('inf') else 0.0)
            time_total += t_seg_s
            
            if dz > 0: elev_gain += dz
            else:      elev_loss += abs(dz)
            if sl > SLOPE_COLORS['dikkat'][0]: danger_segs += 1
            prev = cur

        final_time_min = (time_total / 60.0) + (total_charge_time_h * 60.0)

        return dict(
            total_distance_m=dist_total,
            total_energy_wh=motor_energy_total,
            total_time_s=time_total,
            total_time_min=final_time_min,
            avg_slope_deg=float(np.mean(slopes)) if slopes else 0.0,
            max_slope_deg=float(np.max(slopes)) if slopes else 0.0,
            elevation_gain_m=elev_gain,
            elevation_loss_m=elev_loss,
            height_profile=heights,
            slope_profile=slopes,
            energy_profile=energies,
            power_profile=powers,
            danger_segments=danger_segs,
            battery_pct=(current_battery/max_bat)*100.0,
            node_count=len(path),
            battery_failed=battery_failed,
            charge_stops=charge_stops,
            total_charge_time_min=(total_charge_time_h * 60.0)
        )

# ─────────────────────────────────────────────────────────────
# KAYDIRILABİLİR PENCERE
# ─────────────────────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=PANEL)
        self.inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.inner, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        canvas.bind_all('<MouseWheel>', lambda e: canvas.yview_scroll(-1*(e.delta//120), 'units'))

# ─────────────────────────────────────────────────────────────
# ANA UYGULAMA (TUA ROVER)
# ─────────────────────────────────────────────────────────────
class RoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("◈ TUA AY KEŞİF ARACI ROTA PLANLAMA SİSTEMİ v10.1")
        self.root.geometry("1520x960")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 700)

        self.terrain        = None
        self.triangles      = None
        self.energy_model   = RoverEnergyModel()
        self.waypoints      = []      
        self.current_path   = None
        self.segment_paths  = None
        self.route_stats    = None
        self.compare_stats  = {}      
        self.point_mode     = None
        self._hover_ann     = None
        self._computing     = False
        
        self.ros_bridge = TUARosBridge(telemetry_callback=self._telemetry_callback)

        self._init_styles()
        self._build_ui()
        self._bind_hotkeys()

    def _init_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TFrame',            background=PANEL)
        s.configure('Dark.TFrame',       background=PANEL)
        s.configure('Panel2.TFrame',     background=PANEL2)
        s.configure('TLabel',            background=PANEL,  foreground=TEXT,  font=('Consolas',9))
        s.configure('Muted.TLabel',      background=PANEL,  foreground=MUTED, font=('Consolas',8))
        s.configure('Head.TLabel',       background=PANEL,  foreground=ACCENT,font=('Consolas',10,'bold'))
        s.configure('Value.TLabel',      background=PANEL,  foreground='#ffffff',font=('Consolas',10,'bold'))
        s.configure('Accent.TButton',    background=ACCENT, foreground='#000',font=('Consolas',8,'bold'), relief='flat', padding=(6,3))
        s.configure('Success.TButton',   background=SUCCESS,foreground='#fff',font=('Consolas',8,'bold'), relief='flat', padding=(6,3))
        s.configure('Danger.TButton',    background=DANGER, foreground='#fff',font=('Consolas',8,'bold'), relief='flat', padding=(6,3))
        s.configure('Warning.TButton',   background=WARNING,foreground='#000',font=('Consolas',8,'bold'), relief='flat', padding=(6,3))
        s.configure('Ghost.TButton',     background=PANEL2, foreground=TEXT,  font=('Consolas',8),        relief='flat', padding=(6,3))
        s.configure('TNotebook',         background=BG,     borderwidth=0)
        s.configure('TNotebook.Tab',     background=PANEL2, foreground=MUTED, padding=(12,5), font=('Consolas',8,'bold'))
        s.map('TNotebook.Tab', background=[('selected', PANEL)], foreground=[('selected', ACCENT)])
        s.configure('TProgressbar',      troughcolor=BORDER, background=SUCCESS, thickness=4)
        s.configure('TEntry',            fieldbackground='#1e293b', foreground=TEXT, insertcolor=ACCENT)
        s.configure('TCombobox',         fieldbackground='#1e293b', foreground=TEXT)

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=BG, height=44)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  TUA AY KEŞİF ARACI YER KONTROL İSTASYONU",
                 font=('Consolas',13,'bold'), fg=ACCENT, bg=BG).pack(side='left', padx=18, pady=8)
        tk.Label(hdr, text="TÜRKİYE UZAY AJANSI  v10.1",
                 font=('Consolas',8), fg=MUTED, bg=BG).pack(side='right', padx=18)

        tk.Frame(self.root, bg=ACCENT, height=1).pack(fill='x')
        self.progress = ttk.Progressbar(self.root, mode='determinate', style='TProgressbar', length=100, maximum=100)

        pw = tk.PanedWindow(self.root, orient='horizontal', bg=BG, sashwidth=5, sashrelief='flat', sashpad=0)
        pw.pack(fill='both', expand=True, padx=4, pady=4)

        left_wrap = tk.Frame(pw, bg=PANEL, width=320)
        pw.add(left_wrap, minsize=290)
        self._build_left(left_wrap)

        right_wrap = tk.Frame(pw, bg=BG)
        pw.add(right_wrap, minsize=700)
        self._build_right(right_wrap)

        sb = tk.Frame(self.root, bg='#080d18', height=24)
        sb.pack(fill='x')
        sb.pack_propagate(False)
        
        self._live_telemetry_lbl = tk.Label(sb, text="Konum: 0.0, 0.0 | Batarya: %100.0", font=('Consolas',8), fg='#4b6080', bg='#080d18')
        self._live_telemetry_lbl.pack(side='right', padx=8)
        
        self._status_var = tk.StringVar(value="  ● Hazır — STL dosyası yükleyin veya Demo Yüzey oluşturun")
        tk.Label(sb, textvariable=self._status_var,
                 font=('Consolas',8), fg='#4b6080', bg='#080d18', anchor='w').pack(fill='both', expand=True, padx=8)

    def _build_left(self, parent):
        sf = ScrollableFrame(parent, bg=PANEL)
        sf.pack(fill='both', expand=True)
        p = sf.inner
        p.columnconfigure(0, weight=1)

        row = [0]
        def R(): r = row[0]; row[0] += 1; return r
        def section(title):
            tk.Frame(p, bg=BORDER, height=1).grid(row=R(), column=0, sticky='ew', padx=4, pady=(10,0))
            tk.Label(p, text=f"  {title}", font=('Consolas',8,'bold'), fg=ACCENT, bg=PANEL).grid(row=R(), column=0, sticky='w', padx=4)

        section("▸ YÜZEY VERİSİ (TERRAIN)")

        bf = tk.Frame(p, bg=PANEL)
        bf.grid(row=R(), column=0, padx=6, pady=3, sticky='ew')
        bf.columnconfigure((0,1), weight=1)
        ttk.Button(bf, text="📂  STL Yükle",    style='Accent.TButton',  command=self._load_stl ).grid(row=0,column=0,sticky='ew',padx=2)
        ttk.Button(bf, text="🌙  Demo Yüzey",  style='Ghost.TButton',   command=self._demo     ).grid(row=0,column=1,sticky='ew',padx=2)

        self._file_lbl = tk.Label(p, text="  Dosya: —", font=('Consolas',7), fg=MUTED, bg=PANEL, anchor='w', wraplength=280)
        self._file_lbl.grid(row=R(), column=0, sticky='ew', padx=4)

        rf = tk.Frame(p, bg=PANEL)
        rf.grid(row=R(), column=0, padx=6, pady=2, sticky='ew')
        rf.columnconfigure(1, weight=1)
        
        tk.Label(rf, text="Grid Çözünürlük:", font=('Consolas',8), fg=MUTED, bg=PANEL).grid(row=0,column=0,sticky='w', pady=2)
        self._res_var = tk.IntVar(value=150)
        self._res_lbl = tk.Label(rf, text="150", font=('Consolas',8,'bold'), fg=ACCENT, bg=PANEL, width=4)
        self._res_lbl.grid(row=0, column=2, sticky='w')
        sc = tk.Scale(rf, from_=20, to=1000, orient='horizontal', variable=self._res_var,
                      bg=PANEL, fg=ACCENT, troughcolor=BORDER, highlightthickness=0,
                      showvalue=False, length=110, command=lambda v: self._res_lbl.config(text=str(v)))
        sc.grid(row=0, column=1, sticky='ew', padx=4)

        tk.Label(rf, text="Harita X (km):", font=('Consolas',8), fg=MUTED, bg=PANEL).grid(row=2,column=0,sticky='w', pady=2)
        self._map_x_var = tk.StringVar(value="")
        tk.Entry(rf, textvariable=self._map_x_var, font=('Consolas',8), bg='#1a2540', fg=TEXT, insertbackground=ACCENT, relief='flat').grid(row=2, column=1, sticky='ew', padx=4)
        tk.Label(rf, text="Boş=Oto", font=('Consolas',7), fg='#4b6080', bg=PANEL).grid(row=2, column=2, sticky='w')

        tk.Label(rf, text="Harita Y (km):", font=('Consolas',8), fg=MUTED, bg=PANEL).grid(row=3,column=0,sticky='w', pady=2)
        self._map_y_var = tk.StringVar(value="")
        tk.Entry(rf, textvariable=self._map_y_var, font=('Consolas',8), bg='#1a2540', fg=TEXT, insertbackground=ACCENT, relief='flat').grid(row=3, column=1, sticky='ew', padx=4)
        tk.Label(rf, text="Boş=Oto", font=('Consolas',7), fg='#4b6080', bg=PANEL).grid(row=3, column=2, sticky='w')

        ttk.Button(rf, text="⟳ Yeniden Yükle/Ölçekle", style='Ghost.TButton', command=self._process_terrain).grid(row=4, column=0, columnspan=3, sticky='ew', pady=5)

        section("▸ TUA ROVER & PANEL PARAMETRELERİ")

        self._param_vars = {}
        param_defs = [
            ('mass_kg',          'Kütle',          'kg',  0, 3000,  10),
            ('max_torque_nm',    'Maks Tork',       'Nm',  0.5,500,  0.5),
            ('battery_wh',       'Batarya',         'Wh',  100,10000,100),
            ('max_slope_deg',    'Maks Eğim',       '°',   5,  60,   1),
            ('solar_area_m2',    'Panel Alanı',     'm²',  0.5, 10.0, 0.1),
            ('solar_efficiency', 'Panel Verimi',    '%',   0.1, 0.5, 0.01),
            # Güneş parametrelerini buradan sildik, onlara özel panel yapacağız.
        ]

        pf = tk.Frame(p, bg=PANEL)
        pf.grid(row=R(), column=0, padx=6, pady=2, sticky='ew')
        pf.columnconfigure(1, weight=1)

        for i, (key, label, unit, lo, hi, step) in enumerate(param_defs):
            tk.Label(pf, text=f"{label}:", font=('Consolas',8), fg=MUTED, bg=PANEL, anchor='w').grid(row=i, column=0, sticky='w', pady=1)
            var = tk.StringVar(value=str(ROVER_DEFAULTS[key]))
            e = tk.Entry(pf, textvariable=var, width=9, font=('Consolas',8), bg='#1a2540', fg=TEXT,
                         insertbackground=ACCENT, relief='flat', highlightthickness=1, highlightcolor=ACCENT, highlightbackground=BORDER)
            e.grid(row=i, column=1, sticky='ew', padx=3, pady=1)
            if unit: tk.Label(pf, text=unit, font=('Consolas',7), fg='#4b6080', bg=PANEL).grid(row=i, column=2, sticky='w')
            self._param_vars[key] = var

        ttk.Button(p, text="⟳  Parametreleri Uygula", style='Success.TButton',
                   command=self._apply_params).grid(row=R(), column=0, padx=6, pady=4, sticky='ew')

        self._force_lbl = tk.Label(p, text="Maks Çekiş: — N", font=('Consolas',8), fg='#7dd3fc', bg=PANEL2, anchor='center', relief='flat', pady=3)
        self._force_lbl.grid(row=R(), column=0, padx=6, sticky='ew')
        self._update_force_label()
        section("▸ GÜNEŞ SİMÜLASYONU (GELİŞ AÇISI)")

        sun_f = tk.Frame(p, bg=PANEL)
        sun_f.grid(row=R(), column=0, padx=6, pady=4, sticky='ew')
        sun_f.columnconfigure(1, weight=1)

        tk.Label(sun_f, text="Güneş Yüksekliği:", font=('Consolas',8), fg=MUTED, bg=PANEL).grid(row=0, column=0, sticky='w')
        self._sun_elev_var = tk.DoubleVar(value=ROVER_DEFAULTS['sun_elevation_deg'])
        elev_lbl = tk.Label(sun_f, text=f"{self._sun_elev_var.get()}°", font=('Consolas',8,'bold'), fg=WARNING, bg=PANEL, width=5)
        elev_lbl.grid(row=0, column=2, sticky='e')
        tk.Scale(sun_f, from_=0, to=90, orient='horizontal', variable=self._sun_elev_var,
                 bg=PANEL, fg=WARNING, troughcolor=BORDER, highlightthickness=0, showvalue=False,
                 command=lambda v: [elev_lbl.config(text=f"{float(v):.0f}°"), self._update_solar_preview()]).grid(row=0, column=1, sticky='ew', padx=4)

        tk.Label(sun_f, text="Güneş Yönü (Azimut):", font=('Consolas',8), fg=MUTED, bg=PANEL).grid(row=1, column=0, sticky='w')
        self._sun_az_var = tk.DoubleVar(value=ROVER_DEFAULTS['sun_azimuth_deg'])
        az_lbl = tk.Label(sun_f, text=f"{self._sun_az_var.get()}°", font=('Consolas',8,'bold'), fg=WARNING, bg=PANEL, width=5)
        az_lbl.grid(row=1, column=2, sticky='e')
        tk.Scale(sun_f, from_=0, to=360, orient='horizontal', variable=self._sun_az_var,
                 bg=PANEL, fg=WARNING, troughcolor=BORDER, highlightthickness=0, showvalue=False,
                 command=lambda v: [az_lbl.config(text=f"{float(v):.0f}°"), self._update_solar_preview()]).grid(row=1, column=1, sticky='ew', padx=4)

        self._solar_preview_lbl = tk.Label(sun_f, text="Düz Zeminde Maks Üretim: — W", font=('Consolas',8), fg=SUCCESS, bg=PANEL)
        self._solar_preview_lbl.grid(row=2, column=0, columnspan=3, pady=(4,0))
        
        # Paneli ilk açılışta güncellemesi için tetikliyoruz
        self.root.after(100, self._update_solar_preview)

        section("▸ SÜRÜŞ VE ŞARJ MODU")

        mf = tk.Frame(p, bg=PANEL)
        mf.grid(row=R(), column=0, padx=6, pady=4, sticky='ew')
        mf.columnconfigure((0,1), weight=1)
        self._mode_var = tk.StringVar(value='dengeli')
        modes = [
            ('⚖  Dengeli',  'dengeli'),
            ('🛡  Güvenli',  'guvenli'),
            ('⚡  Enerjili', 'enerji'),
            ('🚀  Hızlı',   'hizli'),
        ]
        for i, (lbl, val) in enumerate(modes):
            rb = tk.Radiobutton(mf, text=lbl, variable=self._mode_var, value=val,
                                font=('Consolas',8), fg=TEXT, bg=PANEL, selectcolor=PANEL2, activebackground=PANEL,
                                indicatoron=0, relief='flat', borderwidth=0, pady=4, padx=6, cursor='hand2')
            rb.grid(row=i//2, column=i%2, sticky='ew', padx=2, pady=1)

        self._auto_charge_var = tk.BooleanVar(value=True)
        tk.Checkbutton(p, text=" ⚡ Akıllı Şarj (Batarya yetmezse oto-durak ekle)", variable=self._auto_charge_var, font=('Consolas',8),
                       fg=SUCCESS, bg=PANEL, selectcolor=PANEL2, activebackground=PANEL, activeforeground=ACCENT).grid(row=R(), column=0, padx=8, sticky='w')

        section("▸ GÜZERGÂH VE ŞARJ İSTASYONLARI")

        wf = tk.Frame(p, bg=PANEL)
        wf.grid(row=R(), column=0, padx=6, pady=4, sticky='ew')
        wf.columnconfigure((0,1,2,3), weight=1)
        ttk.Button(wf, text="🟢 Baş.", style='Success.TButton', command=lambda: self._set_mode('start')).grid(row=0,column=0,sticky='ew',padx=1)
        ttk.Button(wf, text="🔵 Ara", style='Accent.TButton', command=lambda: self._set_mode('waypoint')).grid(row=0,column=1,sticky='ew',padx=1)
        ttk.Button(wf, text="🟡 Şarj", style='Warning.TButton', command=lambda: self._set_mode('charge')).grid(row=0,column=2,sticky='ew',padx=1)
        ttk.Button(wf, text="🔴 Bitiş", style='Danger.TButton', command=lambda: self._set_mode('end')).grid(row=0,column=3,sticky='ew',padx=1)

        self._mode_lbl = tk.Label(p, text="  Tıklama modu: —", font=('Consolas',8,'bold'), fg=WARNING, bg=PANEL2, pady=3, anchor='w')
        self._mode_lbl.grid(row=R(), column=0, padx=6, sticky='ew')

        self._wp_list = tk.Listbox(p, height=5, bg='#0f1a2e', fg=TEXT, font=('Consolas',8), relief='flat',
                                   selectbackground='#1e4080', selectforeground='#ffffff', highlightthickness=1,
                                   highlightcolor=BORDER, highlightbackground=BORDER)
        self._wp_list.grid(row=R(), column=0, padx=6, pady=2, sticky='ew')
        self._wp_list.bind('<Double-Button-1>', self._remove_selected_wp)

        wf2 = tk.Frame(p, bg=PANEL)
        wf2.grid(row=R(), column=0, padx=6, pady=2, sticky='ew')
        wf2.columnconfigure((0,1,2), weight=1)
        ttk.Button(wf2, text="✕ Son Sil",   style='Danger.TButton',  command=self._pop_wp).grid(row=0,column=0,sticky='ew',padx=1)
        ttk.Button(wf2, text="💾 Kaydet",   style='Ghost.TButton',   command=self._save_wp).grid(row=0,column=1,sticky='ew',padx=1)
        ttk.Button(wf2, text="📂 Yükle",    style='Ghost.TButton',   command=self._load_wp).grid(row=0,column=2,sticky='ew',padx=1)

        section("▸ ROTA KONTROLÜ")

        # Eksik olan arayüz değişkenleri ve kontroller eklendi
        self._compare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(p, text=" Tüm modları karşılaştır", variable=self._compare_var, font=('Consolas',8),
                       fg=TEXT, bg=PANEL, selectcolor=PANEL2, activebackground=PANEL, activeforeground=ACCENT).grid(row=R(), column=0, padx=8, sticky='w')

        self._eps_var = tk.DoubleVar(value=1.5) # Gizli epsilon değeri (A* algoritması için)

        cf = tk.Frame(p, bg=PANEL)
        cf.grid(row=R(), column=0, padx=6, pady=4, sticky='ew')
        cf.columnconfigure((0,1), weight=1)
        self._calc_btn = ttk.Button(cf, text="◈  ROTAYI HESAPLA", style='Accent.TButton', command=self._compute)
        self._calc_btn.grid(row=0, column=0, sticky='ew', padx=1, ipady=4)
        ttk.Button(cf, text="🗑  Sıfırla", style='Danger.TButton', command=self._reset).grid(row=0, column=1, sticky='ew', padx=1, ipady=4)

        section("▸ ROS ENTEGRASYONU (NAV2)")
        ros_f = tk.Frame(p, bg=PANEL)
        ros_f.grid(row=R(), column=0, padx=6, pady=4, sticky='ew')
        ros_f.columnconfigure(1, weight=1)
        
        tk.Label(ros_f, text="IP/Host:", font=('Consolas',8), fg=MUTED, bg=PANEL).grid(row=0, column=0, sticky='w')
        self._ros_ip_var = tk.StringVar(value='127.0.0.1')
        tk.Entry(ros_f, textvariable=self._ros_ip_var, font=('Consolas',8), bg='#1a2540', fg=TEXT, width=12,
                 insertbackground=ACCENT, relief='flat').grid(row=0, column=1, sticky='ew', padx=3, pady=1)
                 
        ttk.Button(ros_f, text="🔗 Bağlan", style='Success.TButton', command=self._connect_ros).grid(row=0, column=2, sticky='ew', padx=1)
        ttk.Button(ros_f, text="📡 ROTAYI ROBOTA GÖNDER (Nav2)", style='Accent.TButton', command=self._publish_route).grid(row=2, column=0, columnspan=3, sticky='ew', pady=(4,2), ipady=3)

        section("▸ ROTA İSTATİSTİKLERİ")

        sf2 = tk.Frame(p, bg=PANEL2)
        sf2.grid(row=R(), column=0, padx=6, pady=4, sticky='ew')
        sf2.columnconfigure(1, weight=1)

        self._stat_widgets = {}
        stat_defs = [
            ('dist',     '📏 Top. Mesafe',  '—',   None),
            ('time',     '⏱ Top. Süre',     '—',   None),
            ('energy',   '⚡ Enerji',       '—',   None),
            ('battery',  '🔋 Batarya',      '—',   None),
            ('max_slope','📐 Maks Eğim',    '—',   None),
            ('elev_up',  '⬆ Yükseliş',      '—',   None),
            ('elev_dn',  '⬇ Alçalış',       '—',   None),
            ('danger',   '⚠ Risk Segm.',    '—',   None),
        ]
        for i, (k, lbl, default, _) in enumerate(stat_defs):
            tk.Label(sf2, text=f"  {lbl}:", font=('Consolas',8), fg=MUTED, bg=PANEL2).grid(row=i, column=0, sticky='w', pady=1, padx=4)
            v = tk.Label(sf2, text=default, font=('Consolas',9,'bold'), fg=ACCENT, bg=PANEL2, anchor='e')
            v.grid(row=i, column=1, sticky='e', pady=1, padx=6)
            self._stat_widgets[k] = v

        # hpad() kaldırıldı, yerine alt hizalama için sabit boşluk eklendi
        tk.Frame(p, bg=PANEL, height=20).grid(row=R(), column=0)

    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill='both', expand=True)

        t3 = tk.Frame(nb, bg=BG)
        nb.add(t3, text='  🌍 TUA 3D GÖRÜNTÜLEYİCİ (GPU)  ')
        
        info_frame = tk.Frame(t3, bg='#0d1520', bd=2, relief='groove')
        info_frame.place(relx=0.5, rely=0.5, anchor='center', width=500, height=200)
        
        tk.Label(info_frame, text="SİNEMATİK 3D RENDER MOTORU\n(PyVista & OpenGL)", font=('Consolas', 14, 'bold'), fg=ACCENT, bg='#0d1520').pack(pady=(30, 10))
        tk.Label(info_frame, text="Yüksek çözünürlüklü donanım ivmeli (GPU) 3D Haritayı\nayrı bir performans penceresinde başlatmak için tıklayın.", font=('Consolas', 10), fg=TEXT, bg='#0d1520').pack(pady=(0, 20))
        
        ttk.Button(info_frame, text="🚀 SİNEMATİK MOTORU BAŞLAT", style='Success.TButton', command=self._launch_pyvista).pack(ipady=10, ipadx=20)

        # [UI FIX]: Eksik sekmeler tek tek geri yüklendi!
        ts = tk.Frame(nb, bg=BG)
        nb.add(ts, text='  📐 EĞİM VE RİSK HARİTASI  ')
        self._fig_sl, self._ax_sl, self._cv_sl = self._make_fig(ts, '2d', toolbar=True)
        self._cv_sl.mpl_connect('button_press_event', self._on_map_click)
        self._cv_sl.mpl_connect('motion_notify_event', self._on_map_hover)

        tp = tk.Frame(nb, bg=BG)
        nb.add(tp, text='  📊 YÜKSEKLİK PROFİLİ  ')
        self._fig_pr, self._ax_pr, self._cv_pr = self._make_fig(tp, '2d')

        te = tk.Frame(nb, bg=BG)
        nb.add(te, text='  ⚡ ENERJİ ANALİZİ  ')
        self._fig_en, self._ax_en, self._cv_en = self._make_fig(te, '2d')

        tc = tk.Frame(nb, bg=BG)
        nb.add(tc, text='  ⚖ SÜRÜŞ MODU KIYASLAMASI  ')
        self._fig_cm, self._cv_cm = self._make_compare_fig(tc)

        tph = tk.Frame(nb, bg=BG)
        nb.add(tph, text='  🔬 ARAÇ FİZİK LİMİTLERİ  ')
        self._fig_ph, self._cv_ph = self._make_physics_fig(tph)
        self._draw_physics()

        self._nb = nb

    def _make_fig(self, parent, kind, toolbar=False):
        fig = Figure(facecolor=BG, tight_layout=True)
        ax = fig.add_subplot(111, facecolor='#0d1520')
        self._style_2d(ax)
        cv = FigureCanvasTkAgg(fig, master=parent)
        cv.get_tk_widget().pack(fill='both', expand=True)
        if toolbar:
            tb_frame = tk.Frame(parent, bg='#0d1520')
            tb_frame.pack(fill='x')
            tb = NavigationToolbar2Tk(cv, tb_frame)
            tb.configure(background='#0d1520')
            tb.update()
        return fig, ax, cv

    def _style_2d(self, ax):
        ax.set_facecolor('#0d1520')
        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    def _launch_pyvista(self):
        if not PYVISTA_AVAILABLE:
            messagebox.showerror("Hata", "PyVista kütüphanesi bulunamadı!\nTerminalde: pip install pyvista")
            return
            
        if self.terrain is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir STL haritası yükleyin veya Demo Yüzey oluşturun.")
            return

        self._set_status("⟳ PyVista GPU Motoru Başlatılıyor...")
        self.root.update()

        try:
            plotter = pv.Plotter(title="TUA Ay Keşif Aracı - Sinematik 3D Motor (GPU İvmeli)")
            plotter.set_background('#0b0f1e')

            t = self.terrain
            grid = pv.StructuredGrid(t.xi, t.yi, t.zi)
            grid["Yükseklik"] = t.zi.flatten(order="F")
            
            plotter.add_mesh(grid, scalars="Yükseklik", cmap="gray", 
                             show_scalar_bar=True, smooth_shading=True,
                             specular=0.0, ambient=0.15, diffuse=0.9,
                             scalar_bar_args={'title': 'Yükseklik (m)'})

            x_range = t.x_max - t.x_min
            z_range = t.z_max - t.z_min
            z_exag = max(1.0, (x_range / max(z_range, 1.0)) * 0.05)
            z_exag = np.clip(z_exag, 1.0, 50.0) 
            plotter.set_scale(zscale=z_exag)

            z_offset = (t.z_max - t.z_min) * 0.05 + 1.0
            point_radius = (t.x_max - t.x_min) * 0.004
            
            for wx, wy, wz, lbl in self.waypoints:
                color = 'green' if lbl == 'BAŞLANGIÇ' else 'red' if lbl == 'BİTİŞ' else 'yellow' if lbl == 'ŞARJ' else 'blue'
                sphere = pv.Sphere(radius=point_radius, center=(wx, wy, wz + z_offset))
                plotter.add_mesh(sphere, color=color, smooth_shading=True)
                plotter.add_point_labels(np.array([[wx, wy, wz + z_offset * 2]]), [lbl], 
                                         point_size=0, text_color='white', 
                                         shape_color=color, font_size=20, margin=3, shape_opacity=0.7)

            if self.route_stats and self.route_stats.get('charge_stops'):
                for (r, c) in self.route_stats['charge_stops']:
                    x, y, z = t.grid_to_world(r, c)
                    sphere = pv.Sphere(radius=point_radius*1.2, center=(x, y, z + z_offset))
                    plotter.add_mesh(sphere, color='orange', smooth_shading=True)
                    plotter.add_point_labels(np.array([[x, y, z + z_offset * 2]]), ["OTO ŞARJ"], 
                                         point_size=0, text_color='black', 
                                         shape_color='orange', font_size=20, margin=3, shape_opacity=0.9)


            tube_radius = (t.x_max - t.x_min) * 0.0008
            
            if self.current_path:
                seg_cols = ['cyan', 'magenta', 'orange', 'green', 'blue', 'yellow']
                if self.segment_paths:
                    for i, seg in enumerate(self.segment_paths):
                        if len(seg) < 2: continue
                        pts = np.array([t.grid_to_world(r,c) for r,c in seg])
                        pts[:, 2] += z_offset * 0.5
                        spline = pv.Spline(pts, len(pts))
                        tube = spline.tube(radius=tube_radius)
                        plotter.add_mesh(tube, color=seg_cols[i % len(seg_cols)], smooth_shading=True)
                else:
                    pts = np.array([t.grid_to_world(r,c) for r,c in self.current_path])
                    pts[:, 2] += z_offset * 0.5
                    spline = pv.Spline(pts, len(pts))
                    tube = spline.tube(radius=tube_radius)
                    plotter.add_mesh(tube, color='cyan', smooth_shading=True)

            if self.route_stats:
                s = self.route_stats
                hud_text = (f"TUA ROTA ANALIZI\n"
                            f"----------------\n"
                            f"Mesafe : {s['total_distance_m']:.1f} m\n"
                            f"Sure   : {s['total_time_min']:.1f} dk\n"
                            f"Motor E: {s['total_energy_wh']:.1f} Wh\n"
                            f"Sarj M.: {s['total_charge_time_min']:.1f} dk\n"
                            f"Batarya: %{s['battery_pct']:.1f}\n")
                
                if s.get('battery_failed'):
                    hud_text += "\n[DIKKAT: BATARYA YETERSIZ!]"
                    
                plotter.add_text(hud_text, position='upper_left', color='#00d4ff', font_size=11, font='courier', shadow=True)

            plotter.show_bounds(grid='front', location='outer', all_edges=True,
                                xlabel='X (Metre)', ylabel='Y (Metre)', zlabel='Z (Metre)',
                                font_size=10, color='white')

            plotter.enable_eye_dome_lighting() 
            plotter.enable_anti_aliasing('ssaa') 
            
            self._set_status("✓ PyVista GPU Motoru Aktif!")
            plotter.show()
            self._set_status("  ● Hazır")
        except Exception as e:
            messagebox.showerror("PyVista Hatası", f"3D Render başarısız:\n{e}")
            self._set_status("✕ 3D Render Hatası")

    def _draw_slope_map(self):
        if self.terrain is None: return
        t = self.terrain
        
        self._fig_sl.clf()
        self._ax_sl = self._fig_sl.add_subplot(111, facecolor='#0d1520')
        ax = self._ax_sl
        self._style_2d(ax)

        im = ax.contourf(t.xi, t.yi, t.slope_deg, levels=np.linspace(0, min(t.slope_deg.max()+1, 45), 60), cmap='RdYlGn_r', vmin=0, vmax=35)
        ax.contour(t.xi, t.yi, t.zi, levels=12, colors='white', alpha=0.12, linewidths=0.6)
        for lo, hi, col in SLOPE_COLORS.values():
            ax.contour(t.xi, t.yi, t.slope_deg, levels=[lo], colors=[col], linewidths=1.2, alpha=0.75)

        try:
            self._fig_sl.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label='Eğim (°)')
        except Exception: pass

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('TUA Eğim ve Risk Haritası  (Tıkla → Nokta Ekle)', color=ACCENT, fontsize=9, fontfamily='monospace')
        
        self._draw_wps_slope()
        self._draw_route_slope()
        self._cv_sl.draw_idle()
                            
    def _draw_wps_slope(self):
        icon_map = {'BAŞLANGIÇ':('*',SUCCESS,160), 'ARA_NOKTA':('o',ACCENT2,100), 'ŞARJ':('P',WARNING,150), 'BİTİŞ':('*',ACCENT,160)}
        ax = self._ax_sl
        for x,y,z,lbl in self.waypoints:
            mk,col,sz = icon_map.get(lbl, ('o','white',80))
            ax.scatter([x],[y], c=[col], s=sz, marker=mk, zorder=10, edgecolors='white', linewidths=0.8)
            ax.annotate(lbl, (x,y), xytext=(6,6), textcoords='offset points', color=col, fontsize=7, fontfamily='monospace', fontweight='bold')
            
        if self.route_stats and self.route_stats.get('charge_stops'):
            for (r, c) in self.route_stats['charge_stops']:
                x, y, z = self.terrain.grid_to_world(r, c)
                ax.scatter([x],[y], c=['orange'], s=150, marker='P', zorder=11, edgecolors='black', linewidths=0.8)
                ax.annotate("OTO ŞARJ", (x,y), xytext=(6,6), textcoords='offset points', color='orange', fontsize=7, fontfamily='monospace', fontweight='bold')


    def _draw_route_slope(self):
        if not self.current_path: return
        t   = self.terrain
        ax  = self._ax_sl
        seg_cols = [ACCENT2, '#f472b6', WARNING, SUCCESS, '#38bdf8', '#a3e635']
        
        if self.segment_paths:
            for i, seg in enumerate(self.segment_paths):
                if len(seg) == 0: continue
                xs = [t.grid_to_world(r,c)[0] for r,c in seg]
                ys = [t.grid_to_world(r,c)[1] for r,c in seg]
                col = seg_cols[i % len(seg_cols)]
                ax.plot(xs,ys, color=col, linewidth=2.5, alpha=0.9, zorder=8)
                
                if self.route_stats and 'segment_times' in self.route_stats:
                    time_m = self.route_stats['segment_times'][i]
                    dist_m = self.route_stats['segment_dists'][i]
                    mid = len(seg) // 2
                    xm,ym,_ = t.grid_to_world(*seg[mid])
                    
                    txt = f"[{i+1}. Rota]\n{dist_m:.0f}m | {time_m:.1f}dk"
                    ax.annotate(txt, (xm,ym), color='white', fontsize=7, fontfamily='monospace', ha='center',
                                bbox=dict(boxstyle='round,pad=0.3', fc='#111827', alpha=0.85, ec=col))

    def _draw_profile(self):
        if not self.route_stats: return
        stats  = self.route_stats
        
        self._fig_pr.clf()
        self._ax_pr = self._fig_pr.add_subplot(111, facecolor='#0d1520')
        ax = self._ax_pr
        self._style_2d(ax)

        heights = stats['height_profile']
        slopes  = stats['slope_profile']
        n       = len(heights)
        d_step  = stats['total_distance_m'] / max(n-1, 1)
        dists   = [i*d_step for i in range(n)]

        ax.fill_between(dists, heights, min(heights)-0.5, alpha=0.25, color=ACCENT2)
        ax.plot(dists, heights, color=ACCENT2, linewidth=2, label='Yükseklik (m)')

        ax2 = ax.twinx()
        d_s = [i*d_step for i in range(len(slopes))]
        ax2.plot(d_s, slopes, color=WARNING, linewidth=1.2, alpha=0.7, label='Eğim (°)')
        ax2.fill_between(d_s, slopes, 0, alpha=0.1, color=WARNING)
        for thresh, col in [(10,'#2ecc71'),(20,'#f1c40f'),(28,'#e74c3c')]:
            ax2.axhline(thresh, color=col, linestyle=':', linewidth=0.8, alpha=0.6)
        ax2.set_ylabel('Eğim (°)', color=WARNING, fontsize=8)
        ax2.tick_params(colors=WARNING, labelsize=6)
        ax2.set_ylim(0, max(max(slopes)*1.3, 35) if slopes else 35)
        for sp in ax2.spines.values(): sp.set_color(BORDER)

        ax.set_xlabel('Mesafe (m)', fontsize=8)
        ax.set_ylabel('Yükseklik (m)', color=ACCENT2, fontsize=8)
        ax.set_title('Yükseklik ve Eğim Profili', color=ACCENT2, fontsize=9, fontfamily='monospace')
        ax.tick_params(colors=MUTED, labelsize=6)
        self._fig_pr.tight_layout()
        self._cv_pr.draw_idle()

    def _draw_energy(self):
        if not self.route_stats: return
        stats  = self.route_stats
        
        self._fig_en.clf()
        self._ax_en = self._fig_en.add_subplot(111, facecolor='#0d1520')
        ax = self._ax_en
        self._style_2d(ax)

        energies = stats.get('energy_profile', [])
        if not energies: return

        norm    = plt.Normalize(0, max(max(energies), 1e-6))
        colors  = [plt.cm.plasma(norm(e)) for e in energies]
        ax.bar(range(len(energies)), energies, color=colors, alpha=0.85, width=1.0, edgecolor='none')

        cumul  = np.cumsum(energies)
        ax2    = ax.twinx()
        ax2.plot(range(len(cumul)), cumul, color=ACCENT2, linewidth=2, label='Kümülatif (Wh)')
        ax2.axhline(self.energy_model.p['battery_wh'], color=DANGER, linestyle='--', linewidth=1.2, alpha=0.7,
                    label=f"Batarya ({self.energy_model.p['battery_wh']:.0f} Wh)")
        ax2.set_ylabel('Kümülatif Motor Enerjisi (Wh)', color=ACCENT2, fontsize=8)
        ax2.tick_params(colors=ACCENT2, labelsize=6)
        ax2.legend(loc='upper left', fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        for sp in ax2.spines.values(): sp.set_color(BORDER)

        ax.set_xlabel('Rota Adımı', fontsize=8)
        ax.set_ylabel('Segment Motor Enerjisi (Wh)', fontsize=8)
        ax.set_title('Enerji Tüketim Analizi (Sadece Motor)', color=ACCENT2, fontsize=9, fontfamily='monospace')
        self._fig_en.tight_layout()
        self._cv_en.draw_idle()

    def _make_compare_fig(self, parent):
        fig = Figure(facecolor=BG, tight_layout=True)
        cv  = FigureCanvasTkAgg(fig, master=parent)
        cv.get_tk_widget().pack(fill='both', expand=True)
        return fig, cv

    def _draw_compare(self):
        fig = self._fig_cm
        fig.clear()
        if not self.compare_stats:
            ax = fig.add_subplot(111, facecolor='#0d1520')
            self._style_2d(ax)
            ax.text(0.5,0.5,"Karşılaştırma için 'Tüm modları karşılaştır'\nseçeneğini işaretleyip rotayı hesaplayın.",
                    transform=ax.transAxes, color=MUTED, ha='center', va='center', fontsize=10, fontfamily='monospace')
            self._cv_cm.draw_idle()
            return

        modes = list(self.compare_stats.keys())
        cols  = [ACCENT2, SUCCESS, WARNING, '#a78bfa']
        metrics = ['total_distance_m','total_energy_wh','total_time_min','max_slope_deg']
        mlabels = ['Mesafe (m)','Enerji (Wh)','Süre (dk)','Maks Eğim (°)']

        axes = [fig.add_subplot(2,2,i+1,facecolor='#0d1520') for i in range(4)]
        for ax,metric,mlbl in zip(axes, metrics, mlabels):
            self._style_2d(ax)
            vals  = [self.compare_stats[m].get(metric, 0) for m in modes]
            bars  = ax.bar(modes, vals, color=cols[:len(modes)], alpha=0.8, width=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02, f"{v:.1f}",
                        ha='center', va='bottom', color=TEXT, fontsize=7, fontfamily='monospace')
            ax.set_title(mlbl, color=ACCENT2, fontsize=8, fontfamily='monospace')
            ax.tick_params(labelsize=7, colors=MUTED)

        fig.suptitle('Sürüş Modu Karşılaştırması', color=ACCENT2, fontsize=10, fontfamily='monospace', y=1.01)
        fig.tight_layout()
        self._cv_cm.draw_idle()

    def _make_physics_fig(self, parent):
        fig = Figure(facecolor=BG, tight_layout=True)
        cv  = FigureCanvasTkAgg(fig, master=parent)
        cv.get_tk_widget().pack(fill='both', expand=True)
        return fig, cv

    def _draw_physics(self):
        fig = self._fig_ph
        fig.clear()
        em  = self.energy_model
        slopes = np.linspace(0, em.p['max_slope_deg'], 200)

        axes = [fig.add_subplot(2,3,i+1, facecolor='#0d1520') for i in range(6)]
        for ax in axes: self._style_2d(ax)

        ax1,ax2,ax3,ax4,ax5,ax6 = axes

        forces = [em.required_force(s) for s in slopes]
        ax1.plot(slopes, forces, color=ACCENT2, linewidth=2)
        ax1.axhline(em.max_available_force(), color=DANGER, linestyle='--', linewidth=1.5, label=f'Maks Çekiş={em.max_available_force():.0f}N')
        ax1.fill_between(slopes, forces, 0, alpha=0.2, color=ACCENT2)
        ax1.set_title('Gerekli Kuvvet (N)', color=ACCENT2, fontsize=8)
        ax1.legend(fontsize=6, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

        energies_m = [em.energy_wh(s, 1.0) if em.can_traverse(s) else 0 for s in slopes]
        ax2.plot(slopes, energies_m, color=WARNING, linewidth=2)
        ax2.fill_between(slopes, energies_m, 0, alpha=0.2, color=WARNING)
        ax2.set_title('Enerji / Metre (Wh/m)', color=ACCENT2, fontsize=8)

        speeds = [em.speed_ms(s)*100 for s in slopes] 
        ax3.plot(slopes, speeds, color=SUCCESS, linewidth=2)
        ax3.fill_between(slopes, speeds, 0, alpha=0.2, color=SUCCESS)
        ax3.set_title('Hız (cm/s)', color=ACCENT2, fontsize=8)

        tractions = [em.traction_ratio(s)*100 for s in slopes]
        colors_tr = [DANGER if t>80 else WARNING if t>50 else SUCCESS for t in tractions]
        ax4.scatter(slopes, tractions, c=colors_tr, s=4, alpha=0.8)
        ax4.axhline(100, color=DANGER, linestyle='--', linewidth=1)
        ax4.set_title('Tork Kullanımı (%)', color=ACCENT2, fontsize=8)

        safety = [em.safety_score(s)*100 for s in slopes]
        ax5.fill_between(slopes, safety, 0, alpha=0.3, color='#2ecc71')
        ax5.plot(slopes, safety, color='#2ecc71', linewidth=2)
        ax5.set_title('Güvenlik Skoru (%)', color=ACCENT2, fontsize=8)

        powers = [em.power_w(s) for s in slopes]
        ax6.plot(slopes, powers, color='#a78bfa', linewidth=2)
        ax6.fill_between(slopes, powers, 0, alpha=0.2, color='#a78bfa')
        ax6.set_title('Motor Gücü (W)', color=ACCENT2, fontsize=8)

        for ax in axes:
            ax.set_xlabel('Eğim (°)', fontsize=7)
            ax.tick_params(labelsize=6, colors=MUTED)
            ax.axvline(em.p['safe_slope_deg'],  color='#2ecc71', linestyle=':', alpha=0.4, linewidth=1)
            ax.axvline(em.p['max_slope_deg'],   color=DANGER,    linestyle=':', alpha=0.4, linewidth=1)

        fig.suptitle('Araç Fizik Limitleri Analizi', color=ACCENT2, fontsize=10, fontfamily='monospace')
        fig.tight_layout()
        self._cv_ph.draw_idle()

    def _on_map_click(self, event):
        if not self.point_mode or self.terrain is None: return
        if event.inaxes != self._ax_sl: return
        if event.xdata is None or event.ydata is None: return
        self._add_waypoint(event.xdata, event.ydata)

    def _on_map_hover(self, event):
        if self.terrain is None or event.inaxes != self._ax_sl: return
        if event.xdata is None or event.ydata is None: return
        t    = self.terrain
        r, c = t.world_to_grid(event.xdata, event.ydata)
        sl   = t.get_slope(r, c)
        ht   = t.get_height(r, c)
        
        asp  = t.get_aspect(r, c)
        p_solar = self.energy_model.solar_power_w(sl, asp)
        
        can  = "✓" if self.energy_model.can_traverse(sl) else "✕"
        txt  = f"X={event.xdata:.1f} Y={event.ydata:.1f}\nZ={ht:.2f}m Eğim={sl:.1f}° {can}\n☀️ Panel Üretim: {p_solar:.1f}W"

        if self._hover_ann:
            try: self._hover_ann.remove()
            except: pass
        self._hover_ann = self._ax_sl.annotate(txt, (event.xdata, event.ydata), xytext=(12, 12), textcoords='offset points',
            fontsize=7, fontfamily='monospace', color=TEXT, bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, alpha=0.85, ec=ACCENT2, lw=0.8), zorder=20)
        self._cv_sl.draw_idle()

    def _set_mode(self, mode):
        if self.terrain is None:
            messagebox.showwarning("Uyarı", "Önce bir yüzey verisi (Terrain) yükleyin!")
            return
        self.point_mode = mode
        msgs = {'start':  '🟢 Harita üzerinde BAŞLANGIÇ noktasını tıklayın',
                'waypoint':'🔵 Harita üzerinde ARA NOKTA tıklayın',
                'charge':'🟡 Harita üzerinde ŞARJ NOKTASI tıklayın',
                'end':    '🔴 Harita üzerinde BİTİŞ noktasını tıklayın'}
        self._mode_lbl.config(text=f"  {msgs[mode]}")
        self._set_status(msgs[mode])
        self._nb.select(1)

    def _add_waypoint(self, x, y):
        t    = self.terrain
        x    = float(np.clip(x, t.x_min, t.x_max))
        y    = float(np.clip(y, t.y_min, t.y_max))
        r, c = t.world_to_grid(x, y)
        z    = t.get_height(r, c)
        lbl  = {'start':'BAŞLANGIÇ','waypoint':'ARA_NOKTA','charge':'ŞARJ','end':'BİTİŞ'}[self.point_mode]

        if lbl == 'BAŞLANGIÇ':
            self.waypoints = [(wx,wy,wz,l) for wx,wy,wz,l in self.waypoints if l != 'BAŞLANGIÇ']
            self.waypoints.insert(0, (x,y,z,'BAŞLANGIÇ'))
        elif lbl == 'BİTİŞ':
            self.waypoints = [(wx,wy,wz,l) for wx,wy,wz,l in self.waypoints if l != 'BİTİŞ']
            self.waypoints.append((x,y,z,'BİTİŞ'))
        else:
            end_i = next((i for i,(wx,wy,wz,l) in enumerate(self.waypoints) if l=='BİTİŞ'), len(self.waypoints))
            self.waypoints.insert(end_i, (x,y,z,lbl))

        self.point_mode = None
        self._mode_lbl.config(text="  Tıklama modu: —")
        self._refresh_wp_list()
        self._set_status(f"✓ {lbl} eklendi: ({x:.1f}, {y:.1f}, Z={z:.2f})")
        self._draw_slope_map()

    def _refresh_wp_list(self):
        self._wp_list.delete(0, 'end')
        icons = {'BAŞLANGIÇ':'🟢','ARA_NOKTA':'🔵','ŞARJ':'🟡','BİTİŞ':'🔴'}
        for i,(x,y,z,l) in enumerate(self.waypoints):
            self._wp_list.insert('end', f"{icons.get(l,'○')} {l[:9]:9s} ({x:.0f}, {y:.0f}, Z={z:.1f})")

    def _remove_selected_wp(self, event=None):
        sel = self._wp_list.curselection()
        if sel:
            self.waypoints.pop(sel[0])
            self._refresh_wp_list()
            self.current_path = None
            self.segment_paths = None
            self._draw_slope_map()

    def _pop_wp(self):
        if self.waypoints:
            self.waypoints.pop()
            self._refresh_wp_list()
            self.current_path = None
            self.segment_paths = None
            self._draw_slope_map()

    def _save_wp(self):
        if not self.waypoints:
            messagebox.showinfo("Bilgi", "Kaydedilecek rota noktası yok.")
            return
        fp = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json'),('All','*.*')], title='Güzergah Kaydet')
        if fp:
            with open(fp,'w', encoding='utf-8') as f: json.dump(self.waypoints, f)
            self._set_status(f"✓ Güzergah kaydedildi: {os.path.basename(fp)}")

    def _load_wp(self):
        fp = filedialog.askopenfilename(filetypes=[('JSON','*.json'),('All','*.*')], title='Güzergah Yükle')
        if fp and os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f: data = json.load(f)
            self.waypoints = [tuple(w) for w in data]
            self._refresh_wp_list()
            self._draw_slope_map()
            self._set_status(f"✓ {len(self.waypoints)} rota noktası yüklendi")

    def _telemetry_callback(self, message):
        try:
            data = json.loads(message['data'])
            self.root.after(0, self._update_telemetry_ui, data)
        except:
            pass

    def _update_telemetry_ui(self, data):
        x = data.get('x', 0.0)
        y = data.get('y', 0.0)
        bat = data.get('battery', 100.0)
        self._live_telemetry_lbl.config(text=f"Konum: {x:.1f}, {y:.1f} | Batarya: %{bat:.1f}")

    def _connect_ros(self):
        ip = self._ros_ip_var.get()
        self._set_status(f"⟳ ROS Bridge sunucusuna ({ip}) bağlanılıyor...")
        self.root.update()
        
        success, msg = self.ros_bridge.connect(host=ip)
        if success:
            messagebox.showinfo("ROS Bağlantısı", msg)
            self._set_status(f"✓ {msg}")
        else:
            messagebox.showerror("ROS Hatası", msg)
            self._set_status(f"✕ {msg}")

    def _publish_route(self):
        if not self.current_path:
            messagebox.showwarning("Uyarı", "Önce bir rota hesaplamalısınız!")
            return
            
        self._set_status("⟳ Rota ROS'a aktarılıyor...")
        self.root.update()
        
        world_coords = []
        for r, c in self.current_path:
            x, y, z = self.terrain.grid_to_world(r, c)
            world_coords.append({"x": round(x,3), "y": round(y,3), "z": round(z,3)})
            
        success, msg = self.ros_bridge.publish_route(world_coords)
        
        if success:
            self._set_status(f"✓ {msg}")
        else:
            messagebox.showerror("ROS Hatası", msg)
            self._set_status(f"✕ {msg}")
            
    def _trigger_estop(self):
        success, msg = self.ros_bridge.trigger_estop()
        if success:
            messagebox.showwarning("E-STOP TETİKLENDİ", "Robota acil durdurma sinyali iletildi!")
            self._set_status(f"🛑 {msg}")
        else:
            messagebox.showerror("ROS Hatası", msg)

    def _update_progress_bar(self, pct, msg):
        self.progress['value'] = pct
        self._set_status(f"⟳ {msg} (%{pct})")
        self.root.update_idletasks()

    def _compute(self):
        if self._computing: return
        if self.terrain is None:
            messagebox.showwarning("Uyarı", "Yüzey verisi yüklenmemiş!")
            return
        has_s = any(l=='BAŞLANGIÇ' for _,_,_,l in self.waypoints)
        has_e = any(l=='BİTİŞ'   for _,_,_,l in self.waypoints)
        if not has_s or not has_e:
            messagebox.showwarning("Uyarı", "Başlangıç 🟢 ve Bitiş 🔴 noktası gerekli!")
            return

        mode = self._mode_var.get()
        eps  = self._eps_var.get()
        do_compare = self._compare_var.get()
        auto_charge = self._auto_charge_var.get()

        self._computing = True
        self._calc_btn.state(['disabled'])
        self.progress.pack(fill='x', padx=8, pady=0)
        self.progress['value'] = 0
        self._set_status("⟳ TUA Navigasyon Algoritması Çalışıyor…")

        thread = threading.Thread(target=self._compute_worker, args=(mode, eps, do_compare, auto_charge), daemon=True)
        thread.start()

    def _compute_worker(self, mode, eps, do_compare, auto_charge):
        try:
            wps_grid = []
            manual_charge_grids = []
            for x,y,z,lbl in self.waypoints: 
                grid_pt = self.terrain.world_to_grid(x, y)
                wps_grid.append(grid_pt)
                if lbl == 'ŞARJ':
                    manual_charge_grids.append(grid_pt)

            def ui_update(segment_idx, total_segments, pct):
                msg = f"İşlem [{segment_idx}/{total_segments}] Rota Hesaplanıyor..."
                self.root.after(0, lambda: self._update_progress_bar(pct, msg))

            planner = RoutePlanner(self.terrain, self.energy_model, mode=mode, epsilon=eps)
            path, segs = planner.plan_route(wps_grid, progress_callback=ui_update)

            if path is None:
                self.root.after(0, lambda: self._compute_done(None, None, {}, "✕ Rota bulunamadı — eğim çok yüksek veya nokta erişilemez"))
                return

            analyzer = RouteAnalyzer(self.terrain, self.energy_model, auto_charge=auto_charge, manual_charge_wps=manual_charge_grids)
            stats    = analyzer.analyze(path)
            
            compare_stats = {}
            if do_compare:
                def calc_mode(m):
                    pl2 = RoutePlanner(self.terrain, self.energy_model, mode=m, epsilon=eps)
                    p2, _ = pl2.plan_route(wps_grid)
                    if p2: return m, RouteAnalyzer(self.terrain, self.energy_model, auto_charge=auto_charge, manual_charge_wps=manual_charge_grids).analyze(p2)
                    return m, None
                    
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(calc_mode, m) for m in ['dengeli','guvenli','enerji','hizli']]
                    for f in concurrent.futures.as_completed(futures):
                        m, res = f.result()
                        if res: compare_stats[m] = res
            
            seg_times = []
            seg_dists = []
            if segs:
                for s in segs:
                    s_stat = analyzer.analyze(s)
                    seg_times.append(s_stat.get('total_time_min', 0.0))
                    seg_dists.append(s_stat.get('total_distance_m', 0.0))
            stats['segment_times'] = seg_times
            stats['segment_dists'] = seg_dists

            self.root.after(0, lambda: self._compute_done(path, segs, stats, compare_stats))
            
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda e=ex: self._compute_done(None, None, {}, f"✕ Hata: {str(e)}"))

    def _compute_done(self, path, segs, stats_or_msg, compare_or_msg=None):
        self.progress.stop()
        self.progress.pack_forget()
        self._calc_btn.state(['!disabled'])
        self._computing = False

        if path is None:
            msg = compare_or_msg if isinstance(compare_or_msg, str) else "✕ Rota Hesaplanamadı"
            messagebox.showerror("Hata", msg)
            self._set_status(msg)
            return

        self.current_path  = path
        self.segment_paths = segs
        self.route_stats   = stats_or_msg
        if isinstance(compare_or_msg, dict): self.compare_stats = compare_or_msg

        self._update_stats(self.route_stats)
        self._draw_slope_map()
        self._draw_profile()
        self._draw_energy()
        self._draw_compare()

        s = self.route_stats
        
        if s.get('battery_failed'):
            messagebox.showwarning("BATARYA YETERSİZ!", "Aracın bataryası bu rotayı tamamlamaya yetmiyor! \nLütfen daha fazla 'Şarj Noktası' ekleyin veya 'Akıllı Şarj' özelliğini açın.")
            self._set_status(f"✕ BATARYA YETERSİZ! Rota çizildi ancak araç hedefe ulaşamaz.")
        else:
            self._set_status(f"✓ Rota hazır  |  {s['total_distance_m']:.1f}m  | Toplam: {s['total_time_min']:.1f}dk  | Maks eğim {s['max_slope_deg']:.1f}°")

    def _update_stats(self, s):
        if s is None:
            for w in self._stat_widgets.values(): w.config(text='—', fg=ACCENT)
            return
        def c(k, txt, warn=False, ok=None):
            col = DANGER if warn else (SUCCESS if ok else ACCENT)
            self._stat_widgets[k].config(text=txt, fg=col)

        c('dist',     f"{s['total_distance_m']:.1f} m")
        c('time',     f"{s['total_time_min']:.1f} dk")
        c('energy',   f"{s['total_energy_wh']:.2f} Wh")
        
        bat_warn = s['battery_failed'] or s['battery_pct'] < 10
        c('battery',  f"%{s['battery_pct']:.1f}", warn=bat_warn)
        
        c('max_slope',f"{s['max_slope_deg']:.1f}°", warn=s['max_slope_deg'] > SLOPE_COLORS['dikkat'][0])
        c('elev_up',  f"+{s['elevation_gain_m']:.1f} m")
        c('elev_dn',  f"-{s['elevation_loss_m']:.1f} m")
        c('danger',   f"{s['danger_segments']} segment", warn=s['danger_segments'] > 0)

    def _load_stl(self):
        fp = filedialog.askopenfilename(title="STL Dosyası Seç", filetypes=[("STL","*.stl"),("All","*.*")])
        if not fp: return
        self._set_status(f"⟳ Yükleniyor: {os.path.basename(fp)}…")
        self.root.update()
        try:
            tris, norms = STLLoader.load(fp)
            self.triangles = tris
            self._file_lbl.config(text=f"  {os.path.basename(fp)}")
            
            self._map_x_var.set("")
            self._map_y_var.set("")
            
            self._process_terrain()
        except Exception as ex:
            messagebox.showerror("Hata", f"STL yüklenemedi:\n{ex}")
            self._set_status("✕ STL yükleme hatası")

    def _demo(self):
        self._set_status("⟳ Demo ay yüzeyi oluşturuluyor…")
        self.root.update()
        N = 40
        x = np.linspace(0, 100, N)
        y = np.linspace(0, 100, N)
        XX, YY = np.meshgrid(x, y)
        ZZ = np.zeros_like(XX)

        for cx,cy,cr,d in [(22,25,10,5),(68,62,7,3.5),(50,80,5,2.5), (14,72,6,2.5),(85,22,9,4),(40,45,4,1.5), (75,40,5,2)]:
            r  = np.sqrt((XX-cx)**2 + (YY-cy)**2)
            ZZ -= d * np.exp(-r**2 / (2*(cr/3)**2))
            ZZ += d*0.35 * np.exp(-((r-cr)**2) / (2*(cr/5)**2))

        ZZ += 7 * np.exp(-((XX-60)**2+(YY-38)**2)/250)
        ZZ += 4 * np.exp(-((XX-30)**2+(YY-60)**2)/150)
        ZZ += 2 * np.sin(np.pi*XX/70) * np.cos(np.pi*YY/55) * 0.7

        np.random.seed(42)
        ZZ += np.random.normal(0, 0.25, ZZ.shape)

        tris = []
        for i in range(N-1):
            for j in range(N-1):
                v00=(XX[i,j],  YY[i,j],  ZZ[i,j])
                v10=(XX[i+1,j],YY[i+1,j],ZZ[i+1,j])
                v01=(XX[i,j+1],YY[i,j+1],ZZ[i,j+1])
                v11=(XX[i+1,j+1],YY[i+1,j+1],ZZ[i+1,j+1])
                tris.append((v00,v10,v01))
                tris.append((v10,v11,v01))
        self.triangles = np.array(tris, dtype=np.float32)
        self._file_lbl.config(text="  [TUA Ay Yüzeyi — Test Alanı]")
        
        self._map_x_var.set("")
        self._map_y_var.set("")
        
        self._process_terrain()

    def _process_terrain(self):
        if self.triangles is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir STL dosyası veya Demo Yüzey yükleyin.")
            return
            
        res = self._res_var.get()
        
        try:
            val_x = self._map_x_var.get().strip().replace(',', '.')
            forced_x_m = float(val_x) * 1000.0 if val_x else None
        except ValueError:
            forced_x_m = None
            
        try:
            val_y = self._map_y_var.get().strip().replace(',', '.')
            forced_y_m = float(val_y) * 1000.0 if val_y else None
        except ValueError:
            forced_y_m = None
            
        self._set_status("⟳ Yüzey Gridleri Oluşturuluyor (Lütfen Bekleyin)...")
        self.root.update()
            
        self.terrain      = TerrainGrid(self.triangles, resolution=res, forced_x_m=forced_x_m, forced_y_m=forced_y_m)
        self.current_path = None
        self.segment_paths= None
        self.route_stats  = None
        self.waypoints    = []
        self._refresh_wp_list()
        self._update_stats(None)
        
        self._draw_slope_map()
        
        x_km_str = f"{(self.terrain.x_max - self.terrain.x_min)/1000:.2f}"
        y_km_str = f"{(self.terrain.y_max - self.terrain.y_min)/1000:.2f}"
        
        self._set_status(f"✓ Harita hazır: {x_km_str}x{y_km_str} km | Grid: {self.terrain.nx}×{self.terrain.ny} | Z: {self.terrain.z_min:.1f}–{self.terrain.z_max:.1f}m")

    def _apply_params(self):
        try:
            for key, var in self._param_vars.items(): 
                self.energy_model.p[key] = float(var.get())
            # Sliderlardan gelen güneş verilerini enerji modeline gönderiyoruz
            self.energy_model.p['sun_elevation_deg'] = self._sun_elev_var.get()
            self.energy_model.p['sun_azimuth_deg'] = self._sun_az_var.get()
            
            self._update_force_label()
            self._update_solar_preview()
            self._set_status("✓ TUA Rover ve Güneş Paneli parametreleri güncellendi")
        except ValueError as ex: 
            messagebox.showerror("Hata", f"Geçersiz değer:\n{ex}")

    def _update_solar_preview(self):
        # Arayüzdeki sliderları çektikçe, o anki açıda panelin ne kadar güç üreteceğini hesaplayan yardımcı fonksiyon
        try:
            elev = self._sun_elev_var.get()
            sun_zenith_rad = math.radians(90.0 - elev)
            cos_theta = math.cos(sun_zenith_rad) # Eğim 0 kabul edilerek
            
            area = float(self._param_vars['solar_area_m2'].get()) if 'solar_area_m2' in self._param_vars else self.energy_model.p['solar_area_m2']
            eff = float(self._param_vars['solar_efficiency'].get()) if 'solar_efficiency' in self._param_vars else self.energy_model.p['solar_efficiency']
            
            power = LUNAR_SOLAR_IRRADIANCE * area * eff * max(0.0, cos_theta)
            self._solar_preview_lbl.config(text=f"Düz Zeminde Maks Üretim: ~{power:.1f} W")
        except:
            pass
    def _update_force_label(self):
        F = self.energy_model.max_available_force()
        self._force_lbl.config(text=f"  Maks Çekiş Kuvveti: {F:.1f} N  |  Motor: {self.energy_model.p['wheel_count']}×{self.energy_model.p['max_torque_nm']:.1f}Nm")

    def _reset(self):
        self.waypoints = []
        self.current_path = None
        self.segment_paths = None
        self.route_stats = None
        self.compare_stats = {}
        self._refresh_wp_list()
        self._update_stats(None)
        if self.terrain:
            self._draw_slope_map()
        self._set_status("● Sıfırlandı")

    def _set_status(self, msg):
        self._status_var.set(f"  {msg}")
        self.root.update_idletasks()

    def _bind_hotkeys(self):
        self.root.bind('<Escape>', lambda e: self._cancel_mode())
        self.root.bind('<Delete>', lambda e: self._pop_wp())
        self.root.bind('<Return>', lambda e: self._compute())

    def _cancel_mode(self):
        self.point_mode = None
        self._mode_lbl.config(text="  Tıklama modu: —")

# ─────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.resizable(True, True)
    try: root.iconbitmap('')
    except: pass
    app = RoverApp(root)
    print("━"*60)
    print("  TUA (Türkiye Uzay Ajansı) Ay Aracı Yer Kontrol İstasyonu v10.1")
    print("  PyVista GPU Render + Oto-Şarj Güneş Motoru + Çoklu Çekirdek + Tüm Sekmeler")
    print("━"*60)
    root.mainloop()

if __name__ == '__main__':
    main()