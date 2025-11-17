

from ursina import *
import numpy as np

MAX_RAYCAST_DIST = 100

class SingleRaySensor(Entity):
    def __init__(self, car, angle):
        super().__init__(
            parent = car,
            model = "line",
            texture = "assets/utils/red.png",
            position = (0,0,0)
        )
        self.car = car
        self.angle = angle
        self.rotation_y = self.angle

        self.sensing_dist = MAX_RAYCAST_DIST
        self.last_hit_point = None
    def cast_ray(self):
        #self.world_position = self.car.world_position
        cast = raycast(origin = self.world_position + (0,1,0), direction = self.forward, distance = MAX_RAYCAST_DIST, ignore = [self.car,])

        if cast.hit:
            self.last_hit_point = cast.world_point
            return cast.distance

        else :
            self.last_hit_point = None
            return MAX_RAYCAST_DIST


    def update(self):
        if self.visible:
            self.sensing_dist = self.cast_ray()
            self.scale_z = self.sensing_dist


class MultiRaySensor(Entity):
    def __init__(self, car, nbr_ray, half_angle):
        super().__init__(
            parent = car,
        )
        self.car = car
        self.half_angle = half_angle

        self.rays = []
        for i in range(nbr_ray):
            ray = SingleRaySensor(self, -half_angle + 2 * half_angle / (nbr_ray-1) * i)
            ray.enable()
            self.rays.append(ray)

    def collect_sensor_values(self):
        if self.enabled:
            return [r.sensing_dist for r in self.rays]
        else:
            return [r.cast_ray() for r in self.rays]

    def set_enabled_rays(self, enable):
        if enable:
            self.enable()
        else:
            self.disable()

        for r in self.rays:
            r.visible = enable
    
    def get_edge_points(self):
        self.collect_sensor_values()

        if len(self.rays) < 2:
            return None, None

        left_ray = self.rays[-1]
        right_ray = self.rays[0]

        lp = left_ray.last_hit_point
        rp = right_ray.last_hit_point

        if lp is None or rp is None:
            return None, None

        p_left = np.array([lp.x, lp.z], dtype=float)
        p_right = np.array([rp.x, rp.z], dtype=float)

        width = float(np.linalg.norm(p_right - p_left))
        center = 0.5 * (p_left + p_right)
        return width, center
