from collections.abc import Callable, Iterable, Mapping
from typing import Any
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
# import yaml
import time
import threading
import pyastar2d
import multiprocessing as mp


class AgentRobot:
    def __init__(self, range, env_grid, number, map_res=0.05, init_position=(10, 10), color=(0, 0, 255), sensor_angle=360, frequency=10, N_rays=360,
                 range_resolution=0.05) -> None:
        self.range = range
        self.position = np.array(init_position)  # array(x,y)
        self.sensor_angle = np.deg2rad(sensor_angle)
        self.rays_number = N_rays
        self.angle_increment = self.sensor_angle / (self.rays_number - 1)
        self.min_angle = - self.sensor_angle / 2
        self.max_angle = self.sensor_angle / 2
        self.angles = np.linspace(self.min_angle, self.max_angle, self.rays_number)
        self.frequency = frequency
        self.next_call = time.time()
        self.range_resolution = range_resolution
        self.map_res = map_res
        self.sigma = 0.01
        self.grid_map = np.ones(env_grid.shape, dtype=np.float32) * -1
        self.number = number
        
        self.env_grid = env_grid
        self.is_moving = False
        self.path = None
        self.color = color
        # with open("config/params.yaml", 'r') as stream:
        #     self.BGMM_params = yaml.safe_load(stream)["BGMM"]

        # self.BGMM = BayesianGaussianMixture(**self.BGMM_params)
        # self.BGMM_is_fitted = False
        self.GMM_is_fitted = False
        self.get_ranges_in_grid_frame()
        cv2.namedWindow(f"robot_map_{self.number}")
        cv2.setMouseCallback(f"robot_map_{self.number}", self.get_location)
        print(f"inti robot:{number}")
        # self.display_map(True)

    def get_location(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
        # for agent in agents:
            if self.is_moving:
                self.is_moving = False
                time.sleep(1)
            self.generate_path(np.array([x, y]), True)
            
            
    
    def display_map(self, with_pos=False):
        img = 1 - self.grid_map.copy()
        img = img * 255
        img[self.grid_map == -1] = 50
        img = np.tile(img[..., None], 3)
        # print(np.unique(img))
        x, y = np.floor(self.position / self.map_res).astype(np.uint32)
        self.robot_map_img = cv2.circle(img, (x, y), radius=3, color=self.color, thickness=-1)
        if self.path is not None:
            for i in range(self.path.shape[0] - 1):
                self.robot_map_img = cv2.line(self.robot_map_img, tuple(self.path[i]), tuple(self.path[i + 1]), self.color, thickness=1)
        cv2.imshow(f"robot_map_{self.number}", self.robot_map_img.astype(np.uint8))
        cv2.imshow(f"robot_map_{self.number}", self.robot_map_img.astype(np.uint8))
        cv2.waitKey(1)

    def add_uncertainty(self, distance, angle):
        # print(f"dist:{distance.shape} and type:{type(distance[0,0])}")
        dist, ang = np.random.multivariate_normal([0, 0], np.eye(2) * self.sigma ** 2)
        dist += distance
        ang += angle
        dist[dist < 0] = 0
        ang = np.where(np.abs(ang) > np.pi, ((ang > np.pi) * 2 - 1) * -2 * np.pi + ang, ang)
        return np.c_[dist, ang]
    
    def get_ranges_in_grid_frame(self, update_grid=True):
        self.next_call += 1 / self.frequency
        reading = self.get_range_reading()
        # x and y are in grid frame
        x = (reading[:, 0] * np.cos(reading[:, 1]) + self.position[0]) / self.map_res
        y = (-reading[:, 0] * np.sin(reading[:, 1]) + self.position[1]) / self.map_res
        x[x < 0] = 0
        y[y < 0] = 0
        
        x, y = np.round(x).astype(np.int32), np.round(y).astype(np.int32)
        x[x >= self.grid_map.shape[1]] = self.grid_map.shape[1] - 1
        y[y >= self.grid_map.shape[0]] = self.grid_map.shape[0] - 1
        points = np.array((x, y, reading[:, 2])).astype(int)

        if update_grid:
            grid_position = self.position / self.map_res
            united_points = np.zeros((0, 2), dtype=np.int32)
            for p, r in zip(points.T, reading[:, 0]):
                n_points = int(r / self.range_resolution)
                if n_points <  2:
                    continue
                dx, dy = (p[0] - grid_position[0]) / (n_points - 1), (p[1] - grid_position[1]) / (n_points - 1)
                # print(f" dxdy:{np.array([dx, dy])}, n_point:{n_points}, \n"
                #       f"position:{grid_position}")
                free_points = (np.array([[dx, dy]]).T * np.arange(n_points)[None,:]).T + grid_position
                # print(f"free: {free_points[-5:]}")
                free_points = np.round(free_points).astype(np.int32)
                # print(f"free after: {free_points[-5:]}")
                united_points = np.concatenate((united_points, free_points), axis=0)

            indexes = np.unique(united_points, axis=0)                            
            # indexes[indexes[:,1]>= self.grid_map.shape[0]] = self.grid_map.shape[0] - 1
            # indexes[indexes[:,0]>= self.grid_map.shape[1]] = self.grid_map.shape[1] - 1
            indexes = indexes[(indexes[:,1] < self.grid_map.shape[0]) & (indexes[:,1] >= 0)]
            indexes = indexes[(indexes[:,0] < self.grid_map.shape[1]) & (indexes[:,0] >= 0)]
            mask = self.grid_map[y, x] == -1 
            self.grid_map[y[mask], x[mask]] = reading[mask, 2]
            mask = self.grid_map[indexes[:,1], indexes[:,0]] == -1
            self.grid_map[indexes[mask, 1], indexes[mask, 0]] = 0.5
            # np.save("map_.npy", self.grid_map)
        threading.Timer(self.next_call - time.time(), self.get_ranges_in_grid_frame).start()

    def get_range_reading(self):
        # Extract the robot's position coordinates
        robot_x, robot_y = self.position

        # Calculate the x and y components of the ray directions for all angles
        noisy_ang = 0.01 * np.random.randn(self.angles.shape[0]) + self.angles
        ray_dir_x = np.cos(noisy_ang)
        ray_dir_y = -np.sin(noisy_ang)

        # Calculate the range values for all rays
        range_values = np.arange(self.range_resolution, self.range + self.range_resolution, self.range_resolution)

        # Calculate the next positions along the rays
        next_x = robot_x + np.outer(range_values, ray_dir_x)
        next_y = robot_y + np.outer(range_values, ray_dir_y)

        # Calculate the grid indices for the next positions
        x_grid = (next_x // self.map_res).astype(int)
        y_grid = (next_y // self.map_res).astype(int)
        x_grid = np.clip(x_grid, 0, self.env_grid.shape[1] - 1)
        y_grid = np.clip(y_grid, 0, self.env_grid.shape[0] - 1)
        # Check if the next positions are within the grid boundaries
        # in_bounds = (x_grid >= 0) & (x_grid < len(self.env_grid[0])) & (y_grid >= 0) & (y_grid < len(self.env_grid))

        # Check if the next positions are obstacles
        obstacles = self.env_grid[y_grid, x_grid] == 1
        # obstacles[~in_bounds] = 1
        # Calculate the range readings based on obstacles and grid boundaries
        range_readings = np.where(obstacles, range_values[:, None], np.inf)
        range_readings = np.amin(range_readings, axis=0)

        occ = range_readings != np.inf
        range_readings[~occ] = self.range
        # Add uncertainty to range readings
        # range_readings = self.add_uncertainty(range_readings, self.angles)
        range_readings = np.c_[range_readings, self.angles]
        # Create the final range readings array
        range_readings = np.column_stack((range_readings, np.ones(range_readings.shape[0]) * 0.5))
        range_readings[occ, 2] = 1

        return range_readings

    def get_frontiers(self, debug=False):
        img_src = np.copy(self.grid_map).astype(np.uint8) * 100 
        img_src[self.grid_map == -1] = 50
        # img_src = cv2.rotate(img_src, cv2.ROTATE_180)
        # img_src = cv2.flip(img_src, 1)

        walls = (img_src > 90).astype(np.uint8)
        walls = cv2.filter2D(walls, 0, np.ones((10,10)))

        # img_src[img_src==-1] = 100
        img_edges = cv2.Canny(img_src.astype(np.uint8), 20, 200)  # 200/30, 200

        self.img_edges = np.where(walls, 0, img_edges)
        contours = cv2.findContours(self.img_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # np.save("cont.npy", contours)
        if debug:
            # cv2.imshow("edges_after_removing_walls", self.img_edges)
            # cv2.waitKey(1000)
            # cv2.imshow("edges_before_removing_walls", img_edges)
            # cv2.waitKey(1000)
            # cv2.imshow("src", img_src * 255 / 100)
            # cv2.waitKey(1000)
            # cv2.imshow("walls", walls * 255)
            # cv2.waitKey(1000)
            # cv2.drawContours(img_src, contours[0], -1, (200), 5)
            if self.GMM_is_fitted:
                img_src = np.tile(img_src[..., None], 3)
                img_src = self.draw_gmm(img_src)
            cv2.imshow("src", img_src * 255 / 200)
            cv2.waitKey(1000)
        
        contours = contours[0]

        if len(contours)==0:
            print(f"Agent {self.number} didn't find frontiers.")
            self.frontiers_points = None
        else:
            self.frontiers_points = np.concatenate(contours).squeeze()
            merged_cnt = [np.array(contours[0][:,0,:])]

            contours.pop()
            while contours:
                flag = -1
                
                for inx, cnt in enumerate(contours):
                    if np.min(cdist(merged_cnt[-1], cnt[:,0,:])) < 15:
                        flag=inx
                        break
                if flag!=-1:
                    merged_cnt[-1] = np.concatenate((merged_cnt[-1], 
                                                     contours[flag][:,0,:]), axis=0)
                    contours.pop(flag)
                
                else:
                    merged_cnt.append(contours[0][:,0,:])
                    contours.pop(0)
            self.init_means = np.array([np.mean(a, axis=0) for a in merged_cnt])        
        
    def define_gmm(self, ):
        if self.frontiers_points is None:
            self.GMM_is_fitted = False
            return None
        self.normalization_factor = [self.frontiers_points.min(axis=0),
                                    self.frontiers_points.max(axis=0) - self.frontiers_points.min(axis=0)]
        
        means = (self.init_means - self.normalization_factor[0]) / self.normalization_factor[1]
        self.gmm = GaussianMixture(means.shape[0], means_init=means, max_iter=100)
        X = (self.frontiers_points - self.normalization_factor[0]) / self.normalization_factor[1]
        print("fitting!!")
        self.gmm.fit(X)
        print("Done")
        cov_factor = self.normalization_factor[1][:, None] * self.normalization_factor[1][None, :]
        self.f_means = self.gmm.means_ * self.normalization_factor[1] + self.normalization_factor[0]
        self.f_covs = self.gmm.covariances_ * cov_factor
        self.f_weigths = self.gmm.weights_
        self.f_weigths /= np.sum(self.f_weigths)
        
        self.GMM_is_fitted = True

    def draw_gmm(self, img):
        ellipsis = img.copy()

        for m, c, w in zip(self.f_means, self.f_covs, self.f_weigths):
            
            val, vector = np.linalg.eigh(c)
            scale = 2 * np.sqrt(val)
            angle = np.arctan2(vector[0,1], vector[1,1]) * 180 / np.pi
            center = tuple(np.round(m).astype(np.int))
            axes = tuple(np.round(scale).astype(np.int))
            ellipsis = cv2.ellipse(ellipsis, center, axes, angle, 0, 360, (255, 102, 102), -1)
        # print(f"scale: {scale} and axes:{axes}")
        return cv2.addWeighted(ellipsis, 0.4, img, 1 - 0.4, 0)

    def drive(self, desire_pos, grid=False, display=False):
        
        if grid:
            desire_pos = desire_pos * self.map_res
        
        while np.linalg.norm(desire_pos - self.position) > 0.05 and self.is_moving:
            dx, dy = (desire_pos - self.position) / np.linalg.norm((desire_pos - self.position))
            dt = 0.1
            self.position = self.position + dt * np.array((dx, dy))
            limits = np.r_[self.grid_map.shape[::-1]] * self.map_res  # Get the shape as (x,y)
            self.position = (self.position > limits) * (limits - self.position) + self.position
            self.position = (self.position < [0, 0]) * (-self.position) + self.position
            # self.get_ranges_in_grid_frame()
            if display:
                self.display_map()

    def generate_cost_map(self, display_cost_map=False):

        self.cost_map = np.ones(self.env_grid.shape).astype(np.float32)
        # Distance cost
        xx = np.arange(self.env_grid.shape[1])[None, :]
        yy = np.arange(self.env_grid.shape[0])[:, None]
        grid_position = self.position / self.map_res
        # rr = np.sqrt((xx - grid_position[0])**2 + (yy - grid_position[1])**2)

        # rr = rr / np.max(rr)


        # Obstacle cost

        walls = self.grid_map > 0.9
        kernel = cv2.getGaussianKernel(50, 30)
        kernel = kernel * kernel.T
        walls = cv2.filter2D(walls.astype(np.uint8), -1, np.ones((5,5)))
        walls[walls > 1] = 1

        obs = cv2.filter2D(walls.astype(np.uint8), 5, kernel)
        obs = obs / np.max(obs)
        obs[walls] = 1
        # self.cost_map = 0.5 * obs + 0.5 * rr
        self.cost_map += obs
        self.cost_map[self.cost_map < 1] = 1
        self.cost_map[walls > 0] = np.inf


        if display_cost_map:
            cv2.imshow(f'cmap_{self.number}', (self.cost_map * 255).astype(np.uint8))
            cv2.waitKey(1000)

    def generate_path(self, destination, drive=True, grid=True):
        self.is_moving = False
        pos = tuple(np.round(self.position / self.map_res).astype(np.int))
        end_pos = tuple(np.round(destination).astype(np.int)) if grid else tuple(np.round(destination / self.map_res).astype(np.int))
        maze = self.env_grid.astype(np.float32)
        maze[maze == 1] = np.inf
        maze = maze + 1
        self.path = pyastar2d.astar_path(maze.T, pos, end_pos, allow_diagonal=True)

        if drive and self.path is not None:
            self.is_moving = True
            for p in self.path:
                
                self.drive(p, grid=True, display=True)
                time.sleep(0.01)
            self.path = None
            self.is_moving = False
        

        # if reduced_path[0] is False:
        #     path = astar(self.grid_map, )
            


        

        
