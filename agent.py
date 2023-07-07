import numpy as np
import cv2

class AgentRobot:
    def __init__(self, range, map_shape, number, init_position=(10, 10), sensor_angle=360, frequency=4, N_rays=200,
                 range_resolution=0.05, map_res=0.05) -> None:
        self.range = range
        self.position = np.array(init_position)
        self.sensor_angle = np.deg2rad(sensor_angle)
        self.rays_number = N_rays
        self.angle_increment = self.sensor_angle / (self.rays_number - 1)
        self.min_angle = - self.sensor_angle / 2
        self.max_angle = self.sensor_angle / 2
        self.angels = np.linspace(self.min_angle, self.max_angle, self.rays_number)
        self.frequency = frequency
        self.range_resolution = range_resolution
        self.map_res = map_res
        self.sigma = 0.0
        self.grid_map = np.ones(map_shape, dtype=np.float32) * -1
        self.number = number

    def display_map(self):
        img = 1 - self.grid_map.copy()
        img = img * 255
        img[self.grid_map == -1] = 50
        # print(np.unique(img))
        cv2.imshow(f"robot_map_{self.number}", img.astype(np.uint8))
        cv2.waitKey(1)

    def add_uncertentiy(self, distance, angle):
        dist, ang = np.random.multivariate_normal([distance, angle], np.eye(2) * self.sigma ** 2)
        dist = np.max(dist, 0)
        if abs(ang) > np.pi:
            ang = ((ang > np.pi) * 2 - 1) * -2 * np.pi + ang
        return [dist, ang]
    
    def get_ranges_in_grid_frame(self, env_grid, update_grid=True):

        reading = self.get_range_reading(env_grid)
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
        return points

    def get_range_reading(self, grid):
        # Extract the robot's position coordinates
        robot_x, robot_y = self.position

        # Initialize the range readings
        range_readings = []

        # Iterate over the number of rays
        for angle in self.angels:

            # Calculate the x and y components of the ray direction
            ray_dir_x = np.cos(angle) * self.range_resolution
            ray_dir_y = -np.sin(angle) * self.range_resolution

            # Initialize the current range reading and position
            curr_range = 0
            curr_x, curr_y = robot_x, robot_y
            # Continue scanning until the range limit is reached or an obstacle is encountered
            while curr_range <= self.range:
                # Calculate the next position along the ray
                next_x = curr_x + ray_dir_x
                next_y = curr_y + ray_dir_y
                # if next_x == curr_x and next_y == curr_y:
                #     curr_range += self.range_resolution
                #     curr_x, curr_y = curr_x + ray_dir_x, curr_y + ray_dir_y
                #     pass

                # Check if the next position is within the grid boundaries
                x_grid, y_grid = int(next_x / self.map_res), int(next_y / self.map_res)
                if 0 <= x_grid < len(grid[0]) and 0 <= y_grid < len(grid):
                    # Check if the next position is an obstacle (e.g., wall)
                    if grid[y_grid, x_grid] == 1:
                        break  # Stop scanning along this ray

                    # Increment the range and update the current position
                    curr_range += self.range_resolution
                    curr_x, curr_y = next_x, next_y
                else:
                    break  # Stop scanning along this ray if out of grid boundaries

            if curr_range > self.range:
                curr_range = np.inf

            range_readings.append(self.add_uncertentiy(curr_range, angle))
        range_readings = np.asarray(range_readings)
        range_readings = np.concatenate((range_readings, np.ones((range_readings.shape[0], 1))*.5), axis=1)
        occ = range_readings[:, 0] != np.inf
        range_readings[occ, 2] = 1
        range_readings[~occ, 0] = self.range
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
        print(contours)
        if debug:
            # cv2.imshow("edges_after_removing_walls", self.img_edges)
            # cv2.waitKey(1000)
            # cv2.imshow("edges_before_removing_walls", img_edges)
            # cv2.waitKey(1000)
            cv2.imshow("src", img_src * 255 / 100)
            # cv2.waitKey(1000)
            # cv2.imshow("walls", walls * 255)
            # cv2.waitKey(1000)
            cv2.drawContours(img_src, contours[0], -1, (200), 5)
            cv2.imshow("src", img_src * 255 / 200)
            cv2.waitKey(1000)