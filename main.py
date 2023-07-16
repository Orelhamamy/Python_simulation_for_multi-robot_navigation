#!/usr/bin/env python3

import time
from env import BuildEnvironment
from agent import AgentRobot
import cv2
import numpy as np
import argparse
import yaml
import multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser(description='Exploration with Multi agents using frontier-based approach.')
    parser.add_argument('-m', '--map', type=str, default="map1.png",
                        help='an integer for the accumulator')
    parser.add_argument('-a', '--agents', type=int,
                        default=1,
                        help='Set how many agents will attend the exploration.')
    parser.add_argument('-r', '--random_init', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Draw random initization position for each agent.')
    return parser.parse_args()

"""def get_location(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # for agent in agents:
        agent.drive(np.array([x, y]), grid=True)
    # print(agent.position)"""


def drive_sim(arg):
    obj, destination = arg
    return obj.generate_path(destination)


def get_united_map(agents):
    img = agents[0].grid_map.copy()
    poses = np.tile(np.zeros(img.shape).astype(np.uint8)[..., None], 3)
    for agent in agents[1:]:
        mask = (img == -1) & (agent.grid_map != -1)
        img[mask] = agent.grid_map[mask]
        x, y = np.floor(agent.position / agent.map_res).astype(np.uint32)
        poses = cv2.circle(poses, (x, y), radius=3, color=agent.color, thickness=-1)
    img = img * 255
    img[img == -1] = 50
    img = np.tile(img[..., None], 3).astype(np.uint8)
    img[poses != 0] = poses[poses != 0]
    return img


def main(input_arg):
    env = BuildEnvironment("maps/map1.png")
    # agent = AgentRobot(5, env.grid, 0, map_res=env.res)
    print(input_arg.random_init)

    def init_pos():
        if input_arg.random_init:
            return np.random.randint(0, env.grid.shape[::-1]) * env.res
        return (2.5, 2.5)

    with open('config/rgb_colors.yml', 'r') as f:
        colors_map = yaml.safe_load(f.read())
    agents = [AgentRobot(5, env.grid, i, map_res=env.res,
                         init_position=init_pos(), color=c)
              for i, c in zip(range(input_arg.agents), colors_map.values())]

    pool = mp.Pool(2)
    pool.map(drive_sim, ((agent, (150, 150)) for agent in agents))
    pool.close()

    # agents = [mp.Process(target=AgentRobot, args=(5, env.grid, i, env.res,
    #                      init_pos(), c)) for i, c in zip(range(input_arg.agents), colors_map.values())]
    c = 20
    timer = time.time()
    while 1:
        env.show_env()
        # time.sleep(1)
        img = get_united_map(agents)
        cv2.imshow("map_united", img)
        cv2.waitKey(1)

        # agent.display_map()
        # agent.generate_cost_map(display_cost_map = True)
        if (time.time() - timer) >= 10:
            print("time")
            pool.join()
            timer = time.time()
            pool = mp.Pool(2)
            c = c + 10
            pool.map(drive_sim, ((agent, (150 + c, 150 + c)) for agent in agents))
            pool.close()

        #     agent.get_frontiers(debug=True) 
        #     agent.define_gmm()
        #     timer = time.time()
        #     # if agent.frontiers_points is not None:
        #     # agent.define_gmm()
        # if not agent.is_moving and agent.GMM_is_fitted:
        #     d = np.linalg.norm(agent.f_means - agent.position / agent.map_res).argmin()
        #     print(agent.f_means)
        #     print(d)
        #     agent.generate_path(agent.f_means[d], grid=True)
            

        # print(f"clicked_point:{env.clicked_point}")

if __name__ == "__main__":
    args = get_args()
    main(args)