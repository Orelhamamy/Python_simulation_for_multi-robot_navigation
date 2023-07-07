import time
from env import BuildEnvironment
from agent import AgentRobot
import cv2
import numpy as np


def get_location(event, x, y, flags, param):
    global agent
    # if event == cv2.EVENT_LBUTTONDBLCLK:
    # print("Update agent position")
    # for agent in agents:
    agent.position = np.array([x, y]) * env.res
    # print(agent.position)


env = BuildEnvironment("maps/map1.png")
agent = AgentRobot(5, env.grid.shape, 0, map_res=env.res)
# agents = [AgentRobot(5, env.grid.shape, i, map_res=env.res) for i in range(20)]
cv2.setMouseCallback("map", get_location)

timer = time.time()
while 1:
    env.show_env()
    # time.sleep(1)
    
    # for agent in agents:
        # agent.get_ranges_in_grid_frame(env.grid)
    agent.get_ranges_in_grid_frame(env.grid)
    agent.display_map()
    if (time.time() - timer) >= 5:
        agent.get_frontiers(debug=True) 
        timer = time.time()

    # print(f"clicked_point:{env.clicked_point}")
