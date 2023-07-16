#!/usr/bin/env python3

"""
Origin: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2 

author: Nicholas Swift
"""

from PIL import Image
import numpy as np
import cv2
import pyastar2d

'''
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)
    closest_node = start_node
    closest_node.h = ((closest_node.position[0] - end_node.position[0]) ** 2) + ((closest_node.position[1] - end_node.position[1]) ** 2)
    
    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)
        
        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return (True, path[::-1]) # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (maze.shape[1] - 1) or node_position[0] < 0 or \
               node_position[1] > (maze.shape[0] -1) or node_position[1] < 0:
                continue
            
            # Make sure walkable terrain
            if maze[node_position[1],node_position[0]] > 50:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            flag = 1
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    flag = 0
                    break

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.h += maze[child.position[1], child.position[0]]
            child.f = child.g + child.h
            
            if child.h < closest_node.h:
                closest_node = child
            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    flag = 0
                    break

            # Add the child to the open list
            if flag:
                open_list.append(child)
    
    path = []
    current = closest_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return (False, path[::-1]) 

import heapq

def heuristic(node, goal):
    return (node[0] - goal[0])**2 + (node[1] - goal[1])**2
    
def astargpt(cost_map, start, goal):
    # Initialize the open and closed sets
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        # Get the node with the lowest f_score from the open set
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reached the goal, reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            # path.append(start)
            path.reverse()
            return path

        # Generate the neighbors of the current node
        neighbors = []
        x, y = current
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx==dy==0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(cost_map[0]) and 0 <= ny < len(cost_map):
                    neighbors.append((nx, ny))

        for neighbor in neighbors:
            if cost_map[neighbor[1], neighbor[0]] > 225:
                continue
            # Calculate the tentative g_score for the neighbor
            new_g_score = g_score[current] + cost_map[neighbor[1]][neighbor[0]] + 1
            
            if neighbor not in g_score or new_g_score < g_score[neighbor]:
                # Update the g_score and f_score
                g_score[neighbor] = new_g_score
                f_score = new_g_score + .25 * heuristic(neighbor, goal)

                # Add the neighbor to the open set
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current

    # No path found
    return []
'''



    
class Graph():
    
    def __init__(self, bounds, start, star=True):
        """
        Graph G(V,E),  the vertex described as (x,y), each have an inx.
        E - edges will be d
        Parameters
        ----------
        bounds : np.array
         bounds = [y_max, x_max] == same as the shape of the grid map
        
        start : np.array

        
        """
        
        self.bounds = np.array(bounds )
        self.vertices = np.array([start])
        self.edges = np.zeros((1, 4))
        self.success = False
        self.eta = 100
        self.star = star
        self.d = self.vertices.shape[1]
        
    def is_in_bounds(self, vertex):
        x = (vertex[0] > 0 & vertex[0] < self.bounds[1])
        y = (vertex[1] > 0 & vertex[1] < self.bounds[0])
        return x & y
    
    def obstacleFree(self, cost_map, vertex, end_vertex=None, dist=None):
        
        if end_vertex is None:
            return cost_map[vertex[1], vertex[0]] > 0.8
        res = np.ceil(dist).astype(int)
        indexes = np.round(np.linspace(0,1, res)[:, None] * (end_vertex - vertex)[None,:] + vertex).astype(int)
        
        try:    
            return np.all(cost_map[indexes[:,1], indexes[:,0]] < 0.8)
        except:
            print(f"Error! vertex:{vertex}, end:{end_vertex}, res:{res}")
            return False
        
    
    def get_nearest(self, ver , indexes=None):
        
        ind = np.ones((self.vertices.shape[0]), dtype=bool)
        if indexes is not None:
            ind = ~ind
            ind[indexes] = True
        d = np.linalg.norm(self.vertices - ver, axis=1)
        d[~ind] = d.max()
        
        min_inx = d.argmin()
        return min_inx, d[min_inx], d
    
    def get_neighbors(self, vertex, radius=None):
        card, d = self.vertices.shape
        radius = radius if radius is not None else min(self.eta  * 10 * (np.log(card) / card) ** (1 / d), self.eta)
        
        distances = np.linalg.norm(self.vertices - vertex, axis=1)
        return np.where(distances < radius)[0]
    
    def sample_v(self, cost_map, display=True):
        found = False
        while not found:
            vertex = np.round(np.random.rand(2) * (self.bounds - 1)).astype(int)[::-1]
            
            while cost_map[vertex[1], vertex[0]] > 0.8:
                vertex = np.round(np.random.rand(2) * (self.bounds - 1)).astype(int)[::-1]
            
            x_min, dist, _ = self.get_nearest(vertex)
            c_min = dist
            if self.obstacleFree(cost_map, self.vertices[x_min], vertex, dist):
                found = True
        
        if self.star:
            neighbors = self.get_neighbors(vertex)
            c_min = self.edges[x_min, 2] + dist
            for n in neighbors:
                dist = np.linalg.norm(self.vertices[n] - vertex)
                if self.obstacleFree(cost_map, vertex, self.vertices[n], dist) and c_min > self.edges[n, 2] + dist:
                    x_min, c_min = n, dist + self.edges[n, 2]
                    
        vertex_inx = self.vertices.shape[0]
        e = np.array([[vertex_inx , x_min, c_min, cost_map[vertex[1], vertex[0]]]])
        self.edges = np.r_[self.edges, e]
        self.vertices = np.r_[self.vertices, vertex[None,:]]
        
        if display:
            img = np.tile(cost_map[..., None], 3)
            for i in range(self.vertices.shape[0]):
                img = cv2.circle(img, tuple(self.vertices[i]), radius=4, color=(255,0,0), thickness=-1)
                indexes = [int(j) for j in self.edges[i,:2]]
                img = cv2.line(img, tuple(self.vertices[indexes[1]]), tuple(self.vertices[indexes[0]]), (0,255,125), thickness=2)
            cv2.imshow('Building the tree', img)
            cv2.waitKey(1)
        if self.star:
            for n in neighbors:
                dist = np.linalg.norm(self.vertices[n] - vertex)
                if self.obstacleFree(cost_map, vertex, self.vertices[n], dist) and c_min + dist < self.edges[n,2]:
                    self.edges[n, 1:3] = [vertex_inx, c_min + dist]
                    if display:
                        img = cv2.circle(img, tuple(vertex), radius=6, color=(255,255,125), thickness=-1)
                        img = cv2.circle(img, tuple(self.vertices[n]), radius=4, color=(0,0,255), thickness=-1)
                        indexes = [int(j) for j in self.edges[n,:2]]
                        img = cv2.line(img, tuple(self.vertices[indexes[1]]), tuple(self.vertices[indexes[0]]), color=(0,0,255), thickness=2)
                        cv2.imshow('Building the tree', img)
                        cv2.waitKey(1)
            
    def find_path(self, cost_map, pos, end_pos, display=False):
        self.edges[:,1:] = -1
        start_inx, _, _ = self.get_nearest(pos)
        end_inx,_ ,_ = self.get_nearest(end_pos)
        # Maybe add more vertexes
        
        
        parent_inx = start_inx
        parent_vertex = self.vertices[parent_inx]        
        self.edges[start_inx, 1:] = [0 , 0, cost_map[parent_vertex[1], parent_vertex[0]]] 
        if display:
            img = np.tile(cost_map[..., None], 3)
            for i in range(self.vertices.shape[0]):
                img = cv2.circle(img, tuple(self.vertices[i]), radius=4, color=(255,0,0), thickness=-1)
            cv2.imshow('Building the tree', img)
            cv2.waitKey(0)
        while self.edges[end_inx, 1] == -1:                
            unconnected_nearest, dist, distances = self.get_nearest(parent_vertex, indexes=self.edges[:,1]==-1)
            max_dist = np.max(distances)
            
            while not self.obstacleFree(cost_map, parent_vertex, self.vertices[unconnected_nearest], dist):
                distances[unconnected_nearest] = max_dist
                unconnected_nearest = np.argmin(distances)
                dist = distances[unconnected_nearest]
            
            c_min = dist + self.edges[parent_inx, 2]
            x_min = parent_inx
            vertex = self.vertices[unconnected_nearest]
            
            if self.star:
                card = np.count_nonzero(G.edges[:,2]!=-1)
                radius = min(self.eta, self.eta * 10 * (np.log(card) / card) ** (1/self.d))
                neighbors = self.get_neighbors(unconnected_nearest, radius=radius)
                for n in neighbors:
                    if self.edges[n, 2] != -1:
                        dist = np.linalg.norm(self.vertices[n] - vertex)
                        if self.obstacleFree(cost_map, vertex, self.vertices[n], dist) and c_min > self.edges[n, 2] + dist:
                            x_min, c_min = n, dist + self.edges[n, 2]
            
            self.edges[unconnected_nearest, 1:] = [x_min, c_min, cost_map[vertex[1], vertex[0]]]
            
            if display:
                for i in np.where(self.edges[:,2]!=-1)[0]:
                    indexes = [int(j) for j in self.edges[i,:2]]
                    img = cv2.line(img, tuple(self.vertices[indexes[1]]), tuple(self.vertices[indexes[0]]), (0,255,125), thickness=2)
                cv2.imshow('Building the tree', img)
                cv2.waitKey(1)
            
            parent_inx = unconnected_nearest
            parent_vertex = self.vertices[parent_inx]        
            
            if self.star:
                for n in neighbors:
                    dist = np.linalg.norm(self.vertices[n] - vertex)
                    if self.obstacleFree(cost_map, vertex, self.vertices[n], dist) and (c_min + dist < self.edges[n,2] or self.edges[n,2]==-1):
                        self.edges[n, 1:3] = [parent_inx, c_min + dist]
                        if display:
                            img = cv2.circle(img, tuple(vertex), radius=6, color=(255,255,125), thickness=-1)
                            img = cv2.circle(img, tuple(self.vertices[n]), radius=4, color=(0,0,255), thickness=-1)
                            indexes = [int(j) for j in self.edges[n,:2]]
                            img = cv2.line(img, tuple(self.vertices[indexes[1]]), tuple(self.vertices[indexes[0]]), color=(0,0,255), thickness=2)
                            cv2.imshow('Building the tree', img)
                            cv2.waitKey(1)
                    
        
        
        
        
if __name__ == "__main__":
    
    
    
    maze = (255 - np.asarray(Image.open("/home/orel/orel_ws/thesis/proof of concept/maps/map1.png")).astype(np.float32))[..., 0]
    
    pos = (100, 100)
    
    end_pos = (500, 550)
    
    # path = astargpt(maze, pos, end_pos)
    cm = maze + 1
    cm[cm>50] = np.inf
    
    path = pyastar2d.astar_path(cm.T, pos, end_pos, allow_diagonal=True)
    # G = Graph(maze.shape, pos, star=True)
    
    # for _ in range(2500):
    #     G.sample_v(maze / 255., display=True)
    
    # G.find_path(maze / 255., (500,350), end_pos, display=True)
    img_maze = np.tile(maze[..., None], 3)
    img_maze = cv2.circle(img_maze, pos, radius=5, color=(0,0,255), thickness=-1)
    img_maze = cv2.circle(img_maze, end_pos, radius=5, color=(255,0,0), thickness=-1)        
    
    # for p_i, e_i in zip(G.vertices, G.edges):
    #     img_maze = cv2.circle(img_maze, tuple(p_i), radius=1, color=(255,0,0), thickness=-1)
    #     indexes = [int(i) for i in e_i[:2]]
    #     img_maze = cv2.line(img_maze, tuple(G.vertices[indexes[1]]), tuple(G.vertices[indexes[0]]), (0,255,125), thickness=2)
    # img_maze = cv2.circle(img_maze, end_pos, radius=5, color=(0,255,125), thickness=-1)
    # cv2.imshow('maze_with_tree', img_maze)
    # i = np.linalg.norm(np.array([end_pos])- G.vertices, axis=1).argmin()
    # path = np.array([G.vertices[i]])
    # while i!=0 and path.shape[0] < 1000:
    #     next_i = int(G.edges[i,1])
    #     path = np.r_[G.vertices[None, next_i], path]
    #     i = next_i
    
    for i in range(path.shape[0] - 1):
        img_maze = cv2.line(img_maze, tuple(path[i]), tuple(path[i+1]), (0,0,255) , thickness=2)
    cv2.imshow('maze_with_path', img_maze)
    while 1:
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k==ord('q'):
            cv2.destroyAllWindows()
            break
    # k = 25
    # M, N = maze.shape
    # pad = ((0, M % k), (0, N % k))
    # maze = np.pad(maze, pad)
    # M, N = maze.shape
    

    # m = M // k
    # n = N // k

    # reduce_cm = maze[:m*k, :n*k].reshape(m, k, n, k).mean(axis=(1,3))
    # reduce_pos = (pos[0] // k, pos[1] // k)
    # reduce_end = (end_pos[0] // k, end_pos[1] // k)
    # path = astar(reduce_cm, reduce_pos, reduce_end)
    # path = np.array(path[1]) * k + k//2
    # # p = np.array(path[1]) * k + k//2
    # img_maze = np.tile(maze[..., None], 3)
    # img_maze = cv2.circle(img_maze, pos, radius=5, color=(0,0,255), thickness=-1)
    
    # path_opt = []
    # d = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    # for i in enumerate(path):
        
    # img_maze = np.tile(maze[..., None], 3)
    # img_maze = cv2.circle(img_maze, pos, radius=5, color=(0,0,255), thickness=-1)
    # opt_path = [path[0]]
    # for i in range(len(path) - 1):
    #     if opt_path[-1] == path[i]:
    #         ind = np.where([any([q==s for s, q in zip(opt_path[-1], p)]) for p in path[i + 1:]])[0]
    #         if ind.shape[0]:
    #             opt_path.append(path[ind[0] + i + 1])
    #             i = ind[0]
            
    #         else:
    #             opt_path.append(path[i+1])
    
    # for p_i in path:
    #     img_maze = cv2.circle(img_maze, tuple(p_i), radius=1, color=(255,0,0), thickness=-1)
    # img_maze = cv2.circle(img_maze, end_pos, radius=5, color=(0,255,125), thickness=-1)
    # cv2.imshow('maze_with_path', img_maze)
    