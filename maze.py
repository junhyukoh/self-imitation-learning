from PIL import Image
import numpy as np
import gym
import copy
from bs4 import BeautifulSoup
import tensorflow as tf

BLOCK=0
AGENT=1
KEY=2
DOOR=3
TREASURE=4
APPLE=5
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]

COLOR = [
        [44, 42, 60], # block
        [105, 105, 105], # agent
        [135, 206, 250], # key
        [152, 251, 152], # door
        [255, 255, 0], #treasure
        [250, 128, 114], #apple
        ]

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def generate_maze(csv_file):
    raw_csv = np.genfromtxt(csv_file, delimiter=',')
    size = raw_csv.shape[0]
    maze_tensor = np.zeros((size, size, 6))
    obj_pos = [[] for _ in range(len(COLOR))] 
    for y in range(raw_csv.shape[0]):
        for x in range(raw_csv.shape[1]):
            if raw_csv[y][x] > 0:
                obj_idx = int(raw_csv[y][x]-1)
                maze_tensor[y][x][obj_idx] = 1

                if obj_idx is not BLOCK:
                    obj_pos[obj_idx].append([y, x])
       
    return maze_tensor, obj_pos

def visualize_maze(maze, img_size=320):
    my = maze.shape[0]
    mx = maze.shape[1]
    colors = np.array(COLOR, np.uint8)
    num_channel = maze.shape[2]
    vis_maze = np.matmul(maze, colors[:num_channel])
    vis_maze = vis_maze.astype(np.uint8)
    for i in range(vis_maze.shape[0]):
        for j in range(vis_maze.shape[1]):
            if maze[i][j].sum() == 0.0:
                vis_maze[i][j][:] = int(255)
    image = Image.fromarray(vis_maze) 
    return image.resize((int(float(img_size) * mx / my), img_size), Image.NEAREST)

def visualize_mazes(maze, img_size=320):
    if maze.ndim == 3:
        return visualize_maze(maze, img_size=img_size)
    elif maze.ndim == 4:
        n = maze.shape[0]
        size = maze.shape[1]
        dim = maze.shape[-1]
        concat_m = maze.transpose((1,0,2,3)).reshape((size, n * size, dim))
        return visualize_maze(concat_m, img_size=img_size)
    else:
        raise ValueError("maze should be 3d or 4d tensor")

def to_string(maze):
    my = maze.shape[0]
    mx = maze.shape[1]
    str = ''
    for y in range(my):
        for x in range(mx):
            if maze[y][x][BLOCK] == 1:
                str += 'x'
            elif maze[y][x][KEY] == 1:
                str += 'k'
            elif maze[y][x][DOOR] == 1:
                str += 'd'
            elif maze[y][x][TREASURE] == 1:
                str += 't'
            elif maze[y][x][APPLE]==1:
                str += 'p'
            elif maze[y][x][AGENT] == 1:
                str += 'a'
            else:
                str += ' '
        str += '\n'
    return str

class Maze(object):
    def __init__(self, csv_file):
        self.csv = csv_file
        self.reset()

    def reset(self):
        self.maze, self.obj_pos = generate_maze(self.csv)
        self.size = self.maze.shape[0]
        
        self.agent_pos = self.obj_pos[AGENT][0]
        self.update_object_state()
        #self.count_state = np.zeros([self.size, self.size]+[2]*len(self.obj_state))
        #self.count_state[tuple(self.state())] = self.count_state[tuple(self.state())]+1

    def is_reachable(self, y, x):
        if x < 0 or x >= self.size or y < 0 and y >= self.size:
            return False
        if self.maze[y][x][BLOCK] == 1:
            return False
        if self.maze[y][x][DOOR] == 1 and self.existing_object(KEY):
            return False
        return True
 
    def move_agent(self, direction):
        y = self.agent_pos[0] + DY[direction]
        x = self.agent_pos[1] + DX[direction]
        if not self.is_reachable(y, x):
            return False
        self.maze[self.agent_pos[0]][self.agent_pos[1]][AGENT] = 0
        self.maze[y][x][AGENT] = 1
        self.agent_pos = [y, x]
        return True

    def is_object_reached(self, obj_idx):
        if self.maze.shape[2] <= obj_idx:
            return False
        return self.maze[self.agent_pos[0]][self.agent_pos[1]][obj_idx]==1
    
    def is_empty(self, y, x):
        return np.sum(self.maze[y][x]) == 0
    
    def update_object_state(self):
        self.obj_state = []
        self.obj_state.append(float(self.existing_object(KEY)))
        self.obj_state.append(float(self.existing_object(DOOR)))
        self.obj_state.append(float(self.existing_object(TREASURE)))
        for [y, x] in self.obj_pos[APPLE]:
            self.obj_state.append(self.maze[y][x][APPLE])
        # print(self.obj_state)

    def remove_object(self, y, x, obj_idx):
        removed = self.maze[y][x][obj_idx] == 1
        self.maze[y][x][obj_idx] = 0
        self.update_object_state()
        return removed
    
    def existing_object(self, obj_idx): 
        if obj_idx == APPLE:
            for [y, x] in self.obj_pos[APPLE]:
                if self.maze[y][x][APPLE] == 1:
                    return True
            return False
        else:
            [y, x] = self.obj_pos[obj_idx][0]
            return self.maze[y][x][obj_idx] == 1
 
    def observation(self, clone=True):
        return np.array(self.maze, copy=clone)
    
    def state(self):
        #x, y, key, door, treasure, apple
        state = [self.agent_pos[1], self.agent_pos[0]]
        return state + self.obj_state

#    def count(self):
#        return self.count_state[tuple(self.state())]

    def visualize(self):
        return visualize_maze(self.maze)
    
    def to_string(self):
        return to_string(self.maze)

class MazeEnv(object):
    def __init__(self, config="", verbose=1):
        self.config = BeautifulSoup(config, "lxml") 
        # map
        self.csv = self.config.maze["csv"]
        self.max_step = int(self.config.maze["time"])
        # reward
        self.rewards = [[] for _ in range(len(COLOR))]
        self.rewards[KEY] = int(self.config.reward["key"])
        self.rewards[DOOR] = int(self.config.reward["door"])
        self.rewards[TREASURE] = int(self.config.reward["treasure"])
        self.rewards[APPLE] = int(self.config.reward["apple"])
        # meta
        self.meta_remaining_time = str2bool(self.config.meta["remaining_time"])
        # log
        self.log_freq = 100
        self.log_t = 0
        self.max_history = 1000
        self.reward_history = []
        self.length_history = []
        self.verbose = verbose

        self.reset()
        self.action_space = gym.spaces.discrete.Discrete(4)
        self.observation_space = gym.spaces.box.Box(0, 1, self.state().shape)
        self.reward_range = [self.rewards[KEY], self.rewards[TREASURE]] 
        self.metadata = {'remaining_time': self.meta_remaining_time}       
        self.spec = None

    def observation(self, clone=True):
        return self.maze.observation(clone=clone)
    
    def state(self):
        return np.array(self.meta() + self.maze.state())

    def reset(self, reset_episode=True):
        if reset_episode:
            self.t = 0
            self.episode_reward = 0
            self.last_step_reward = 0.0
            self.terminated = False

        self.maze = Maze(self.csv) 
        return self.state()

    def remaining_time(self, normalized=True):
        return float(self.max_step - self.t) / float(self.max_step)
    
    def last_reward(self):
        return self.last_step_reward
    
    def meta(self):
        meta = []
        if self.meta_remaining_time:
            meta.append(self.remaining_time())
        return meta

    def visualize(self):
        return self.maze.visualize() 
    
    def to_string(self):
        return self.maze.to_string()

    def step(self, act):
        assert self.action_space.contains(act), "invalid action: %d" % act
        assert not self.terminated, "episode is terminated"
        self.t += 1

        self.maze.move_agent(act)
        reward = 0
        
        if self.maze.is_object_reached(KEY):
            reward = self.rewards[KEY]
            self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], KEY)
        if self.maze.is_object_reached(DOOR):
            if not self.maze.existing_object(KEY):
                reward = self.rewards[DOOR]
                self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], DOOR)
        if self.maze.is_object_reached(TREASURE):
            if not self.maze.existing_object(DOOR):
                reward = self.rewards[TREASURE]
                self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], TREASURE)
                self.terminated = True
        if self.maze.is_object_reached(APPLE):
            reward = self.rewards[APPLE]
            self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], APPLE)

        if self.t >= self.max_step:
            self.terminated = True

        self.episode_reward += reward
        self.last_step_reward = reward

        to_log = None
        if self.terminated:
            self.log_episode(self.episode_reward, self.t)
            if self.log_t < self.log_freq:
                self.log_t += 1
            else:
                to_log = {}
                to_log["global/episode_reward"] = self.reward_mean(self.log_freq)
                to_log["global/episode_length"] = self.length_mean(self.log_freq)
                self.log_t = 0

        # self.maze.count_state[tuple(self.state())] = self.maze.count_state[tuple(self.state())]+1
        return self.state(), reward, self.terminated, {} #{'count':self.maze.count()}
    
    def log_episode(self, reward, length):
        self.reward_history.insert(0, reward)
        self.length_history.insert(0, length)
        while len(self.reward_history) > self.max_history:
            self.reward_history.pop()
            self.length_history.pop()

    def reward_mean(self, num):
        return np.asarray(self.reward_history[:num]).mean()
    
    def length_mean(self, num):
        return np.asarray(self.length_history[:num]).mean()  

