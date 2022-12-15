import random 
import sys
import glob
import os
from collections import deque
import numpy as np 
import cv2
import math
import time 
import tensorflow as tf
from keras.backend import set_session
from keras.models import load_model
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

MODEL_PATH = 'models/Improved_DQNmodel'
MEMORY_FRACTION = 0.8

IM_WIDTH = 960
IM_HEIGHT = 480
SHOW_PREVIEW  = False
SECONDS_PER_EPISODE = 30

class ACTIONS:
    forward = 0
    left = 1
    right = 2
    forward_left = 3
    forward_right = 4
    brake = 5
    brake_left = 6
    brake_right = 7

ACTION_CONTROL = {
    0: [1, 0, 0],
    1: [0, 0, -1],
    2: [0, 0, 1],
    3: [1, 0, -1],
    4: [1, 0, 1],
    5: [0, 1, 0],
    6: [0, 1, -1],
    7: [0, 1, 1],
}

ACTIONS_NAMES = {
    0: 'forward',
    1: 'left',
    2: 'right',
    3: 'forward_left',
    4: 'forward_right',
    5: 'brake',
    6: 'brake_left',
    7: 'brake_right',
}

ACTION = ['forward', 'left', 'right', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0   ## full turn for every single time
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    action_space_size = len(ACTION)

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(200.0)
      
        #self.world = self.client.load_world('Town05')
        #self.map = self.world.get_map()   
        
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.model_3 = self.blueprint_library.filter("model3")[0]  ## grab tesla model3 from library
        
        self.actions = [getattr(ACTIONS, action) for action in ACTION]

    def reset(self):
        self.collision_hist = []    
        self.actor_list = []
        self.laneInvasion_hist = []
        
        self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=5.0)

        self.filtered_waypoints = []   
        i = 0
        for self.waypoint in self.waypoints:
            if(self.waypoint.road_id == 10):
                self.filtered_waypoints.append(self.waypoint)
                for i in range(len(self.filtered_waypoints)):
                    self.world.debug.draw_string(self.filtered_waypoints[i].transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=0, g=255, b=0), life_time=40,
                                   persistent_lines=True)
                    i = i+1

        self.spawn_point = self.filtered_waypoints[1].transform

        self.spawn_point.location.z += 2
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)  ## changed for adding waypoints


        self.actor_list.append(self.vehicle)
        
        #Set Camera Attributes
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")  ## fov, field of view
        
        #Spawn and attach the camera
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        
        #Set up the collision sensor
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        #Set up the Lane Sensor
        lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lanesensor = self.world.spawn_actor(lanesensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_invasion_data(event))

        while self.front_camera is None:  ## return the observation
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)
        
    def lane_invasion_data(self, event):
        #Lane Invasion Detected, return reward of -100
        self.laneInvasion_hist.append(event)
        

    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera = i3  ## remember to scale this down between 0 and 1 for CNN input purpose


    def step(self, action):
        
        #if action == 0:
        #    self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0.0 ))
            
        #if action == 1:
        #    self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0*self.STEER_AMT))
        #if action == 2:
        #    self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0*self.STEER_AMT))
        
        #Apply the action, an action must be taken every step 
        self.vehicle.apply_control(carla.VehicleControl(throttle=ACTION_CONTROL[self.actions[action]][0], steer=ACTION_CONTROL[self.actions[action]][2]*self.STEER_AMT, brake=ACTION_CONTROL[self.actions[action]][1]))
           
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
      
        
        #Define rewards
        i = 2
        for i in range(2, len(self.filtered_waypoints)):


            if len(self.collision_hist) != 0:
                done = True
                reward = -300
            elif kmh < 30:
                done = False
                reward = -15
            elif carla.Location.distance(carla.Actor.get_location(self.actor_list[0]), self.filtered_waypoints[i].transform.location) == 0:
                done = False
                reward = 25
            elif len(self.laneInvasion_hist) != 0:
                done = False
                reward = -10
            else:
                done = False
                reward = 30
            i = i + 1

            if self.episode_start + SECONDS_PER_EPISODE < time.time():  ## when to stop
                done = True

            return self.front_camera, reward, done, None
            
if __name__ == '__main__':

    #Set GPU Options 
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
    
    #Load the Model
    model = load_model(MODEL_PATH)
    print("Loading Model from: ", MODEL_PATH)
    
    #Create the environment for the car 
    env = CarEnv()
    
    #For agent speed measurements, keep last 60 frames
    fps_counter = deque(maxlen=60)
    
    #Initialize the predictions 
    model.predict(np.ones((1, IM_HEIGHT, IM_WIDTH, 3)))
    
    #Main loop 
    while True:

        print('Restarting Episodes')

        current_state = env.reset()
        env.collision_hist = []

        done = False

        while True:
            step_start = time.time()
        
            #cv2.imshow(f'Agent - preview', current_state)
            #cv2.waitKey(1)
        
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qs)
        
            new_state, reward, done, _ = env.step(action)
        
            current_state = new_state
        
            if done:
                break
        
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: {action}')
        
        #Destroy an actor at the end of an episode
        for actor in env.actor_list:
            actor.destroy()