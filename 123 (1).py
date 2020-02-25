#!/usr/bin/env python

#INIT
import glob
import os
from tkinter import *
import threading
import sys
import numpy as np
import random
import os
import keras
import time
import traceback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from carla import *

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from Carla_utils import *
from keras.models import load_model
from GE import *
#END OF INIT

SENSOR_REFRESH_DELAY = 0.02
SENSOR_DISTANCE = 50
REFRESH_DELAY_NEYRO = 0.1

left = SENSOR_DISTANCE
right = SENSOR_DISTANCE
forward = SENSOR_DISTANCE
collision_flag=False
compass=0
position_right=Transform(Location(x=-304.450623, y=409.865082, z=2.331627), Rotation(yaw=180))
position_left=Transform(Location(x=-400, y=10, z=1), Rotation(yaw=180))
def get_obstacle_forward(data):
    global forward
    if data.other_actor.type_id=="static.road" or data.other_actor.type_id=="static.roadline":
        return 0
    forward=data.distance
    
def get_obstacle_right(data):
    global right
    right=data.distance

    
def get_obstacle_left(data):
    global left
    left=data.distance

def get_collision_data(data):
    global collision_flag
    if not collision_flag:
        collision_flag=True

def get_compass_data(data):
    global compass
    compass=math.degrees(data.compass)
    
def create_obstacle(actor,angle):
    global world
    blueprint = world.get_blueprint_library().find('sensor.other.obstacle')
    blueprint.set_attribute('debug_linetrace','False')
    blueprint.set_attribute('distance',str(SENSOR_DISTANCE))
    blueprint.set_attribute('hit_radius','1')
    blueprint.set_attribute('sensor_tick',str(SENSOR_REFRESH_DELAY))
    transform = Transform(Location(x=1, z=1.2),Rotation(yaw=angle))
    sensor = world.spawn_actor(blueprint, transform, attach_to=actor)
    if angle==0:
        sensor.listen(get_obstacle_forward)
    elif angle==45:
        sensor.listen(get_obstacle_right)
    elif angle==315:
        sensor.listen(get_obstacle_left)  
    return sensor

def create_collision_sensor(actor):
    global world
    blueprint = world.get_blueprint_library().find('sensor.other.collision')
    transform = Transform(Location(x=0, z=0),Rotation(yaw=180))
    sensor = world.spawn_actor(blueprint, transform, attach_to=actor)
    sensor.listen(get_collision_data)
    return sensor


def create_vehicle(position):
    global world
    vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.ford.*'))
    actor = world.spawn_actor(vehicle_bp, position)
    return actor

def speed_control(actor):
    v=actor.get_velocity()
    speed=3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    if speed>40:
        return 0
    elif speed<35:
        return 1
    elif speed<40:
        return 0.3


def speed_checker(actor):
    v=actor.get_velocity()
    speed=3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    print("speed: ",speed)
    if speed<15:
        return 1
    return 0

def create_imu(actor):
    global world
    blueprint = world.get_blueprint_library().find('sensor.other.imu')
    transform = Transform(Location(x=0, z=0),Rotation(yaw=180))
    sensor = world.spawn_actor(blueprint, transform, attach_to=actor)
    sensor.listen(get_compass_data)
    return sensor


def start_race(model):
    position = position_left
    global world
    global collision_flag
    global right
    global left
    global forward
    global compass
    actor=create_vehicle(position)
    sensor_forward=create_obstacle(actor,0)
    sensor_right=create_obstacle(actor,45)
    sensor_left=create_obstacle(actor,315)
    sensor_collision=create_collision_sensor(actor)
    sensor_compass=create_imu(actor)
    speed_control(actor)

    time.sleep(1)
    start_time=time.time()
    compass_time=0
    compass_data=[]
    flag_fail_race=0
    while not collision_flag:
        if len(compass_data)<15:
            compass_data.append(compass)
            
        else:
            #print(compass_data)
            for i in compass_data:
                for j in compass_data:
                    if abs(i-j)>120 and abs(i-j)<240:
                        accuracy=0
                        flag_fail_race=1
                        break       #drop first for
                if flag_fail_race:
                    break  #drop seconf for
            if flag_fail_race:  #drop while
                print(compass_data)
                break
            compass_data=[]

            


        time.sleep(REFRESH_DELAY_NEYRO)
        throttle=speed_control(actor)
        
        sensor_data=np.array([1/left,1/forward,1/right]).reshape((1,3))
        predicted_steer=model.predict(sensor_data)

        predicted_steer=predicted_steer[0][0]//0.02
        predicted_steer=predicted_steer*0.02
        v=actor.get_velocity()
        speed=3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        #print(predicted_steer)
        control=actor.get_control()
        print("Speed: ",speed, "  throttle:", control.throttle )
        control.throttle=throttle
        control.steer=predicted_steer
        actor.apply_control(control)
        
        forward=right=left=SENSOR_DISTANCE

    if not flag_fail_race:
        accuracy=(time.time()-start_time)/1000    
    sensor_forward.stop()
    sensor_left.stop()
    sensor_right.stop()
    sensor_collision.stop()
    sensor_compass.stop()
    sensor_compass.destroy()
    actor.destroy()
    sensor_forward.destroy()
    sensor_left.destroy()
    sensor_right.destroy()
    sensor_collision.destroy()
    collision_flag=0
    return accuracy




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("--model", help="Your model to run")
    args = parser.parse_args()

    position_forward=Transform(Location(x=331.461670, y=15.913572, z=4.062667),Rotation(yaw=180))
    client = Client('localhost', 2000)
    client.set_timeout(10.0)
    #world = client.get_world() # no reload, only get world
    world = client.load_world('Town04') # reload world
    
    spc=world.get_spectator()
    #spc.set_location(Location(26.723200, 293.131073, 2.608377))
    if (args.model is None):
        evolution(start_race)
    else:
        model = load_model(args.model)
        start_race(model)