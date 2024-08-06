import numpy as np
import carla
import random

import carla_gym.utils.transforms as trans_utils
from carla_gym.core.obs_manager.object_finder.vehicle import ObsManager as OmVehicle
from carla_gym.core.obs_manager.object_finder.pedestrian import ObsManager as OmPedestrian
from carla_gym.core.obs_manager.navigation.waypoint_plan import ObsManager as OmWaypoint


from carla_gym.utils.traffic_light import TrafficLightHandler
from carla_gym.utils.hazard_actor import is_straight_line, lbc_hazard_walker, wp_hazard_vehicle ,curve_speed, is_lanechange_or_junction ,wp_hazard_vehicle_future, EgoModel
from carla_gym.core.obs_manager.birdview.chauffeurnet import  ObsManager as OmBev

class ValeoAction(object):

    def __init__(self, ego_vehicle):
        self._ego_vehicle = ego_vehicle

        self.om_vehicle = OmVehicle({'max_detection_number':30, 'distance_threshold': 40})
        # self.om_pedestrian = OmPedestrian({'max_detection_number': 40, 'distance_threshold': 33})
        self.om_waypoint = OmWaypoint({'steps':40})
        self.om_vehicle.attach_ego_vehicle(self._ego_vehicle)
        # self.om_pedestrian.attach_ego_vehicle(self._ego_vehicle)
        self.om_waypoint.attach_ego_vehicle(self._ego_vehicle)

        # self._maxium_speed = speed_limit/3.6 #m/s
        self.last_speed_limit = 0.0 #kph
        self.last_random_speed = 0.0 #mps

        self._last_steer = 0.0
        # self._last_throttle = 0.0
        self._tl_offset = -0.8 * self._ego_vehicle.vehicle.bounding_box.extent.x
        self._vehicle_model = EgoModel(dt=(1.0 / 20))###

    def get(self, terminal_reward):
        ev_transform = self._ego_vehicle.vehicle.get_transform()
        ev_control = self._ego_vehicle.vehicle.get_control()
        ev_vel = self._ego_vehicle.vehicle.get_velocity()
        ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))

        speed_limit = self._ego_vehicle.vehicle.get_speed_limit()  # kph

        # Check if the speed limit has changed
        if self.last_speed_limit != speed_limit:
            self.last_speed_limit = speed_limit

            # Set the random speed based on the speed limit
            if speed_limit == 30.0:
                # If the speed limit is 30 km/h, choose a random speed from [5, 6, 7] m/s
                random_speed = random.choice([5.0, 6.0, 7.0])
            elif speed_limit == 60.0:
                random_speed = random.choice([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
            elif speed_limit == 90.0:
                random_speed = random.choice([17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0])
            else:  # speed_limit == 40.0
                random_speed = random.choice([8.0, 9.0, 10.0])
            self.last_random_speed = random_speed
        else:
            # If the speed limit hasn't changed, use the previously chosen random speed
            random_speed = self.last_random_speed

        # Set the maximum speed to the chosen random speed
        _maxium_speed = random_speed

        waypoint_plan = self.om_waypoint.get_observation() 
        # print("###waypoint_plan:",waypoint_plan['location'])

        # action
        steer_shake = abs(ev_control.steer - self._last_steer)
        # straight = is_straight_line(waypoint_plan)
        # if steer_shake > 0.0 :#and straight is True:
        if steer_shake > 0.01:
            r_steer = -0.1 #-0.5*steer_shake
        else:
            r_steer = 0.0
        self._last_steer = ev_control.steer

        # if (ev_control.throttle - self._last_throttle) > 0.7:
        #     r_throttle = -0.1
        # else:
        #     r_throttle = 0.0
        # self._last_throttle = ev_control.throttle

        # desired_speed
        obs_vehicle = self.om_vehicle.get_observation()
        hazard_vehicle_loc = wp_hazard_vehicle(waypoint_plan, obs_vehicle, max_distance=2.65)#assume lane width3.2m, half1.6
        # obs_pedestrian = self.om_pedestrian.get_observation()
        # hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=33.0)#9.5
        light_state, light_loc, _ = TrafficLightHandler.get_light_state(self._ego_vehicle.vehicle,
                                                                        offset=self._tl_offset, dist_threshold=40.0)#18.0

        desired_spd_veh  = desired_spd_rl = slow_before_cur = desired_spd_stop = desired_spd_fu = _maxium_speed#self._maxium_speed = desired_spd_ped 

        # future_hazard
        # Check if the waypoint plan includes a lane change or junction
        lc_ju_wp = is_lanechange_or_junction(waypoint_plan)
        a = False
        dist_fu = 0.0  # Distance to future hazard
        
        # If lane change or junction is present
        if lc_ju_wp:
            a = True
            
            # Get future hazard location
            future_hazard_loc = wp_hazard_vehicle_future(lc_ju_wp,  # Waypoint plan
                                                        obs_vehicle,  # Observed vehicles
                                                        self._vehicle_model,  # Vehicle model
                                                        max_distance=2.75,  # Maximum distance
                                                        num_frames=60)  # Number of frames
            
            # If future hazard location is found
            if future_hazard_loc is not None:
                # Calculate distance to future hazard
                dist_fu = max(0.0, np.linalg.norm(future_hazard_loc[0:2]) - 8.0)
    
                # Calculate desired speed for future hazard
                desired_spd_fu = round(dist_fu * 4.0 / 3.6, 3)

        # If the location of the maximum curvature is None, then the curvature is also set to None.
        curvature, cur_spd, curve_loc = curve_speed(waypoint_plan)

        if curve_loc is not None:
            # Calculate the distance to the maximum curvature
            dist_cur = max(0.0, np.linalg.norm(curve_loc) - 1.0)
            slow_before_cur = round(dist_cur * 3.5 / 3.6, 3) + cur_spd

        if hazard_vehicle_loc is not None:
            dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2])-8.0)
            desired_spd_veh = round(dist_veh * 3.5 / 3.6, 3)

        # if hazard_ped_loc is not None:
        #     dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2])-9.0) #6
        #     # desired_spd_ped = _maxium_speed * np.clip(dist_ped, 0.0, 5.0)/5.0 #self._maxium_speed
        #     desired_spd_ped = round(dist_ped * 4.0 / 3.6, 2)

        if (light_state == carla.TrafficLightState.Red or light_state == carla.TrafficLightState.Yellow):
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2])-8.0)#5
            desired_spd_rl = round(dist_rl * 3.5 / 3.6, 3)

        # stop sign
        stop_sign = self._ego_vehicle.criteria_stop._target_stop_sign
        stop_loc = None
        if (stop_sign is not None) and (not self._ego_vehicle.criteria_stop._stop_completed):
            trans = stop_sign.get_transform()
            tv_loc = stop_sign.trigger_volume.location
            loc_in_world = trans.transform(tv_loc)
            loc_in_ev = trans_utils.loc_global_to_ref(loc_in_world, ev_transform)
            stop_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
            dist_stop = max(0.0, np.linalg.norm(stop_loc[0:2])-6.0)
            desired_spd_stop = round(dist_stop * 3.5 / 3.6, 2)

        desired_speed = min(_maxium_speed, desired_spd_veh, desired_spd_rl, slow_before_cur, desired_spd_stop, desired_spd_fu)###desired_spd_ped

        # r_speed
        r_speed = (1.0 - np.abs(ev_speed-desired_speed) / _maxium_speed)*1.0 #self._maxium_speed

        # bouns
        if desired_speed > 0 and ev_speed+1 < desired_speed and ev_control.throttle > 0:
            r_speed+=0.2
        # if desired_speed > 17 and ev_speed < desired_speed and ev_speed >16:
        #     r_speed+=0.1


        # r_position
        wp_transform = self._ego_vehicle.get_route_transform()

        d_vec = ev_transform.location - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

        lateral_distance = np.abs(np.dot(np_wp_unit_right, np_d_vec))
        r_position = -0.8 * lateral_distance #0.5

        # r_rotation
        angle_difference = np.deg2rad(np.abs(trans_utils.cast_angle(
            ev_transform.rotation.yaw - wp_transform.rotation.yaw)))
        r_rotation = -0.8 * angle_difference

        reward = r_speed + r_position + r_rotation + terminal_reward + r_steer #+ r_brake # r_throttle 

        # if hazard_vehicle_loc is None:
        #     txt_hazard_veh = '[]'
        # else:
        #     txt_hazard_veh = np.array2string(hazard_vehicle_loc[0:2], precision=1, separator=',', suppress_small=True)
        # if hazard_ped_loc is None:
        #     txt_hazard_ped = '[]'
        # else:
        #     txt_hazard_ped = np.array2string(hazard_ped_loc[0:2], precision=1, separator=',', suppress_small=True)
        # if light_loc is None:
        #     txt_light = '[]'
        # else:
        #     txt_light = np.array2string(light_loc[0:2], precision=1, separator=',', suppress_small=True)
        # if stop_loc is None:
        #     txt_stop = '[]'
        # else:
        #     txt_stop = np.array2string(stop_loc[0:2], precision=1, separator=',', suppress_small=True)

        debug_texts = [
            f'rs:{r_speed:5.2f} rp:{r_position:5.2f} rr:{r_rotation:5.2f}',
            f'r:{reward:5.2f} lim:{speed_limit} maxs:{_maxium_speed:5.2f}',
            f'ds:{desired_speed:5.2f}, s:{ev_speed:5.2f}, ',
            f'cur:{curvature:5.2f},{cur_spd:5.2f} slow:{slow_before_cur:5.2f}',###
            # f'straight:{straight}, wd:{weight_double}',###
            f'l&j:{a} fu_ds:{dist_fu:5.2f} s:{desired_spd_fu:5.2f}'
            # f'veh_ds:{desired_spd_veh:5.2f} {txt_hazard_veh}',
            # f'ped_ds:{desired_spd_ped:5.2f} {txt_hazard_ped}',
            # f'tl_ds:{desired_spd_rl:5.2f} {light_state}{txt_light}',
            # f'st_ds:{desired_spd_stop:5.2f} {txt_stop}',
            # f'r_term:{terminal_reward:5.2f}' #straight:{straight}'
        ]
        reward_debug = {
            'debug_texts': debug_texts
        }
        return reward, reward_debug
