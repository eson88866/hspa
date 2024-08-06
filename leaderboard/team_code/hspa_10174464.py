import os
import json
import datetime
import pathlib
import time
import cv2

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
import numpy as np
from omegaconf import OmegaConf

from carla_gym.core.task_actor.common.criteria import run_stop_sign
from hspa.obs_manager.birdview.chauffeurnet_21b import ObsManager
from carla_gym.utils.config_utils import load_entry_point
import carla_gym.utils.transforms as trans_utils
from carla_gym.utils.traffic_light import TrafficLightHandler
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import downsample_route
from agents.navigation.local_planner import RoadOption

from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)

def get_entry_point():
	return 'HSPAgent'


class HSPAgent(autonomous_agent.AutonomousAgent):#roach-11833344 #hspa-10174464
	def setup(self, path_to_conf_file, ckpt="leaderboard/team_code/hspa/log/ckpt_10174464.pth"):
		self._render_dict = None
		self.supervision_dict = None
		self._ckpt = ckpt
		cfg = OmegaConf.load(path_to_conf_file)
		cfg = OmegaConf.to_container(cfg)
		self.cfg = cfg
		self._obs_configs = cfg['obs_configs']
		self._train_cfg = cfg['training']
		self._policy_class = load_entry_point(cfg['policy']['entry_point'])
		self._policy_kwargs = cfg['policy']['kwargs']
		print('###',self._ckpt)
		if self._ckpt is None:
			self._policy = None
		else:
			self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
			self._policy = self._policy.eval()
		self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
		self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self._3d_bb_distance = 50

		self.prev_lidar = None

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))


			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'measurements').mkdir()
			(self.save_path / 'supervision').mkdir()
			(self.save_path / 'bev').mkdir()
			(self.save_path / 'visualize').mkdir()

	def _init(self):
		self._waypoint_planner = RoutePlanner(4.0, 50)
		self._waypoint_planner.set_route(self._plan_gps_HACK, True)

		self._command_planner = RoutePlanner(7.5, 25.0, 257)
		self._command_planner.set_route(self._global_plan, True)

		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self._world = CarlaDataProvider.get_world()
		self._map = self._world.get_map()
		self._ego_vehicle = CarlaDataProvider.get_ego()
		self._last_route_location = self._ego_vehicle.get_location()
		self._criteria_stop = run_stop_sign.RunStopSign(self._world)
		self.birdview_obs_manager = ObsManager(self.cfg['obs_configs']['birdview'], self._criteria_stop)
		self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)

		self.navigation_idx = -1


		# for stop signs
		self._target_stop_sign = None # the stop sign affecting the ego vehicle
		self._stop_completed = False # if the ego vehicle has completed the stop sign
		self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign

		TrafficLightHandler.reset(self._world)
		print("initialized")

		self.initialized = True

	def _get_angle_to(self, pos, theta, target):
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta),  np.cos(theta)],
			])

		aim = R.T.dot(target - pos)
		angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
		angle = 0.0 if np.isnan(angle) else angle 

		return angle
	

	def _truncate_global_route_till_local_target(self, windows_size=5):
		ev_location = self._ego_vehicle.get_location()
		closest_idx = 0
		for i in range(len(self._global_route)-1):
			if i > windows_size:
				break

			loc0 = self._global_route[i][0].transform.location
			loc1 = self._global_route[i+1][0].transform.location

			wp_dir = loc1 - loc0
			wp_veh = ev_location - loc0
			dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

			if dot_ve_wp > 0:
				closest_idx = i+1
		if closest_idx > 0:
			self._last_route_location = carla.Location(self._global_route[0][0].transform.location)

		self._global_route = self._global_route[closest_idx:]

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._command_planner.mean) * self._command_planner.scale

		return gps

	def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp_route):
		"""
		Set the plan (route) for the agent
		"""
		self._global_route = wp_route
		ds_ids = downsample_route(global_plan_world_coord, 50)
		self._global_plan = [global_plan_gps[x] for x in ds_ids]
		self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]

		self._plan_gps_HACK = global_plan_gps
		self._plan_HACK = global_plan_world_coord

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 900, 'height': 256, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data, timestamp):
		self._truncate_global_route_till_local_target()

		birdview_obs = self.birdview_obs_manager.get_observation(self._global_route)
		control = self._ego_vehicle.get_control()
		throttle = np.array([control.throttle], dtype=np.float32)
		steer = np.array([control.steer], dtype=np.float32)
		brake = np.array([control.brake], dtype=np.float32)
		gear = np.array([control.gear], dtype=np.float32)

		# You can specify the target speed or dynamically adjust it according to the road speed limits.
		speed_limit = self._ego_vehicle.get_speed_limit() / 3.6###
		np_speed_limit = np.array([speed_limit], dtype=np.float32)###
		# np_speed_limit = np.array([16.0], dtype=np.float32) #m/s

		
		ev_transform = self._ego_vehicle.get_transform()
		vel_w = self._ego_vehicle.get_velocity()
		vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)
		vel_xy = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)


		self._criteria_stop.tick(self._ego_vehicle, timestamp)

		state_list = []
		state_list.append(np_speed_limit)###
		state_list.append(throttle)
		state_list.append(steer)
		state_list.append(brake)
		state_list.append(gear)
		state_list.append(vel_xy)
		state = np.concatenate(state_list)
		obs_dict = {
			'state': state.astype(np.float32),
			'birdview': birdview_obs['masks'],
		}

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)

		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		target_gps, target_command = self.get_target_gps(input_data['gps'][1], compass)

		weather = self._weather_to_dict(self._world.get_weather())

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'weather': weather,
				}
		next_wp, next_cmd = self._route_planner.run_step(self._get_position(result))

		result['next_command'] = next_cmd.value
		result['x_target'] = next_wp[0]
		result['y_target'] = next_wp[1]

		
		return result, obs_dict, birdview_obs['rendered'], target_gps, target_command

	def im_render(self, render_dict):
		im_birdview = render_dict['rendered']
		h, w, c = im_birdview.shape
		im = np.zeros([h, w, c], dtype=np.uint8)
		im[:h, :w] = im_birdview

		# speed_str = np.array2string(3.6 * np.abs(float(render_dict['speed'])), precision=2, separator=',', suppress_small=True)
		# txt_1 = f'speed:{speed_str}km/h'
		txt_1 = f"speed:{3.6 * np.abs(float(render_dict['speed'])):.1f}km/h"
		im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
		# txt_1 = f'action{action_str}'
		# im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
					
		# for i, txt in enumerate(debug_texts):
		# 	im = cv2.putText(im, txt, (3, 25+ int(0.3 * 24)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
		return im

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()

		self.step += 1

		if self.step < 20:

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			self.last_control = control
			return control

		if self.step % 2 != 0:
			return self.last_control
		tick_data, policy_input, rendered, target_gps, target_command = self.tick(input_data, timestamp)

		gps = self._get_position(tick_data)

		near_node, near_command = self._waypoint_planner.run_step(gps)
		far_node, far_command = self._command_planner.run_step(gps)

		actions, values, log_probs, mu, sigma, features = self._policy.forward(
			policy_input, deterministic=True, clip_action=True)
		control = self.process_act(actions)

		# render_dict = {"rendered": rendered, "action": actions}
		render_dict = {"rendered": rendered, "speed": tick_data['speed']}###
		# render_dict = {"rendered": rendered, "action": actions,}
		
			
		render_img = self.im_render(render_dict)

		supervision_dict = {
			'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
			'value': values[0],
			'action_mu': mu[0],
			'action_sigma': sigma[0],
			'features': features[0],
			'speed': tick_data['speed'],
			'target_gps': target_gps,
			'target_command': target_command,
		}
		# if SAVE_PATH is not None and self.step % 10 == 0:
		if SAVE_PATH is not None :
			self.save(near_node, far_node, near_command, far_command, tick_data, supervision_dict, render_img)

		# steer = control.steer
		# control.steer = steer + 1e-2 * np.random.randn()
		self.last_control = control
		return control

	def save(self, near_node, far_node, near_command, far_command, tick_data, supervision_dict, render_img):
		frame = self.step - 20

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%05d.png' % frame))
		Image.fromarray(render_img).save(self.save_path / 'bev' / ('%04d.png' % frame))
		
		# Read tick_data['rgb'] and render_img and convert them into PIL image objects
		rgb_image = Image.fromarray(tick_data['rgb'])
		render_image = Image.fromarray(render_img)
		# Calculate the new width of render_img after adjusting its height, keeping the height ratio
		new_width = int(render_image.width * (rgb_image.height / render_image.height))
		# Resize render_img to match the height of tick_data['rgb']
		render_image_resized = render_image.resize((new_width, rgb_image.height))
		# Join two pictures horizontally
		concatenated_image = Image.new('RGB', (rgb_image.width + render_image_resized.width, rgb_image.height))
		concatenated_image.paste(rgb_image, (0, 0))
		concatenated_image.paste(render_image_resized, (rgb_image.width, 0))
		
		# Save the connected image
		concatenated_image.save(self.save_path / 'visualize' / ('%05d.png' % frame))		
		
		###
		pos = self._get_position(tick_data)
		theta = tick_data['compass']
		speed = tick_data['speed']

		data = {
				'x': pos[0],
				'y': pos[1],
				'theta': theta,
				'speed': speed,
				'x_command_far': far_node[0],
				'y_command_far': far_node[1],
				'command_far': far_command.value,
				'x_command_near': near_node[0],
				'y_command_near': near_node[1],
				'command_near': near_command.value,
				'x_target': tick_data['x_target'],
				'y_target': tick_data['y_target'],
				'target_command': tick_data['next_command'],
				}
		outfile = open(self.save_path / 'measurements' / ('%04d.json' % frame), 'w')
		json.dump(data, outfile, indent=4)
		outfile.close()
		with open(self.save_path / 'supervision' / ('%04d.npy' % frame), 'wb') as f:
			np.save(f, supervision_dict)
		
			
	def get_target_gps(self, gps, compass):
		"""
		Calculate the target GPS coordinates and the corresponding road option.

		Parameters:
			gps (numpy.ndarray): The current GPS coordinates.
			compass (float): The compass heading.

		Returns:
			numpy.ndarray: The target GPS coordinates.
			numpy.ndarray: The corresponding road option.
		"""
		# Convert GPS coordinates to carla.Location objects
		def gps_to_location(gps):
			"""
			Convert GPS coordinates to carla.Location objects.

			Parameters:
				gps (numpy.ndarray): The GPS coordinates.

			Returns:
				carla.Location: The carla.Location object.
			"""
			lat, lon, z = gps
			lat = float(lat)
			lon = float(lon)
			z = float(z)

			location = carla.Location(z=z)
			xy =  (gps[:2] - self._command_planner.mean) * self._command_planner.scale
			location.x = xy[0]
			location.y = -xy[1]
			return location
		
		# Calculate the target GPS coordinates and the corresponding road option
		global_plan_gps = self._global_plan
		next_gps, _ = global_plan_gps[self.navigation_idx+1]
		next_gps = np.array([next_gps['lat'], next_gps['lon'], next_gps['z']])
		next_vec_in_global = gps_to_location(next_gps) - gps_to_location(gps)
		ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
		loc_in_ev = trans_utils.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

		# If the target point is within a certain distance and in the opposite direction,
		# move to the next point in the global plan
		if np.sqrt(loc_in_ev.x**2+loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
			self.navigation_idx += 1

		# Ensure that the navigation index is within the valid range
		self.navigation_idx = min(self.navigation_idx, len(global_plan_gps)-2)

		# Get the road option before and after the target point
		_, road_option_0 = global_plan_gps[max(0, self.navigation_idx)]
		gps_point, road_option_1 = global_plan_gps[self.navigation_idx+1]
		gps_point = np.array([gps_point['lat'], gps_point['lon'], gps_point['z']])

		# If the road option before the target point is a lane change option and the road option
		# after the target point is not a lane change option, use the road option after the target point
		if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
				and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
			road_option = road_option_1
		else:
			road_option = road_option_0

		# Return the target GPS coordinates and the corresponding road option
		return np.array(gps_point, dtype=np.float32), np.array([road_option.value], dtype=np.int8)


	def process_act(self, action):
		"""
		Process the action received from the agent and return a carla.VehicleControl object.

		Parameters:
			action (numpy.ndarray): The action received from the agent.

		Returns:
			carla.VehicleControl: The processed control values.
		"""
		# Extract the acceleration and steering values from the action.
		# Note: The action is a numpy array of shape (batch_size, action_size),
		# where action_size is the size of the action vector.
		acc = action[0][0]
		steer = action[0][1]

		# Calculate the throttle and brake values based on the acceleration.
		# If the acceleration is greater than or equal to 0, then the throttle value is set to the acceleration.
		# Otherwise, the brake value is set to the absolute value of the acceleration.
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		# Clip the throttle, steering and brake values to their valid ranges.
		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)

		# Create a carla.VehicleControl object with the processed control values.
		control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

		# Return the processed control values.
		return control

	def _weather_to_dict(self, carla_weather):
		weather = {
			'cloudiness': carla_weather.cloudiness,
			'precipitation': carla_weather.precipitation,
			'precipitation_deposits': carla_weather.precipitation_deposits,
			'wind_intensity': carla_weather.wind_intensity,
			'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
			'sun_altitude_angle': carla_weather.sun_altitude_angle,
			'fog_density': carla_weather.fog_density,
			'fog_distance': carla_weather.fog_distance,
			'wetness': carla_weather.wetness,
			'fog_falloff': carla_weather.fog_falloff,
		}

		return weather


	def _get_3d_bbs(self, max_distance=50):

		bounding_boxes = {
			"traffic_lights": [],
			"stop_signs": [],
			"vehicles": [],
			"pedestrians": []
		}

		bounding_boxes['traffic_lights'] = self._find_obstacle_3dbb('*traffic_light*', max_distance)
		bounding_boxes['stop_signs'] = self._find_obstacle_3dbb('*stop*', max_distance)
		bounding_boxes['vehicles'] = self._find_obstacle_3dbb('*vehicle*', max_distance)
		bounding_boxes['pedestrians'] = self._find_obstacle_3dbb('*walker*', max_distance)

		return bounding_boxes


	def _find_obstacle_3dbb(self, obstacle_type, max_distance=50):
		"""Returns a list of 3d bounding boxes of type obstacle_type.
		If the object does have a bounding box, this is returned. Otherwise a bb
		of size 0.5,0.5,2 is returned at the origin of the object.

		Args:
			obstacle_type (String): Regular expression
			max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.

		Returns:
			List: List of Boundingboxes
		"""        
		obst = list()
		
		_actors = self._world.get_actors()
		_obstacles = _actors.filter(obstacle_type)

		for _obstacle in _obstacles:    
			distance_to_car = _obstacle.get_transform().location.distance(self._ego_vehicle.get_location())

			if 0 < distance_to_car <= max_distance:
				
				if hasattr(_obstacle, 'bounding_box'): 
					loc = _obstacle.bounding_box.location
					_obstacle.get_transform().transform(loc)

					extent = _obstacle.bounding_box.extent
					_rotation_matrix = self.get_matrix(carla.Transform(carla.Location(0,0,0), _obstacle.get_transform().rotation))

					rotated_extent = np.squeeze(np.array((np.array([[extent.x, extent.y, extent.z, 1]]) @ _rotation_matrix)[:3]))

					bb = np.array([
						[loc.x, loc.y, loc.z],
						[rotated_extent[0], rotated_extent[1], rotated_extent[2]]
					])

				else:
					loc = _obstacle.get_transform().location
					bb = np.array([
						[loc.x, loc.y, loc.z],
						[0.5, 0.5, 2]
					])

				obst.append(bb)

		return obst

	def get_matrix(self, transform):
		"""
		Creates matrix from carla transform.
		"""

		rotation = transform.rotation
		location = transform.location
		c_y = np.cos(np.radians(rotation.yaw))
		s_y = np.sin(np.radians(rotation.yaw))
		c_r = np.cos(np.radians(rotation.roll))
		s_r = np.sin(np.radians(rotation.roll))
		c_p = np.cos(np.radians(rotation.pitch))
		s_p = np.sin(np.radians(rotation.pitch))
		matrix = np.matrix(np.identity(4))
		matrix[0, 3] = location.x
		matrix[1, 3] = location.y
		matrix[2, 3] = location.z
		matrix[0, 0] = c_p * c_y
		matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
		matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
		matrix[1, 0] = s_y * c_p
		matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
		matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
		matrix[2, 0] = s_p
		matrix[2, 1] = -c_p * s_r
		matrix[2, 2] = c_p * c_r
		return matrix

