import numpy as np
import math


def is_within_distance_ahead(target_location, max_distance, up_angle_th=60):
    distance = np.linalg.norm(target_location[0:2])
    if distance < 0.001:
        return True
    if distance > max_distance:
        return False
    x = target_location[0]
    y = target_location[1]
    angle = np.rad2deg(np.arctan2(y, x))
    return abs(angle) < up_angle_th


def lbc_hazard_vehicle(obs_surrounding_vehicles, ev_speed=None, proximity_threshold=9.5):
    for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
        if not is_valid:
            continue

        sv_yaw = obs_surrounding_vehicles['rotation'][i][2]
        same_heading = abs(sv_yaw) <= 150

        sv_loc = obs_surrounding_vehicles['location'][i]
        with_distance_ahead = is_within_distance_ahead(sv_loc, proximity_threshold, up_angle_th=45)
        if same_heading and with_distance_ahead:
            return sv_loc
    return None

def wp_hazard_vehicle(waypoint_plan, obs_surrounding_vehicles, max_distance=1.5):###
    """
    Find the first vehicle within a certain distance to the waypoints.
    
    Args:
        waypoint_plan (dict): The plan of the waypoints.
        obs_surrounding_vehicles (dict): The information of the surrounding vehicles.
        max_distance (float): The maximum distance to the waypoints. Default is 1.5.

    Returns:
        numpy.ndarray or None: The location of the vehicle if found, otherwise None.
    """

    # Extract waypoint locations
    waypoint_locations = waypoint_plan['location']

    # Iterate over surrounding vehicles
    for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
        # Skip invalid vehicles
        if not is_valid:
            continue

        # Extract vehicle location
        sv_loc = obs_surrounding_vehicles['location'][i][:2]
        
        # Iterate over waypoints
        for wp_loc in waypoint_locations:
            # Calculate distance between vehicle and waypoint
            distance = np.linalg.norm(sv_loc - wp_loc)
            
            # If distance is within the threshold, return the location
            if distance < max_distance:
                return sv_loc
    
    # If no vehicle is found, return None
    return None

def lbc_hazard_walker(obs_surrounding_pedestrians, ev_speed=None, proximity_threshold=9.5):
    """
    Identify a pedestrian hazard within a certain proximity threshold. 
    The proximity threshold is calculated based on the distance of the pedestrian.

    Args:
        obs_surrounding_pedestrians (dict): The information of the surrounding pedestrians.
        ev_speed (float, optional): The speed of the ego-vehicle. Defaults to None.
        proximity_threshold (float, optional): The threshold distance to consider a pedestrian as a hazard. Defaults to 9.5.

    Returns:
        numpy.ndarray or None: The location of the pedestrian if found, otherwise None.
    """
    # Iterate over surrounding pedestrians
    for i, is_valid in enumerate(obs_surrounding_pedestrians['binary_mask']):
        # Skip invalid pedestrians
        if not is_valid:
            continue

        # Skip pedestrians on the sidewalk
        if int(obs_surrounding_pedestrians['on_sidewalk'][i]) == 1:
            continue

        # Extract pedestrian location
        ped_loc = obs_surrounding_pedestrians['location'][i]

        # Calculate distance of the pedestrian
        dist = np.linalg.norm(ped_loc)

        # Calculate angle threshold based on distance
        degree = 162 / (np.clip(dist, 1.5, 10.5)+0.3) #1.5~10.5

        # Check if pedestrian is within the proximity threshold
        if is_within_distance_ahead(ped_loc, proximity_threshold, up_angle_th=degree):
            return ped_loc

    # If no pedestrian is found, return None
    return None


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


def challenge_hazard_walker(obs_surrounding_pedestrians, ev_speed=None):
    p1 = np.float32([0, 0])
    v1 = np.float32([10, 0])

    for i, is_valid in enumerate(obs_surrounding_pedestrians['binary_mask']):
        if not is_valid:
            continue

        ped_loc = obs_surrounding_pedestrians['location'][i]
        ped_yaw = obs_surrounding_pedestrians['rotation'][i][2]
        ped_vel = obs_surrounding_pedestrians['absolute_velocity'][i]

        v2_hat = np.float32([np.cos(np.radians(ped_yaw)), np.sin(np.radians(ped_yaw))])
        s2 = np.linalg.norm(ped_vel)

        if s2 < 0.05:
            v2_hat *= s2

        p2 = -3.0 * v2_hat + ped_loc[0:2]
        v2 = 8.0 * v2_hat

        collides, collision_point = get_collision(p1, v1, p2, v2)

        if collides:
            return ped_loc
    return None


def challenge_hazard_vehicle(obs_surrounding_vehicles, ev_speed):
    # np.linalg.norm(_numpy(self._vehicle.get_velocity())
    o1 = np.float32([1, 0])
    p1 = np.float32([0, 0])
    s1 = max(9.5, 2.0 * ev_speed)
    v1_hat = o1
    v1 = s1 * v1_hat

    for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
        if not is_valid:
            continue

        sv_loc = obs_surrounding_vehicles['location'][i]
        sv_yaw = obs_surrounding_vehicles['rotation'][i][2]
        sv_vel = obs_surrounding_vehicles['absolute_velocity'][i]

        o2 = np.float32([np.cos(np.radians(sv_yaw)), np.sin(np.radians(sv_yaw))])
        p2 = sv_loc[0:2]
        s2 = max(5.0, 2.0 * np.linalg.norm(sv_vel[0:2]))
        v2_hat = o2
        v2 = s2 * v2_hat

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue

        return sv_loc

    return None


def behavior_hazard_vehicle(ego_vehicle, actors, route_plan, proximity_th, up_angle_th, lane_offset=0, at_junction=False):
    '''
    ego_vehicle: input_data['ego_vehicle']
    actors: input_data['surrounding_vehicles']
    route_plan: input_data['route_plan']
    '''
    # Get the right offset
    #print(ego_vehicle['lane_id'])
    if ego_vehicle['lane_id'] < 0 and lane_offset != 0:
        lane_offset *= -1

    for i, is_valid in enumerate(actors['binary_mask']):
        if not is_valid:
            continue

        if not at_junction and (actors['road_id'][i] != ego_vehicle['road_id'] or
                                actors['lane_id'][i] != ego_vehicle['lane_id'] + lane_offset):

            #print(route_plan['road_id'])
            next_road_id = route_plan['road_id'][5]
            next_lane_id = route_plan['lane_id'][5]

            if actors['road_id'][i] != next_road_id or actors['lane_id'][i] != next_lane_id + lane_offset:
                continue

        if is_within_distance_ahead(actors['location'][i], proximity_th, up_angle_th):
            return actors['location'][i]
        
    return None


def behavior_hazard_walker(ego_vehicle, actors, route_plan, proximity_th, up_angle_th, lane_offset=0, at_junction=False):
    '''
    ego_vehicle: input_data['ego_vehicle']
    actors: input_data['surrounding_vehicles']
    route_plan: input_data['route_plan']
    '''
    # Get the right offset
    if ego_vehicle['lane_id'] < 0 and lane_offset != 0:
        lane_offset *= -1

    for i, is_valid in enumerate(actors['binary_mask']):
        if not is_valid:
            continue

        if int(actors['on_sidewalk'][i]) == 1:
            continue

        if not at_junction and (actors['road_id'][i] != ego_vehicle['road_id'] or
                                actors['lane_id'][i] != ego_vehicle['lane_id'] + lane_offset):

            next_road_id = route_plan['road_id'][5]
            next_lane_id = route_plan['lane_id'][5]

            if actors['road_id'][i] != next_road_id or actors['lane_id'][i] != next_lane_id + lane_offset:
                continue

        if is_within_distance_ahead(actors['location'][i], proximity_th, up_angle_th):
            return i
    return None

def is_straight_line(waypoint_plan, num_points=5):###
    for i ,command in enumerate(waypoint_plan['command'][:num_points]):
        if command not in [3, 4]:
            return None
    return True

def is_lanechange_or_junction(waypoint_plan, num_points=40):###
    """
    Finds points where lane change or junction is happening in the given waypoint plan.

    Args:
        waypoint_plan (dict): The waypoint plan containing information about the route.
        num_points (int, optional): The number of points to consider in the route plan. Defaults to 40.

    Returns:
        list: A list of locations where lane change or junction is happening.
    """
    # Initialize an empty list to store the points where lane change or junction is happening
    lanechange_or_junction_points = []

    # Get the commands and locations from the waypoint plan
    commands = waypoint_plan['command']
    locations = waypoint_plan['location']
    is_junctions = waypoint_plan['is_junction']

    # Iterate over the commands and locations up to the specified number of points
    for i in range(min(num_points, len(commands))):
        command = commands[i]  # Get the command for the current point

        # Check if the command indicates a lane change
        if command in [5, 6]:
            lanechange_or_junction_points.append(locations[i])  # Add the location to the list

        # Check if the point is a junction
        elif is_junctions[i]:
            lanechange_or_junction_points.append(locations[i])  # Add the location to the list

    # Return the list of points where lane change or junction is happening
    return lanechange_or_junction_points


def is_lanechange(waypoint_plan, num_points=10):###
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
    """
    for i, command in enumerate(waypoint_plan['command'][:num_points]):
        if command in [5, 6]:#if any 56
            return waypoint_plan['lane_id'][i+1]
    return None

def curve_speed(waypoint_plan):###
    """
    Calculate the maximum curvature of a given waypoint plan and return the maximum curvature,
    speed, and the location of the maximum curvature.

    Args:
        waypoint_plan (dict): A dictionary containing the waypoint plan with keys 'location' and 'command'.

    Returns:
        tuple: A tuple containing the maximum curvature, speed, and the location of the maximum curvature.
               The speed is calculated based on the maximum curvature.
    """
    waypoint_locations = waypoint_plan['location']
    max_curvature = 0.0  # Initialize the maximum curvature to 0
    
    # Iterate through the waypoint plan excluding the last 4 points
    for i in range(len(waypoint_locations) - 4):
        p0 = np.array(waypoint_locations[i])  # Get the current waypoint location
        p1 = np.array(waypoint_locations[i+2])  # Get the next waypoint location
        p2 = np.array(waypoint_locations[i+4])  # Get the next next waypoint location
        
        # Continue to the next iteration if the waypoints are the same
        if np.allclose(p0, p1) or np.allclose(p1, p2):
            continue

        v1 = p1 - p0  # Calculate the vector from p0 to p1
        v2 = p2 - p1  # Calculate the vector from p1 to p2

        # Calculate the cosine of the angle between v1 and v2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Clip the cosine value to [-1, 1]

        # Calculate the angle between v1 and v2
        angle = np.arccos(cos_angle)

        # Calculate the curvature based on the angle
        curvature = 2 * np.sin(angle) / np.linalg.norm(v1)

        # Update the maximum curvature and location if the current curvature is greater
        if curvature > max_curvature+0.02:
            max_curvature = curvature
            max_curvature_location = waypoint_locations[i]

    # Calculate the speed based on the maximum curvature
    if max_curvature < 0.06:
        return 0.0, 25,None
    elif max_curvature > 0.14:
        return max_curvature, 3, max_curvature_location
    else:
        return max_curvature, 15 - ((max_curvature - 0.06) / (0.14 - 0.06)) * (15 - 3), max_curvature_location


# def is_curve(waypoint_plan):###
#     """
#     VOID = -1
#     LEFT = 1
#     RIGHT = 2
#     STRAIGHT = 3
#     LANEFOLLOW = 4
#     CHANGELANELEFT = 5
#     CHANGELANERIGHT = 6
#     """
#     for i, command in enumerate(waypoint_plan['command']):
#         if command in [1, 2]:
#             return waypoint_plan['location'][i]

#     return None

def _get_forward_speed(rotations, velocities):###
    """ Convert the vehicle transforms directly to forward speeds """
    pitch = np.deg2rad(rotations[:, 1])
    yaw = np.deg2rad(rotations[:, 2])
    orientation = np.column_stack((np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)))
    speeds = np.dot(velocities, orientation.T)
    return speeds


def wp_hazard_vehicle_future(waypoint_plan, obs_surrounding_vehicles, vehicle_model, max_distance=2.0, num_frames=20):###
    """
    Calculate the future locations of hazard vehicles based on the waypoint plan and surrounding vehicles.

    Args:
        waypoint_plan (dict): The plan of the waypoints.
        obs_surrounding_vehicles (dict): The information of the surrounding vehicles.
        vehicle_model (class): The vehicle model to predict the future location.
        max_distance (float): The maximum distance to the waypoints. Default is 2.0.
        num_frames (int): The number of frames to predict. Default is 20.

    Returns:
        numpy.ndarray or None: The future location of the hazard vehicle if found, otherwise None.
    """
    # Extract waypoint locations
    waypoint_locations = waypoint_plan#['location']

    # Initialize future locations, yaws, speeds, and actions
    future_loc = []
    future_yaw = []
    future_spd = []
    future_act = []

    # Extract valid indices and corresponding information of surrounding vehicles
    valid_indices = np.where(obs_surrounding_vehicles['binary_mask'])[0]
    traffic_locations = obs_surrounding_vehicles['location'][valid_indices]
    traffic_rotations = obs_surrounding_vehicles['rotation'][valid_indices]
    traffic_controls = obs_surrounding_vehicles['control'][valid_indices]
    traffic_velocities = obs_surrounding_vehicles['absolute_velocity'][valid_indices]

    # If there are no valid rotations, return None
    if len(traffic_rotations) == 0:
        return None
    
    # Calculate traffic speeds based on rotations and velocities
    traffic_speeds = _get_forward_speed(traffic_rotations, traffic_velocities)

    # Update future locations, yaws, speeds, and actions
    future_loc = traffic_locations[:, :2]
    future_yaw = traffic_rotations[:, 2] / 180.0 * np.pi
    future_spd = traffic_speeds.reshape(-1, 1)
    future_act = np.array(np.stack([traffic_controls[:, 1], traffic_controls[:, 0], traffic_controls[:, 2]], axis=-1))
    
    # If there are no future locations, return None
    if len(future_loc) == 0:
        return None
    
    # Predict future locations for each frame
    for _ in range(num_frames): 
        for i, fff_loc in enumerate(future_loc):  # Chose 1 car
            # Predict the next location based on the current location, yaw, speed, and action
            next_loc, next_yaw, next_speed = vehicle_model.forward(future_loc[i], future_yaw[i], future_spd[i], future_act[i])
            future_loc[i] = next_loc  # Replace with the position of the second frame
            future_yaw[i] = next_yaw
            future_spd[i] = next_speed

        # Check if any future location is within the maximum distance to the waypoints
        for wp_loc in waypoint_locations:
            distances = np.linalg.norm(future_loc - wp_loc, axis=1)
            for j, distance in enumerate(distances):
                if distance < max_distance:
                    return future_loc[j]
            
    # If no future location is within the maximum distance, return None
    return None

class EgoModel():###
    def __init__(self, dt=1. / 4):
        self.dt = dt

        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb = -0.090769015
        self.rear_wb = 1.4178275

        self.steer_gain = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837


    def forward(self, locs, yaws, spds, acts):
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds



