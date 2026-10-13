# required to subscribe to LIDAR topic
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# required to publish to Throttle, Steering topics
from std_msgs.msg import Float32

import math
import statistics

class WallFollowNode(Node):
    def __init__(self): # Handles Topics
        super().__init__('wall_follow_node')

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/autodrive/roboracer_1/lidar',
            self.lidar_callback,
            # self.test_callback,
            qos_profile_sensor_data,
        )

        self.throttle_publisher = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/throttle_command',
            10,
        )

        self.steering_publisher = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/steering_command',
            10,
        )

        self.straight_count = 0
    
    def publish_commands(self, throttle, steering): # Publishes to Steering, Throttle Topics
        if not math.isfinite(throttle) or not math.isfinite(steering):
            throttle = 0.0
            steering = 0.0
        
        throttle_msg = Float32()
        throttle_msg.data = float(max(-1.0, min(1.0, throttle)))

        steering_msg = Float32()
        steering_msg.data = float(max(-1.0, min(1.0, steering)))

        self.steering_publisher.publish(steering_msg)
        self.throttle_publisher.publish(throttle_msg)
    
    def distance_at(self, scan, angle, half_window=4): # Estimates distance using several nearby LIDAR beams
        index = round(
            (angle - scan.angle_min) / scan.angle_increment
        )

        start = max(0, index - half_window)
        end = min(len(scan.ranges), index + half_window + 1)

        return self.valid_median(scan, scan.ranges[start:end])
    
    def valid_median(self, scan, readings): # Removes invalid readings to calculate median
        valid = [
            value for value in readings
            if math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
        ]

        return statistics.median(valid) if valid else float('nan')

    def lidar_callback(self, scan): # Main
        # Wall distance at 4 distinct directions
        a_dist = self.distance_at(scan, math.pi / 4)
        b_dist = self.distance_at(scan, math.pi / 2)
        c_dist = self.distance_at(scan, -math.pi / 4)
        d_dist = self.distance_at(scan, -math.pi / 2)

        # Checks validity of readings
        lvalid = all(
            math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
            for value in (a_dist, b_dist)
        )
        rvalid = all(
            math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
            for value in (c_dist, d_dist)
        )

        # Defaults for N/A walls
        lwall_rad = lwall_dist = float('nan')
        rwall_rad = rwall_dist = float('nan')

        # Angle & Distance of left / right walls relative to car
        if lvalid:
            a_proj = a_dist * math.sqrt(2) / 2
            lwall_rad = math.atan2(
                a_proj - b_dist,
                a_proj,
            )
            lwall_dist = b_dist * math.cos(lwall_rad)
        if rvalid:
            c_proj = c_dist * math.sqrt(2) / 2
            rwall_rad = math.atan2(
                c_proj - d_dist,
                c_proj,
            )
            rwall_dist = d_dist * math.cos(rwall_rad)

        # Front clearance (used by throttle control and straight detection)
        fl_rad = round((math.pi / 12 - scan.angle_min) / scan.angle_increment)
        fr_rad = round((-math.pi / 12 - scan.angle_min) / scan.angle_increment)

        distances = scan.ranges
        f_med = self.valid_median(scan, distances[fr_rad:(fl_rad + 1)])

        if not math.isfinite(f_med):
            self.straight_count = 0
            self.publish_commands(0.0, 0.0)
            return

        # Increase steering gain as the nearest side-wall reading decreases.

        # Use sideways beams even when a diagonal beam points into open space.
        side_distances = [
            value for value in (b_dist, d_dist)
            if math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
        ]
        if not side_distances:
            self.straight_count = 0
            self.publish_commands(0.0, 0.0)
            return

        nearest_wall = min(side_distances)
        near_distance = 0.4 # meters: maximum gain at or below this distance
        far_distance = 1.0 # meters: minimum gain at or above this distance
        kp_far = 0.60
        kp_near = 0.75
        proximity = max(0.0, min(
            1.0,
            (far_distance - nearest_wall) / (far_distance - near_distance),
        ))
        kp = kp_far + proximity * (kp_near - kp_far)

        # Tweak these values to improve best lap time.
        L = 0.55 # lookahead distance
        lideal = 0.9 # minimum distance from car to left wall
        rideal = 0.4 # minimum distance from car to right wall

        # Future Distance to left / right walls given L
        lwall_pred = lwall_dist + L * math.sin(lwall_rad)
        rwall_pred = rwall_dist + L * math.sin(rwall_rad)

        # Calculates steering given prior calculations
        if lvalid and rvalid:
            steering = kp * ((lwall_pred - lideal) - (rwall_pred - rideal))
        elif lvalid:
            steering = kp * (lwall_pred - lideal)
        elif rvalid:
            steering = kp * (rideal - rwall_pred)
        else:
            self.straight_count = 0
            self.publish_commands(0.0, 0.0) # error occured
            return

        # Calculates throttle given steering strength
        throttle_max = 0.145
        throttle_min = 0.12
        turn_fraction = min(abs(steering) / 0.5, 1.0)
        throttle = (
            throttle_max - turn_fraction * (throttle_max - throttle_min)
        )

        angle_limit = math.radians(10) # Trial tolerance
        clearance_limit = 5 # Trial distance in meters
        straight_candidate = (
            lvalid
            and rvalid
            and math.isfinite(f_med)
            and abs(lwall_rad) < angle_limit
            and abs(rwall_rad) < angle_limit
            and f_med > clearance_limit
            and math.isfinite(steering)
            and abs(steering) < 0.10  # Trial threshold
        )
        if straight_candidate:
            self.straight_count += 1
        else:
            self.straight_count = 0
        is_straight = self.straight_count >= 5
        straight_boost = 1.0
        if is_straight:
            throttle = straight_boost

        slow_start = 5.0
        slow_full = 1.0
        far_throttle = 0.20
        near_throttle = 0.10

        proximity = max(
            0.0,
            min(1.0, (slow_start - f_med) / (slow_start - slow_full)),
        )

        front_throttle_cap = (
            far_throttle
            - proximity * (far_throttle - near_throttle)
        )

        throttle = min(throttle, front_throttle_cap)

        # Publishes to Throttle, Steering Topics
        self.publish_commands(throttle, steering)

    # def test_callback(self, scan):
    #     if not scan.ranges or not math.isfinite(scan.angle_increment) or scan.angle_increment <= 0:
    #         self.publish_commands(0.0, 0.0)
    #         return
    #     low = round(
    #         (-math.pi / 3 - scan.angle_min) / scan.angle_increment
    #     )
    #     high = round(
    #         (math.pi / 3 - scan.angle_min) / scan.angle_increment
    #     )
    #     low = max(0, low)
    #     high = min(len(scan.ranges) - 1, high)
    #     beams = scan.ranges[low:(high + 1)]

    #     valid_beams = [
    #         scan.range_max if val == math.inf
    #         else val if (
    #             math.isfinite(val)
    #             and scan.range_min <= val <= scan.range_max
    #         )
    #         else 0.0
    #         for val in beams
    #     ]

    #     obstacles = [
    #         (i, val)
    #         for i, val in enumerate(beams)
    #         if math.isfinite(val)
    #         and scan.range_min <= val <= scan.range_max
    #     ]

    #     if obstacles:
    #         n_index, n_dist = min(
    #             obstacles, key = lambda item: item[1]
    #         )
        
    #         bubble_radius = 0.25

    #         if n_dist <= bubble_radius:
    #             self.publish_commands(0.0,0.0)
    #             return

    #         bubble_angle = math.asin(bubble_radius / n_dist)
    #         bubble_beams = math.ceil(
    #             bubble_angle / abs(scan.angle_increment)
    #         )

    #         start = max(0, n_index - bubble_beams)
    #         end = min(len(valid_beams), n_index + bubble_beams + 1)

    #         for i in range(start, end):
    #             valid_beams[i] = 0.0

    #     # Find consecutive beams with enough clearance after applying the bubble.
    #     gap_clearance = 1.0 # meters; experimental opening threshold
    #     gaps = []
    #     gap_start = None
    #     for i, distance in enumerate(valid_beams):
    #         if distance >= gap_clearance:
    #             if gap_start is None:
    #                 gap_start = i
    #         elif gap_start is not None:
    #             gaps.append((gap_start, i - 1))
    #             gap_start = None
    #     if gap_start is not None:
    #         gaps.append((gap_start, len(valid_beams) - 1))

    #     if not gaps:
    #         self.publish_commands(0.0, 0.0)
    #         return

    #     # Prefer the widest angular gap; break ties toward straight ahead.
    #     def gap_score(gap):
    #         start, end = gap
    #         midpoint = (start + end) / 2
    #         angle = scan.angle_min + (low + midpoint) * scan.angle_increment
    #         return (end - start + 1, -abs(angle))

    #     gap_start, gap_end = max(gaps, key=gap_score)
    #     target_index = (gap_start + gap_end) / 2
    #     target_angle = scan.angle_min + (low + target_index) * scan.angle_increment

    #     # Target bearing is converted to a normalized steering request.
    #     steering_gain = 1.0 # normalized command per radian; tune at low speed
    #     steering = max(-1.0, min(1.0, steering_gain * target_angle))
    #     self.publish_commands(0.1, steering)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
