# required to subscribe to LIDAR topic
from numpy.f2py.crackfortran import c
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# required to publish to Throttle, Steering topics
from std_msgs.msg import Float32

import math
import statistics

class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/autodrive/roboracer_1/lidar',
            self.lidar_callback,
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
    
    def publish_commands(self, throttle, steering):
        
        throttle_msg = Float32()
        throttle_msg.data = float(max(-1.0, min(1.0, throttle)))

        steering_msg = Float32()
        steering_msg.data = float(max(-1.0, min(1.0, steering)))

        self.steering_publisher.publish(steering_msg)
        self.throttle_publisher.publish(throttle_msg)
    
    def distance_at(self, scan, angle, half_window=4):
        index = round(
            (angle - scan.angle_min) / scan.angle_increment
        )

        start = max(0, index - half_window)
        end = min(len(scan.ranges), index + half_window + 1)

        valid = [
            value for value in scan.ranges[start:end]
            if math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
        ]

        if not valid:
            return float('nan')

        return statistics.median(valid)

    def lidar_callback(self, scan):
        # Read wall distances at 4 distinct points
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

        # Angle & Distance of left / right walls relative to car
        lwall_rad = math.atan(
            (a_dist * math.sqrt(2) / 2 - b_dist) 
            / 
            (a_dist * math.sqrt(2) / 2)
        )
        lwall_dist = b_dist * math.cos(lwall_rad)
        rwall_rad = math.atan(
            (c_dist * math.sqrt(2) / 2 - d_dist) 
            / 
            (c_dist * math.sqrt(2) / 2)
        )
        rwall_dist = d_dist * math.cos(rwall_rad)

        # Corner detection (provisional)
        l_rad = round((math.pi / 3 - scan.angle_min) / scan.angle_increment)
        fl_rad = round((math.pi / 12 - scan.angle_min) / scan.angle_increment)
        fr_rad = round((-math.pi / 12 - scan.angle_min) / scan.angle_increment)
        r_rad = round((-math.pi / 3 - scan.angle_min) / scan.angle_increment)

        distances = scan.ranges
        FrontLeft = distances[fl_rad:(l_rad + 1)]
        Front = distances[fr_rad:(fl_rad + 1)]
        FrontRight = distances[r_rad:(fr_rad + 1)]

        fl_med = statistics.median(FrontLeft)
        f_med = statistics.median(Front)
        fr_med = statistics.median(FrontRight)

        corner_threshold = 2.0
        direction_margin = 0.25

        corner = "None"

        if f_med < corner_threshold:
            if fl_med > fr_med + direction_margin:
                corner = "Left"
            elif fr_med > fl_med + direction_margin:
                corner = "Right"
            else:
                corner = "N.A."
        
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
        rideal = 0.5 # minimum distance from car to right wall

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
        throttle_1 = 0.145
        throttle_2 = 0.12
        turn_fraction = min(abs(steering) / 0.5, 1.0)
        throttle = (
            throttle_1 - turn_fraction * (throttle_1 - throttle_2)
        )

        angle_limit = math.radians(3)  # Trial tolerance
        clearance_limit = 4.0         # Trial distance in meters
        straight_candidate = (
            lvalid
            and rvalid
            and math.isfinite(f_med)
            and abs(lwall_rad) < angle_limit
            and abs(rwall_rad) < angle_limit
            and f_med > clearance_limit
            and math.isfinite(steering)
            and abs(steering) < 0.08  # Trial threshold
        )
        if straight_candidate:
            self.straight_count += 1
        else:
            self.straight_count = 0
        is_straight = self.straight_count >= 5
        straight_boost = 0.16
        if is_straight:
            throttle = straight_boost

        if corner != "None":
            throttle = min(throttle, 0.11)

        # Debug
        # mode = (
        #     "both" if lvalid and rvalid
        #     else "left" if lvalid
        #     else "right" if rvalid
        #     else "neither"
        # )

        # self.get_logger().info(
        #     f"mode={mode} corner={corner} | "
        #     f"a={a_dist:.2f} b={b_dist:.2f} "
        #     f"c={c_dist:.2f} d={d_dist:.2f} | "
        #     f"Lpred={lwall_pred:.2f} Rpred={rwall_pred:.2f} | "
        #     f"front={f_med:.2f} near={nearest_wall:.2f} | "
        #     f"kp={kp:.3f} steer_raw={steering:.3f} "
        #     f"throttle={throttle:.3f}",
        #     throttle_duration_sec=0.1,
        # )

        # Publishes to Throttle, Steering Topics
        self.publish_commands(throttle, steering)

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
