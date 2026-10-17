# required to subscribe to LiDAR topic.
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# required to publish to Throttle, Steering topics.
from std_msgs.msg import Float32

import math
import statistics

cos_45 = math.sqrt(2) / 2

class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        # subscribes to LiDAR topic.
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/autodrive/roboracer_1/lidar',
            self.lidar_callback,
            qos_profile_sensor_data,
        )

        # creates publishers for Throttle, Steering.
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

        self.boost_active = False
        self.steering_failure_since = None
        self.straight_count = 0 # counter increments when straight path identified in single frame.

        # used to calculate Derivative (K_d) in a PID.
        self.prev_error = None
        self.prev_time = None
        self.kd = 0.025
        self.prev_mode = None

        self.lideal = 0.9 # optimal distance from car to left wall
        self.rideal = 0.35 # optimal distance from car to right wall
    
    def end_boost(self):
        """Restore normal wall targets when leaving boost."""
        if self.boost_active:
            self.prev_error = None
            self.prev_time = None
        self.boost_active = False

    def publish_commands(self, throttle, steering): # publishes to Throttle, Steering topics.
        if not math.isfinite(throttle) or not math.isfinite(steering): # failsafe in case of invalid value/s.
            throttle = 0.0
            steering = 0.0
        
        throttle_msg = Float32()
        throttle_msg.data = float(max(-1.0, min(1.0, throttle)))
        self.throttle_publisher.publish(throttle_msg)

        steering_msg = Float32()
        steering_msg.data = float(max(-1.0, min(1.0, steering)))
        self.steering_publisher.publish(steering_msg)
    
    def distance_at(self, scan, angle, half_window=4): # estimates distance using several nearby LiDAR beams.
        index = round(
            (angle - scan.angle_min) / scan.angle_increment
        )

        start = max(0, index - half_window)
        end = min(len(scan.ranges), index + half_window + 1)

        return self.valid_median(scan, scan.ranges[start:end]) # ~±1° around target direction.
    
    def valid_median(self, scan, readings): # removes invalid readings to calculate median.
        valid = [
            value for value in readings
            if math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
        ]

        return statistics.median(valid) if valid else float('nan')

    def lidar_callback(self, scan): # Node
        # wall distance at 90°, 45°, -45°, -90°.
        p90_dist = self.distance_at(scan, math.pi / 2)
        p45_dist = self.distance_at(scan, math.pi / 4)
        n45_dist = self.distance_at(scan, -math.pi / 4)
        n90_dist = self.distance_at(scan, -math.pi / 2)

        # checks validity of above distances.
        lvalid = all(
            math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
            for value in (p45_dist, p90_dist)
        )
        rvalid = all(
            math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
            for value in (n45_dist, n90_dist)
        )

        # defaults for unknown walls.
        lwall_rad = lwall_dist = float('nan')
        rwall_rad = rwall_dist = float('nan')

        # angle, perpendicular distance to left / right walls.
        if lvalid:
            p45_dist2 = p45_dist * cos_45
            lwall_rad = math.atan2(
                p45_dist2 - p90_dist,
                p45_dist2,
            )
            lwall_dist = p90_dist * math.cos(lwall_rad)
        if rvalid:
            n45_dist2 = n45_dist * cos_45
            rwall_rad = math.atan2(
                n45_dist2 - n90_dist,
                n45_dist2,
            )
            rwall_dist = n90_dist * math.cos(rwall_rad)



        # ↓ increases steering kp as nearest side-wall reading decreases.

        # keeps valid p90_dist, n90_dist
        lr_dista = [
            value for value in (p90_dist, n90_dist)
            if math.isfinite(value)
            and scan.range_min <= value <= scan.range_max
        ]
        if not lr_dista:
            self.straight_count = 0
            self.end_boost()
            self.steering_failure_since = None
            self.publish_commands(0.0, 0.0)
            return

        lr_min = min(lr_dista)
        lb_dist = 0.4 # maximum kp at or below this distance
        ub_dist = 1.0 # minimum kp at or above this distance
        kp_min = 0.5
        kp_max = 0.7
        prox = max(0.0, min(
            1.0,
            (ub_dist - lr_min) / (ub_dist - lb_dist),
        )) # scale: 0-1
        kp = kp_min + prox * (kp_max - kp_min)



        # Tweak these values to improve best lap time.
        L = 0.55 # lookahead distance

        # Future Distance to left / right walls given L
        lwall_pred = lwall_dist + L * math.sin(lwall_rad)
        rwall_pred = rwall_dist + L * math.sin(rwall_rad)

        # Calculates steering given prior calculations
        # Calculate error
        if lvalid and rvalid:
            error = (lwall_pred - self.lideal) - (rwall_pred - self.rideal)

        elif lvalid:
            error = lwall_pred - self.lideal

        elif rvalid:
            error = self.rideal - rwall_pred

        else:
            self.straight_count = 0
            self.end_boost()
            self.steering_failure_since = None
            self.prev_error = None
            self.prev_time = None
            self.publish_commands(0.0, 0.0)
            return


        # Calculate elapsed time
        # After calculating error; the no-wall branch already returned.
        mode = (lvalid, rvalid)
        current_time = self.get_clock().now()
        derivative = 0.0

        if (
            mode == self.prev_mode
            and self.prev_time is not None
            and self.prev_error is not None
        ):
            dt = (current_time - self.prev_time).nanoseconds / 1e9
            if dt > 0:
                derivative = (error - self.prev_error) / dt

        steering = kp * error + self.kd * derivative

        self.prev_error = error
        self.prev_time = current_time
        self.prev_mode = mode

        # Calculates throttle given steering strength
        throttle_max = 0.17
        throttle_min = 0.12
        turn_fraction = min(abs(steering) / 0.5, 1.0)
        throttle = (
            throttle_max - turn_fraction * (throttle_max - throttle_min)
        )

        # front clearance (used by throttle control and straight detection)
        p15_ind = round((math.pi / 12 - scan.angle_min) / scan.angle_increment)
        n15_ind = round((-math.pi / 12 - scan.angle_min) / scan.angle_increment)
        f_dist = self.valid_median(scan, scan.ranges[n15_ind:(p15_ind + 1)])
        if not math.isfinite(f_dist):
            self.straight_count = 0
            self.end_boost()
            self.steering_failure_since = None
            self.publish_commands(0.0, 0.0)
            return

        angle_limit = math.radians(20) # Trial tolerance
        clearance_limit = 5.0 # Trial distance in meters
        # Geometry and invalid-data failures disable boost immediately.
        geometry_ok = (
            lvalid and rvalid
            and math.isfinite(f_dist)
            and abs(lwall_rad) < angle_limit
            and abs(rwall_rad) < angle_limit
            and f_dist > clearance_limit
            and math.isfinite(steering)
        )
        steering_limit = 0.15 if self.boost_active else 0.10
        straight_candidate = geometry_ok and abs(steering) < steering_limit
        steering_exit_delay = 0.05 # seconds of continuous steering failure

        if not geometry_ok:
            self.end_boost()
            self.straight_count = 0
            self.steering_failure_since = None
        elif self.boost_active:
            if straight_candidate:
                self.steering_failure_since = None
                self.straight_count += 1
            else:
                self.straight_count = 0
                if self.steering_failure_since is None:
                    self.steering_failure_since = current_time
                elapsed = (current_time - self.steering_failure_since).nanoseconds / 1e9
                if elapsed < 0 or elapsed >= steering_exit_delay:
                    self.end_boost()
                    self.steering_failure_since = None
        else:
            self.steering_failure_since = None
            self.straight_count = self.straight_count + 1 if straight_candidate else 0
            self.boost_active = self.straight_count >= 3

        is_straight = self.boost_active
        straight_boost = 1.0

        slow_start = 3.0
        slow_full = 1.0
        far_throttle = 0.17
        near_throttle = 0.15

        # Apply the clearance cap last, only below 3 meters.
        if f_dist < slow_start:
            proximity = max(
                0.0,
                min(
                    1.0,
                    (slow_start - f_dist) / (slow_start - slow_full),
                ),
            )

            front_throttle_cap = (
                far_throttle
                - proximity * (far_throttle - near_throttle)
            )

            throttle = min(throttle, front_throttle_cap)

        if is_straight:
            throttle = straight_boost

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
