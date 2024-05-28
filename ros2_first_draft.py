#code for the line following control
import rclpy
import scipy.signal
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Float32MultiArray, String
import numpy as np
import time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Twist

class followerControl(Node):
    def __init__(self):
        super().__init__('PID_CONTROLLER_NODE_PIERREBORRACHO')

        qos_profile = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.pubCompVal = self.create_publisher(Float32, '/compensationVal', 10)
        self.offsetSub = self.create_subscription(Float32, '/offset', self.offset_callback, 10)
        self.encL = self.create_subscription(Float32, '/VelocityEncL', self.encoderL_callback, qos_profile)
        self.encR = self.create_subscription(Float32, '/VelocityEncR', self.encoderR_callback, qos_profile)

        self.get_logger().info('Node initializer')

        self.get_logger().info('line followedsdads controller node initialized')
        self.timer_period = 0.01
        self.radius = 0.05
        self.wheelbase = 0.175
        print("aqui no MAMA")
        self.offset = Float32()
        #print("EN ESTA PARTE ES DONDE PETA")
        self.offset = 0.0
        self.velL = 0.0
        self.velR = 0.0
        self.derivative = 0.0
        self.integral = 0.0
        self.proportional = 0.0
        self.lastError = 0.0
        self.compensation = 0.0
        self.kp = 13.0
        self.ki = 6.5
        self.kd = 3.5
        #print("ODIO FUZZY LOGIC JAJA XD")

    def offset_callback(self, offset_mm):
        self.offset = offset_mm.data
        twist_msg = Twist()
        self.integral = self.offset * self.timer_period
        self.derivative = (self.offset - self.lastError)/self.timer_period
        self.compensation = self.kp*self.offset + self.ki*self.integral + self.kd*self.derivative
        self.lastError = self.offset
        twist_msg.linear.x = 0.05
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = np.clip(self.compensation, -0.05, 0.05)
        #self.compensation*0.01
        print("IIIIIIII apajetson")
        self.pubCompVal.publish(Float32(data=self.compensation))
        #print("OOOOOOO VALIO ORTO OOOOOO")
        self.publisher.publish(twist_msg)
        print(self.offset)

    def encoderL_callback(self, msg_cmdL):
        self.velL = msg_cmdL.data

    def encoderR_callback(self, msg_cmdR):
        self.velR = msg_cmdR.data


def main(args=None):
    rclpy.init(args=args)
    follow = followerControl()
    rclpy.spin(follow)
    follow.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()