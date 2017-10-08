
GAS_DENSITY = 2.858
ONE_MPH = 0.44704
PI = 3.14159265359

from yaw_controller import YawController
from pid import PID

class Controller(object):
    def __init__(self, *args, **kwargs):

        self.__dict__.update(kwargs)
        # print 'Controller: max_steer',self.max_steer_angle,'max_throttle',self.max_throttle,'max_brake',self.max_brake

        kp = -5.0
        ki = -0.05
        kd = -5.0
        self.spidcontroller = PID(kp, ki, kd, -self.max_brake, self.max_throttle)

        kp = -1.0 #-0.2
        ki = -0.05
        kd = -20.0
        self.pidcontroller = PID(kp, ki, kd, -self.max_steer_angle, self.max_steer_angle)

        min_speed = 0.0
        self.yawcontroller = YawController(self.wheel_base, self.steer_ratio, min_speed, self.max_lat_accel, self.max_steer_angle)

        self.pedal = self.max_throttle

    def control(self, *args): #, **kwargs):

        # input arguments
        cte = args[0]
        delta_time = args[1]
        linear_velocity = args[2]
        angular_velocity = args[3]
        current_velocity = args[4]

        # Throttle - PID control
        pedal = self.spidcontroller.step(current_velocity-linear_velocity, delta_time)

        if (pedal>0):
            throttle = pedal
            brake = 0.0
        else:
            throttle = 0.0
            brake = -pedal
            # Keep car stopped since car will naturally creep forward.
            if current_velocity < 0.2:
                brake = 0.1

        # Steer
        if (current_velocity==0.0):
            steer = 0.0

        else:
            # Steer - PID control
            steer_pid = self.pidcontroller.step(cte, delta_time)

            # Steer - YAW control
            steer_yaw = self.yawcontroller.get_steering(linear_velocity, angular_velocity, current_velocity)

            steer = steer_pid + steer_yaw

        return throttle, brake, steer
