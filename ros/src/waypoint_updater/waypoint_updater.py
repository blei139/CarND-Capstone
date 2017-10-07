#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped

from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import yaml
from bisect import bisect

import time
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.currvelocity_cb)


        self.line_pos_wp = []
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        green_light_time = rospy.get_param('~green_light_time', 2.)
        stop_distance = rospy.get_param('~stop_distance', 35.)
        stop_buffer = rospy.get_param('~stop_buffer', 0.0)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.waypoints = None
        self.my_current_velocity = 0
        self.setspeed = (10.0)*(1.60934*1000/3600); # Mph to m/s
        self.trafficlight_status  = -1
        self.green_light_time = green_light_time
        self.stop_distance = stop_distance
        self.stop_buffer = stop_buffer
        print 'green light time',self.green_light_time, 'stop distance',self.stop_distance,'stop buffer',self.stop_buffer
        # self.stop_distance = 37 #6*self.setspeed*self.setspeed # roughly - TBD
        self.nextstop = -1
        self.ignorestop = False
        self.initialstate = True
        self.startwaypoint = -1
        self.state = 'move'
        self.detection_ready = False

        # start loop
        self.loop()

        # rospy.spin()

    def currvelocity_cb(self,velocity):
        self.my_current_velocity = velocity.twist.linear.x

    def pose_cb(self, msg):
        self.pose = msg

    def loop(self):
        dt = 0.1
        rate = rospy.Rate(1/dt) # 10Hz or 50Hz
        green_start_time = -1
        while not rospy.is_shutdown():

            if (self.pose is not None) and (self.waypoints is not None): # in case update comes before /base_waypoints

                # Get waypoints of all traffic light lines if not already done
                if self.line_pos_wp == []:
                    stop_line_positions = self.config['stop_line_positions']
                    for line_pos in stop_line_positions:
                        # Search the waypoints for the closest distance to this line
                        closest_dist = 10**6
                        closest_ind = 0
                        for i,waypoint in enumerate(self.waypoints.waypoints):
                            way = waypoint.pose.pose.position
                            dist = ((line_pos[0] - way.x)**2 + (line_pos[1] - way.y)**2)**0.50
                            if dist < closest_dist:
                                closest_ind = i
                                closest_dist = dist

                        self.line_pos_wp.append(closest_ind)

                num_wp = len(self.waypoints.waypoints)
                closestWPi = self.get_closest_waypoint()

                if self.startwaypoint <0:
                    self.startwaypoint = closestWPi

                # clear initial state after some distance - starting sim with speed 0
                if self.distance(self.waypoints.waypoints,self.startwaypoint,closestWPi)>0.05:
                    self.initialstate = False

                # next traffic light stop line
                if (self.line_pos_wp is not None):

                    pos_in_list = bisect(self.line_pos_wp,closestWPi)

                    # loop back to beginning
                    if pos_in_list == len(self.line_pos_wp):
                        pos_in_list = 0

                    # pass light, reset flag
                    if self.line_pos_wp[pos_in_list] != self.nextstop:
                        self.ignorestop = False

                    # set next stop waypoint
                    self.nextstop = self.line_pos_wp[pos_in_list]


                startindex = min(closestWPi,num_wp)
                endindex = min(closestWPi+LOOKAHEAD_WPS,num_wp)

                # ----------- Car State Machine -----------------------
                # print 'status:', self.state, self.nextstop, self.trafficlight_status, self.ignorestop, self.initialstate, self.my_current_velocity

                # STOP
                if self.state == 'stop':
                    if self.trafficlight_status < 0 and self.detection_ready == True: # insert traffic light check here
                        self.state = 'move'
                        print '=========  M O V E  =========', self.nextstop
                # MOVE
                elif self.state == 'move':
                    if (self.initialstate is False) and (self.ignorestop is False) and (self.distance(self.waypoints.waypoints,startindex,self.nextstop)<self.stop_distance):
                        self.state = 'brake'
                        print '-=-=-=-=- B R A K E -=-=-=-=-', self.nextstop
                # BRAKE
                elif self.state == 'brake':
                    if self.my_current_velocity<0.0001*self.setspeed:
                        self.ignorestop = True
                        self.state = 'stop'
                        print '---------  S T O P  ---------', self.nextstop
                    elif self.trafficlight_status<0: # light turns green
                        if green_start_time < 0:
                            green_start_time = time.time()
                            green_elapsed_time = 0
                        else:
                            green_elapsed_time = time.time() - green_start_time
                        # enough time to cross the line before ligth turns red?
                        distance2line = self.distance(self.waypoints.waypoints,closestWPi,self.nextstop)+self.stop_buffer
                        speedprojected = self.setspeed #min(self.my_current_velocity+1.0,self.setspeed)
                        time_before_red = self.green_light_time - green_elapsed_time
                        # print distance2line, distance2line/speedprojected, time_before_red
                        if (distance2line/speedprojected)<time_before_red:
                            self.ignorestop = True
                            self.state = 'move'
                            print '>>>>>>>>>   Vroom!  <<<<<<<<<'
                    else:
                        green_start_time = -1
                # -----------------------------------------------------

                # ---------- Construct waypoints ----------------------
                wp2pub = []
                # =========== STOP ==============
                if self.state == 'stop':
                    for wpi in range(startindex,endindex):
                        self.waypoints.waypoints[wpi].twist.twist.linear.x = 0.0
                        wp2pub.append(self.waypoints.waypoints[wpi])

                # =========== MOVE ==============
                elif self.state == 'move':
                    for wpi in range(startindex,endindex):
                        if self.initialstate is False:
                            self.waypoints.waypoints[wpi].twist.twist.linear.x = self.setspeed
                        else:
                            self.waypoints.waypoints[wpi].twist.twist.linear.x = 1.5 #0.1*self.setspeed
                        wp2pub.append(self.waypoints.waypoints[wpi])

                # =========== BRAKE ==============
                elif self.state == 'brake':
                    brakedistfinal = 2.0
                    brakedist1 = self.stop_distance-brakedistfinal
                    for wpi in range(startindex,endindex):
                        distance2line = self.distance(self.waypoints.waypoints,wpi,self.nextstop)

                        # speed_for_WP = (distance2line/self.stop_distance)*self.my_current_velocity

                        if distance2line > self.stop_distance: # overshot line, loaded next waypoint
                            speed_for_WP = self.setspeed
                            self.state = 'move'
                            print '========= O o p s 1 =========', self.nextstop

                        elif distance2line > brakedist1: # stage 1
                            if self.my_current_velocity<0.8*self.setspeed: # initial start
                                speed_for_WP = 0.0 #(distance2line/brakedist1)*(3.0)
                            else:
                                speed_for_WP = (distance2line/brakedist1)*(self.setspeed-1.5) # set to 1.5 m/s in last brakedist
                        else: # stage 0
                            speed_for_WP = 0.0 #(distance2line/brakedistfinal)*self.my_current_velocity # 0.0

                        # if speed_for_WP > self.setspeed: # in case car failed to stop before light, and treat next light as the target
                        #     speed_for_WP = self.setspeed
                        #     self.state = 'move'
                        #     print '========= O o p s 2 ========='

                        self.waypoints.waypoints[wpi].twist.twist.linear.x = speed_for_WP
                        wp2pub.append(self.waypoints.waypoints[wpi])
                # -----------------------------------------------------

                # Header
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)
                lane.waypoints = wp2pub

                # print 'WP_updater:',closestWPi,self.pose.pose.position.x, self.pose.pose.position.y
                self.final_waypoints_pub.publish(lane)

            rate.sleep()

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.detection_ready = True
        self.trafficlight_status = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_closest_waypoint(self):
        """Identifies the closest path waypoint to the current pose position
        Returns: int: index of the closest waypoint in self.waypoints
        """
        best_dist = 9999.99
        best_i = -1
        if (self.pose is not None):
            dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
            for i,wp in enumerate(self.waypoints.waypoints): # range(len(self.waypoints.waypoints)):
                # this_dist = dl(self.pose.pose.position,self.waypoints.waypoints[i].pose.pose.position)
                this_dist = dl(self.pose.pose.position,wp.pose.pose.position)
                if (this_dist<best_dist):
                    best_dist = this_dist
                    best_i = i
        else:
            rospy.logerr('Could not find current pose.')

        return best_i

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
