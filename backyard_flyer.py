import argparse
import time
from enum import Enum

import numpy as np
from numpy import linalg as LA

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class BackyardFlyer(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = self.calculate_box()
        self.in_mission = True
        self.check_state = {}
        self.error = 0.5

        # initial state
        self.flight_state = States.MANUAL

        # Register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        """
        This triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data
        """
        # print("local_position_callback: local_position: ", self.local_position, ", global_position: ", self.global_position, ", global_home: ", self.global_home)
        if self.is_takeoff_state() and self.at_target_position() and self.at_slow_velocity():
            self.waypoint_transition()
        elif self.is_waypoint_state() and self.at_target_position() and self.at_slow_velocity():
            if len(self.all_waypoints) > 0:
                self.waypoint_transition()
            else:
                self.landing_transition()
        elif self.is_landing_state() and self.at_target_position():
            self.disarming_transition()

    def velocity_callback(self):
        """
        This triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data
        """
        # print("velocity callback")

    def state_callback(self):
        """
        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        # print("state callback: flight_state: ", self.flight_state ,"armed: ", self.armed, ", guided: ", self.guided)
        if self.is_manual_state(): 
            self.arming_transition()
        elif self.is_arming_state():
            self.takeoff_transition()
        elif self.is_disarming_state():
            self.manual_transition()

    def calculate_box(self):
        """
        1. Return waypoints to fly a box
        """
        return [[8.0, 0.0, 3.0], [8.0, 8.0, 3.0], [0.0, 8.0, 3.0], [0.0, 0.0, 3.0]]

    def arming_transition(self):
        """
        1. Take control of the drone
        2. Pass an arming command
        3. Set the home location to current position
        4. Transition to the ARMING state
        """
        print("---arming transition")
        self.take_control()
        self.arm()
        self.set_home_position(self.global_position[0], self.global_position[1], self.global_position[2])
        self.flight_state = States.ARMING

    def takeoff_transition(self):
        """
        1. Set target_position altitude to 3.0m
        2. Command a takeoff to 3.0m
        3. Transition to the TAKEOFF state
        """
        print("---takeoff transition")
        self.target_position[2] = 3
        self.takeoff(self.target_position[2])
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        """
        1. Command the next waypoint position
        2. Transition to WAYPOINT state
        """
        print("---waypoint transition")
        if len(self.all_waypoints) == 0:
            return
        self.target_position = np.array(self.all_waypoints.pop(0))
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], 0)
        self.flight_state = States.WAYPOINT

    def landing_transition(self):
        """
        1. Command the drone to land
        2. Transition to the LANDING state
        """
        print("---landing transition")
        self.target_position = np.zeros(3)
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        """
        1. Command the drone to disarm
        2. Transition to the DISARMING state
        """
        print("---disarm transition")
        self.disarm()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        """This method is provided
        
        1. Release control of the drone
        2. Stop the connection (and telemetry log)
        3. End the mission
        4. Transition to the MANUAL state
        """
        print("---manual transition")

        self.release_control()
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        """This method is provided
        
        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()

    def is_armed_and_guided(self):
        return self.armed and self.guided
    
    def not_armed_and_not_guided(self):
        return not self.armed and not self.guided
    
    def is_manual_state(self):
        return self.flight_state == States.MANUAL and self.not_armed_and_not_guided()
    
    def is_arming_state(self):
        return self.flight_state == States.ARMING and self.is_armed_and_guided()
    
    def is_takeoff_state(self):
        return self.flight_state == States.TAKEOFF and self.is_armed_and_guided()
    
    def is_waypoint_state(self):
        return self.flight_state == States.WAYPOINT and self.is_armed_and_guided()
    
    def is_landing_state(self):
        return self.flight_state == States.LANDING and self.is_armed_and_guided()
    
    def is_disarming_state(self):
        return self.flight_state == States.DISARMING and not self.armed
    
    def at_target_position(self):
        local_position = np.array(self.local_position)
        local_position[2] = -local_position[2]
        # print("at_target_position: ", LA.norm(self.target_position - local_position))
        return LA.norm(self.target_position - local_position) < self.error
    
    def at_slow_velocity(self):
        # print("at_slow_velocity: ", LA.norm(self.local_velocity))
        return LA.norm(self.local_velocity) < self.error
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn)
    time.sleep(2)
    drone.start()
