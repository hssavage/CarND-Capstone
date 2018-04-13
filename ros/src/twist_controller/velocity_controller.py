'''
###############################################################################
# velocity_controller.py                                                      #
# --------------------------------------------------------------------------- #
#                                                                             #
# Description:                                                                #
# ------------                                                                #
# This module contains the source for the velocity controller which outputs   #
# throttle and brakes values to help the car maintain a given desired speed   #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 2/27/2018          | Henry Savage  | Initial pass on the code           | #
# +--------------------+---------------+------------------------------------+ #
# | 3/13/2018          | Henry Savage  | Updated max *celeration values to  | #
# |                    |               | always use the properly signed     | #
# |                    |               | values -- fixed a bug where the    | #
# |                    |               | vehicle wouldn't brake             | #
# +--------------------+---------------+------------------------------------+ #
# | 3/19/2018          | Henry Savage  | Removed duplicate multiplication   | #
# |                    |               | of density when calculating mass   | #
# +--------------------+---------------+------------------------------------+ #
# | 3/24/2018          | Xiao He       | Edited the throttle input to reach | #
# |                    |               | the speed limit                    | #
# +--------------------+---------------+------------------------------------+ #
# | 3/29/2018          | Xiao He       | Updated the calculation for        | #
# |                    |               | throttle                           | #
# +-------------------------------------------------------------------------| #
# | 3/30/2018          | Xiao He       | Updated the comments               | #
# +-------------------------------------------------------------------------| #
# | 4/12/2018          | Henry Savage  | Started a table based modifier     | #
# |                    |               | method of helping edge the speed   | #
# |                    |               | to the actual limit. This helps    | #
# |                    |               | throttle get close and hold since  | #
# |                    |               | its not over estimating at all. We | #
# |                    |               | should really determine the proper | #
# |                    |               | math to solve this in the future   | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# Gas density constant to help with weight
GAS_DENSITY = 2.858 # Almost definitely Kg/Gallon

class VelocityController(object):
    def __init__(self, vehicle_mass=1e-6, max_accel=0.0, max_decel=0.0,
                 max_input_accel=0.0, max_input_decel=0.0, deadband=0.0,
                 wheel_radius=0.0, fuel_capacity=0.0):
        '''
        Initializes the controller with thresholds and vehicle constants
        '''

        # Calculate the weight of the vehicle with gas
        self.vehicle_mass = vehicle_mass
        self.fuel_weight = fuel_capacity * GAS_DENSITY
        self.fuel_percentage = 1.0
        self.total_vehicle_mass = self.vehicle_mass + (self.fuel_weight * self.fuel_percentage)

        # Vehicle constants for Torque calculations
        self.wheel_radius = wheel_radius
        self.deadband = deadband

        # Acceleration limit
        self.max_accel = max_accel # Actual acceleration value threshold (m/s^2)
        self.max_decel = max_decel # Actual deceleration value threshold (m/s^2)

        # Force signs
        if(self.max_accel < 0):
            self.max_accel *= -1
        if(self.max_decel > 0):
            self.max_accel *= -1

        # Torque limits
        max_accel_torque = self.wheel_radius * self.max_accel * self.total_vehicle_mass
        self.max_accel_input = max_input_accel # Max throttle input to the DBW system (% engaged)
        self.max_accel_torque = max_accel_torque # Max acceleration Torque (Nm)
        self.max_decel_torque = max_input_decel # Max deceleration Torque (Nm)

    def set_fuel_percentage(self, fuel_percentage):
        '''
        Set the vehicle's current fuel percentage. Updates the vehicles current
        "total" weight as well.

        Args:
            float: A decimal represented percentage in the range [0.0, 1.0]
        Returns:
            None
        '''
        self.fuel_percentage = fuel_percentage
        self.total_vehicle_mass = self.vehicle_mass + (self.fuel_weight * GAS_DENSITY * self.fuel_percentage)
        self.max_accel_torque = self.wheel_radius * self.max_accel * self.total_vehicle_mass

    # Create a velocity modifier to increase the target velocity up so
    # we can actuall hit the desired velocity (due to acceleration based
    # calculations always peering ahead in time). We can pick this based
    # on the target velocity.
    # DATA POINTS: (target in MPH for table)
    # target | old top | modifier | new avg | multiplier |
    # ----------------------------------------------------
    # 6.2137 | 7.15    | 0.518    | 6.21    | 0.08336417 |
    # ----------------------------------------------------
    # 12.427 | 10.47   | 0.980    | 12.37   | 0.07869960 |
    # ----------------------------------------------------
    # 18.641 | 15.97   | 1.540    | 18.60   | 0.08261359 |
    # ----------------------------------------------------
    # 24.85  | 21.04   | 2.075    | 24.85   | 0.08350100 |
    # ----------------------------------------------------
    # 31.068 | 26.18   | 2.595    | 31.06   | 0.08350100 |
    # ----------------------------------------------------
    def get_modifier(self, linear_velocity):
        linear_velocity *= 2.23694
        print(linear_velocity)
        if(linear_velocity >= 31.06):
            return 2.595
        elif(linear_velocity >= 24.85):
            return 2.075
        elif(linear_velocity >= 18.64):
            return 1.540
        elif(linear_velocity >= 12.42):
            return 0.980
        elif(linear_velocity >= 6.21):
            return 0.518
        return 0.0

    def get_throttle_brake(self, linear_velocity, angular_velocity, current_velocity):
        '''
        Returns a throttle and brake amount based on the current and target
        velocities

        Throttle value is calculated as proportional to the velocity error
        (target velocity - current velocity) and the proportional gain can be tuned.
        Throttle ranges from 0 to 1 and it is limited by a parameter, which can
        be set in the dbw_node.py

        Brake value is in the units of torque:
        Torque = F * r
        F = M * a
        a = (velocity error) / <time of execution>
          = (target velocity - current velocity) / <time of execution>

        We'll use the wheel torque (as opposed to the engine torque) and
        assume that there's no resistence forces on the wheels/tires. This
        means we can use the wheel radius for (r)

        The current and target velocities are given based on the status of
        the car. The mass thats being moved is the weight of the car. We'll
        assume for now that the weight doesn't change (i.e. no fuel loss)

        Args:
            float: target linear velocity (velocity in the x direction)
            float: target angular velocity of the vehicle
            float: current linear velocity of the vehicle
        Returns:
            list<float>: [throttle, brake]
        '''

        # Placeholders
        throttle = 0.0
        brake = 0.0

        # pick our velocity modifier
        velocity_modifier = self.get_modifier(linear_velocity)

        # Get the velocity error
        d_vel = ((linear_velocity + velocity_modifier) - current_velocity)

        # Figure out a proper acceleration over an arbitrary time. For
        # now we'll try to execute the acceleration over a second
        # and we'll limit the value to a desired amount based on whether
        # we want to accelerate or decelerate
        dt = 1.0 # seconds
        acc = d_vel / dt # m/s^2

        # Check the acceleration against the deadband
        if(acc < 0.0 and acc >= self.deadband):
            return throttle, brake

        # Check min/max values and gate the input acceleration
        if(acc < 0.0):
            acc = max(acc, self.max_decel)
        elif(acc > 0.0):
            acc = min(acc, self.max_accel)

        # Use acceleration, mass of car, and wheel radius to get the desired
        # torque value
        torque = self.total_vehicle_mass * self.wheel_radius * acc

        # When we're on the throttle, we're off the brake and vice versa
        if(torque >= 0):

            # control the throttle to be proportional to the velocity delta
            # and gate the throttle to the limit
            throttle = (torque / self.max_accel_torque) * self.max_accel_input
            throttle, brake = min(self.max_accel_input, throttle), 0.0
        else:
            # Gates the braking input value which is just a torque value
            throttle, brake = 0.0, max(abs(torque), self.max_decel_torque)

        # Return our final value
        return throttle, brake
