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
            self.max_decel *= -1

        # Torque limits
        max_accel_torque = self.wheel_radius * self.max_accel * self.total_vehicle_mass
        self.max_accel_input = max_input_accel # Max input to the DBW system
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

    def get_throttle_brake(self, linear_velocity, angular_velocity, current_velocity):
        '''
        Returns a throttle and brake amount based on the current and target
        velocities

        To mix the throttle and brake amounts together we need a method of
        mapping a 'Nm' torque amount (the brake-by-wire system's value) to
        a percentage (the throttle-by-wire system's value). The thought is
        that:

        Torque = F * r
        F = M * a
        a = (target velocity - current velocity) / <time of execution>

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

        # Get the velocity delta
        d_vel = (linear_velocity - current_velocity)

        # Figure out a proper time/acceleration value that is underneath our
        # thresholds. It's different for acceleration and deceleration. For
        # Now we'll try to execute the acceleration over a second
        acc = d_vel
        if(d_vel > 0): # Accel
            acc = min(d_vel, self.max_accel)
        else:          # Decel
            acc = max(d_vel, self.max_decel)

        # Check the acceleration against the deadband
        if(acc < 0.0 and acc >= self.deadband):
            return throttle, brake

        # Use acceleration, mass of car, and wheel radius to get the desired
        # torque value
        torque = self.total_vehicle_mass * self.wheel_radius * acc

        # if we're accelerating, normalize this the range [0, 1]. We can use
        # the max torque value created above (mass * wheel radius * max acc)
        # When we're on the throttle, we're off the brake and vice versa
        if(torque >= 0):
            # Gates the throttle to the input limit given which is a ration
            # based on the torque outputted
            throttle = (torque / self.max_accel_torque) * self.max_accel_input
            throttle, brake = min(self.max_accel_input, throttle), 0.0
        else:
            # Gates the braking input value which is just a torque value
            throttle, brake = 0.0, max(abs(torque), self.max_decel_torque)

        # Return our final value
        return throttle, brake
