import math

def custom_reward(state, action):
    pole_angle = state[2] * (180 / math.pi)

    if pole_angle > 0:
        direction = 1
    else:
        direction = 0 

    if action == direction:
        return min(abs(pole_angle) / 12, 1.0)
    else:
        return -min(abs(pole_angle) / 12, 1.0)