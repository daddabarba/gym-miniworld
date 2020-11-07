#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Maze-v0')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--save_to', type=str, help='where to save maze map', default=None)
parser.add_argument('--load_from', type=str, help='where to load maze map from', default=None)
parser.add_argument('--domain_rand', type=lambda x : x.lower()=='true', help='Set to true to use stockastic world', default=False)
parser.add_argument('--size', nargs=2, type=int, help='HxW of maze', default=(5,5))
parser.add_argument('--max_steps', type=int, help='Max number of steps per episode', default=None)
parser.add_argument('--base_punishment', type=float, help='Base reward at each time-step', default=-1)
args = parser.parse_args()

env = gym.make(args.env_name, save_to=args.save_to, load_from=args.load_from, domain_rand=args.domain_rand, base_punishment=args.base_punishment, num_rows=args.size[0], num_cols=args.size[1], max_episode_steps=args.max_steps)

view_mode = 'top' if args.top_view else 'agent'
do_render = True

env.reset()

# Create the display window
env.render('pyglet', view=view_mode)

def step(action):

    obs, reward, done, info = env.step(action)
    print('step {}/{}: a: {}, r: {}, done: {}'.format(env.step_count, env.max_episode_steps, env.actions(action).name, reward, done))

    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        env.reset()

    if do_render:
        env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    global view_mode
    global do_render

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render('pyglet', view=view_mode)
        return

    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    if symbol == key.UP:
        step(env.actions.move_forward)
    elif symbol == key.DOWN:
        step(env.actions.move_back)

    elif symbol == key.LEFT:
        step(env.actions.turn_left)
    elif symbol == key.RIGHT:
        step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)

    elif symbol == key.V:
        view_mode = 'top' if view_mode=='agent' else 'agent'

    elif symbol == key.R:
        do_render = not do_render

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    if do_render:
        env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

# Enter main event loop
pyglet.app.run()

env.close()
