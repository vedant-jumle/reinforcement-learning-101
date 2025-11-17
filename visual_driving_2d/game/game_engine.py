"""
Car Racing Game Engine with RL Interface
Adapted from OpenAI Gym CarRacing environment
"""

import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import (fixtureDef, polygonShape, contactListener)

import pyglet
from pyglet import gl

from . import config
from .car_dynamics import Car


class FrictionDetector(contactListener):
    """Detects when car wheels touch road tiles"""

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # Reset tile color
        tile.color[0] = config.ROAD_COLOR[0]
        tile.color[1] = config.ROAD_COLOR[1]
        tile.color[2] = config.ROAD_COLOR[2]

        if not obj or "tiles" not in obj.__dict__:
            return

        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += config.REWARD_PER_TILE / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)


class RacingGame:
    """
    Car Racing Game with RL Interface

    Supports continuous control (steer, gas, brake) with visual observations.
    Compatible with socket-based distributed training.
    """

    def __init__(self, render_mode='human', total_episode_steps=None, verbose=1, seed=None):
        """
        Initialize racing game

        Args:
            render_mode: 'human', 'rgb_array', or None (headless)
            total_episode_steps: Maximum steps per episode
            verbose: Print debug info (0 or 1)
            seed: Random seed for track generation
        """
        self.render_mode = render_mode
        self.verbose = verbose
        self.total_episode_steps = total_episode_steps or config.DEFAULT_EPISODE_STEPS

        # Random number generator
        self.np_random = np.random.RandomState(seed)

        # Box2D physics world
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        # Rendering
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None

        # Game state
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.current_steps = 0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        # Track tile fixture definition
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

    def seed(self, seed=None):
        """Set random seed"""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        """Reset game to initial state"""
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.current_steps = 0

        # Generate new track
        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("Retrying track generation...")

        # Create car at starting position
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):
        """
        Execute one timestep

        Args:
            action: dict with keys 'steer', 'gas', 'brake' or None (for reset)
                   steer: [-1, +1]
                   gas: [0, +1]
                   brake: [0, +1]

        Returns:
            state: RGB image (96x96x3) if render_mode != None, else None
            reward: float
            done: bool
            info: dict with telemetry
        """
        self.current_steps += 1

        if action is not None:
            self.car.steer(action.get('steer', 0.0))
            self.car.gas(action.get('gas', 0.0))
            self.car.brake(action.get('brake', 0.0))

        self.car.step(1.0 / config.FPS)
        self.world.Step(1.0 / config.FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / config.FPS

        step_reward = 0
        done = False

        if action is not None:  # First step without action, called from reset()
            self.reward += config.REWARD_PER_FRAME
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # Check if completed track
            if self.tile_visited_count == len(self.track):
                done = True

            # Check if out of bounds
            x, y = self.car.hull.position
            if abs(x) > config.PLAYFIELD or abs(y) > config.PLAYFIELD:
                done = True
                step_reward = config.REWARD_OUT_OF_BOUNDS

            # Check if exceeded max steps
            if self.current_steps > self.total_episode_steps:
                done = True

        # Get state (render if needed)
        state = self.render('state_pixels') if self.render_mode is not None else None

        # Build info dict with telemetry
        info = self.get_telemetry()

        return state, step_reward, done, info

    def get_telemetry(self):
        """Get car telemetry data"""
        if self.car is None:
            return {}

        x, y = self.car.hull.position
        vx, vy = self.car.hull.linearVelocity
        angle = self.car.hull.angle
        angular_vel = self.car.hull.angularVelocity

        # Calculate speed
        speed = np.sqrt(vx**2 + vy**2)

        # Wheel data
        wheel_speeds = [w.omega for w in self.car.wheels] if self.car.wheels else [0, 0, 0, 0]

        return {
            'car_x': float(x),
            'car_y': float(y),
            'car_vx': float(vx),
            'car_vy': float(vy),
            'car_angle': float(angle),
            'car_angular_vel': float(angular_vel),
            'speed': float(speed),
            'wheel_speeds': wheel_speeds,
            'tiles_visited': self.tile_visited_count,
            'total_tiles': len(self.track) if self.track else 0,
            'steps': self.current_steps,
        }

    def render(self, mode='human'):
        """
        Render game

        Args:
            mode: 'human', 'state_pixels', or 'rgb_array'

        Returns:
            None for 'human', RGB array for others
        """
        assert mode in ['human', 'state_pixels', 'rgb_array']

        if self.viewer is None:
            from pyglet import gl
            self.viewer = pyglet.window.Window(width=config.WINDOW_W, height=config.WINDOW_H)
            self.score_label = pyglet.text.Label(
                '0000',
                font_size=36,
                x=20,
                y=config.WINDOW_H * 2.5 / 40.00,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255)
            )
            self.transform = Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Calculate camera transform
        zoom = config.ZOOM * config.SCALE
        zoom_state = config.ZOOM * config.SCALE * config.STATE_W / config.WINDOW_W
        zoom_video = config.ZOOM * config.SCALE * config.VIDEO_W / config.WINDOW_W

        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity

        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])

        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            config.WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            config.WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle))
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        # Setup viewport
        if mode == 'rgb_array':
            VP_W = config.VIDEO_W
            VP_H = config.VIDEO_H
        elif mode == 'state_pixels':
            VP_W = config.STATE_W
            VP_H = config.STATE_H
        else:
            VP_W = config.WINDOW_W
            VP_H = config.WINDOW_H

        # Clear and render
        gl.glClearColor(0.4, 0.8, 0.4, 1.0)
        self.viewer.clear()
        gl.glViewport(0, 0, VP_W, VP_H)

        self.transform.enable()
        self._render_road()
        self.transform.disable()

        self._render_indicators(config.WINDOW_W, config.WINDOW_H)

        if mode == 'human':
            self.viewer.flip()
            return self.viewer.is_visible

        # Capture pixels
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]  # Flip and remove alpha

        return arr

    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _destroy(self):
        """Destroy existing track and car"""
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        if self.car:
            self.car.destroy()

    def _create_track(self):
        """Generate procedural racing track"""
        CHECKPOINTS = config.TRACK_CHECKPOINTS

        # Create checkpoints in a circle
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = self.np_random.uniform(config.TRACK_RAD / 3, config.TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * config.TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * config.TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        self.road = []

        # Generate track path between checkpoints
        x, y, beta = 1.5 * config.TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False

        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            # Find destination checkpoint
            while True:
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy

            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi

            prev_beta = beta
            proj *= config.SCALE
            if proj > 0.3:
                beta -= min(config.TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(config.TRACK_TURN_RATE, abs(0.001 * proj))

            x += p1x * config.TRACK_DETAIL_STEP
            y += p1y * config.TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))

            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break

        if self.verbose == 1:
            print(f"Track generation: {i1}..{i2} -> {i2-i1}-tiles track")

        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        # Check if track is well-connected
        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > config.TRACK_DETAIL_STEP:
            return False

        # Mark borders on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(config.BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > config.TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == config.BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(config.BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create track tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - config.TRACK_WIDTH * math.cos(beta1), y1 - config.TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + config.TRACK_WIDTH * math.cos(beta1), y1 + config.TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - config.TRACK_WIDTH * math.cos(beta2), y2 - config.TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + config.TRACK_WIDTH * math.cos(beta2), y2 + config.TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [config.ROAD_COLOR[0] + c, config.ROAD_COLOR[1] + c, config.ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)

            # Add borders on turns
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * config.TRACK_WIDTH * math.cos(beta1), y1 + side * config.TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (config.TRACK_WIDTH + config.BORDER) * math.cos(beta1),
                        y1 + side * (config.TRACK_WIDTH + config.BORDER) * math.sin(beta1))
                b2_l = (x2 + side * config.TRACK_WIDTH * math.cos(beta2), y2 + side * config.TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (config.TRACK_WIDTH + config.BORDER) * math.cos(beta2),
                        y2 + side * (config.TRACK_WIDTH + config.BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))

        self.track = track
        return True

    def _render_road(self):
        """Render road using OpenGL"""
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(*config.GRASS_COLOR, 1.0)
        gl.glVertex3f(-config.PLAYFIELD, +config.PLAYFIELD, 0)
        gl.glVertex3f(+config.PLAYFIELD, +config.PLAYFIELD, 0)
        gl.glVertex3f(+config.PLAYFIELD, -config.PLAYFIELD, 0)
        gl.glVertex3f(-config.PLAYFIELD, -config.PLAYFIELD, 0)
        gl.glColor4f(*config.GRASS_LIGHT_COLOR, 1.0)
        k = config.PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def _render_indicators(self, W, H):
        """Render speed/wheel indicators at bottom of screen"""
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


class Transform:
    """Simple 2D transform for rendering"""

    def __init__(self):
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.translate_x = 0.0
        self.translate_y = 0.0
        self.rotation = 0.0

    def set_scale(self, sx, sy):
        self.scale_x = sx
        self.scale_y = sy

    def set_translation(self, tx, ty):
        self.translate_x = tx
        self.translate_y = ty

    def set_rotation(self, angle):
        self.rotation = angle

    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(self.translate_x, self.translate_y, 0)
        gl.glRotatef(math.degrees(self.rotation), 0, 0, 1)
        gl.glScalef(self.scale_x, self.scale_y, 1)

    def disable(self):
        gl.glPopMatrix()
