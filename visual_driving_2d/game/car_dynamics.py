"""
Car dynamics using Box2D physics engine.
Adapted from OpenAI Gym's box2d car_dynamics.
"""

import math
import numpy as np
import Box2D
from Box2D.b2 import (
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    circleShape,
)

from . import config


class Car:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x / config.SCALE, y / config.SCALE) for x, y in config.HULL_POLY1]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x / config.SCALE, y / config.SCALE) for x, y in config.HULL_POLY2]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x / config.SCALE, y / config.SCALE) for x, y in config.HULL_POLY3]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x / config.SCALE, y / config.SCALE) for x, y in config.HULL_POLY4]
                    ),
                    density=1.0,
                ),
            ],
        )
        self.hull.color = config.CAR_COLORS
        self.wheels = []
        self.fuel_spent = 0.0
        self.tiles = set()
        WHEEL_POLY = [
            (-config.WHEEL_W, +config.WHEEL_R),
            (+config.WHEEL_W, +config.WHEEL_R),
            (+config.WHEEL_W, -config.WHEEL_R),
            (-config.WHEEL_W, -config.WHEEL_R),
        ]
        for wx, wy in config.WHEELPOS:
            w = self.world.CreateDynamicBody(
                position=(init_x + wx / config.SCALE, init_y + wy / config.SCALE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x / config.SCALE, y / config.SCALE) for x, y in WHEEL_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = config.WHEEL_R / config.SCALE
            w.color = config.WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx / config.SCALE, wy / config.SCALE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * config.SCALE * config.SCALE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]

    def gas(self, gas):
        """Apply gas to rear wheels"""
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase gas
            w.gas += diff

    def brake(self, b):
        """Apply brakes to all wheels"""
        b = np.clip(b, 0, 1)
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """Steer front wheels"""
        s = np.clip(s, -1, 1)
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        """Update car physics"""
        for w in self.wheels:
            # Steer front wheels
            if w.steer != 0:
                w.joint.motorSpeed = float(np.sign(w.steer) * 50)
                w.joint.maxMotorTorque = 10000000 * config.SCALE * config.SCALE
            else:
                w.joint.motorSpeed = 0
                w.joint.maxMotorTorque = 0

            # Position => friction_limit
            grass = True
            friction_limit = config.FRICTION_LIMIT * 0.6  # Grass friction
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, config.FRICTION_LIMIT * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W = F*v
            # F = WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt / v
            # domega = dt*F/WHEEL_MOMENT_OF_INERTIA/w.omega * v
            # since E = 0.5*I*omega^2
            # if we change omega by domega, energy will change by dE ~ I*omega*domega
            w.omega += (
                dt
                * config.ENGINE_POWER
                * w.gas
                / config.WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )  # small coef not to divide by zero
            self.fuel_spent += dt * config.ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000 * config.SCALE * config.SCALE
            p_force *= 205000 * config.SCALE * config.SCALE
            force = (f_force * forw[0] + p_force * side[0], f_force * forw[1] + p_force * side[1])

            # Skid trace
            if abs(force[0]) > friction_limit or abs(force[1]) > friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            # Apply force
            w.ApplyForceToCenter(
                (
                    np.clip(force[0], -friction_limit, friction_limit),
                    np.clip(force[1], -friction_limit, friction_limit),
                ),
                True,
            )

    def draw(self, viewer, draw_particles=True):
        """Draw car using OpenGL (pyglet)"""
        from pyglet import gl

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = trans * f.shape.pos
                    gl.glColor4f(*obj.color, 1)
                    vertices = []
                    for angle in np.linspace(0, 2 * np.pi, 40):
                        vertices.extend([
                            t[0] + f.shape.radius * np.cos(angle),
                            t[1] + f.shape.radius * np.sin(angle)
                        ])
                    gl.glBegin(gl.GL_POLYGON)
                    for i in range(0, len(vertices), 2):
                        gl.glVertex2f(vertices[i], vertices[i+1])
                    gl.glEnd()
                else:
                    path = [trans * v for v in f.shape.vertices]
                    gl.glColor4f(*obj.color, 1)
                    gl.glBegin(gl.GL_POLYGON)
                    for v in path:
                        gl.glVertex3f(v[0], v[1], 0)
                    gl.glEnd()
                    # Wheel highlights
                    if "phase" not in obj.__dict__:
                        continue
                    a1 = obj.phase
                    a2 = obj.phase + 1.2  # radians
                    s1 = np.sin(a1)
                    s2 = np.sin(a2)
                    c1 = np.cos(a1)
                    c2 = np.cos(a2)
                    if s1 > 0 and s2 > 0:
                        continue
                    if s1 > 0:
                        c1 = np.sign(c1)
                    if s2 > 0:
                        c2 = np.sign(c2)
                    white_poly = [
                        (-config.WHEEL_W, +config.WHEEL_R * c1),
                        (+config.WHEEL_W, +config.WHEEL_R * c1),
                        (+config.WHEEL_W, +config.WHEEL_R * c2),
                        (-config.WHEEL_W, +config.WHEEL_R * c2),
                    ]
                    white_path = [trans * ((v[0] / config.SCALE, v[1] / config.SCALE)) for v in white_poly]
                    gl.glColor4f(*config.WHEEL_WHITE, 1)
                    gl.glBegin(gl.GL_POLYGON)
                    for v in white_path:
                        gl.glVertex3f(v[0], v[1], 0)
                    gl.glEnd()

    def _create_particle(self, point1, point2, grass):
        """Create skid particle for rendering"""
        class Particle:
            pass

        p = Particle()
        p.color = config.MUD_COLOR if not grass else (0.4, 0.8, 0.4, 0.0)
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        return p

    def destroy(self):
        """Clean up car bodies"""
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
