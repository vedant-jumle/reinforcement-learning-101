#!/usr/bin/env python
"""Test different pyglet rendering backends"""

import os
import sys

print("Testing pyglet backends...\n")

# Test 1: Check available platforms
try:
    import pyglet
    print(f"Pyglet version: {pyglet.version}")
    print(f"Pyglet platform: {pyglet.options['debug_gl']}")

    # Try to get platform
    from pyglet import gl
    print(f"GL info available: {hasattr(gl, 'gl_info')}")

except Exception as e:
    print(f"Error importing pyglet: {e}")
    sys.exit(1)

# Test 2: Try EGL backend
print("\n" + "="*60)
print("TEST 1: Trying headless EGL backend")
print("="*60)

try:
    os.environ['PYGLET_HEADLESS'] = '1'
    os.environ['PYGLET_EGL_DEVICE'] = '0'

    # Force reimport
    import importlib
    importlib.reload(pyglet.gl)
    importlib.reload(pyglet.window)

    window = pyglet.window.Window(width=100, height=100, visible=False)
    print("✓ EGL backend works!")
    window.close()
except Exception as e:
    print(f"✗ EGL backend failed: {e}")

# Test 3: Try xlib backend with specific config
print("\n" + "="*60)
print("TEST 2: Trying xlib with specific GL config")
print("="*60)

try:
    del os.environ['PYGLET_HEADLESS']

    # Specific GL config
    config = pyglet.gl.Config(
        double_buffer=True,
        depth_size=24,
        major_version=3,
        minor_version=3,
    )

    window = pyglet.window.Window(width=100, height=100, visible=False, config=config)
    print("✓ xlib with specific config works!")
    window.close()
except Exception as e:
    print(f"✗ xlib with config failed: {e}")

# Test 4: Try with minimal config
print("\n" + "="*60)
print("TEST 3: Trying minimal GL config")
print("="*60)

try:
    config = pyglet.gl.Config()
    window = pyglet.window.Window(width=100, height=100, visible=False, config=config)
    print("✓ Minimal config works!")
    window.close()
except Exception as e:
    print(f"✗ Minimal config failed: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("If all tests failed, pyglet cannot create GL contexts on this system.")
print("Possible solutions:")
print("  1. Use pygame-based rendering instead")
print("  2. Run on native Linux/Windows (not WSL2)")
print("  3. Use headless training only (--render none)")