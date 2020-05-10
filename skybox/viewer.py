#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys 

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader

from PIL import Image               # load images for textures
from itertools import cycle

from transform import Trackball, identity, vec, scale, translate, rotate, lerp, quaternion_slerp, quaternion_matrix, quaternion, quaternion_from_euler, lookat
from bisect import bisect_left

import math

# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""

    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers.append(GL.glGenBuffers(1))
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(
                    loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


# ------------  Scene object classes ------------------------------------------
class Node:
    """ Scene graph transform and parameter broadcast node """

    def __init__(self, children=(), transform=identity()):
        self.transform = transform
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model):
        """ Recursive draw, passing down updated model matrix. """
        for child in self.children:
            child.draw(projection, view, model @ self.transform)

    def key_handler(self, key):
        """ Dispatch keyboard events to children """
        for child in self.children:
            if hasattr(child, 'key_handler'):
                child.key_handler(key)


# -------------- Phong rendered Mesh class -----------------------------------
# mesh to refactor all previous classes
class Mesh:

    def __init__(self, shader, attributes, index=None):
        self.shader = shader
        names = ['view', 'projection', 'model']
        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(self.loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(primitives)


class PhongMesh(Mesh):
    """ Mesh with Phong illumination """

    def __init__(self, shader, attributes, index=None,
                 light_dir=(0, -1, 0),   # directionnal light (in world coords)
                 k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):
        super().__init__(shader, attributes, index)
        self.light_dir = light_dir
        self.k_a, self.k_d, self.k_s, self.s = k_a, k_d, k_s, s

        # retrieve OpenGL locations of shader variables at initialization
        names = ['light_dir', 'k_a', 's', 'k_s', 'k_d', 'w_camera_position']

        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # setup light parameters
        GL.glUniform3fv(self.loc['light_dir'], 1, self.light_dir)

        # setup material parameters
        GL.glUniform3fv(self.loc['k_a'], 1, self.k_a)
        GL.glUniform3fv(self.loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(self.loc['k_s'], 1, self.k_s)
        GL.glUniform1f(self.loc['s'], max(self.s, 0.001))

        # world camera position for Phong illumination specular component
        w_camera_position = (0, 0, 0)   # TODO: to update
        GL.glUniform3fv(self.loc['w_camera_position'], 1, w_camera_position)

        super().draw(projection, view, model, primitives)

class TexturedPhongMesh(Mesh):
    """ Simple first textured object """

    def __init__(self, shader, texture, attributes, index=None,
                 light_dir=(0, -1, 0),   # directionnal light (in world coords)
                 k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):

        super().__init__(shader, attributes, index)
        self.light_dir = light_dir
        self.k_a, self.k_d, self.k_s, self.s = k_a, k_d, k_s, s

        names = ['light_dir', 'k_a', 's', 'k_s', 'k_d', 'w_camera_position']

        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc
        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)
        # setup texture and upload it to GPU
        self.texture = texture

    def key_handler(self, key):
        # some interactive elements
        if key == glfw.KEY_F6:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
        if key == glfw.KEY_F7:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)

        # setup light parameters
        GL.glUniform3fv(self.loc['light_dir'], 1, self.light_dir)

        # setup material parameters
        GL.glUniform3fv(self.loc['k_a'], 1, self.k_a)
        GL.glUniform3fv(self.loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(self.loc['k_s'], 1, self.k_s)
        GL.glUniform1f(self.loc['s'], max(self.s, 0.001))

        # world camera position for Phong illumination specular component
        w_camera_position = (0, 0, 0)   # TODO: to update
        GL.glUniform3fv(self.loc['w_camera_position'], 1, w_camera_position)

        super().draw(projection, view, model, primitives)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
    

# -------------- 3D resource loader -----------------------------------------
def load_phong_mesh(file, shader, light_dir):
    """ load resources from file using assimp, return list of ColorMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # prepare mesh nodes
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        mesh = PhongMesh(shader, [mesh.mVertices, mesh.mNormals], mesh.mFaces,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes

def load_phong_tex_mesh(file, shader, tex_file, light_dir):
    """ load resources from file using assimp, return list of ColorMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            mat.properties['diffuse_map'] = Texture(tex_file)

    # prepare mesh nodes
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mNormals, mesh.mTextureCoords[0]]
        mesh = TexturedPhongMesh(shader, mat['diffuse_map'], attributes, mesh.mFaces,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=1280, height=720):
        super().__init__()

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())
        

        GL.glDepthFunc(GL.GL_LEQUAL)

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.61,0.87,1.61, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)    # depth test now enabled (TP2)
        # GL.glEnable(GL.GL_CULL_FACE)     # backface culling enabled (TP2)

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            # view = view + lookat(vec(-4, 20, 0), vec(-4, 1, 1), vec(0, 1, 0))
            projection = self.trackball.projection_matrix(win_size)

            # draw our scene objects
            self.draw(projection, view, identity())

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

            if key == glfw.KEY_R:
                glfw.set_time(0)

            self.key_handler(key)

# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """

    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            tex = np.asarray(Image.open(file).convert('RGBA'))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


# -------------- Example texture plane class ----------------------------------
class TexturedPlane(Mesh):
    """ Simple first textured object """

    def __init__(self, file, shader):

        vertices = 100 * np.array(
            ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
        faces = np.array(((0, 1, 2), (0, 2, 3)), np.uint32)
        
        super().__init__(shader, [vertices], faces)

        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = file

        # setup texture and upload it to GPU
        self.texture = Texture(file, self.wrap_mode, *self.filter_mode)

    def key_handler(self, key):
        # some interactive elements
        if key == glfw.KEY_F6:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(
                self.file, self.wrap_mode, *self.filter_mode)
        if key == glfw.KEY_F7:
            self.filter_mode = next(self.filter)
            self.texture = Texture(
                self.file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv
        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        super().draw(projection, view, model, primitives)

class WaterPlane(Mesh):
    """ Simple first textured object """

    def __init__(self, shader):

        vertices = 100 * np.array(
            ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)), np.float32)
        faces = np.array(((3, 2, 0), (2, 1, 0)), np.uint32)
        super().__init__(shader, [vertices], faces)

        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)
        
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        super().draw(projection, view, model, primitives)

class Ground(Mesh):

    def __init__(self, shader, file):

        vertices = np.zeros((1,3),np.float32)    
        faces = np.zeros((1,3), np.uint32) 
        normals = np.zeros((1,3), np.uint32)    
        texCoords= np.zeros((1,2), np.float32)    

        size = 100

        shift_y = -5
        shift_x= -50
        shift_z = -50
 
        x = -1
        z = -1
        for i in range(0, size):
            x = i + shift_x
            for j in range(0, size):
                z = j + shift_z
                y = math.sin(x) * math.cos(z) + shift_y
                vertices = np.vstack((vertices, np.array(((x, y, z)), np.float32)))

        vertices = 100 *  np.delete(vertices, (0), axis=0)
        
        for i in range(0, size - 1):
            for j in range(0, size - 1):
                f1 = i * size + j
                f2 = i * size + 1 + j 
                f3 = (i + 1) * size + j

                texCoords = np.vstack((texCoords, np.array(((0,.1), (.1,.1), (0,0)), np.float32) ))
                
                f4 = f3
                f5 = f2
                f6 = (i + 1) * size + 1 + j

                texCoords = np.vstack((texCoords, np.array(((0, .2), (.2,.3), (0,0)), np.float32) ))

                faces = np.vstack((faces, np.array(((f1, f2, f3), (f4, f5, f6)), np.uint32)))

        faces = np.delete(faces, (0), axis=0)
        texCoords = np.delete(texCoords, (0), axis=0)


        super().__init__(shader, [vertices, texCoords], faces)

        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = file

        # setup texture and upload it to GPU
        self.texture = Texture(file, self.wrap_mode, *self.filter_mode)
    
    def key_handler(self, key):
        # some interactive elements
        if key == glfw.KEY_F8:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
        if key == glfw.KEY_F9:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):

        GL.glDepthMask(GL.GL_FALSE)
        GL.glDepthFunc(GL.GL_LEQUAL)

        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv
        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        super().draw(projection, view, model, primitives)

        GL.glDepthMask(GL.GL_TRUE)

         # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


# -------------- Example texture mesh class ----------------------------------
class TexturedMesh(Mesh):
    """ Simple first textured object """

    def __init__(self, shader, texture, attributes, index=None):
        super().__init__(shader, attributes, index)

        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc
        # setup texture and upload it to GPU
        self.texture = texture

         # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)

    def key_handler(self, key):
        # some interactive elements
        if key == glfw.KEY_F6:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
        if key == glfw.KEY_F7:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        super().draw(projection, view, model, primitives)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

class TexturedCubeMapMesh(Mesh):
    """ Simple first textured object """

    def __init__(self, shader, texture, attributes, index=None):
        super().__init__(shader, attributes, index)

        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc
        # setup texture and upload it to GPU
        self.texture = texture

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):

        GL.glDepthMask(GL.GL_FALSE)
        GL.glDepthFunc(GL.GL_LEQUAL)

        GL.glUseProgram(self.shader.glid)
        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        super().draw(projection, view, model, primitives)

        GL.glDepthMask(GL.GL_TRUE)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
        GL.glUseProgram(0)


class SkyboxTexture:
    """ Helper class to create and automatically destroy textures """

    def __init__(self, files):
        self.glid = GL.glGenTextures(1)
        try:
            # GL.glEnable(GL.GL_TEXTURE_CUBE_MAP)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.glid)
            
            # imports image as a numpy array in exactly right format
            for i in range(len(files)):

                tex = np.asarray(Image.open(files[i]).convert('RGBA'))
                texturing = GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i

                GL.glTexImage2D(texturing, 0, GL.GL_RGBA, tex.shape[1],
                                tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                               GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                               GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                               GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                               GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                               GL.GL_TEXTURE_WRAP_R, GL.GL_REPEAT)
            message = 'Loaded skybox texture'
            # GL.glDisable(GL.GL_TEXTURE_CUBE_MAP)
            print(message)
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

class Skybox(Node):
    """ Very simple skybox"""
    def __init__(self, shader):
        super().__init__()
        sky_texs = [
            "./skybox/underwater01_RT.jpg",
            "./skybox/underwater01_LF.jpg",
            "./skybox/underwater01_UP.jpg",
            "./skybox/underwater01_DN.jpg",
            "./skybox/underwater01_FR.jpg",
            "./skybox/underwater01_BK.jpg"]

        self.add(*load_skybox('./skybox/skybox.obj', shader, sky_texs))  # just load cylinder from file

class Cube(Node):
    """ Very simple cube"""
    def __init__(self, shader, texture):
        super().__init__()
        self.add(*load_textured('./cube/cube.obj', shader, texture))  # just load cube from file

class Fish(Node):
    """ Very simple fish"""
    def __init__(self, shader, obj, texture):
        super().__init__()
        self.add(*load_textured(obj, shader, texture))  # just load cube from file

class Submarine(Node):
    """ Very simple fish"""
    def __init__(self, shader,):
        super().__init__()
        self.add(*load_textured('./submarine/Seaview submarine/Seaview submarine.obj', shader, './submarine/Seaview submarine/Maps/fna1.jpg'))  # just load cube from file

class Diver(Node):
    """ Very simple fish"""
    def __init__(self, shader, light_dir):
        super().__init__()
        self.add(*load_phong_mesh('./diver/diver.obj', shader, light_dir))  # just load cube from file

class PhongFish(Node):
    """ Very simple fish"""
    def __init__(self, shader, obj, texture, light_dir):
        super().__init__()
        self.add(*load_phong_tex_mesh(obj, shader, texture, light_dir))  # just load cube from file


class RotationControlNode(Node):
    def __init__(self, key_left, key_right, key_fwd, key_bwd, key_up, key_down, axis, angle=0):
        super().__init__()
        x = 0
        z = 0
        self.angle, self.axis, self.x, self.z = angle, axis, x, z
        self.key_left, self.key_right, self.key_fwd, self.key_bwd, self.key_up, self.key_down= key_left, key_right, key_fwd, key_bwd, key_up, key_down

    def key_handler(self, key):
        self.angle += 0.5 * int(key == self.key_left)
        self.angle -= 0.5 * int(key == self.key_right)
        self.x += 1 * (key == self.key_fwd)
        self.x -= 1 * (key == self.key_bwd)
        self.z += 1 * (key == self.key_up)
        self.z -= 1 * (key == self.key_down)
        # self.transform = 
        self.transform = translate(0, self.z, self.x) @ rotate(self.axis, self.angle)
        super().key_handler(key)
    
    def draw(self, projection, view, model):
        super().draw(projection, lookat(vec(-4, 20, 0), vec(-4, 1, 1), vec(0, 1, 0)), model)

def load_skybox(file, shader, tex_files=None):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_files and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
        if tex_files:
            mat.properties['diffuse_map'] = SkyboxTexture(tex_files)

    # prepare textured mesh
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mTextureCoords[0]]
        mesh = TexturedCubeMapMesh(
            shader, mat['diffuse_map'], attributes, mesh.mFaces)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes

def load_textured(file, shader, tex_file=None):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            mat.properties['diffuse_map'] = Texture(tex_file)

    # prepare textured mesh
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mTextureCoords[0]]
        mesh = TexturedMesh(shader, mat['diffuse_map'], attributes, mesh.mFaces)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes

class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time <= self.times[0] or time >= self.times[-1]:
            return self.values[0 if time <= self.times[0] else -1]
        # 2. search for closest index entry in self.times, using bisect_left function
        _i = bisect_left(self.times, time) # _i is the time index just before t

        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        fraction = ( time - self.times[ _i - 1 ] ) / ( self.times[ _i ] - self.times[ _i - 1 ])
        return self.interpolate( self.values[ _i - 1 ], self.values[ _i ], fraction )

class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys): #,rotate_keys, scale_keys
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keys = KeyFrames(translate_keys)
        self.rotate_keys = KeyFrames(rotate_keys, interpolation_function= quaternion_slerp)
        # self.scale_keys = KeyFrames(scale_keys)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translate_mat = translate(self.translate_keys.value(time))
        rotate_mat = quaternion_matrix(self.rotate_keys.value(time))
        # scale_mat = vec(self.scale_keys.value(time))
        return translate_mat @ rotate_mat #* scale_mat

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        super().__init__()
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys) #, rotate_keys, scale_keys

    def draw(self, projection, view, model):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model)

class ProdecuralAnimationNode(Node):

    def draw(self, projection, view, model):
        self.transform = translate(0, 3 * math.sin(glfw.get_time()), glfw.get_time() % 1000000)
        super().draw(projection, view, model)

class ProdecuralAnimationNode2(Node):
        
    def draw(self, projection, view, model):
        self.transform = translate(4*math.sin(glfw.get_time()), 3 * math.sin(glfw.get_time()), 4*math.cos(glfw.get_time()))
        super().draw(projection, view, model)


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    shader = Shader("texture.vert", "texture.frag")
    phong_shader = Shader("phong.vert", "phong.frag")
    skybox_shader = Shader("skybox.vert", "skybox.frag")
    color_shader = Shader("color.vert", "color.frag")
    ground_shader = Shader("ground.vert", "ground.frag")
    tex_phong_shader = Shader("tex_phong.vert", "tex_phong.frag")

    light_dir = (0, -1, 0)

    skybox = Skybox(skybox_shader)
    sky_shape = Node(transform = scale(1000,1000,1000))
    sky_shape.add(skybox)
    viewer.add(sky_shape)

    ground = Ground(ground_shader, "./skybox/underwater01_DN.jpg")
    viewer.add(ground)
    
    cube = Cube(shader, "./cube/cube.png")
    cube_shape = Node(transform =translate(0.3, 0.03, 10) @ scale(1, 1, 1))  
    cube_shape.add(cube)                   
    viewer.add(cube_shape)


    clown_fish = Fish(shader, "./ClownFish/ClownFish2.obj","./ClownFish/ClownFish2_Base_Color.png")
    clown_fish_shape = Node(transform =translate(3, 1, 1) @ scale(1, 1, 1))     
    clown_fish_shape.add(clown_fish)                   
    viewer.add(clown_fish_shape)

    clown_anim = ProdecuralAnimationNode()
    clown_anim.add(clown_fish_shape)
    viewer.add(clown_anim)

    barracuda_fish = Fish(shader, "./Barracuda/Barracuda2anim.obj","./Barracuda/Barracuda_Base Color.png")
    barracuda_fish_shape = Node(transform =translate(1, 2, 1) @ scale(1, 1, 1) @rotate(vec(0, 1, 0), 90))   
    barracuda_fish_shape.add(barracuda_fish)                   

    translate_keys = {0: vec(1, 1, 1), 2: vec(25, 1, 0), 3: vec(50, 0, 0), 5: vec(25, 0, 0), 7: vec(0, 0, 0)}

    rotate_keys = {1: quaternion(), 4: quaternion_from_euler(0, 180, 0),
                   6: quaternion(0)}

    scale_keys = {0: 1, 2: 1, 4: 0.5}

    keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
    keynode.add(barracuda_fish_shape)
    viewer.add(keynode)

    submarine = Submarine(shader)
    submarine_shape = Node(transform = translate(-4, 1, 1) @ scale(0.1, 0.1, 0.1))     # make a thin cylinder
    submarine_shape.add(submarine)                    # scaled cylinder shape
    submarine_rot = RotationControlNode(glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_SPACE, glfw.KEY_LEFT_SHIFT , vec(0, 1, 0))
    submarine_rot.add(submarine_shape)
    viewer.add(submarine_rot)


    phong_fish = PhongFish(tex_phong_shader, "./Barracuda/Barracuda2anim.obj","./Barracuda/Barracuda_Base Color.png", light_dir)
    phong_fish_shape = Node(transform = translate(-1, -1, -1) @ scale(0.1, 0.1, 0.1) @ rotate((0,1,0), 90))
    phong_fish_shape.add(phong_fish)
    viewer.add(phong_fish_shape)

    phong_fish_2 = PhongFish(tex_phong_shader, "./fish-new/fish.obj","./fish-new/fish.jpg", light_dir)
    phong_fish_2_shape = Node(transform = translate(4, 1, -5) @ scale(0.05, 0.05, 0.05))
    phong_fish_2_shape.add(phong_fish_2)
    viewer.add(phong_fish_2_shape)

    # diver = Diver(phong_shader, light_dir)
    # viewer.add(diver)


    # Hierarchical fishes
    b1 = Node(transform = translate(1, 2, 20))
    b1.add(barracuda_fish)

    b2 = Node(transform = translate(4, 4, 23))
    b2.add(barracuda_fish)

    b3 = Node(transform = translate(3, 3, 25))
    b3.add(barracuda_fish)

    b_group = Node()
    b_group.add(b1, b2, b3)

    b_anim = ProdecuralAnimationNode2()
    b_anim.add(b_group)

    viewer.add(b_anim)



    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
