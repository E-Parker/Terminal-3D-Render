# This program was writen by Ethan Parker.
# This program is a self-contained 3d render engine that runs in the terminal.

# EXPLANATIONS:
# the constant, DITHER is a 4x4 bayer matrix. Basically, it's a noise map that I use to offset what colour is displayed
# at a given pixel. This approximates colours that cannot be displayed natively with the 231 colours used in the
# terminal.

# The Mesh class is a bit useless, it would be more efficient to just store 4 lists and have a bunch of functions to
# handle appending, removing, and replacing. the clipMesh method is also very slow and one of the main limiting
# factors when it comes to speed besides the slowness of calling the draw polygon function for every face, 
# one at a time. 

# NOTES:
# do NOT try to run this program using the standard python IDE as it does not use ANSI escape codes.

# I understand it would be way easier to import packages, there are tons of better ways to do what I've done here
# in a lot less code. Writing all of this was meant to be an exercise in what I could do with the knowledge I have.

# Because there aren't any built-in keyboard input models python 3.7.12, that are included in the windows,
# Linux, macOS, and Unix versions I had to resort to using the msvcrt package which is a windows specific package.
# before you say it, yes, I know the keyboard package exists, but it's not installed by default on Windows, requiring a
# pip install to get it running. It would be a lot easier to just start using non-standard python packages but that
# defeats the whole point of making this project.

# if you are running this on a non-Widows system, replace the getInput function with one of your own design. 
# All it needs to do is return a string of all the keys pressed since the last time it was called, plus the K_ESCAPE
# constant if the escape key is pressed. That should be enough to get this to run on anything. 

import math
import time
import os
import sys
import array
import msvcrt

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# DEFINING CONSTANTS AND SYSTEM FUNCTIONS


# Display:
WIDTH, HEIGHT = 64, 48  # Screen Size
MASTER_OFFSET = (24, 8)  # offsets all screen objects
FOV = 70
NEARCLIP = 0.01  # Near clipping distance.
FARCLIP = 64     # Far clipping distance.
MSPEED = 0.1     # Default movement speed.
RSPEED = 0.075   # Default rotation speed.

# Math:
INT_255_TO_INT_6 = 6 / 255      # Constant to convert an int[0-255] to int[0-6].
INT_255_TO_INT_24 = 24 / 255    # Constant to convert an int[0-255] to int[0-24].

FPS = 60
LIGHTING = (0, -0.75, 0.25)  # Vector for angle of lighting, *MUST BE A UNIT VECTOR*.
DITHER = (((0 / 16) - 0.5, (8 / 16) - 0.5, (2 / 16) - 0.5, (10 / 16) - 0.5),
          ((12 / 16) - 0.5, (4 / 16) - 0.5, (14 / 16) - 0.5, (6 / 16) - 0.5),
          ((3 / 16) - 0.5, (11 / 16) - 0.5, (1 / 16) - 0.5, (9 / 16) - 0.5),
          ((15 / 16) - 0.5, (7 / 16) - 0.5, (13 / 16) - 0.5, (5 / 16) - 0.5))

# Colours:
DARK = (0, 10, 40)
LIGHT = (255, 240, 230)
TRANSPARENCY = (255, 0, 255)
SF_ERR_CLR = (255, 60, 60)    # Soft error message colour
TEXT_CLR_1 = (240, 240, 240)  # Main text colour
TEXT_CLR_2 = (200, 200, 200)  # Menu text colour
TEXT_CLR_3 = (100, 100, 100)  # Background text colour

# Text:
CURSER = '\033[' + str(round((HEIGHT - (HEIGHT / 6) + MASTER_OFFSET[1]) / 2)) + ';' + str(MASTER_OFFSET[0] + 4) + 'H' +\
         u'\u001b[38;5;255m' + u'\u001b[48;5;235m'+'-> '

# Constants for key input:
K_ESCAPE = "k_escape"       # detecting escape key
K_UP = "k_up_arrow"         # detecting up arrow key
K_DOWN = "k_down_arrow"     # detecting down arrow key
K_LEFT = "k_left_arrow"     # detecting left arrow key
K_RIGHT = "k_right_arrow"   # detecting right arrow key


def terminate():
    sys.exit()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# LOADING FILES


def loadMesh(filename):
    """ This function loads a .obj file and stores it into a mesh object. .obj files are extremely simple. Each line
    consists of a tag followed by the data for that item. for example, the tag 'v' is for vertex and the following
    information should be three floating point numbers stored in raw text, while the tag 'p' stands for polygon,
    and the information should be 3 integers for the index of each vertex that forms that face. This is done to avoid
    repeating vertices. Seriously, if you want to try  something like this on your own just open a .obj in a text
    editor and see how it all goes together. """

    if filename[len(filename) - 4:] not in ['.obj', '.OBJ']:
        raise Exception('File is not an .obj, file unable to decode ')

    obj = open(filename, "r")
    vertices = ()
    uv_vertices = ()
    polygons = ()
    uv_polygons = ()

    try:  # Try loading mesh
        for line in obj:
            line = line.strip("\n")
            line = line.split(" ")

            if line[0] == 'v':  # Vertex
                point = Vect3(float(line[1]), float(line[2]), float(line[3]))
                vertices += (point,)

            if line[0] == 'vt':  # UV texture information
                uv = Vect2(float(line[1]), float(line[2]))
                uv_vertices += (uv,)

            elif line[0] == 'f':  # Polygon
                face1, face2, face3 = line[1].split('/'), line[2].split('/'), line[3].split('/')
                polygon = (int(face1[0]) - 1, int(face2[0]) - 1, int(face3[0]) - 1, Vect3(0, 0, 0))
                uv_polygon = (int(face1[1]) - 1, int(face2[1]) - 1, int(face3[1]) - 1,)

                polygons += (polygon,)
                uv_polygons += (uv_polygon,)
        # Calls update to write depth information.
        new_mesh = Mesh(vertices, polygons, uv_vertices, uv_polygons)
        new_mesh.updateNormals()

    finally:
        obj.close()

    return new_mesh


def loadBitmap(filename):
    """ This function decodes a bitmap image and stores it to a surface object.
        Bitmap images are very simple. the first 54 bytes are reserved for the header
        and can be skipped. To verify that the header is indeed 54 bytes long,
        check the value at 0000, A. the width and height of the image are stored in 
        four byte ints found at 0010, 2 to  0010, 5 and 0010, 6 to 0010, 9 respectively."""

    file = open(filename, "rb")
    rawBytes = file.read()
    # Bytes are writen in reverse order.

    width = int.from_bytes(rawBytes[18:21], 'little')
    height = int.from_bytes(rawBytes[22:25], 'little')
    pixelArray = []

    for x in range(int(rawBytes[10]), len(rawBytes), 3):
        colour = Colour(int(rawBytes[x + 2]), int(rawBytes[x + 1]), int(rawBytes[x]))   # Convert byte to float(0 - 1)
        pixelArray.append(colour)

    # Write data to texture:
    texture = Surface(width, height)
    for y in range(height):
        for x in range(width):
            texture.set_at((x, y), pixelArray[(y * width) + x])

    return texture


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# MATHEMATICS


class Vect2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        x = self.x + other[0]
        y = self.y + other[1]
        return Vect2(x, y)

    def __sub__(self, other):
        x = self.x - other[0]
        y = self.y - other[1]
        return Vect2(x, y)

    def __mul__(self, val):
        x = self.x * val
        y = self.y * val
        return Vect2(x, y)

    def __truediv__(self, val):
        x = self.x / val
        y = self.y / val
        return Vect2(x, y)

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value

    def update(self, x, y):
        self.x, self.y = x, y

    def length(self):
        length = math.sqrt((self.x * self.x) + (self.y * self.y))
        return length


class Vect3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        x = self.x + other[0]
        y = self.y + other[1]
        z = self.z + other[2]
        return Vect3(x, y, z)

    def __sub__(self, other):
        x = self.x - other[0]
        y = self.y - other[1]
        z = self.z - other[2]
        return Vect3(x, y, z)

    def __mul__(self, val):
        x = self.x * val
        y = self.y * val
        z = self.z * val
        return Vect3(x, y, z)

    def __truediv__(self, val):
        x = self.x / val
        y = self.y / val
        z = self.z / val
        return Vect3(x, y, z)

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value

    def __eq__(self, other):
        return self.x == other[0] and self.y == other[1] and self.z == other[2]

    def update(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def length(self):
        return math.sqrt((self.x * self.x) + (self.y * self.y) + (self.z * self.z))

    def dot(self, other):
        return (self.x * other[0]) + (self.y * other[1]) + (self.z * other[2])

    def rotate_x(self, angle):
        angle = math.radians(angle)
        y1 = (self.y * math.cos(angle)) - (self.z * math.sin(angle))
        z1 = (self.z * math.cos(angle)) + (self.y * math.sin(angle))
        self.y, self.z = y1, z1
        return self

    def rotate_y(self, angle):
        angle = math.radians(angle)
        x1 = (self.x * math.cos(angle)) - (self.z * math.sin(angle))
        z1 = (self.z * math.cos(angle)) + (self.x * math.sin(angle))
        self.x, self.z = x1, z1
        return self

    def rotate_z(self, angle):
        angle = math.radians(angle)
        x1 = (self.x * math.cos(angle)) - (self.y * math.sin(angle))
        y1 = (self.y * math.cos(angle)) + (self.x * math.sin(angle))
        self.x, self.y = x1, y1
        return self

    def lerp(self, b, amount):
        a = Vect3(self.x, self.y, self.z)
        return a + ((b - a) * amount)


class Plane:
    """ This class defines a basic 3d plane as defined by 3 points. """
    def __init__(self, a, b, c):

        self.n = getNormal(a, b, c)
        self.p = (a + b + c) / 3
        self.d = self.p.dot(self.n)

    def pointToPlane(self, p):
        """ This function calculates the point-to-plane distance from any given point. """
        distance = self.n.dot(p) - self.d
        return distance

    def vectorPlaneIntersect(self, start, end):
        """ This function calculates the intersection point of a ray and a plane. """
        ad = start.dot(self.n)
        t = (self.d - ad) / ((end.dot(self.n)) - ad)
        ray = end - start
        intersect = (ray * t) + start
        return intersect

    def vertexPlaneIntersect(self, start, end):
        """ This function calculates the intersection point of a vertex with texture coordinate and a plane. """
        ad = start[0].dot(self.n)
        t = (self.d - ad) / ((end[0].dot(self.n)) - ad)

        intersect = ((end[0] - start[0]) * t) + start[0]
        tx_intersect = ((end[1] - start[1]) * t) + start[1]

        return intersect, tx_intersect


def quickInterp(x1, y1, x2, inv_dist, y):
    """ This function interpolates between two points at a given y. """
    try:
        result = x1 + ((x2 - x1) * (y - y1)) * inv_dist
        return result
    except ZeroDivisionError:
        return x1


def QuickSort(sort, index):
    """my implementation of the QuickSort algorithm originally writen by Tony Hoare, 1960. """

    elements = len(sort)

    # Base case
    if elements < 2:
        return sort, index

    current_position = 0

    for i in range(1, elements):
        if sort[i] < sort[0]:
            current_position += 1
            sort[i], sort[current_position] = sort[current_position], sort[i]
            index[i], index[current_position] = index[current_position], index[i]
    sort[0], sort[current_position], = sort[current_position], sort[0]
    index[0], index[current_position] = index[current_position], index[0]

    # recursively sort blocks
    left = QuickSort(sort[0:current_position], index[0:current_position])
    right = QuickSort(sort[current_position + 1:elements], index[current_position + 1:elements])

    # recombine lists into one list
    return sort, left[1] + [index[current_position]] + right[1]


def getNormal(a, b, c):
    """ This function gets the normal vector of a face formed by three Vect3 objects """
    u, v = b - a, c - a

    normal = Vect3((u.y * v.z) - (u.z * v.y),
                   (u.z * v.x) - (u.x * v.z),
                   (u.x * v.y) - (u.y * v.x))

    length = normal.length()
    # since a divided by length of 0 is undefined, check for valid length first.
    if length > 0:
        normal = normal / length

    return normal


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# MESH CLASS & METHODS


class Mesh:
    """ This class stores a 3D object consisting of vertices and polygons that connect said vertices.
        The structure of this object is identical to the .obj file format. """
    def __init__(self, vertices=None, polygons=None, uv_vertices=None, uv_polygons=None):
        """ Vertices must be a list containing Vect3 [point 1, point 2, point 3 ...]
            Polygons must be a list containing lists containing index of vertices forming a face: [[0,1,2], ...].
            Data structure is the same for uv_vertices, and uv_polygons, just with Vect2 for texture coordinates."""

        if vertices is None:
            vertices = ()
        if polygons is None:
            polygons = ()
        if uv_vertices is None:
            uv_vertices = ()
        if uv_polygons is None:
            uv_polygons = ()

        self.vertices = vertices        # List of vertices
        self.polygons = polygons        # List of faces connecting vertices
        self.depth = []                 # Average depth information for each face
        self.uv_vertices = uv_vertices  # UV Texture coordinates
        self.uv_polygons = uv_polygons  # Faces connecting UV

        self.rotation_vel = [0, 0, 0]  # Current rotational velocity
        self.rotation = [0, 0, 0]  # Rotational offset
        self.position = Vect3(0, 0, 0)  # Positional offset
        self.facing = Vect3(0, 0, 1)  # Forward direction of mesh

    def __len__(self):
        """ Returns the number of polygons in the mesh. """
        return len(self.polygons)

    def __getitem__(self, key):
        """ Return the polygon formed by vertices at index."""
        try:
            a = (self.vertices[self.polygons[key][0]], self.uv_vertices[self.uv_polygons[key][0]])
            b = (self.vertices[self.polygons[key][1]], self.uv_vertices[self.uv_polygons[key][1]])
            c = (self.vertices[self.polygons[key][2]], self.uv_vertices[self.uv_polygons[key][2]])
            return a, b, c, self.polygons[key][3]
        except IndexError:
            print(self.polygons[key])
            print(len(self.vertices))
            raise IndexError("polygon at index dones not exist!")

    def __delitem__(self, key):
        if key != len(self.polygons):
            self.polygons = self.polygons[:key] + self.polygons[key + 1:]
            self.uv_polygons = self.uv_polygons[:key] + self.uv_polygons[key + 1:]
        else:
            self.polygons = self.polygons[:key]
            self.uv_polygons = self.uv_polygons[:key]

    def __setitem__(self, key, trigon):
        a, b, c, n = trigon[0], trigon[1], trigon[2], trigon[3]
        """ This function handles adding a new trigon to the mesh. """
        a_index, b_index, c_index = self.setVertex(a[0]), self.setVertex(b[0]), self.setVertex(c[0])
        u_index, v_index, w_index = self.setTexVertex(a[1]), self.setTexVertex(b[1]), self.setTexVertex(c[1])

        if n is None:
            n = getNormal(a, b, c)
        self.polygons = self.polygons[:key] + ((a_index, b_index, c_index, n),) + self.polygons[key + 1:]
        self.uv_polygons = self.uv_polygons[:key] + ((u_index, v_index, w_index),) + self.uv_polygons[key + 1:]

    def append(self, a, b, c, n=None):
        """ This function handles appending a new trigon to the mesh. """
        a_index, b_index, c_index = self.setVertex(a[0]), self.setVertex(b[0]), self.setVertex(c[0])
        u_index, v_index, w_index = self.setTexVertex(a[1]), self.setTexVertex(b[1]), self.setTexVertex(c[1])

        if n is None:
            n = getNormal(a, b, c)
        self.polygons += ((a_index, b_index, c_index, n),)
        self.uv_polygons += ((u_index, v_index, w_index),)

    def rotate_x(self, angle):
        """ Update x-axis rotation of all points."""
        self.rotation_vel[0] = angle
        self.facing.rotate_x(angle)

        iterator = map(self.vertex_rotate_x, self.vertices)
        self.vertices = tuple(iterator)

    def rotate_y(self, angle):
        """ Update y-axis rotation of all points."""
        self.rotation_vel[1] = angle
        self.facing.rotate_y(angle)

        iterator = map(self.vertex_rotate_y, self.vertices)
        self.vertices = tuple(iterator)

    def rotate_z(self, angle):
        """ Update z-axis rotation of all points."""
        self.rotation_vel[2] = angle
        self.facing.rotate_z(angle)

        iterator = map(self.vertex_rotate_z, self.vertices)
        self.vertices = tuple(iterator)

    def move_to(self, position):
        if self.position != position:
            self.position = self.position * -1
            iterator = map(self.vertex_move, self.vertices)
            self.vertices = tuple(iterator)

            self.position = position
            iterator = map(self.vertex_move, self.vertices)
            self.vertices = tuple(iterator)

    def move(self, position):
        self.position = position
        iterator = map(self.vertex_move, self.vertices)
        self.vertices = tuple(iterator)
        self.position = Vect3(0, 0, 0)

    def update(self):
        """ Reset average depth for sorting later. """
        if self.rotation_vel != 0:
            self.rotation += self.rotation_vel

        iterator = map(self.average_depth, self.polygons)
        self.depth = list(iterator)

    def updateNormals(self):
        """ This function updates all the normal vectors for each face."""
        for face in self.polygons:
            n = getNormal(self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]])
            face[3].x, face[3].y, face[3].z = n.x, n.y, n.z

    def setVertex(self, item):
        """ This function adds a single vertex to the list of vertices and returns its location in said list.
            Note: A vertex is only added if it does not already exist in the list of vertices."""
        if item in self.vertices:
            index = self.vertices.index(item)
        else:
            self.vertices += (item,)
            index = len(self.vertices) - 1

        return index

    def setTexVertex(self, item):
        """ This function adds a single texture coordinate to the list of uv_vertices. """
        if item in self.uv_vertices:
            index = self.uv_vertices.index(item)
        else:
            self.uv_vertices += (item,)
            index = len(self.uv_vertices) - 1

        return index

    def vertex_rotate_x(self, vertex):
        return (vertex - self.position).rotate_x(self.rotation_vel[0]) + self.position

    def vertex_rotate_y(self, vertex):
        return (vertex - self.position).rotate_y(self.rotation_vel[1]) + self.position

    def vertex_rotate_z(self, vertex):
        return (vertex - self.position).rotate_z(self.rotation_vel[2]) + self.position

    def vertex_move(self, vertex):
        return vertex + self.position

    def average_depth(self, face):
        return (self.vertices[face[0]].z + self.vertices[face[1]].z + self.vertices[face[2]].z) / 3


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# CAMERA CLASS

class Camera:
    """ This class is a generic camera class used for rendering. """

    def __init__(self, near, far, fov, resolution):
        # Display & rendering pre-calculations:
        self.resolution = resolution
        self.h_screen_center = self.resolution[0] / 2
        self.v_screen_center = self.resolution[1] / 2
        self.fov = fov
        self.aspect_ratio = resolution[0] / resolution[1]
        self.scale = 1 / (math.tan(0.5 * math.radians(self.fov)))
        self.h_scale_const = 0.5 * (resolution[0] * self.scale / min((1, self.aspect_ratio)))
        self.v_scale_const = 0.5 * (resolution[1] * self.scale * max((1, self.aspect_ratio)))
        self.h_center = 0.5 * resolution[0]
        self.v_center = 0.5 * resolution[1]
        self.deviation = (self.fov / 360)  # used checking of an object can be seen by the camera.
        self.state = 0
        self.near = near
        self.far = far
        self.clip = self.genClip()

        # Other:
        self.active = True

        # Movement:
        self.x_rotation = 0
        self.y_rotation = 0
        self.rSpeed = RSPEED
        self.mSpeed = MSPEED
        self.maxRotation = self.rSpeed * 2
        self.maxSpeed = self.mSpeed * 2
        self.smoothness = 0.025
        self.snap = (-0.01, 0.01)

        # Vectors
        self.position = Vect3(0, 0, 0)
        self.rotation = Vect3(0, 0, 1)
        self.velocity = Vect3(0, 0, 0)
        self.r_velocity = Vect2(0, 0)

    def genClip(self):
        """ This function handles generating the points that form the clipping plane used when rendering. """

        # Solve for top and left sides of the far plane:
        horizontal_far_side_length = ((0 - self.h_center) * self.far) / self.h_scale_const
        vertical_far_side_length = ((0 - self.v_center) * self.far) / self.v_scale_const

        # Solve for top and left sides of the near plane:
        horizontal_near_side_length = ((0 - self.h_center) * self.near) / self.h_scale_const
        vertical_near_side_length = ((0 - self.v_center) * self.near) / self.v_scale_const

        # generate points:
        c1 = Vect3(horizontal_far_side_length, vertical_far_side_length, self.far)
        c2 = Vect3(horizontal_far_side_length, 0 - vertical_far_side_length, self.far)
        c3 = Vect3(horizontal_near_side_length, vertical_near_side_length, self.near)
        c4 = Vect3(horizontal_near_side_length, 0 - vertical_near_side_length, self.near)
        c5 = Vect3(0 - horizontal_far_side_length, vertical_far_side_length, self.far)
        c6 = Vect3(0 - horizontal_far_side_length, 0 - vertical_far_side_length, self.far)
        c7 = Vect3(0 - horizontal_near_side_length, vertical_near_side_length, self.near)
        c8 = Vect3(0 - horizontal_near_side_length, 0 - vertical_near_side_length, self.near)

        # Generate planes from points.
        planes = (Plane(c7, c4, c3), Plane(c1, c5, c7), Plane(c6, c2, c4), Plane(c3, c2, c1),
                  Plane(c5, c6, c8), Plane(c6, c5, c1),)  # Last is far plane

        return planes

    def setProperties(self):
        """ This function handles changing different aspects of the camera's behavior. """

        self.maxRotation = self.rSpeed * 2
        self.maxSpeed = self.mSpeed * 2
        self.aspect_ratio = self.resolution[0] / self.resolution[1]
        self.scale = 1 / (math.tan(0.5 * math.radians(self.fov)))
        self.h_scale_const = 0.5 * (self.resolution[0] * self.scale / min((1, self.aspect_ratio)))
        self.v_scale_const = 0.5 * (self.resolution[1] * self.scale * max((1, self.aspect_ratio)))
        self.h_center = 0.5 * self.resolution[0]
        self.v_center = 0.5 * self.resolution[1]

        self.deviation = (self.fov / 360)
        self.clip = self.genClip()

    def update(self, frameDelta):
        """ This function handles updating the camera. """
        mSpeed = self.mSpeed * frameDelta
        rSpeed = self.rSpeed * frameDelta
        smoothness = self.smoothness * frameDelta
        snap = (self.snap[0] * frameDelta, self.snap[1] * frameDelta)

        if self.velocity.x > 0:     self.velocity.x -= smoothness
        elif self.velocity.x < 0:   self.velocity.x += smoothness
        if self.velocity.y > 0:     self.velocity.y -= smoothness
        elif self.velocity.y < 0:   self.velocity.y += smoothness
        if self.velocity.z > 0:     self.velocity.z -= smoothness
        elif self.velocity.z < 0:   self.velocity.z += smoothness

        if self.r_velocity.x > 0:   self.r_velocity.x -= smoothness
        elif self.r_velocity.x < 0: self.r_velocity.x += smoothness
        if self.r_velocity.y > 0:   self.r_velocity.y -= smoothness
        elif self.r_velocity.y < 0: self.r_velocity.y += smoothness

        if snap[0] < self.r_velocity.x < snap[1]: self.r_velocity.x = 0
        if snap[0] < self.r_velocity.y < snap[1]: self.r_velocity.y = 0

        if snap[0] < self.velocity.x < snap[1]: self.velocity.x = 0
        if snap[0] < self.velocity.y < snap[1]: self.velocity.y = 0
        if snap[0] < self.velocity.z < snap[1]: self.velocity.z = 0

        x_rotation = self.r_velocity.x
        y_rotation = self.r_velocity.y
        vel = Vect3(self.velocity[0],self.velocity[1],self.velocity[2])
        rotating = False
        moving = False

        # Get a string of all keypresses since last call:
        keypress = getInput()

        if K_ESCAPE in keypress:
            self.active = False
        else:
            for char in keypress:
                # Check for state toggle:
                if char == 't':
                    self.state += 1
                    if self.state >= 2:
                        self.state = 0
                else:
                    # Check for camera rotation:
                    if char in 'ijkl':
                        rotating = True
                        if char == 'i':   x_rotation += rSpeed
                        elif char == 'k': x_rotation -= rSpeed
                        elif char == 'j': y_rotation += rSpeed
                        elif char == 'l': y_rotation -= rSpeed

                    # Check for player movement:
                    if char in 'wasdrf':
                        moving = True
                        if char == 'w':   vel.z += mSpeed
                        elif char == 's': vel.z -= mSpeed
                        elif char == 'a': vel.x += mSpeed
                        elif char == 'd': vel.x -= mSpeed
                        elif char == 'r': vel.y += mSpeed
                        elif char == 'f': vel.y -= mSpeed

        if rotating:
            if -self.maxRotation < x_rotation < self.maxRotation: self.r_velocity.x = x_rotation
            if -self.maxRotation < y_rotation < self.maxRotation: self.r_velocity.y = y_rotation

        if moving:
            if -self.maxSpeed < vel.x < self.maxSpeed: self.velocity.x = vel.x
            if -self.maxSpeed < vel.y < self.maxSpeed: self.velocity.y = vel.y
            if -self.maxSpeed < vel.z < self.maxSpeed: self.velocity.z = vel.z

        vel.rotate_y(-self.y_rotation)

        # Update Vectors:
        self.x_rotation += math.degrees(self.r_velocity.x)
        self.y_rotation += math.degrees(self.r_velocity.y)

        self.rotation.rotate_y(self.r_velocity.y)
        self.rotation.rotate_x(self.r_velocity.x)

        self.position += vel


    def clipMesh(self, mesh):
        """ This method clips a mesh against the camera's view. """
        # Variables:

        # index         -   Current index of the mesh. The length of mesh changes so a while loop is needed.
        # a_distance    -   Point-to-plane distance for point "a" of current face. (same for b and c)
        # a_inside      -   Bool for if "a" is inside the camera's view. (same for b and c)
        # inside        -   Number of points on screen.

        # First pass removes invalid faces so that the much slower clipping is only run on faces which must be clipped.
        index = 0
        while index < len(mesh):
            face = mesh[index]
            if (face[3].dot(self.rotation) > self.deviation or
                    (face[0][0].z < self.near and face[1][0].z < self.near and face[2][0].z < self.near)):
                del mesh[index]
            else:
                index += 1

        # clip mesh against all planes:
        for plane in self.clip:
            index = 0
            while index < len(mesh):
                face = mesh[index]
                a, b, c, n = face[0], face[1], face[2], face[3]
                a_distance = plane.pointToPlane(a[0])
                b_distance = plane.pointToPlane(b[0])
                c_distance = plane.pointToPlane(c[0])

                a_inside, b_inside, c_inside = a_distance > 0.001, b_distance > 0.001, c_distance > 0.001,
                inside = int(a_inside) + int(b_inside) + int(c_inside)

                if inside == 0:  # Face is off-screen, don't render. (0 faces)
                    del mesh[index]
                else:
                    if inside == 1:  # Two points off-screen, clip into trigon, update face.
                        if a_inside:
                            b, c = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                        elif b_inside:
                            a, c = plane.vertexPlaneIntersect(b, a), plane.vertexPlaneIntersect(b, c)
                        elif c_inside:
                            b, a = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                        mesh[index] = (a, b, c, n)

                    elif inside == 2:  # One point off-screen. Clip into quad then trigon, update and append face.
                        if not a_inside:  # A is off-screen
                            ab, ac = plane.vertexPlaneIntersect(a, b), plane.vertexPlaneIntersect(a, c)
                            mesh.append(b, ab, ac, n)
                            mesh[index] = (c, b, ac, n)

                        elif not b_inside:  # B is off-screen
                            bc, ba = plane.vertexPlaneIntersect(b, c), plane.vertexPlaneIntersect(b, a)
                            mesh.append(a, ba, bc, n)
                            mesh[index] = (a, c, bc, n)

                        elif not c_inside:  # C is off-screen
                            cb, ca = plane.vertexPlaneIntersect(c, b), plane.vertexPlaneIntersect(c, a)
                            mesh.append(b, cb, ca, n)
                            mesh[index] = (b, a, ca, n)
                    index += 1
        return mesh

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# COLOUR CLASS AND METHODS


def threshold(value, maximum, minimum):
    if value <= minimum: return minimum
    elif value >= maximum: return maximum
    return value


def linear_interp(a, b, factor):
    result = a + ((b - a) * factor)
    return result


class Colour:
    def __init__(self, r=0, g=0, b=0):
        """ tuple or list object can be passed, or r, g, b values. """

        # Input is a list or tuple (r, g, b):
        if type(r) is tuple or type(r) is list:
            self.r = threshold(int(r[0]), 255, 0)
            self.g = threshold(int(r[1]), 255, 0)
            self.b = threshold(int(r[2]), 255, 0)

        # Input is separate r, g, b:
        else:
            self.r = threshold(int(r), 255, 0)
            self.g = threshold(int(g), 255, 0)
            self.b = threshold(int(b), 255, 0)

    def __getitem__(self, key):
        if key == 0: return self.r
        elif key == 1: return self.g
        elif key == 2: return self.b

    def __setitem__(self, key, value):
        if key == 0: self.r = int(value)
        elif key == 1: self.g = int(value)
        elif key == 2: self.b = int(value)

    def __eq__(self, colour):
        return self.r == colour[0] and self.g == colour[1] and self.b == colour[2]

    def mix(self, colour, factor):
        """ This method mixes two colours by a factor, float [0:1]. """
        r = linear_interp(self.r, colour[0], factor)
        g = linear_interp(self.g, colour[1], factor)
        b = linear_interp(self.b, colour[2], factor)
        mixed = Colour(r, g, b)
        return mixed


def colour_mix(c1, c2, factor):
    """ This function mixes two coloures. used when input colour may not be stored as a colour. """
    r = linear_interp(c1[0], c2[0], factor)
    g = linear_interp(c1[1], c2[1], factor)
    b = linear_interp(c1[2], c2[2], factor)
    mixed = Colour(r, g, b)
    return mixed


def RGB_to_256_Colour(colour, d):
    if colour[0] != colour[1] or colour[1] != colour[2]:  # Colour is not grayscale, (16 - 231)
        result = str(16 + (36 * threshold(int(colour[0] * INT_255_TO_INT_6 + d), 5, 0) +
                           (6 * threshold(int(colour[1] * INT_255_TO_INT_6 + d), 5, 0) +
                                threshold(int(colour[2] * INT_255_TO_INT_6 + d), 5, 0))))
    else:  # colour is grayscale, (232 - 255)
        result = str(232 + threshold(int(colour[0] * INT_255_TO_INT_24 + d), 23, 0))
    return result


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# SURFACE CLASS AND METHODS


def genDistanceTable(width, factor=4):
    """ This function generates a look-up-tabel for 1D distances. """
    result = [0]
    inv_factor = 1 / factor
    for i in range((width * factor), 0, -1):
        inv = -1 / (i * inv_factor)
        result.append(inv)
    fast_array = array.array('f', result)
    return fast_array, factor


def setCurser(x, y):
    return '\033' + f'[{y // 2};{x}f'


def setPixel(x, y, fColour, bColour):
    """ This function generates the ANSI escape sequence for setting a pair of pixels. """
    fc = RGB_to_256_Colour(fColour, DITHER[x % 4][y % 4])
    bc = RGB_to_256_Colour(bColour, DITHER[x % 4][(y + 1) % 4])

    return u"\u001b[38;5;" + fc + "m" + u"\u001b[48;5;" + bc + "m▀"


clear = setPixel(0, 0, DARK, DARK)  # I know this is out of place but I sorta had to put it here :P
CLEAR_END = clear + clear + clear + clear


class Surface:
    """ This class acts as an array of pixels which can be set using a screen coordinate and a colour value.
        Once a surface has been defined is properties are immutable and WILL break if changed. """
    def __init__(self, width, height):
        # Variables:
        self.width = width
        self.height = height
        self.surface = []
        self.line = [Colour(0, 0, 0) for _ in range(self.height)]
        self.clear()

    def get_at(self, x, y):
        try:
            return self.surface[x][y]
        except IndexError:  # Value error, return nearest pixel instead.
            x = threshold(x, 0, self.width - 1)
            y = threshold(y, 0, self.height - 1)
            return self.surface[x][y]

    def set_at(self, xy, colour):
        """ Set the value of a pixel in the surface. """
        try:
            self.surface[xy[0]][xy[1]] = colour
        except IndexError:
            pass

    def get_scale_values(self, size):
        """ This method returns the properties used to scale a surface. """
        inv_x_size = 1 / size[0]
        inv_y_size = 1 / size[1]
        x_vals = []
        y_vals = []

        # Precalculate lines:
        for y in range(size[1]):
            # Set up y values for scanline:
            v = y * inv_y_size
            v_scaled = v * self.height  # Scale v from float [1:0], to an index for the image
            v_factor = v_scaled % 1
            vi = int(v_scaled)
            y_vals.append((v_factor, vi))

        for x in range(size[0]):
            # set up x values for scanline:
            u = x * inv_x_size
            u_scaled = u * self.width  # Scale u from float [1:0], to an index for the image
            u_factor = u_scaled % 1
            ui = int(u_scaled)
            x_vals.append((u_factor, ui))

        return x_vals, y_vals

    def scale(self, size):
        scaled_surface = Surface(size[0], size[1])

        vals = self.get_scale_values(size)
        x_vals, y_vals = vals[0], vals[1]

        # Write to new Surface:
        for y in range(size[1]):
            v_factor, v0 = y_vals[y]
            v1 = v0 + 1
            for x in range(size[0]):
                u_factor, u0 = x_vals[x]
                u1 = u0 + 1

                # Filter texture:
                s1 = self.get_at(u0, v0).mix(self.get_at(u0, v1), v_factor)
                s2 = self.get_at(u1, v0).mix(self.get_at(u1, v1), v_factor)
                colour = s1.mix(s2, u_factor)

                scaled_surface.surface[x][y] = colour
        return scaled_surface

    def fill(self, colour):
        """ This function fills a surface with a given colour. """
        self.surface = []
        line = [colour for _ in range(self.height)]
        for _ in range(self.width):
            self.surface.append(line[:])

    def clear(self):
        """ This function clears a surface, writing 0's to each pixel. ever so slightly faster than filling with
        black. """
        self.surface = []
        for _ in range(self.width):
            self.surface.append(self.line[:])

    def blit(self, surface, x, y):
        """ This method copies the data from one surface to another and a specific x, y. """
        for lclY in range(y, min(self.height, (surface.height + y))):
            for lclX in range(x, min(self.width, (surface.width + x))):
                self.surface[lclX][lclY] = surface.surface[lclX - x][lclY - y]

    def t_blit(self, surface, x, y):
        """ This method copies the data from one surface to another and a specific x, y, with basic transparency. """
        for lclY in range(y, min(self.height, (surface.height + y))):
            for lclX in range(x, min(self.width, (surface.width + x))):
                n_colour = surface.surface[lclX - x][lclY - y]
                if not n_colour == TRANSPARENCY:
                    self.surface[lclX][lclY] = n_colour

    def s_blit(self, surface):
        """ This method copies the data from one surface to another, while scaling and applying transparency. """
        size = (self.width, self.height)

        vals = surface.get_scale_values(size)
        x_vals, y_vals = vals[0], vals[1]

        # Write to new Surface:
        for y in range(size[1]):
            v_factor, v0 = y_vals[y]
            v1 = v0 + 1
            for x in range(size[0]):
                u_factor, u0 = x_vals[x]
                u1 = u0 + 1

                # Filter texture:
                s1_1, s1_2 = surface.get_at(u0, v0), surface.get_at(u0, v1)
                s2_1, s2_2 = surface.get_at(u1, v0), surface.get_at(u1, v1)

                current_colour = self.get_at(x, y)

                # Mix with colour already present to smooth edge:
                if s1_1 == TRANSPARENCY:
                    s1_1 = current_colour
                if s1_2 == TRANSPARENCY:
                    s1_2 = current_colour
                if s2_1 == TRANSPARENCY:
                    s2_1 = current_colour
                if s2_2 == TRANSPARENCY:
                    s2_2 = current_colour

                s1 = colour_mix(s1_1, s1_2, v_factor)
                s2 = colour_mix(s2_1, s2_2, v_factor)
                colour = colour_mix(s1, s2, u_factor)

                self.surface[x][y] = colour

    def flip(self, x_offset=MASTER_OFFSET[0] + 2, y_offset=MASTER_OFFSET[1] + 1):
        """ This function draws the Surface to the console output. """
        # Draw screen boarder:
        output = ""
        # for each line, update and add it to the output:
        for y in range(0, self.height, 2):
            current_y = y + y_offset
            output += setCurser(x_offset, current_y)
            for x in range(self.width):
                try:
                    output += setPixel(x + x_offset, current_y, self.surface[x][y], self.surface[x][y + 1])
                except IndexError:
                    pass
            output += setCurser(self.width + x_offset, current_y) + CLEAR_END
        return output


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# TEXTBOX CLASS AND METHODS


def getInput():
    """ This function returns input from the keyboard.
        Replace this function with whatever keyboard input method works on your system. """
    keypress = ''
    for _ in range(5):  # Check for input 5 times
        if msvcrt.kbhit():  # a key as been pressed:
            char = msvcrt.getche()
            if ord(char) == 27:
                keypress += K_ESCAPE
            if ord(char) >= 32:
                keypress += str(char)
                break

    return keypress


class TextBox:
    """ Class to store and display text on the screen. """
    def __init__(self, position, size, text='', colour=(240, 240, 240), border='│─╭╮╰╯', l_height=3.5, l_Offset=2.5):
        self.x = position[0] + MASTER_OFFSET[0]
        self.y = (position[1] * l_height) + MASTER_OFFSET[1] - l_Offset
        self.size = size
        self.text = text
        self.d_colour = u'\u001b[38;5;240m' + u'\u001b[48;5;235m'
        self.colour = u'\u001b[38;5;' + RGB_to_256_Colour(colour, 0) + 'm'
        self.border = border  # Default characters for drawing boarder

    def draw(self, style='t'):
        """ This method handles drawing a textbox to the screen. It's a bit messy, but it does work.
            style 't' is for top left,
            style 'c' is for centered. """

        text = self.text[:]

        if style == 't':  # Top left
            xOffset, yOffset = round(self.x - 1), round(self.y - 1)
        elif style == 'c':  # Centered
            xOffset, yOffset = self.x - round((self.size[0] // 2) - 1), round(self.y - (self.size[1] / 2) - 1)
        else:
            xOffset, yOffset = 0, 0

        # Split text by size of frame:
        if len(text) < self.size[0]:    # If the text fits on one line, center it.
            whitespace = ''
            for _ in range(self.size[0] * self.size[1]):
                whitespace += ' '
            gap = (self.size[0] - len(text)) // 2
            lineOffset = (self.size[1] // 2) * self.size[0]
            text = whitespace[:gap + lineOffset] + text + whitespace[len(text) + lineOffset:]
        else:
            for _ in range((self.size[0] * self.size[1]) - len(self.text)):
                text += ' '

        # Generate top line:
        line = self.d_colour + '\033[' + f'{yOffset};{xOffset}H' + self.border[2]
        for _ in range(self.size[0]):
            line += self.border[1]
        line += self.border[3]

        # Generate Middle:
        for y in range(self.size[1]):
            line += '\033[' + f'{y + 1 + yOffset};{xOffset}H' + self.border[0] + self.colour
            for x in range(self.size[0]):
                line += text[x + (y * self.size[0])]
            line += self.d_colour + self.border[0]

        # Generate End:
        line += '\033[' + f'{self.size[1] + 1 + yOffset};{xOffset}H' + self.border[4]
        for _ in range(self.size[0]):
            line += self.border[1]
        line += self.border[5]

        print(line)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# DRAWING FUNCTIONS


def interpLine(v1, v2):
    """ This function interpolates a line on the screen from v1 to v2."""
    # Check that interpolation is even possible:
    start, end = round(v1[1]), round(v2[1])
    if start - end == 0:
        return ()

    # Get Inverse of distance between a and b:
    inv_dist = 1 / (v2[1] - v1[1])

    line = (v1,)

    # interpolate points from start to end:
    for y in range(start + 1, end):
        # generate new position
        line += ((int(quickInterp(v1[0], v1[1], v2[0], inv_dist, y)), y),)

    return line


def interpVertexLine(a, b):
    """ This function is intended for rasterization and handles generating a line segment from x1,y1,z1 to x2,y2,z2.
    Y values will be treated as an int."""
    # Check that interpolation is even possible:
    a, b, u, v = a[0], b[0], a[1], b[1]
    start, end = round(a.y), round(b.y)
    if start - end == 0:
        return ()

    # Get Inverse of distance between a and b:
    inv_dist = 1 / (b.y - a.y)

    # Multiply texture coordinate by inverted Z.
    # This is done so that when we interpolate between z values, they'll be in screen-space (1 / z).
    ui, vi = u * a.z, v * b.z
    line = ((a, ui),)

    # interpolate points from start to end:
    for y in range(start + 1, end, 1):
        # generate new position
        point = Vect3(a.x + ((b.x - a.x) * (y - a.y)) * inv_dist, y, a.z + ((b.z - a.z) * (y - a.y)) * inv_dist)
        tx_point = Vect2(ui.x + ((vi.x - ui.x) * (y - a.y)) * inv_dist, ui.y + ((vi.y - ui.y) * (y - a.y)) * inv_dist)
        line += ((point, tx_point,),)
    return line


def drawLine(surface, v1, v2, colour):
    """ This function draws a line on the screen from v1 to v2."""
    try:
        slope = (v2[1] - v1[1]) / (v2[0] - v1[0])
    except ZeroDivisionError:  # Slope is undefined, line must be vertical.
        if v1[1] > v2[1]:
            v1, v2 = v2, v1
        x = int(v1[0])
        for y in range(round(v1[1]), round(v2[1])):
            surface.set_at((x, y), colour)
        return

    if -1 <= slope <= 1:  # slope is mostly vertical so only set a pixel for every y value.
        if v1[0] > v2[0]:
            v1, v2 = v2, v1
        # Check that interpolation is even possible:
        Start, End = round(v1[0]), round(v2[0])
        if Start - End == 0:
            return

        # Get Inverse of distance between a and b:
        inv_dist = 1 / (v2[0] - v1[0])

        surface.set_at((int(v1[0]), int(v1[1])), colour)

        # interpolate points from start to end:
        for x in range(Start + 1, End):
            # generate new position
            surface.set_at((x, int(quickInterp(v1[1], v1[0], v2[1], inv_dist, x))), colour)

    else:  # Slope is mostly horizontal, set a pixel for every x value.
        if v1[1] > v2[1]: v1, v2 = v2, v1

        # Check that interpolation is even possible:
        Start, End = round(v1[1]), round(v2[1])
        if Start - End == 0:
            return

        # Get Inverse of distance between a and b:
        inv_dist = 1 / (v2[1] - v1[1])

        surface.set_at((int(v1[0]), int(v1[1])), colour)
        # interpolate points from start to end:
        for y in range(Start + 1, End):
            # generate new position
            surface.set_at((int(quickInterp(v1[0], v1[1], v2[0], inv_dist, y)), y), colour)


def drawPolygon(surface, v1, v2, v3, brightness, colour):
    """ This function draws flat-shaded polygon. """

    if v1[1] > v2[1]: v1, v2 = v2, v1
    if v2[1] > v3[1]: v2, v3 = v3, v2
    if v1[1] > v2[1]: v1, v2 = v2, v1

    lSide = interpLine(v1, v2) + interpLine(v2, v3)
    rSide = interpLine(v1, v3)

    yOffset = round(v1[1])

    if len(rSide) != 0:
        middle = len(rSide) // 2
        if lSide[middle][0] < rSide[middle][0]:
            lSide, rSide = rSide, lSide

        for lclY in range(len(rSide)):
            rs, ls = rSide[lclY], lSide[lclY]
            rsOffset, lsOffset, y = int(rs[0]), int(ls[0]), lclY + yOffset
            for lclX in range(lsOffset - rsOffset):
                x = lclX + rsOffset

                if DITHER[x % 4][y % 4] > brightness:
                    surface.set_at((x, y), Colour(0, 0, 0))
                else:
                    surface.set_at((x, y), colour)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# TEXTURE MAPPING:


def texturedPolygon(surface, depth, lookUpTable, a, b, c, brightness, texture, state=0):
    """ This function draws a textured polygon with perspective correction and anti-aliasing. """

    txW, txH = texture.width - 1, texture.height - 1

    if brightness > 0:
        shade = DARK
    else:
        shade = LIGHT

    brightness = abs(brightness)

    if a[0].y > b[0].y: a, b = b, a
    if b[0].y > c[0].y: b, c = c, b
    if a[0].y > b[0].y: a, b = b, a

    yOffset = round(a[0].y)

    lSide = interpVertexLine(a, b) + interpVertexLine(b, c)
    rSide = interpVertexLine(a, c)

    if len(rSide) != 0:
        middle = len(rSide) // 2
        if lSide[middle][0].x < rSide[middle][0].x:
            lSide, rSide = rSide, lSide

        for lclY in range(len(rSide)):
            rs, ls, trs, tls = rSide[lclY][0], lSide[lclY][0], rSide[lclY][1], lSide[lclY][1]
            rsOffset, lsOffset, y = round(rs.x), round(ls.x), lclY + yOffset
            inv_dist = lookUpTable[0][int((rs.x - ls.x) * lookUpTable[1])]
            for lclX in range(lsOffset - rsOffset):
                x = lclX + rsOffset
                inv_z = quickInterp(ls.z, ls.x, rs.z, inv_dist, x)
                try:
                    if depth[x][y] > inv_z:

                        z = 1 / inv_z

                        if state == 0:  # ANTI-ALIASING:

                            # Bilinear filtering:
                            u = (quickInterp(tls.x, ls.x, trs.x, inv_dist, x) * z) % 1
                            v = (quickInterp(tls.y, ls.x, trs.y, inv_dist, x) * z) % 1
                            utex, vtex = (u * txW), (v * txH)
                            ufac, vfac = utex % 1, vtex % 1
                            u0, v0 = int(utex), int(vtex)
                            u1, v1 = u0 + 1, v0 + 1

                            # Sample texture at four points:
                            # Mix by horizontal sub-pixel value:
                            s1 = texture.get_at(u0, v0).mix(texture.get_at(u0, v1), vfac)
                            s2 = texture.get_at(u1, v0).mix(texture.get_at(u1, v1), vfac)
                            # Mix by vertical sub-pixel value:
                            raw_colour = s1.mix(s2, ufac)

                            # Apply Lighting:
                            colour = raw_colour.mix(shade, brightness)

                        elif state == 1:  # WITHOUT AA:

                            u = (quickInterp(tls.x, ls.x, trs.x, inv_dist, x) * z) % 1
                            v = (quickInterp(tls.y, ls.x, trs.y, inv_dist, x) * z) % 1

                            raw_colour = texture.get_at(int(u * txW), int(v * txH))

                            # Apply Lighting:
                            colour = raw_colour.mix(shade, brightness)

                        else:  # FLAT
                            colour = colour_mix(LIGHT, shade, brightness)

                        # Write to surface:
                        surface.set_at((x, y), colour)
                        depth[x][y] = inv_z
                except IndexError:
                    pass

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# GAME CLASS AND MAINLINE


class Game:
    def __init__(self, fov, far, near, resolution):

        # Load background image:
        try:
            tittle = loadBitmap("tittle.bmp")
            tittle = tittle.scale((WIDTH, HEIGHT))
        except:
            tittle = Surface(WIDTH, HEIGHT)
            tittle.fill(Colour(16, 16, 16))
        self.tittleScreen = tittle.flip()

        try:
            self.hud = loadBitmap("hud.bmp")
        except:
            self.hud = Surface(WIDTH, HEIGHT)
            self.hud.fill(TRANSPARENCY)

        # Define Textboxes:
        mdl = WIDTH // 2
        self.pauseScreen = (TextBox((mdl, 3), (20, 6),'',TEXT_CLR_1),
                            TextBox((mdl, 3), (18, 4),'    Game Paused   1: Camera settings2: Load new models3:    Exit program',TEXT_CLR_2,'      '),
                            )

        self.cameraSettings = (TextBox((mdl, 3), (20, 8),'',TEXT_CLR_1),
                               TextBox((mdl, 3), (18, 6),' Camera Settings  1:      Change FOV2:    Change speed3: Change clipping4:  Reset position5:  Reset settings',TEXT_CLR_2,'      '),
                               )

        # Error messages:
        self.error1 = TextBox((mdl, 3), (50, 1), 'Error, File missing or invalid! Press any key.',SF_ERR_CLR,'  ####')
        self.error2 = TextBox((mdl, 3), (50, 1), 'Error, Not enough textures specified! Press any key.',SF_ERR_CLR,'  ####')

        # Text Prompts:
        self.prompt0 = TextBox((mdl, 3), (50, 1), 'Enter another, or nothing to continue',TEXT_CLR_2,'  ####')
        self.prompt1 = TextBox((mdl, 2), (50, 1), 'Please enter the name of a mesh ending in .obj', TEXT_CLR_2,'  ####')
        self.prompt2 = TextBox((mdl, 2), (50, 1), 'Please enter the name of a image ending in .bmp', TEXT_CLR_2,'  ####')
        self.prompt3 = TextBox((mdl, 3), (50, 1), 'Are you sure you want to quit? (y/n)',SF_ERR_CLR,'  ####')

        self.prompt4 = TextBox((mdl, 3), (50, 1), 'Please enter a new FOV, or nothing to cancel.', TEXT_CLR_2,'  ####')
        self.prompt5 = TextBox((mdl, 3), (54, 1), 'Please enter a new camera speed, or nothing to cancel.', TEXT_CLR_2,'  ####')
        self.prompt6 = TextBox((mdl, 3), (54, 1), 'Please enter a new move speed, or nothing to cancel.', TEXT_CLR_2,'  ####')
        self.prompt7 = TextBox((mdl, 3), (50, 1), 'Please enter a new near clip, or nothing to cancel.', TEXT_CLR_2,'  ####')
        self.prompt8 = TextBox((mdl, 3), (50, 1), 'Please enter a new far clip, or nothing to cancel.', TEXT_CLR_2,'  ####')

        # Normal Init:
        self.meshes = ()
        self.textures = ()
        self.depth = []
        self.lut = genDistanceTable(resolution[0], 2)  # Look-up-table for pixel distances
        self.camera = Camera(near, far, fov, resolution)
        self.frameDelta = 1
        self.clear_depth()
        self.startMenu()

    def startMenu(self):
        """ This function prompts the user to load a mesh and texture for said mesh."""

        objLoaded = False
        texLoaded = False
        texture = ()
        mesh = ()

        print(self.tittleScreen, end='')

        while not objLoaded:
            try:
                # Draw background and prompts:
                print(self.tittleScreen, end='')
                if len(mesh) > 0:
                    self.prompt0.draw('c')
                self.prompt1.draw('c')

                # Get input:
                filename = input(CURSER)

                # Handle input:
                if filename != "":
                    mesh += (loadMesh(filename),)

                elif filename == "" and len(mesh) > 0:
                    objLoaded = True

            except:
                print(self.tittleScreen, end='')
                self.error1.draw('c')
                input(CURSER)

        # Mesh(s) Successfully loaded! continue to loading textures:

        print(self.tittleScreen, end='')

        while not texLoaded:
            try:
                # Draw background and prompts:
                print(self.tittleScreen, end='')
                if len(texture) > 0:
                    self.prompt0.draw('c')
                self.prompt2.draw('c')

                # Get input:
                filename = input(CURSER)

                # Handle input:
                if filename != "":
                    tex = loadBitmap(filename)
                    texture += (tex,)

                elif filename == "" and len(texture) == len(mesh):
                    texLoaded = True

                elif len(texture) == len(mesh) and len(texture) < 0:
                    texLoaded = True

                else:
                    print(self.tittleScreen, end='')
                    self.error2.draw('c')
                    input(CURSER)
            except:
                print(self.tittleScreen, end='')
                self.error1.draw('c')
                input(CURSER)

        self.meshes = mesh
        self.textures = texture

    def pauseMenu(self):
        """ This method handles drawing and interacting with the pause menu. """
        # Draw background
        print(self.tittleScreen, end='')
        for item in self.pauseScreen: item.draw('c')

        # reset the camera's velocity to prevent a massive frame delta
        self.camera.r_velocity = Vect3(0, 0, 0)
        self.camera.velocity =   Vect3(0, 0, 0)

        while not self.camera.active:
            print(CURSER, end='')

            char = getInput()

            # Check conditions:
            if K_ESCAPE in char:
                self.camera.active = True
            elif '1' in char:
                self.cameraPrompt()
                print(self.tittleScreen, end='')
                for item in self.pauseScreen:
                    item.draw('c')
            elif '2' in char:
                self.startMenu()
                print(self.tittleScreen, end='')
                for item in self.pauseScreen:
                    item.draw('c')
            elif '3' in char:
                self.exitPrompt()
                print(self.tittleScreen, end='')
                for item in self.pauseScreen:
                    item.draw('c')

    def cameraPrompt(self):
        """ This method handles all settings prompts for the game's Camera. """
        # Draw background
        print(self.tittleScreen, end='')
        for item in self.cameraSettings:
            item.draw('c')

        while True:
            print(CURSER, end='')
            char = getInput()

            # Check conditions
            if K_ESCAPE in char:
                break

            elif '1' in char:  # update camera FOV:
                self.prompt4.draw('c')
                fov = input(CURSER)
                if fov.isnumeric():
                    self.camera.fov = int(fov)
                print(self.tittleScreen, end='')
                for item in self.cameraSettings: item.draw('c')

            elif '2' in char:  # Update movement speed:
                self.prompt5.draw('c')
                rSpeed = input(CURSER)
                if rSpeed.isnumeric():
                    self.camera.rSpeed = float(rSpeed)

                self.prompt6.draw('c')
                mSpeed = input(CURSER)
                if mSpeed.isnumeric():
                    self.camera.mSpeed = float(mSpeed)

                print(self.tittleScreen, end='')
                for item in self.cameraSettings:
                    item.draw('c')

            elif '3' in char:  # Update near and far clip planes:
                self.prompt7.draw('c')
                near = input(CURSER)
                if near.isnumeric():
                    near = float(near)
                    if near <= 0:
                        near = 0.0001
                    self.camera.near = near
                self.prompt8.draw('c')
                far = input(CURSER)
                if far.isnumeric():
                    self.camera.far = float(far)

                print(self.tittleScreen, end='')
                for item in self.cameraSettings:
                    item.draw('c')

            elif '4' in char:  # Reset position:
                self.camera.position = Vect3(0, 0, 0)
                print('Done!')

            elif '5' in char:  # Reset all properties:
                self.camera.fov = FOV
                self.camera.near = NEARCLIP
                self.camera.far = FARCLIP
                self.camera.mSpeed = MSPEED
                self.camera.rSpeed = RSPEED
                self.camera.position = Vect3(0, 0, 0)
                self.camera.rotation = Vect3(0, 0, 1)
                self.camera.setProperties()
                print('Done!')

        # Assume that a change as been made, only call set method once.
        self.camera.setProperties()

    def exitPrompt(self):
        print(self.tittleScreen, end='')
        self.prompt3.draw('c')
        while True:
            print(CURSER, end='')
            keypress = getInput()
            for char in keypress:
                if char in 'nN':
                    self.camera.active = True
                    return
                elif char in 'yY':
                    terminate()

    def clear_depth(self):
        """ This function clears the depth buffer. """
        # for all every line of the buffer:
        line = [self.camera.far for _ in range(self.camera.resolution[1])]
        self.depth = [line[:] for _ in range(self.camera.resolution[0])]

    def projectPoints(self,index):
        """ This method rotates and projects a mesh to world space. """
        mesh = self.meshes[index]
        # Update normals, clip mesh, rotate and move to camera's view
        points = Mesh(mesh.vertices[:], mesh.polygons[:], mesh.uv_vertices[:], mesh.uv_polygons[:])
        points.move(Vect3(0, 0, 0) - self.camera.position)
        points.rotate_y(self.camera.y_rotation)
        points.rotate_x(self.camera.x_rotation)
        points.updateNormals()

        points = self.camera.clipMesh(points)
        points.update()

        # Project vertices
        for p in points.vertices:
            try:
                inv_z = -1 / p.z
            except ZeroDivisionError:
                inv_z = 0.0001
            p.x = ((p.x * inv_z) * self.camera.h_scale_const) + self.camera.h_center
            p.y = ((p.y * inv_z) * self.camera.v_scale_const) + self.camera.v_center
            p.z = inv_z

        return points

    def run_logic(self):
        # Update Camera:
        self.camera.update(self.frameDelta)

        # Check for exit game / pause menu condition:
        if not self.camera.active:
            self.pauseMenu()

    def render(self, time_start, windowSurface):
        """ This function renders the scene to a surface. """
        # Update lighting vector:
        # it's inefficient, but because I used a class for Vect3 I have to do this to get the methods.
        # if I re-write this I'll do away with the class all together and just use functions.
        lighting = Vect3(LIGHTING[0], LIGHTING[1], LIGHTING[2])
        lighting.rotate_y(self.camera.y_rotation)
        lighting.rotate_x(self.camera.x_rotation)

        # Clear screen:
        self.clear_depth()
        windowSurface.fill((0, 0, 0))

        # Render mesh(s):
        for index in range(len(self.meshes)):

            # Project:
            projected = self.projectPoints(index)

            # Sort by depth from camera:
            indices = list(range(len(projected)))
            indices = QuickSort(projected.depth, indices)[1]

            # Draw the nearest first to avoid doing 1 / z more than is absolutely necessary:
            for i in indices:
                p = projected[i]
                l = p[3].dot(lighting) * 0.5
                texturedPolygon(windowSurface, self.depth, self.lut, p[0], p[1], p[2], l, self.textures[index], self.camera.state)

        windowSurface.s_blit(self.hud)
        print(windowSurface.flip())

        # Calculate real frame delta after writing to the "screen":
        self.frameDelta = (time.perf_counter() - time_start)


def main():
    os.system('')                     # os call to enable ANSI codes:
    print('\033[2;J\033[H\033[25;l')  # Clear screen and reset cursor position:

    windowSurface = Surface(WIDTH, HEIGHT)
    game = Game(FOV, FARCLIP, NEARCLIP, (WIDTH, HEIGHT))

    game.camera.position = Vect3(0, 0, -4)

    while True:
        time_start = time.perf_counter()
        game.run_logic()
        game.render(time_start, windowSurface)


main()

