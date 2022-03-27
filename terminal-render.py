# This program was writen by Ethan Parker.
# This program is a self-contained 3d render engine that runs in the terminal.
# do NOT try to run this program using the standard python IDE as it does not use ANSI escape codes.


# EXPLAINATIONS:
# Due to the time it takes to use the print command, sometimes the screen will refresh while the program is in the
# process of clearing and redrawing the screen. this causes the screen to flash rapidly and the effect worsens the
# higher your refresh rate is. At 60Hz it isn't that bad, but I would still be careful running this if you suffer from
# any conditions that cause light sensitivity.

# the constant, DITHER is a 4x4 bayer matrix. Basically, its a noise map that I use to offset what colour is displayed
# at a given pixel. This aproximates colours that cannot be displayed nativly with the 8 colours used in the terminal.

# There are only 8 colours because to draw two pixels per scan line i used the "▀" character, and set the foreground
# and background colours independantly. The problem is that the upper 8 colours are only for the foreground and so 
# im limited to just the lower 8.

# The Mesh class is a bit useless, it would be more efficient to just store 4 lists and have a bunch of functions to
# handdle appending, removing, and repacing. the clipMesh method is also very slow and one of the main limiting 
# factors when it comes to speed. 


# NOTES:
# if your terminal uses more colours or just different ones, replace the values in the "COLOURS" constant with those
# colours. if you have them as int[0:255], simply divide the R,G,B channels by 255.

import math
import time
import copy
import os

# CONSTANTS:

# If Machine is running on Windows, use cls
if os.name in ('nt', 'dos'): CLS = "CLS"
else: CLS = "clear"

CURSER = '\033[1;37;40m -> '

FOV = 60
NEARCLIP = 0.1
FARCLIP = 64

WIDTH, HEIGHT = 64, 48
FPS = 60
FPMS = FPS * 0.001
LIGHTING = (0, -0.75, 0.25)  # Vector for angle of lighting, must be normalized.
BRIGHTNESS = 0.25

RST = "\033[0;37;40m"
CLR_TO_DEC = 1 / 255
COLOURS = ((0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0),
           (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.75, 0.75, 0.75))

DITHER = (((0 / 16), (8 / 16), (2 / 16), (10 / 16)),
          ((12 / 16), (4 / 16), (14 / 16), (6 / 16)),
          ((3 / 16), (11 / 16), (1 / 16), (9 / 16)),
          ((15 / 16), (7 / 16), (13 / 16), (5 / 16)))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# LOADING FILES


def loadMesh(filename):
    """ This function loads a .obj file and stores it into a mesh object. .obj files are extremely simple. Each line
    consists of a tag followed by the data for that item. for example, the tag 'v' is for vertex and the following
    information should be three floating point numbers stored in raw text, while the tag 'p' stands for polygon,
    and the information should be 3 integers for the index of each vertex that forms that face. This is done to avoid
    repeating vertices. Seriously, if you want to try  something like this on your own just open a .obj in a text
    editor and see how it all goes together. """

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
    try:
        file = open(filename, "rb")
        rawBytes = file.read()

        width = int(rawBytes[18]) + (int(rawBytes[19]) * 256) + (int(rawBytes[20]) * 65536) + (int(rawBytes[21]) * 16777216)
        height = int(rawBytes[22]) + (int(rawBytes[23]) * 256) + (int(rawBytes[24]) * 65536) + (int(rawBytes[25]) * 16777216)

        pixelArray = []
        for x in range(int(rawBytes[10]), len(rawBytes), 3):
            colour = (float(rawBytes[x]) * CLR_TO_DEC, float(rawBytes[x + 1]) * CLR_TO_DEC, float(rawBytes[x + 2]) * CLR_TO_DEC)  # Convert byte to float(0 - 1)
            pixelArray.append(colour)

        pixelArray.reverse()  # Reverse the image to make it apper properly
        # Write data to texture:
        texture = Surface(width, height)
        for y in range(height):
            for x in range(width - 1, -1, -1):
                texture.set_at((x, y), pixelArray[(y * width) + x])

        return texture
    except:
        raise Exception(" Error reading file, file does not exist or is corrupted. Validate that file type is .BMP ")
    

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
        if key == 0: return self.x
        if key == 1: return self.y

    def __setitem__(self, key, value):
            if key == 0: self.x = value 
            elif key == 1: self.y = value

    def update(self, x, y):
        self.x, self.y = x, y

    def length(self):
        return math.sqrt((self.x * self.x) + (self.y * self.y))


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
        if key == 0: return self.x
        elif key == 1: return self.y
        elif key == 2: return self.z

    def __setitem__(self, key, value):
        if key == 0: self.x = value 
        elif key == 1: self.y = value
        elif key == 2: self.z = value

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


def interp(x1, y1, x2, y2, y):
    """ This function interpolates between two points at a given y. """
    try:
        result = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
        return result
    except ZeroDivisionError:
        return x1


def quickInterp(x1, y1, x2, inv_dist, y):
    """ This function interpolates between two points at a given y. """
    try:
        result = x1 + ((x2 - x1) * (y - y1)) * inv_dist
        return result
    except ZeroDivisionError:
        return x1


def QuickSort(sort, index):
    """my implementation of the QuickSort algorithm originally writen by Tony Hoare, 1960."""

    elements = len(sort)

    # Base case
    if elements < 2:
        return sort, index

    current_position = 0

    for i in range(1, elements):
        if sort[i] < sort[0]:
            current_position += 1
            sort[i], sort[current_position], index[i], index[current_position] = sort[current_position], sort[i], index[current_position], index[i]
    sort[0], sort[current_position], index[0], index[current_position] = sort[current_position], sort[0], index[current_position], index[0]

    # recursively sort blocks
    left = QuickSort(sort[0:current_position], index[0:current_position])
    right = QuickSort(sort[current_position + 1:elements], index[current_position + 1:elements])

    # recombine lists into one list
    return sort, left[1] + [index[current_position]] + right[1]


def getNormal(a, b, c):
    """ This function gets the normal vector of a Vect3 or similar object. """
    u, v = b - a, c - a

    normal = Vect3((u[1] * v[2]) - (u[2] * v[1]),
                   (u[2] * v[0]) - (u[0] * v[2]),
                   (u[0] * v[1]) - (u[1] * v[0]))

    if normal != (0, 0, 0):
        normal = normal / normal.length()

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

        if vertices is None: vertices = ()
        if polygons is None: polygons = ()
        if uv_vertices is None: uv_vertices = ()
        if uv_polygons is None: uv_polygons = ()

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
        u_index, v_index, w_index = self.setTextureVertex(a[1]), self.setTextureVertex(b[1]), self.setTextureVertex(c[1])

        if n is None: n = getNormal(a, b, c)
        self.polygons = self.polygons[:key] + ((a_index, b_index, c_index, n),) + self.polygons[key + 1:]
        self.uv_polygons = self.uv_polygons[:key] + ((u_index, v_index, w_index),) + self.uv_polygons[key + 1:]

    def append(self, a, b, c, n=None):
        """ This function handles appending a new trigon to the mesh. """
        a_index, b_index, c_index = self.setVertex(a[0]), self.setVertex(b[0]), self.setVertex(c[0])
        u_index, v_index, w_index = self.setTextureVertex(a[1]), self.setTextureVertex(b[1]), self.setTextureVertex(c[1])

        if n is None: n = getNormal(a, b, c)
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

    def setTextureVertex(self, item):
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
        self.near = near
        self.far = far
        self.clip = self.genClip()

        # Movement:
        self.x_rotation = 0
        self.y_rotation = 0
        self.z_rotation = 0

        # Vectors
        self.position = Vect3(0, 0, 0)
        self.rotation = Vect3(0, 0, 1)
        self.velocity = Vect3(0, -1, 0)

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
        c2 = Vect3(horizontal_far_side_length, -1 * vertical_far_side_length, self.far)
        c3 = Vect3(horizontal_near_side_length, vertical_near_side_length, self.near)
        c4 = Vect3(horizontal_near_side_length, -1 * vertical_near_side_length, self.near)
        c5 = Vect3(-1 * horizontal_far_side_length, vertical_far_side_length, self.far)
        c6 = Vect3(-1 * horizontal_far_side_length, -1 * vertical_far_side_length, self.far)
        c7 = Vect3(-1 * horizontal_near_side_length, vertical_near_side_length, self.near)
        c8 = Vect3(-1 * horizontal_near_side_length, -1 * vertical_near_side_length, self.near)

        # Generate planes from points.
        planes = (Plane(c7, c4, c3), Plane(c1, c5, c7), Plane(c6, c2, c4), Plane(c3, c2, c1),
                  Plane(c5, c6, c8), Plane(c6, c5, c1),)  # Last is far plane

        return planes

    def setProperties(self, near, far, fov, resolution):
        """ This function handles changing different aspects of the camera's behavior. """
        self.near = near
        self.far = far
        self.fov = fov
        self.aspect_ratio = resolution[0] / resolution[1]
        self.scale = 1 / (math.tan(0.5 * math.radians(self.fov)))
        self.h_scale_const = 0.5 * (resolution[0] * self.scale / min((1, self.aspect_ratio)))
        self.v_scale_const = 0.5 * (resolution[1] * self.scale * max((1, self.aspect_ratio)))
        self.h_center = 0.5 * resolution[0]
        self.v_center = 0.5 * resolution[1]
        self.resolution = resolution
        self.deviation = (self.fov / 360)

        self.clip = self.genClip()

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
                a_distance, b_distance, c_distance = plane.pointToPlane(a[0]), plane.pointToPlane(b[0]), plane.pointToPlane(c[0])
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

# SURFACE CLASS AND METHODS


def colourDistance(c1, c2):
    """ This function returns the relative distance between two colours, or rather how similar they are. """
    r1, g1, b1 = c1[0], c1[1], c1[2]
    r2, g2, b2 = c2[0], c2[1], c2[2]

    d = math.sqrt(((r2-r1) * (r2-r1)) + ((g2-g1) * (g2-g1)) + ((b2-b1) * (b2-b1)))
    return d


def nearestColour(colour, dither, comparison):
    colour = (colour[0] + dither, colour[1] + dither, colour[2] + dither)
    checked = []
    
    for index in comparison:
        checked.append(colourDistance(colour, index))
    nearest = min(checked)

    return checked.index(nearest)


class Surface:
    """ This class acts as an array of pixels which can be set using a screen coordinate and a colour value. 
        Once a surface has been defined is properties are immutable and WILL break if changed. """
    def __init__(self, width, height):
        # Variables:
        self.width = width
        self.height = height
        self.surface = []
        self.line = [(0, 0, 0) for _ in range(self.height)]
        self.clear()

        # Stuff for drawing screen border:
        self.tLine = RST + "\n╭"
        for _ in range(self.width): self.tLine += "─"
        self.tLine += "╮"

        line = ""
        for _ in range((self.width // 2) - 6): line += "─"
        self.bLine = "╰" + line + "ETHAN─PARKER" + line + "╯"

    def get_at(self, x, y):
        colour = self.surface[x][y]
        return colour
        
    def set_at(self, xy, colour):
        """ Set the value of a pixel in the surface. """
        try:
            self.surface[xy[0]][xy[1]] = colour
        except IndexError:
            pass
    
    def fill(self, colour):
        """ This function fills a surface with a given colour. """
        self.surface = []
        line = [colour for _ in range(self.height)]
        for _ in range(self.width):
            self.surface.append(copy.deepcopy(line))
    
    def clear(self):
        """ This function clears a surface, writing 0's to each pixel. ever so slightly faster than filling with
        black. """
        self.surface = []
        for _ in range(self.width):
            self.surface.append(copy.deepcopy(self.line))

    def blit(self, surface, x, y):
        for lclY in range(y, min(self.height, (surface.height + y))):
            for lclX in range(x, min(self.width, (surface.width + x))):
                self.surface[lclX][lclY] = surface.surface[lclX - x][lclY - y]

    def flip(self):
        """ This function draws the Surface to the console output. """
        # Draw screen boarder:
        sLine = RST + "│"
        output = (self.tLine,)
        
        # for each line, update and add it to the output:
        for y in range(0, self.height, 2):
            
                line = "" + sLine 
                for x in range(self.width):
                    try: 
                        tc = str(nearestColour(self.surface[x][y], DITHER[x % 4][y % 4] - 0.5 * 0.25, COLOURS))
                        bc = str(nearestColour(self.surface[x][y + 1], DITHER[x % 4][(y + 1) % 4] - 0.5 * 0.25, COLOURS)) 
                        line += "\033[0;3" + tc + ";4" + bc + "m▀"
                    except IndexError:
                        pass
                line += sLine
                output += (line,)
            
        output += (self.bLine,)

        return output
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# DRAWING FUNCTIONS


def interpLine(v1, v2):
    """ This function draws a line on the screen from v1 to v2."""
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
    except ZeroDivisionError: # Slope is undefined, line must be vertical.
        if v1[1] > v2[1]: v1, v2 = v2, v1
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


def drawPoint(surface, point, colour):
    surface.set_at(point, colour)


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
                    surface.set_at((x, y), (0, 0, 0))
                else: 
                    surface.set_at((x, y), colour)


def texturedPolygon(surface, depth, a, b, c, brightness, texture):
    """ This function draws a textured polygon with perspective correction. """

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
 
            for lclX in range(lsOffset - rsOffset):
                x = lclX + rsOffset
                inv_z = interp(ls.z, ls.x, rs.z, rs.x, x)
                if depth[x][y] > inv_z:
                    z = 1 / inv_z
                    u = (interp(tls.x, ls.x, trs.x, rs.x, x) * z) % 1
                    v = (interp(tls.y, ls.x, trs.y, rs.x, x) * z) % 1
                    
                    colour = texture.get_at(int(u * (texture.width - 1)), int(v * (texture.height - 1)))
                    surface.set_at((x, y), (colour[0] - brightness, colour[1] - brightness, colour[2] - brightness))
                    depth[x][y] = inv_z

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# GAME CLASS AND MAINLINE


class Game:
    def __init__(self, mesh, texture, fov, far, near, resolution):
        self.world = mesh
        self.texture = texture
        self.depth = []
        self.camera = Camera(near, far, fov, resolution)
        self.frameDelta = 1
        self.clear_depth()
    
    def clear_depth(self):
        """ This function clears the depth buffer. """
        # for all every line of the buffer:
        line = [self.camera.far for _ in range(self.camera.resolution[1])]
        self.depth = [line[:] for _ in range(self.camera.resolution[0])]
    
    def projectPoints(self):
        """ This method rotates and projects a mesh to world space. """

        # Update normals, clip mesh, rotate and move to camera's view
        points = copy.deepcopy(self.world)
        points.move(Vect3(0,0,0) - self.camera.position)
        points.rotate_x(self.camera.x_rotation)
        points.rotate_y(self.camera.y_rotation)
        points.rotate_z(self.camera.z_rotation)
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

    def render(self, time_start, windowSurface):
        """ This function renders the scene to a surface. """
        
        self.clear_depth()
        projected = self.projectPoints()

        indices = list(range(len(projected)))
        indices = QuickSort(projected.depth, indices)[1]
        
        windowSurface.fill((-1, -1, -1))
       
        for i in indices: 
            p = projected[i]
            brightness = ((p[3].dot(LIGHTING) + 1) * 0.75)
            texturedPolygon(windowSurface, self.depth, p[0], p[1], p[2], brightness, self.texture)

        frame = windowSurface.flip()
        frameDelta = (time.perf_counter() - time_start)
        time.sleep(frameDelta % FPMS)  # Wait for next opportunity to draw to screen:

        # Draw Screen:
        os.system(CLS)
        for line in frame:
            print(line)

        # Calculate real frame delta after writing to the "screen":
        self.frameDelta = (time.perf_counter() - time_start)


# THESE FUNCTIONS ARE FOR THE TITTLE SCREEN AND INPUT.

def drawTitle():
    print('\033[1;37;40m╭─────────────────────────────────────────────────────────────╮')
    print('│\033[1;37;40m   888888 888888 88""Yb 8b    d8 88 88b 88    db    88       \033[1;37;40m│')
    print('│\033[1;31;40m     88   88__   88__dP 88b  d88 88 88Yb88   dPYb   88       \033[1;37;40m│')
    print('│\033[1;33;40m     88   88""   88"Yb  88YbdP88 88 88 Y88  dP__Yb  88  .o   \033[1;37;40m│')
    print('│\033[1;32;40m     88   888888 88  Yb 88 YY 88 88 88  Y8 dP""""Yb 88ood8   \033[1;37;40m│')
    print('│\033[1;37;40m                                                             \033[1;37;40m│')
    print('│\033[1;36;40m          88""Yb 888888 88b 88 8888b.  888888 88""Yb         \033[1;37;40m│')
    print('│\033[1;34;40m          88__dP 88__   88Yb88  8I  Yb 88__   88__dP         \033[1;37;40m│')
    print('│\033[1;35;40m          88"Yb  88""   88 Y88  8I  dY 88""   88"Yb          \033[1;37;40m│')
    print('│\033[1;37;40m          88  Yb 888888 88  Y8 8888Y"  888888 88  Yb         \033[1;37;40m│')
    print('│\033[1;37;40m                                                             \033[1;37;40m│')


def invalidInput():
    os.system(CLS)
    drawTitle()
    print('\033[1;37;40m│      Error, File Not Found! \033[1;31;40mPress any key to continue.\033[1;37;40m      │')
    print('\033[1;37;40m╰─────────────────────────────────────────────────────────────╯')
    input(CURSER)


def main():
    
    
    objLoaded = False
    texLoaded = False

    while not objLoaded:
        try:
            os.system(CLS)
            drawTitle()
            print('\033[1;37;40m│       Please enter the name of a mesh ending in\033[1;31;40m .obj\033[1;37;40m        │')
            print('\033[1;37;40m╰─────────────────────────────────────────────────────────────╯')
            filename = input(CURSER)
            mesh = loadMesh(filename)
            objLoaded = True
        except FileNotFoundError:
            invalidInput()
            objLoaded = False

    while not texLoaded:
        try:
            os.system(CLS)
            drawTitle()
            print('\033[1;37;40m│      Please enter the name of a texture ending in\033[1;31;40m .bmp\033[1;37;40m      │')
            print('\033[1;37;40m╰─────────────────────────────────────────────────────────────╯')
            filename = input(CURSER)
            texture = loadBitmap(filename)
            texLoaded = True
        except:
            invalidInput()
            texLoaded = False

    windowSurface = Surface(WIDTH, HEIGHT)
    game = Game(mesh, texture, FOV, FARCLIP, NEARCLIP, (WIDTH, HEIGHT))
    
    game.camera.position = Vect3(0, 0, -4)
    
    # Main "game" loop.
    while True:
        time_start = time.perf_counter()
        game.world.rotate_x(1)
        game.world.rotate_y(2)
        game.world.rotate_z(3)
        game.render(time_start, windowSurface)


main()

