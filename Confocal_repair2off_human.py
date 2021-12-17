import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import itertools
import seaborn as sns
import pandas as pd
import os.path
from fnc import *
import trimesh

# Human 40 apical:
##SpineID2longs = pd.read_pickle("SpineID2longs_Apical-Human.pkl")
##path = 'E:\\Confocal-Spines\\Human\\40\\apical\\off_files'

# Human 40 basal:
SpineID2longs1 = pd.read_pickle("SpineID2longs_Basal-Human-Repair.pkl")
SpineID2longs2 = pd.read_pickle("SpineID2longs_Human40-cing-basal-more20.pkl")
SpineID2longs = pd.concat([SpineID2longs1, SpineID2longs2], axis=0)
path = 'E:\\Confocal-Spines\\Human\\40\\basal\\off_files'

# Human 85 apical:
##SpineID2longs = pd.read_pickle("SpineID2longs_Human85-cing-apical.pkl")
##path = 'E:\\Confocal-Spines\\Human\\85\\apical\\off_files'

# Human 85 basal:
##SpineID2longs = pd.read_pickle("SpineID2longs_Human85-cing-basal.pkl")
##path = 'E:\\Confocal-Spines\\Human\\85\\basal\\off_files'


files_list = []
for root, subFolders , files in os.walk(path):
  for f in files:
    if f.endswith('.off'):
      files_list.append('%s\\%s' % (root , f))
files_list.sort(key=natural_keys)


for path in files_list:
    mesh = trimesh.load(path, force='mesh')
    mesh.vertices[:,2] = mesh.vertices[:,2] * 0.84 # Z-distortion correction
    if len(mesh.split()) > 2:
        continue
    if len(mesh.split())==2:
        meshA , meshB = mesh.split()
        A = np.mean(meshA.vertices, axis=0)
        B = np.mean(meshB.vertices, axis=0)
        lenA = []
        for i in range(len(meshB.vertices)):
            lenA.append(distance3d(meshB.vertices[i][0], meshB.vertices[i][1], meshB.vertices[i][2], A[0], A[1], A[2]))
        lenB = []
        for j in range(len(meshA.vertices)):
            lenB.append(distance3d(meshA.vertices[j][0], meshA.vertices[j][1], meshA.vertices[j][2], B[0], B[1], B[2]))
        BB = meshB.vertices[np.argmin(lenA)]
        AA = meshA.vertices[np.argmin(lenB)]
        vertsA = generate_sphere_vertices( AA, rad=0.17 , n=9)
        vertsB = generate_sphere_vertices( BB , rad=0.17, n=9)
        vrts   = np.vstack( [ vertsA, vertsB  ] )
        pc      = trimesh.PointCloud( vrts )
        lng_cls    = pc.convex_hull
        ##
        vertices = lng_cls.vertices
        faces = lng_cls.faces
        new_vertices , new_faces = trimesh.remesh.subdivide(vertices, faces, face_index=None, vertex_attributes=None)
        new_vertices , new_faces = trimesh.remesh.subdivide(new_vertices, new_faces, face_index=None, vertex_attributes=None)
        lng_cls = trimesh.Trimesh(new_vertices, new_faces)
        ##
        mesh = trimesh.boolean.union([mesh, lng_cls] , engine='blender')
    if len(mesh.split())==1:
        A = SpineID2longs['longs'][path.split("\\")[-1].split(".of")[0]][-1]
        A[2] = A[2] * 0.84
        disA = []
        for i in range(len(mesh.vertices)):
            disA.append(distance3d(mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2], A[0], A[1], A[2]))
        if min(disA) > 0.2:
            closest_vert = mesh.vertices[np.argmin(disA)]
            vertsB = generate_sphere_vertices( closest_vert, rad=0.17 , n=9) # 0.17 is the median human neck radius
            vertsA = generate_sphere_vertices( A, rad=0.17 , n=9)
            vrts   = np.vstack( [ vertsA, vertsB  ] )
            pc      = trimesh.PointCloud( vrts )
            lng_cls    = pc.convex_hull
            ##
            vertices = lng_cls.vertices
            faces = lng_cls.faces
            new_vertices , new_faces = trimesh.remesh.subdivide(vertices, faces, face_index=None, vertex_attributes=None)
            new_vertices , new_faces = trimesh.remesh.subdivide(new_vertices, new_faces, face_index=None, vertex_attributes=None)
            lng_cls = trimesh.Trimesh(new_vertices, new_faces)
            mesh = trimesh.boolean.union([mesh, lng_cls] , engine='blender')
    vertices = mesh.vertices
    faces = mesh.faces
    ## writing a new off file:
    f = open("%s\\off_files_connected/%s.off" % (path.rsplit('\\',2)[0], path.split("\\")[-1].split(".o")[-2]) , "w+")
    f.write("COFF\n")
    f.write("{a} {b} 0\n".format(a=len(vertices) , b=len(faces)))
    for i in range(len(vertices)):
        f.write("{a} {b} {c}\n".format(a=vertices[i][0] , b= vertices[i][1] , c= vertices[i][2]))
    for i in range(len(faces)):
        f.write("3 {a} {b} {c}\n".format(a=faces[i][0] , b=faces[i][1] , c=faces[i][2]))
    f.close()
