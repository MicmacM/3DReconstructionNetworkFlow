import numpy as np
import networkx as nx
from math import exp, radians
from PIL import Image
from solid_angle import solidAngle

class VoxelModel(object):
    def __init__(self, shapeList, cameraPos, cameraRot, cameraDist, FoV, volumeSize, volumeOrigin, dVolume):
        """
        Parameters
        ----------
        shapeList : PIL.Image_object array Liste contenant les objets images de chaque photo
        cameraPos : (float array) tableau contenant pour chaque camera sa position suivant les 3 axes
        cameraRot : (float array) tableau contenant pour chaque camera sa rotation en radians suivant les axes
        cameraDist : float distance fixe de la camera à l'objet sur toutes les photos
        FoV : float (radians) FoV de la caméra
        volumeSize : int array [width, depth, height] nombre de point selon chaque axe pour la reconstruction
        volumeOrigin : float array : [x0, y0, z0] position du premier point de la reconstruction
        dVolume : float array : [dx, dy, dz] espacement des points de la reconstruction suivant les 3 axes
        """
        self.radius = cameraDist
        self.cameraRot = cameraRot
        self.w, self.d, self.h = volumeSize #Garder w*d*h <= 200 000
        self.volumeSize = [self.w, self.d, self.h]
        self.volumeOrigin = volumeOrigin
        self.dVolume = dVolume
        self.nbCamera = len(shapeList)
        self.cameraPos = cameraPos
        self.shapeList = shapeList
        self.FoV = FoV
        #Constantes
        self.REGIONALPENALTYBKG = 255
        self.LAMBDA = -1.1
        self.ALPHA = 0.1
        self.NU = 1.4
        self.createConversionDict()
        self.calculate_Ov()

    def voxel(self):
        """
        Returns : 
            generator_object : contenant tous les couples (x,y,z) de vertex du volume w*d*h
        """
        for z in range(self.h):
            for y in range(self.d):
                for x in range(self.w):
                    yield (x,y,z)

    def createConversionDict(self): #Conversion Volume <-> Graphe
        """
        Crée deux dictionnaires de conversion voxel(volume) en vertex(graphe)
        """
        i = 0
        self.voxel_to_vertex, self.vertex_to_voxel = {}, {}
        for p in self.voxel():
            self.voxel_to_vertex[p] = i
            self.vertex_to_voxel[i] = p
            i += 1
        return self.voxel_to_vertex, self.vertex_to_voxel

    def neighbours6(self, x, y, z): #les 6 voisins, on en calcule seulement 3.
        """
        Arguments :
            x, y, z : coordonnées du voxel
        
        Returns :
            (tuple comprehension) : tuple contenant toutes les coordonnées des voisins accessibles dans un voisinage 6 (3D)
        """
        return ((i, j, k) for (i, j, k) in [(x+1, y, z), (x, y+1, z), (x, y, z+1)]
                 if 0 <= i < self.w and 0 <= j < self.d and 0 <= k < self.h and (i != x or j != y or k != z))

    def regionalPenaltyObj(self, voxel):
        n = self.O[self.voxel_to_vertex[voxel]]
        return self.LAMBDA * (1 - exp(n / self.NU))
    
    def boundaryPenalty(self, voxel1, voxel2):
        N1 = self.O[self.voxel_to_vertex[voxel1]]
        N2 = self.O[self.voxel_to_vertex[voxel2]]
        return self.ALPHA * N1 * N2 / (self.avg_O_v)**2
    
    def calculate_Ov(self):
        """
        Calcul du nombre de camera voyant chaque voxel
        Arguments :
            O (dict) : dictionnaire contenant O[voxel] = NbCamera voyant le voxel
            avg_O_v (float) : valeur moyenne du nombre de caméra voyant un voxel aléatoire
        """
        self.O = dict()
        self.avg_O_v = 0
        for vertex in range(self.w*self.h*self.w):
            self.O[vertex] = 0
        for i in range(self.nbCamera):
            s = solidAngle(self.shapeList[i], self.cameraPos[i],self.cameraRot[i], self.FoV, self.volumeSize, self.volumeOrigin, self.dVolume)
            vertexSet = s.vertexInsideShape()
            self.DEBUGGING = vertexSet
            
            for vertex in vertexSet:
                self.O[vertex] += 1
                self.avg_O_v += 1
        self.avg_O_v /= len(self.O.keys())
    
    def createGraph(self):
        self.graph = nx.Graph()
        g = self.graph

        #Creation du volume de reconstruction : (ensemble V)
        for v in self.voxel():
            g.add_node(self.voxel_to_vertex[v])

        self.s = self.w*self.h*self.d + 2 #le noeud s ne prend pas la valeur 0 car la numérotation des voxel commence à 0
        self.t = self.w*self.h*self.d + 1
        g.add_node(self.s) 
        g.add_node(self.t) 

        #Creation de E:
        for v in self.voxel():
            #Aretes de frontière
            vert = self.voxel_to_vertex[v]
            for n_v in self.neighbours6(*v):
                    neigh = self.voxel_to_vertex[n_v]
                    g.add_edge(vert, neigh, capacity = self.boundaryPenalty(v, n_v)) #, capacity=self.boundary_costs[p][n_p])
            #Aretes de région
            g.add_edge(self.s, vert, capacity = self.regionalPenaltyObj(v))
            g.add_edge(vert, self.t, capacity = self.regionalPenaltyBkg)
        

    def cut(self):      
        self.cut = nx.minimum_cut(self.graph, self.s, self.t)
        self.cut = self.cut[1]
        self.object = [self.vertex_to_voxel[v] for v in self.cut[0] if v != self.s]
        self.background = [self.vertex_to_voxel[v] for v in self.cut[1] if v != self.t]
        return self.background, self.object
    

if __name__ == "__main__":
    shapeList = []
    li_url = ["lapleft2.png","lapright2.png","lapback2.png","lapfront2.png","lapfrontright2.png","lapfrontleft2.png","lapbackright2.png","lapbackleft2.png"]
    for i in range(len(li_url)):
        imagei = Image.open(li_url[i])
        shapeList.append(imagei)
    w, d, h = 28*2 , 18*2, 24*2
    
    cameraDist = 0.75
    cameraPos = [[0    , -0.75 ,0.15], 
                 [0    , 0.75  ,0.15],
                 [-0.75, 0     ,0.15],
                 [0.75 , 0     ,0.15],
                 [0.530330 , 0.530330  ,0.15],
                 [0.530330 , -0.530330 ,0.15],
                 [-0.530330, 0.530330  ,0.15],
                 [-0.530330, -0.530330 ,0.15]]
    cameraRot = np.radians([
                 [90,0,0], 
                 [90,0,180],
                 [90,0,270],
                 [90,0,90],
                 [90,0,135],
                 [90,0,45],
                 [90,0,225],
                 [90,0,315]])
    FoV = radians(45)
    volumeSize = [w,d,h]
    dx, dy, dz = 0.34993/(w-1), 0.218706/(d-1)  ,  0.284318/(h-1)
    volumeOrigin = [-0.170, -0.109, 0]
    dVolume = [dx, dy, dz]
    
    im2 = Image.open("lapfront.png")
    shapeList2 = [im2]
    cameraPos2 = [cameraPos[3]]
    cameraRot2 = [cameraRot[3]]
    
    
    reconstruction = VoxelModel(shapeList, cameraPos, cameraRot, cameraDist, FoV, volumeSize, volumeOrigin, dVolume)
    T = reconstruction.debug()
    background,obj = reconstruction.cut()








