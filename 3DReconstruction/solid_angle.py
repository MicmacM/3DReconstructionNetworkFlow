import numpy as np
from math import sin, cos, tan

        
class solidAngle():
    def __init__(self, image, cameraPos, cameraRotation, FoV, volumeSize, volumeOrigin, dVolume):
        #Camera
        self.cameraPos = np.array(cameraPos)
        self.cameraRx,self.cameraRy,self.cameraRz = cameraRotation
        self.image = image
        self.imageW, self.imageH = self.image.size
        self.FoV = FoV
        self.FoVY = FoV*self.imageH/self.imageW
        self.imageGlobalWidth = 2*tan(self.FoV/2)
        self.imageGlobalHeight = 2*tan(self.FoVY/2)
        self.dLocalX = self.imageGlobalWidth/self.imageW #variation en x entre deux pixels sur la silhouette
        self.dLocalY = self.imageGlobalHeight/self.imageH #idem en y
        self.dLoc = (self.dLocalX + self.dLocalY) / 2
        self.topLeftPixelVector = np.array([self.cameraPos[0]-sin(self.cameraRz)-cos(self.cameraRz)*tan(self.FoV/2),self.cameraPos[1]+cos(self.cameraRz)-sin(self.cameraRz)*tan(self.FoV/2), self.cameraPos[2]+tan(self.FoVY/2)])-self.cameraPos #vecteur originel
        self.topLeftPixelPos = self.topLeftPixelVector + self.cameraPos
        #Volume
        self.vX, self.vY, self.vZ = volumeSize
        self.x0, self.y0, self.z0 = volumeOrigin
        self.dx, self.dy, self.dz = dVolume
        self.d = (self.dx + self.dy + self.dz)/3
        #Global coord
        self.createConversionDict()
        self.computeGlobalCoordPixel()
        self.shapeFromImage()

    def computeGlobalCoordPixel(self):
        """
        Cree et remplis la matrice contenant les coordonnées dans R^3 des pixels
        en partant du pixel en haut à gauche et en itérant
        """
        self.globalCoordPixel = np.zeros((self.imageH+1,self.imageW+1,3))
        for i in range(self.imageH+1):
            for j in range(self.imageW+1):
                delta = [j*self.dLoc*cos(self.cameraRz) , j*self.dLoc*sin(self.cameraRz) , (-i)*self.dLoc]
                self.globalCoordPixel[i][j] = np.around(self.topLeftPixelPos + np.array(delta),decimals=16)
    
    def createConversionDict(self):
        i = 0
        self.voxel_to_vertex, self.vertex_to_voxel = {}, {}
        for p in self.voxel():
            self.voxel_to_vertex[p] = i
            self.vertex_to_voxel[i] = p
            i += 1
        return self.voxel_to_vertex, self.vertex_to_voxel
    
    def voxel(self):
        """
        Returns : 
            generator_object : contenant tous les couples (x,y,z) de vertex du volume w*d*h
        """
        for z in range(self.vZ):
            for y in range(self.vY):
                for x in range(self.vX):
                    yield (x,y,z)
                    
    
    def vertexAbovePlane(self, planVector1, planVector2, vertex):
        """
        Indique si le point est au dessus du plan engendré par v1 et v2 en regardant le signe du déterminant

        Arguments : 
            planVector1 (np.array) : 3x1 numpy array representant v1
            planVector2 (np.array) : 3x1 numpy array representant v2
            vertex (list) : coordonnees cartésiennes du vertex [x,y,z]

        Returns : 
            bool: le vertex est au dessus du plan
        """
        return np.linalg.det(np.array([planVector1,planVector2,np.array(vertex)-self.cameraPos])) > 0

    def shapeFromImage(self):
        """
        Renvoie les coordonnées des pixels de l'image appartenant à l'objet
        L'image utilisée est en fait un masque binaire renvoyé par la segmentation de l'image 2D
        """
        imageArray = np.array(self.image)
        self.shapeArray = []
        for i in range(len(imageArray)):
            for j in range(len(imageArray[0])):
                if imageArray[i][j] == 255:
                    self.shapeArray.append([j,i]) #on inverse car on passe en coordonnées locales image

    def vertexInsideSolidAngle(self,pixel):
        """
        On regarde si le vertex est compris dans la pyramide de sommet l'origine de la caméra et de base le pixel

        Arguments : 
            pixel (list) : [j,i] renvoyé par shapeFromImage(self)

        Returns :
            vertexList (list) : liste des pixels dans l'angle solide

        """
        vertexList = []
        
        if pixel[1] > 0 and pixel[1] < self.imageH and pixel[0] > 0 and pixel[0] < self.imageW:
            topleft = self.globalCoordPixel[pixel[1]][pixel[0]] - self.cameraPos
            topright = self.globalCoordPixel[pixel[1]][pixel[0]+1] - self.cameraPos
            botleft = self.globalCoordPixel[pixel[1]+1][pixel[0]] - self.cameraPos
            botright = self.globalCoordPixel[pixel[1]+1][pixel[0]+1] - self.cameraPos
            for (x,y,z) in self.voxel(): 
                point = [self.x0 + x*self.d, self.y0 + y*self.d, self.z0 + z*self.d]
                if self.vertexAbovePlane(botleft,topleft,point): #plan de gauche
                    if self.vertexAbovePlane(topleft,topright,point): #plan du haut
                        if self.vertexAbovePlane(botright,botleft,point): #plan du bas
                            if self.vertexAbovePlane(topright,botright,point): #plan de droite
                                vertexList.append(self.voxel_to_vertex[(x,y,z)])
        return vertexList

    def vertexInsideShape(self):
        """
        Pour chaque pixel de la silhouette, on regarde quels sont les vertex dans l'angle solide

        Returns : 
            vertexSet (set) : ensemble des vertex dans l'angle solide de la silhouette
        """
        vertexSet = set()
        for pixel in self.shapeArray:
            inside = self.vertexInsideSolidAngle(pixel)
            for voxelId in inside:
                vertexSet.add(voxelId)
        return vertexSet