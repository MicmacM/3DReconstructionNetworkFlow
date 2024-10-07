# -*- coding: utf-8 -*-
from math import exp, sqrt, pi, log
import numpy as np
from PIL import Image
from collections import Counter
import networkx as nx
from networkx.algorithms.flow import *
from scipy.ndimage import convolve1d
import time
import image_drawing





def norm_pdf(x, mu, sigma):
    factor = (1 / (abs(sigma) * sqrt(2 * pi)))
    return factor * exp( -(x-mu)**2 / (2 * sigma**2))

class SegmentedImage(object):
    def __init__(self, image_path, OBJECT, BACKGROUND):
        self.img = Image.open(image_path)
        self.img = self.img.convert("RGB")
        self.w, self.h = self.img.size

        self.pixel_values = {p: self.img.getpixel(p) for p in self.pixels()}

        # SELECTION DE L'UTILISATEUR
        self.obj_seeds = OBJECT
        self.back_seeds = BACKGROUND
        # CONSTANTES
        self.LAMBDA = 0.2
        self.SIGMA = 50
        self.PX_SMOOTHING = 2
        # UTILITAIRE
        self.utilitaire()

        # CORPS DU PROGRAMME
        self.computeHistograms(self.obj_seeds,self.back_seeds)
        self.calculateBoundaryCosts()
        self.calculateRegionalCosts()
        self.createGraph()
    
    def pixels(self):
        """
        Returns :
            generator_object : contenant tous les couples (x,y) de [0,w]*[0,h]
        """
        for x in range(self.w):
            for y in range(self.h):
                yield (x, y)

    def utilitaire(self): #Conversion Image <-> Graphe
        i = 1
        self.pixel_to_vertex, self.vertex_to_pixel = {}, {}
        for p in self.pixels():
            self.pixel_to_vertex[p] = i
            self.vertex_to_pixel[i] = p
            i += 1
        return self.pixel_to_vertex, self.vertex_to_pixel
    
    def neighbours_8(self, x, y):
        """
        Arguments :
            x (int) : abscisse du pixel
            y (int) : ordonnée du pixel
        Returns :
            generator_object : generateur des 4 voisins du pixel pour
            former un 8-voisinage
        """
        return ((i, j) for (i, j) in [(x+1, y), (x, y+1), (x+1, y+1), (x+1, y-1)] if 0 <= i < self.w and 0 <= j < self.h and (i != x or j != y))
    
    def distance(self,p_a,p_b):
        """
        Arguments :
            p_a : pixel
            p_b : pixel
        Returns :
            float : distance euclidienne entre deux pixels
        """
        return abs(p_a[0] - p_b[0]) + abs(p_a[1] - p_b[1])
    
    def i_delta(self,a,b,fun):
        """
        Arguments :
            a (r,g,b)
            b (r,g,b)
            fun (int) : determine la fonction de calcul de i_delta
        Returns :
            float : i_delta
        """
        match fun:
            case 0:
                return self.L(*a) - self.L(*b)
            case 1:
                return sum([abs(a[i] - b[i]) for i in range(len(a))])
            case 2:
                return sum([255/(1+ abs(a[i] - b[i])) for i in range(len(a))])
            case 3:
                return sum([a[i] + b[i] for i in range(len(a))])
            case 4:
                return sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))]))
            case 5:
                return sqrt(sum([v**2 for v in self.vectorial_product(a,b)]))

    def L(self,R,G,B):
        """
        Renvoie la couleur d'un pixel en noir et blanc perçue par l'oeil humain
        """
        return int(R * 299/1000 + G * 587/1000 + B * 114/1000)
    
    def vectorial_product(self,a,b):
        (xa,ya,za),(xb,yb,zb) = a,b
        return (ya*zb - yb*za, xb*za-xa*zb, xa*yb - ya*xb)
    
    ################ CALCUL DU COUT DE FRONTIERE ################

    def B_pq(self, p_a, p_b):
        """
        Arguments :
            p_a : pixel (xa,ya)
            p_b : pixel (xb,yb)
        Returns :
            float : cout de frontiere entre les deux pixels
        """
        delta = self.i_delta(self.pixel_values[p_a], self.pixel_values[p_b], 3)
        return exp(- delta**2 / (2 * self.SIGMA**2)) / self.distance(p_a,p_b)
    

    def calculateBoundaryCosts(self):
        self.boundary_costs = {}
        for p in self.pixels():
            self.boundary_costs[p] = {}
            for n_p in self.neighbours_8(*p):
                self.boundary_costs[p][n_p] = self.B_pq(p, n_p)

        # Calcul de K
        self.K = 1. + max(sum(self.boundary_costs[p].values()) for p in self.pixels())

    
    ################ CALCUL DU COUT DE REGION ################

    def getHistogram(self, points):
        """
        Arguments :
            points : int array in range(0,256)
        Returns :
            dict : probabilité normalisée, telle qu'aucune valeur ne soit nulle
        """
        values_count = Counter(self.L(*self.pixel_values[p]) for p in points)
        for intensite in range(256):
            if intensite not in values_count.keys():
                values_count[intensite] = 1/256
        return {value : float(count) / (len(points)) for value, count in values_count.items()}
    
    def gaussian_smoothing(self,histogram, sigma=1.0):
        """
        Lissage de l'histogramme avec un noyau gaussien
        Arguments :
            histogram (array) : Input histogram.
            sigma (float, optional): std of the Gaussian kernel
        Returns :
            array : Smoothed histogram
        """
        histogram = np.array(histogram, dtype=np.float32)
        kernel = np.exp(-0.5 * np.arange(-5 * sigma, 5 * sigma + 1)**2 / sigma**2)
        smoothed = convolve1d(histogram, kernel, mode='reflect') / kernel.sum()
        return smoothed
    
    def computeHistograms(self,obj,bkg):
        self.bkgh = self.getHistogram(bkg)
        self.objh = self.getHistogram(obj)
        intensite = [i for i in range(255)]
        self.bkg_hist = self.gaussian_smoothing([self.bkgh[i] for i in intensite if i in self.bkgh.keys()], self.PX_SMOOTHING)
        self.obj_hist = self.gaussian_smoothing([self.objh[i] for i in intensite if i in self.objh.keys()], self.PX_SMOOTHING)

    def R_p(self, point, hist):
        """
        Arguments :
            point : pixel (x,y)
            hist : array histogram
        Returns :
            float : cout de région du pixel de coordonnées (x,y)
        """
        Ip = self.L(*self.pixel_values[point])
        proba = hist[Ip]
        return - self.LAMBDA * log(proba)
    
    def calculateRegionalCosts(self):
        self.regional_penalty_obj = {p: 0 if p in self.obj_seeds else self.K if p in self.back_seeds else self.R_p(p,self.obj_hist) for p in self.pixels()}
        self.regional_penalty_bkg = {p: self.K if p in self.obj_seeds else 0 if p in self.back_seeds else self.R_p(p,self.bkg_hist) for p in self.pixels()}

    def createGraph(self):
        self.graph = nx.Graph()
        g = self.graph

        #Creation de P
        for p in self.pixels():
            g.add_node(self.pixel_to_vertex[p])

        #Creation de la source et du puit
        self.s = 0
        self.t = self.w*self.h + 1
        g.add_node(self.s)
        g.add_node(self.t)

        #Creation des aretes de frontière:
        for x in range(self.w):
            for y in range(self.h):
                p = (x,y)
                for n_p in self.neighbours_8(x,y):
                    g.add_edge(self.pixel_to_vertex[p], self.pixel_to_vertex[n_p], capacity=self.boundary_costs[p][n_p])

        #Creation des aretes de région:
        for p in self.pixels():
            g.add_edge(self.s, self.pixel_to_vertex[p], capacity=self.regional_penalty_obj[p])
            g.add_edge(self.pixel_to_vertex[p], self.t, capacity=self.regional_penalty_bkg[p])

    def cut(self):
        cut = nx.minimum_cut(self.graph, self.s, self.t,
        flow_func=shortest_augmenting_path)
        return cut[1]
    
    def generate_mask(self, partition):
        S,T = partition
        imarray = np.zeros((self.h,self.w), dtype=np.uint8)
        for vertex in T:
            if vertex != self.t and vertex != 0:
                (x,y) = self.vertex_to_pixel[vertex]
                imarray[y][x] = 255
        return Image.fromarray(imarray,mode='L').convert('1')





if __name__ == "__main__":
    t0 = time.time()
    IMAGE_URL = "front1.png"

    bkg,obj = image_drawing.interactive_segmentation(IMAGE_URL)
    bkg = bkg[0]
    obj = obj[0]

    img = SegmentedImage(IMAGE_URL,obj, bkg)
    partition = img.cut()
    mask = img.generate_mask(partition)
    mask.show()
    mask.save("maskman.png")
    print("Segmentation time :", time.time() - t0)
