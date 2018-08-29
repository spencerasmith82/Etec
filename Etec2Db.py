#This python module will contain all the functions and classes needed to implement the 2D Etec algorithm
#It also includes (*v2*) the modifications to enable construction of the weight transition matrix (which allows us to find the stretching of each initial edge - working in the mode where every initial edge is given weight one)
#(*v3*) speeds up the algorithm (cut out some computationally expensive parts of the code)
#(*V4*) Uses a constrained Delaunay triangulation when a band is passed in (i.e. not in the initial net configuration)
#(*v5b*) similar to v5a, but with the constrained Delaunay triangulation (splitting the versions into a and b since b requires an additional module ... triangle)

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import triangle  #this is from: http://dzhelil.info/triangle/delaunay.html ... unfortunately not a standard module 
#change this whenever the scipy.spatial implements a constrained Delaunay triangulation.
from operator import itemgetter, attrgetter  #for making sorting more efficient.
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from scipy.optimize import curve_fit


#To more flexibly interact with the core Etec2D algorithm, I have been doing much of the high-level control in separate ipython notebook files.  However, this function is a good template to base further uses of Etec2D on (I'll put some additional commented-out ideas in this function).  I'll restrict this function to taking in a file name that has a particular structure (like a matlab .mat file with a particular structure), and outputting the weights vs. time list and the estimated topological entropy and error
def GetWeightsAndEntropy(filename):

    filein_contents = sio.loadmat(filename)  #load the matlab file in a local structure
    times = filein_contents['times'][:,0].tolist()  #the makes a list of the trajectory times
    xin = filein_contents['x_coordinates'].tolist() #list of lists for each trajectory point for each time (x position)
    yin = filein_contents['y_coordinates'].tolist()
    numtimes = len(times)
    numtraj = len(xin)
    traj = []
    for t in range(0,numtimes):
        traj.append([[xin[i][t],yin[i][t]] for i in range(0,numtraj)])   #put all of this in the trajectory structure we use
    #Here is another way to load a trajectory file ... in this case just a space separated text file of my own data
    #[times,traj] = OpenTrajectoryFile(filename)  #note for me ... this data usually needs to be downsampled
        
    tri = triangulation2D(traj[0])  #initialize the triangulation ... can also initialize with a set of edges having non-zero weight:
    #tri = triangulation2D(traj[0],ptpairs)  #where ptpairs is a list of pairs of adjacent points (adjacent in the initial Delaunay triangulation) i.e. [[p1,p2],[p3,p4],...].  Repeated point pairs will add weight to that initial edge (integer weight corresponding to the number of times the pair is included in the list)
    #tri = Et.triangulation2D(traj[0],None,rbands)  # where rbands is a list of rubber bands, each represented by a list of points in counter clockwise orientation (important!), where each adjacent pair in the list represents an edge in the initial triangulation (and wraps around).
    #tri = Et.triangulation2D(traj[0],ptpairs,rbands)  # this allows for the anchored points of ptpairs and for rubber bands simultaneously.
    #tri.TriangulationPlot()  #plot the initial Delaunay triangulation or
    #tri.TriangulationPlotWhole()  #use this to see the extra auxiliary points that bound the actual data points
    Weights = []
    Weights.append(tri.GetWeightTotal())
    skipprint = 20     #only print the info to screen every skipprint intervals
    numsteps = numtimes #can vary the fraction of the trajectory you analyze (set to numtimes for the full trajectory)
    for i in range(1,numsteps):
        tri.Evolve(traj[i])
        #could print the triangulation plot with non-zero weights identified to file
        #tri.TriangulationPlotWeights(name)  #where name is the file name string (if path contains a folder, make sure this folder exists first)
        #could print these out with sequential names so that they can be stitched into a movie externally
        Weights.append(tri.GetWeightTotal())  #This repeatedly evolves the triangulation forward
        if i%skipprint == 0:
            print("Evolution at timestep: ",i,", time: ",times[i], ", with weight: ",Weights[-1])
    #now output the weights and times to file if wanted (will comment this out)
    #file = open("WeightsOut.dat","w")
    #for i in range(0,len(Weights)):
    #    file.write(str(times[i])+" "+str(Weights[i]))
    #file.close()
    
    #now calculate the bound on the topological entropy
    LWeights = [math.log(Weights[i]) for i in range(0,len(Weights))]
    indend = len(Weights)-1   #the fitting will by default go all the way to the end.  Can change if wanted
    fracstart = 100           #the end index divided by this will be the starting index
    indst = int(indend/fracstart)   #not set to zero ... will assume some transistory behavior, and start at a fraction of the final index
    popt, pcov = curve_fit(linear_func, times[indst:indend], LWeights[indst:indend])  #fitting to a linear function ax+b
    #popt has the optimal fits for a and b (in that order), and pcov has the covariance
    perr = np.sqrt(np.diag(pcov))  #the one standard deviation errors
    
    return [[popt[0],perr[0]],[times,Weights]]
    
#just a quick definition for the linear fit of the log of the weights    
def linear_func(x, a, b):
    return a*x+b
    

#The simplex class.  Each simplex object will hold the point ID of each of the 3 points that define the simplex.  Their order (up to even permutation) will by convention reflect the x-y coordinate system handedness (RHR -> positive z direction).  The class also holds a reference to each of the three adjacent simplices (in the same order as the points they are across from).  This data will be initialized after each object is created (points at the time of object instantiation, and simplex references after all simplices in a triangulation are created).  Finally, each simplex holds the weights of each edge (same order as the points they are opposite to).  So each weight is represented in the triangulation twice
class simplex2D:
    
    _count = 0
    
    #IDlist is a simple list of the 3 point IDs from the master list (as part of the tranguation class).  It is assumed that IDlist already refers to points in the proper permutation (this need to be checked before being passed in).
    def __init__(self, IDlist):
        simplex2D._count += 1
        self.points = []
        for i in range(0,len(IDlist)):
            self.points.append(IDlist[i])
        #initialize the list that will hold references to the correct opposite simplices ... for now have None as placeholders
        self.simplices = [None,None,None]
        #initialize the weights (for the rubber net initial conditions, all weights are 1)
        self.weights = [1,1,1]
        #initialize the id for each edge (same order as the weights) to None.  These will be filled in in the constructor for the triangulation.
        self.edgeids = [None,None,None]
        self.SLindex = None   #This is needed to be able to quickly retrieve the position in the big simplist of this simplex.  It will hold the index of this simplex in the simplist
    
    def __del__(self):
        simplex2D._count -= 1
    
    def GetCount(self):
        return simplex2D._count
    
    def __eq__(self, other):
        return self is other  #I'm just doing this to have all comparisions be by object id, not value.  Now can use some of the functionality of a list of simplex objects, like remove, more freely
    
    #this function takes an input point id and outputs the internal id of this point (0,1,2)
    def LocalID(self, IDin):
        stpt = -1
        for i in range(0,len(self.points)):
            if IDin == self.points[i]:
                stpt = i
                break
        if not stpt == -1:
            return stpt
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None
    
    #This returns a list comprised of two elements: each a reference to a simplex adjacent to the given point in this simplex (and sharing an edge).  There are two of these (possibly the same), and they are ordered in a CCW manner: simplex1, this simplex, simplex2.  The input is an ID (assumed to be one of the set for this simplex), so we need to search for the point
    def AdjSimp(self, IDin):
        simp2 = []
        stpt = self.LocalID(IDin)
        if not stpt == None:
            simp2.append(self.simplices[(stpt+2)%3])
            simp2.append(self.simplices[(stpt+1)%3])
            return simp2
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None
    
    #returns the reference to the simplex opposite the given ID.  Again, the ID must be checked to see if it is contained in this simplex
    def OppSimp(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            return self.simplices[stpt]
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None      
        
    #This returns the neighboring simplex which shares the edge to the right of the given point (IDin)
    def RightSimp(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            return self.simplices[(stpt+2)%3]
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None         
        
    #This returns the neighboring simplex which shares the edge to the left of the given point (IDin)
    def LeftSimp(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            return self.simplices[(stpt+1)%3]
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None
        
    #This returns the number of edges that emanate from a given point in this simplex.  Alternatively, this counts the number of simplices attached to this point.
    def NumSimpNeighbors(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            simpcounter = 1
            lsimp = self.simplices[(stpt+1)%3]
            while (not self is lsimp) and (not lsimp is None):
                simpcounter += 1
                nsimp = lsimp.LeftSimp(IDin)
                lsimp = nsimp
            if lsimp is None:  #this deals with the boundary simplex case
                rsimp = self.simplices[(stpt+2)%3]
                while (not self is rsimp) and (not rsimp is None):
                    simpcounter += 1
                    nsimp = rsimp.RightSimp(IDin)
                    rsimp = nsimp
            return simpcounter
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None        
        
    #This returns the sum of the edge weights around the given point.
    def PointWeightSum(self, IDin):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            weightcounter = self.weights[(stpt+1)%3]
            lsimp = self.simplices[(stpt+1)%3]
            while (not self is lsimp) and (not lsimp is None):
                npt = lsimp.LocalID(IDin)
                weightcounter += lsimp.weights[(npt+1)%3]
                nsimp = lsimp.simplices[(npt+1)%3]
                lsimp = nsimp
            if lsimp is None:  #this deals with the boundary simplex case
                weightcounter += self.weights[(stpt+2)%3]
                rsimp = self.simplices[(stpt+2)%3]
                while (not self is rsimp) and (not rsimp is None):
                    npt = rsimp.LocalID(IDin)
                    weightcounter += rsimp.weights[(npt+2)%3]
                    nsimp = rsimp.simplices[(npt+2)%3]
                    rsimp = nsimp
            return weightcounter
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None 
    
    #This returns a list of all the simplices (in CCW cyclical order) adjacent to a point (IDin).
    def SimpNeighbors(self, IDin):
        NeighborList = []
        stpt = self.LocalID(IDin)
        if not stpt == None:
            NeighborList.append(self)
            lsimp = self.simplices[(stpt+1)%3]
            while (not self is lsimp) and (not lsimp is None):
                NeighborList.append(lsimp)
                nsimp = lsimp.LeftSimp(IDin)
                lsimp = nsimp
            if lsimp is None:  #this deals with the boundary simplex case
                rsimp = self.simplices[(stpt+2)%3]
                while (not self is rsimp) and (not rsimp is None):
                    NeighborList.append(rsimp)
                    nsimp = rsimp.RightSimp(IDin)
                    rsimp = nsimp
            return NeighborList
        else:
            print("The input ID does not correspond to an ID in this simplex")
            return None        
 

    #This changes a weight to the right of the given point (and changes the appropriate weight in the adjacent simplex)
    def WeightChangeRight(self, IDin, neweight, firstuse = True):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            self.weights[(stpt+2)%3] = neweight
            if firstuse:
                rsimp = self.simplices[(stpt+2)%3]
                rsimp.WeightChangeLeft(IDin,neweight,False)
        else:
            print("The input ID does not correspond to an ID in this simplex")
            
    #This changes a weight to the left of the given point (and changes the appropriate weight in the adjacent simplex)
    def WeightChangeLeft(self, IDin, neweight, firstuse = True):
        stpt = self.LocalID(IDin)
        if not stpt == None:
            self.weights[(stpt+1)%3] = neweight
            if firstuse:
                rsimp = self.simplices[(stpt+2)%3]
                rsimp.WeightChangeRight(IDin,neweight,False)
        else:
            print("The input ID does not correspond to an ID in this simplex")

    
    #this returns True/False for whether the current simplex is ajacent to the given simplex along an edge
    def IsEdgeAdj(self, SimpIn):
        isadj = False
        for i in range(0,3):
            if self.simplices[i] is SimpIn:
                isadj = True
                break
        return isadj
    
    #This returns True/False for whether the current simplex is ajacent to the given simplex or outertriangle along at least point (i.e. share a point and possibly an edge).  Can use in conjunction with IsEdgeAdj to determine if they share only a point (though a single function that looks for two shared points would be more efficient)
    def IsPtAdj(self, ObjIn):
        isadj = False
        MyPoints = self.points
        if isinstance(ObjIn,outertriangle) or isinstance(ObjIn,simplex2D):
            for i in range(0,3):
                if MyPoints[i] in ObjIn.points:
                    isadj = True
                    break
        else:
            print("Passed in incompatible Object type:", type(ObjIn))                    
        return isadj
    
#End of simplex2D class ****************************************************************************************************************************************************************************************************************************************************************
    

    
    
    
#This class is a container for the data that define the outer triangle of a point (The triangle that represents the part of the rubber band close to the point) + a few simple methods to make it act like simplex object in a few respects
class outertriangle:
   
    _count = 0  #this is mainly a way to debug if I'm not properly getting rid of outertriangle objects
    
    def __init__(self, pointin, Rsimplexin, Lsimplexin):
        outertriangle._count += 1  #increment the counter ... the
        self.point = pointin        #The point about which the band is wrapped
        self.Rsimplex = Rsimplexin  #the simplex interior to the outer triangle and sharing its right edge (from current point looking out toward the the outer triangle)
        self.Lsimplex = Lsimplexin  #same as above, but for the simplex sharing the left boundary
        self.area = None          #will contain the area of this triangle
        self.arearate = 1
        self.points = self.GetPoints()  #This is simply to have the same data name as the simplex object, since some of the code is written to handle either object without using control structures
        self.OTLindex = None #This is needed to be able to quickly retrieve the position in the big otrilist of this outertriangle.  It will hold the index of this outertriangle in the otrilist
    
    def __del__(self):
        outertriangle._count -= 1

    def GetCount(self):
        return outertriangle._count
    
    def __eq__(self, other):
        return self is other  #I'm just doing this to have all comparisions be by object id, not value.  Now can use some of the functionality of a list of outer triangle objects, like remove, more freely.
    
    def GetPoints(self):
        RlocalID = self.Rsimplex.LocalID(self.point)   #find the id of the current point within the right simplex
        Rpoint = self.Rsimplex.points[(RlocalID+1)%3]  #use this id to return the point immediatly to its right
        LlocalID = self.Lsimplex.LocalID(self.point)   #find the id of the current point within the left simplex
        Lpoint = self.Lsimplex.points[(LlocalID+2)%3]  #use this id to return the point immediatly to its left
        return [self.point,Rpoint,Lpoint]        #The list comprising the outer triangle points [current point, right point, left point].  Here right and left refer to the view from the current point looking toward the interior of this outer triangle
    
    def SetPoints(self):
        self.points = self.GetPoints()
    
    #This returns True/False for whether the current outer triangle is ajacent to the given simplex or outer triangle at at least one point
    def IsPtAdj(self, ObjIn):
        isadj = False
        MyPoints = self.points
        if isinstance(ObjIn,outertriangle) or isinstance(ObjIn,simplex2D):
            for i in range(0,3):
                if MyPoints[i] in ObjIn.points:
                    isadj = True
                    break
        else:
            print("Passed in incompatible Object type:", type(ObjIn))                    
        return isadj        
    
    #This returns the sum of the weights about the central point in this outer triangle
    def PointWeightSum(self):
        return self.Rsimplex.PointWeightSum(self.point)
    
    #get the weight along the left edge of this outer triangle (again triangle faced acute angle up)
    def GetLeftWeight(self):
        lid = self.Lsimplex.LocalID(self.point)
        return self.Lsimplex.weights[(lid+1)%3]
    
    #get the weight along the right edge of this outer triangle (again triangle faced acute angle up)
    def GetRightWeight(self):
        rid = self.Rsimplex.LocalID(self.point)
        return self.Rsimplex.weights[(rid+2)%3]
    
    #This returns true if the given outer triangle is completely folded over (i.e. two of the three points are the same)
    #This type of outer triangle has zero area, despite not being a candidate for an event.
    def IsFolded(self):
        if self.points[1] == self.points[2]:
            return True
        else:
            return False
    
#End of OuterTriangle class ****************************************************************************************************************************************************************************************************************************************************************
        

#This is the WeightOperator class.  Each Operator stores the event type (core simplex collapse (et = 0), outer triangle collapse (et = 1), and combination collapse (et = 2)), the subtype (for et = 2,3 only) (Wstar == Wleft (include case of Wstar == Wleft == Wright) (st = 0), Wstar = Wright (st = 1), Wstar = Nloop (st = 2)), and the column indices involved (in order: [alpha, beta, gamma^0], and for et = 2: [alpha, beta, gamma^0,delta])
#the methods will define how each operator acts on the weight transfer matrix
class WeightOperator:
    
    #the constructor
    def __init__(self, EventType, ColumnIndices, Time, SubType = None):
        self.et = EventType
        self.st = SubType
        self.ind = ColumnIndices
        self.time = Time      #having each operator time-stamped will allow for takeing specific subsets of the full operator list
    
    #This updates the given Matrix M (here assumed to be either a numpy array or a scipy sparse matrix of lil type (list of lists)... both options have the same manipulation interface) with the stored weight operator.
    #The operations will be on rows due to the efficiency of the lil sparse type
    def Update(self,M):
        #first split out the three cases
        if self.et == 0:       #core simplex collapse
            M[self.ind[0],:] += M[self.ind[2],:]
            M[self.ind[1],:] += M[self.ind[2],:]
            M[self.ind[2],:] = 0
        elif self.et == 1:     #outer triangle collapse
            if self.st == 0:
                M[self.ind[2],:] = M[self.ind[0],:]
                M[self.ind[1],:] -= M[self.ind[0],:]
                M[self.ind[0],:] = 0
            elif self.st == 1:
                M[self.ind[2],:] = M[self.ind[1],:]
                M[self.ind[0],:] -= M[self.ind[1],:]
                M[self.ind[1],:] = 0
            elif self.st == 2:
                for i in range(0,M.shape[0]):
                    s = M[self.ind[0],i] - M[self.ind[1],i]
                    if s > 0.5:  #really looking for > 0, 0, < 0
                        M[self.ind[2],i] = M[self.ind[1],i]
                        M[self.ind[0],i] = s
                        M[self.ind[1],i] = 0
                    elif s < -0.5:
                        M[self.ind[2],i] = M[self.ind[0],i]
                        M[self.ind[0],i] = 0
                        M[self.ind[1],i] = -s
                    else:
                        M[self.ind[2],i] = M[self.ind[0],i]
                        M[self.ind[0],i] = 0
                        M[self.ind[1],i] = 0
        elif self.et == 2:     #combination collapse
            if self.st == 0:
                M[self.ind[2],:] = M[self.ind[0],:]
                M[self.ind[1],:] -= M[self.ind[0],:]
                M[self.ind[0],:] = 0
            elif self.st == 1:
                M[self.ind[2],:] = M[self.ind[1],:]
                M[self.ind[0],:] -= M[self.ind[1],:]
                M[self.ind[1],:] = 0
            elif self.st == 2:
                for i in range(0,M.shape[0]):
                    s = M[self.ind[0],i] - M[self.ind[1],i]
                    if s > 0.5:  #really looking for > 0, 0, < 0
                        M[self.ind[2],i] = M[self.ind[1],i]
                        M[self.ind[0],i] = s
                        M[self.ind[1],i] = 0
                    elif s < -0.5:
                        M[self.ind[2],i] = M[self.ind[0],i]
                        M[self.ind[0],i] = 0
                        M[self.ind[1],i] = -s
                    else:
                        M[self.ind[2],i] = M[self.ind[0],i]
                        M[self.ind[0],i] = 0
                        M[self.ind[1],i] = 0
            M[self.ind[0],:] += M[self.ind[3],:]
            M[self.ind[1],:] += M[self.ind[3],:]
            M[self.ind[3],:] = 0
        else:
            print("The wrong event type was recorded for this weight operator")
    
#End of WeightOperator class ****************************************************************************************************************************************************************************************************************************************************************


#This is the triangulation class, the central class in the overall 2DEtec algorithm.  It can be initialized using a Delaunay triangulation
class triangulation2D:

    #The constructor for triangulation2D.  ptlist is the list of [x,y] positions for the points at the initial time.
    #the defaut is to have a "foam" where each edge weight is initially set to 1, and each edge is anchored to the two adjacent points.  There are two other options:
    #ptpairs (optional), will initialize all weights to zero (except for those in ptpair, which are given weight 1).  They are input in pairs: ptpairs = [[p1,p2],[p3,p4], ...].  Repeated pairs give the edge additional weight (+1 for each additional pair).
    #rbands (optional), will also initialize all weights to zero, (except for those in rbands, which are given weight 1)
    #However rbands will also create the required outer triangles to make the weights correspond to an unanchored set (i.e. a rubber band).  The input looks like: rbands = [[p1,p2,p3,...],[p20,p21,p22,...], ...], where each list corresponds to sequentially adjacent points (and loops around so that p5,p1 form an edge) in a rubber band.  Each band set must be disjoint (can't do bands around bands that share edges yet ... that would additionally require figuring out which one is the outer band to get the outer triangles correct), and be disjoint from any ptpairs (ptpairs and rbands can be done at the same time.)
    #**Important: it is assumed that the points listed in rbands are given in counter-clockwise order about the closed band
    #The option controlptlist inputs a set of point that will be used to surround the data set and act as a boundary.  The default option of None will use a standard set of control points which enclose the domain the data initially starts out in in two concentric circles of 24 points
    #new with *v4*, we are using a constrained Delaunay triangulation for the ptpairs and rbands case
    def __init__(self, ptlist, ptpairs = None, rbands = None, controlptlist = None):
        #some initial error/debugging related items:
        self.atstep = 0     #This will be incremented each time the evolve method is used (useful for error handling)
        self.printstep = []  #this triggers a bunch of print items at this evolution step or list of steps (ex [105], or [105,107]).
        #now for the core variables
        self.extranum = 24   #the number of extra boundary points to use (must be even)
        if not controlptlist is None:
            self.extranum = len(controlptlist)  #user input number of extra boundary points (could be any number)
        self.ptnum = len(ptlist)+self.extranum   #put in the number of points, since this does not change, and will be used often. 
        #the additional 24 (or number input by the user) are for the points of the large bounding buffer simplices (to be computed) ... these are not picked to be too far (as that was found to make triangulation difficult).  There are two layers of points to help insulate boundary simplices from any of the dynamics (being involved in events)
        
        #Now calculate the extra boundary points
        self.extrapoints = []
        if controlptlist is None:
            temppoints0 = np.array(ptlist)
            ptcenter = np.average(temppoints0,axis = 0)   #the center (averaged)
            ptrange = (np.max(temppoints0,axis = 0)-np.min(temppoints0,axis = 0))/2.0   #the bounding box half widths
            rad = np.max(ptrange)   #effective bounding radius
            sfactor = 3  #the factor used to scale up the bounding points ***Using a really large factor here seems to cause the Delaunay trangulation function below to give bad output.  Will keep any bounding points not too far away, though we will need to make sure (in later additions to the code) that the points won't wander outside of these bounds at some-point in the full trajectory data.
            sfactor2 = 4
            numcirc = int(self.extranum/2)
            #now add extranum/2 points around a circle of radius sfactor*rad starting at theta = 0
            self.extrapoints = [[ptcenter[0]+sfactor*rad*math.cos(2*math.pi*i/numcirc),ptcenter[1]+sfactor*rad*math.sin(2*math.pi*i/numcirc)] for i in range(0,numcirc)]
            #now add another extranum/2 points around a larger circle of radius sfactor2*rad starting at 2*pi/(2*numcirc) (i.e. rotated to be angularly half-way between any pair of the first set of points)
            angoffset = 2*math.pi/(2*numcirc)
            self.extrapoints = self.extrapoints + [[ptcenter[0]+sfactor2*rad*math.cos(2*math.pi*i/numcirc+angoffset),ptcenter[1]+sfactor2*rad*math.sin(2*math.pi*i/numcirc+angoffset)] for i in range(0,numcirc)]
        else:
            self.extrapoints = controlptlist
        #Now put all the points together and triangulate
        self.pointpos = ptlist + self.extrapoints   #combining the bounding simplex points with the regular points' positions (adding the extra point to the end rather than the beginin is dicated by plotting concerns later)
        self.pointposfuture = self.pointpos    #pointposfuture holds the point positions at the next iterate.  Here we are just initializing it to be equal to the original point positions.
        temppoints = np.array(self.pointpos)  #put the total point set (regular plus large bounding simplex) into numpy array form
        temptri = []
        if ptpairs is None and rbands is None:
            temptri = Delaunay(temppoints,qhull_options="QJ Pp")   #create the initial Delaunay triangulation.  The option forces the creation of simplices for degenerate points by applying a random perturbation.
        else:
            #first need to find the outer edges as the edges in the convex hull of the control points 
            #(faster than the full point set ... but it is an assumption that only the control points form the convex hull)
            hull = ConvexHull(self.extrapoints)
            pllen = len(ptlist)
            hullptsz = hull.vertices.size
            constraintlines = [[hull.vertices[i]+pllen,hull.vertices[(i+1)%hullptsz]+pllen] for i in range(hullptsz)]
            #now add on the constraint lines from ptpairs and rbands
            if not ptpairs is None:
                for i in range(len(ptpairs)):
                    constraintlines.append(ptpairs[i])
            if not rbands is None:
                for i in range(len(rbands)):
                    rbl = len(rbands[i])
                    for j in range(rbl):
                        constraintlines.append([rbands[i][j],rbands[i][(j+1)%rbl]])
            #now we need to create the PSLG dictionary that is the input to triangle
            pslg = {'vertices': None, 'segments': None}
            pslg['vertices']= np.array(temppoints)
            pslg['segments'] = np.array(constraintlines)
            
            #now create the triangulation
            temptri = triangle.triangulate(pslg,'p')
            #unfortunately, this doesen't represent the triangles as simplices ... will need to manually get their neighbors
            
        
        #Now we need to store the triangulation data in a local data structure
        numsimp = 0
        if ptpairs is None and rbands is None:
            numsimp = temptri.simplices.shape[0]   #the number of simplices in the triangulation
        else:
            numsimp = temptri['triangles'].shape[0]  #number of triangles in the triangulation
        self.simplist = []
        
        #first create the list of simplex2D objects (not linked together yet ... need to create every object first)
        if ptpairs is None and rbands is None:  #the default option where each edge is assigned a weight of one
            for i in range(0,numsimp):
                self.simplist.append(simplex2D(temptri.simplices[i].tolist()))
                self.simplist[-1].SLindex = i
        else:  #this will assign everything a weight of zero, and then later we will put in the specific weights
            for i in range(0,numsimp):
                triptlist = temptri['triangles'][i].tolist()
                #make sure that this triangle has the correct area (>0)
                temparea = self.TriArea([temppoints[triptlist[0]],temppoints[triptlist[1]],temppoints[triptlist[2]]])
                if temparea < 0:
                    print("fixing negative area")
                    tempval1 = triptlist[0]
                    triptlist[0] = triptlist[1]
                    triptlist[1] = tempval1
                self.simplist.append(simplex2D(triptlist))
                self.simplist[-1].weights = [0,0,0]
                self.simplist[-1].SLindex = i
                
        #now create the links
        if ptpairs is None and rbands is None:
            for i in range(0,numsimp):
                linklist = temptri.neighbors[i].tolist()
                for j in range(0,len(linklist)):
                    if not linklist[j] == -1:
                        self.simplist[i].simplices[j] = self.simplist[linklist[j]]    #if -1 then the simplex already points to None (true for neighbors of boundary simplices) 
        else:
            edgestore = []
            for i in range(len(temppoints)):
                edgestore.append([])
                for j in range(i):  #this is an upper triangle matrix
                    edgestore[i].append([])
            #now store the triangle ids that have these edges (always with highest pt id first)
            for k in range(numsimp):
                tripts = sorted(temptri['triangles'][k].tolist(),reverse=True)
                edgestore[tripts[0]][tripts[1]].append(k)
                edgestore[tripts[0]][tripts[2]].append(k)
                edgestore[tripts[1]][tripts[2]].append(k)
            for i in range(len(temppoints)):
                for j in range(i):
                    if len(edgestore[i][j]) == 2: #(has the index of the two simplices)
                        s1 = edgestore[i][j][0]
                        s2 = edgestore[i][j][1]
                        SimpLink(self.simplist[s1],self.simplist[s2])  #this links the two simplices together
                        
                
        
        #now create the pointlist with links to individual simplices
        #first initialize the list
        self.pointlist = []
        for i in range(0,self.ptnum):
            self.pointlist.append(None)
            
        #now go through each simplex and add that simplex to each slot in the pointlist that corresponds to an included point if the slot contains None (possibly more efficient way to do this)
        for i in range(0,numsimp):
            for j in range(0,3):
                 if self.pointlist[self.simplist[i].points[j]] is None:
                        self.pointlist[self.simplist[i].points[j]] = self.simplist[i]
        
        #this initializes the list of anchor edges (the number of edges permanently attached to each point).  The three cases (full initial net, individual anchored pairs, and rubber band) are treated separately
        self.anchoredges = []
        #initialize to zero for the case of ptpairs or rbands
        if not (ptpairs is None and rbands is None):
            for i in range(0,self.ptnum):
                self.anchoredges.append(0)
        
        #This puts the weights back in for those in the ptpairs list (if it is triggered ... i.e. not the default option)
        if not ptpairs is None:
            for i in range(0,len(ptpairs)):
                STM = self.GetEdgePairSimp(ptpairs[i])
                if not STM is None:  #want to ignore the case of two points that are on the boundary or are non-adjacent in the triangulation
                    #Now set the weight of the shared edge
                    SimpWeightSet(STM[0],STM[1],1,True) #don't just set = to 1, adding 1 (True option) will allow us to get higher weights by including more copies of the pair
                else:
                    print("Input ptpairs contains points on the boundary or points that are not connected by an edge in the initial triangulation")
            #now correct the anchor list for those points in ptpairs
            for i in range(0,len(ptpairs)):
                p1 = ptpairs[i][0]
                p2 = ptpairs[i][1]
                self.anchoredges[p1] = self.pointlist[p1].PointWeightSum(p1)
                self.anchoredges[p2] = self.pointlist[p2].PointWeightSum(p2)
                
                
        #Set the Outer Triangle list (empty to begin with).  This will be filled with outertriangle objects as they are created 
        self.otrilist = []
        #set the outer triangle auxilliary list.  This will contain the index of otrilist slots that do not have an outer triangle
        self.otriauxlist = []
        
        #We also create the analogue of pointlist for outer triangles.  Each element in this list has a reference to the outer triangle that has its main point corresponding to the list index.  If a outer triangle is not yet associated with a given point, then "None" is put in this slot.
        self.otriptlist = []
        for i in range(0,self.ptnum):
            self.otriptlist.append(None)            
            
        #This puts the weights back in for those in the rbands list (if it is triggered ... i.e. not the default option)
        if not rbands is None:
            for i in range(0,len(rbands)):
                rblen = len(rbands[i])
                STM = []
                for j in range(0,rblen):
                    #get 2 simplices that share these points as an edge
                    STM.append(self.GetEdgePairSimp([rbands[i][j],rbands[i][(j+1)%rblen]]))
                    #want to ignore the case of a boundary edge, or two points not adjacent in the initial triangulation
                    if not STM[j] is None:  
                        SimpWeightSet(STM[j][0],STM[j][1],1)
                        #note that for rbands we are not adding, but keeping the weight at 1
                    else:
                        print("Input rbands contains points on the boundary or points that are not connected by an edge in the initial triangulation")
                #Now that we have all of the simplex pairs that border each edge in the band, we need to add outer triangles
                #**Important: it is assumed that the points listed in rbands are given in counter-clockwise order
                for j in range(0,rblen):
                    p1 = rbands[i][j]
                    p2 = rbands[i][(j+1)%rblen]
                    p3 = rbands[i][(j+2)%rblen]
                    if self.TriArea([self.pointpos[p1],self.pointpos[p2],self.pointpos[p3]]) >= -0.0000001:
                        #the accute angle defined by this ordered triple of points is INWARD with respect to the rubber band
                        leftsimp = STM[j][0]  #guessing that this is the correct left simplex for the outer triangle ... now check
                        p1loc = leftsimp.LocalID(p1)
                        if not leftsimp.points[(p1loc+1)%3] == p2:  #guess was wrong, choose other
                            leftsimp = STM[j][1]
                        rightsimp = STM[(j+1)%rblen][0]
                        p2loc = rightsimp.LocalID(p2)
                        if not rightsimp.points[(p2loc+1)%3] == p3:
                            rightsimp = STM[(j+1)%rblen][1]
                        #now create the new outer triangle
                        self.otriptlist[p2] = outertriangle(p2, rightsimp, leftsimp)  #put into the outer triangle point list
                        self.otrilist.append(self.otriptlist[p2])  #put in the outer triangle list
                        self.otriptlist[p2].OTLindex = len(self.otrilist)-1   #store the otrilist index
                    else:  
                        #the accute angle defined by the ordered triple of points is OUTWARD with respect to the rubber band
                        leftsimp = STM[(j+1)%rblen][0]  #guessing that this is the correct left simplex for the outer triangle ... now check
                        p3loc = leftsimp.LocalID(p3)
                        if not leftsimp.points[(p3loc+1)%3] == p2:  #guess was wrong, choose other
                            leftsimp = STM[(j+1)%rblen][1]
                        rightsimp = STM[j][0]
                        p2loc = rightsimp.LocalID(p2)
                        if not rightsimp.points[(p2loc+1)%3] == p1:
                            rightsimp = STM[j][1]
                        #now create the new outer triangle
                        self.otriptlist[p2] = outertriangle(p2, rightsimp, leftsimp)  #put into the outer triangle point list
                        self.otrilist.append(self.otriptlist[p2])  #put in the outer triangle list
                        self.otriptlist[p2].OTLindex = len(self.otrilist)-1   #store the otrilist index
                        
                        
        #Each initial simplex edge was given a weight of 1.  Here we set the weights corresponding to all edges adjacent to the extra points (auxiliary points).  Only do the default option, since the rbands and ptpairs already set these = to zero
        if ptpairs is None and rbands is None:
            for i in range(0,self.extranum):
                thispt = self.ptnum-1-i
                #print("Extra numbers: ", thispt)
                Neighbors = self.pointlist[thispt].SimpNeighbors(thispt)
                #print("Neighbors: ", [Neighbors[k].points for k in range(0,len(Neighbors))])
                for j in range(0,len(Neighbors)):
                    locid = Neighbors[j].LocalID(thispt)
                    Neighbors[j].weights[(locid+1)%3] = 0
                    Neighbors[j].weights[(locid+2)%3] = 0
        
        #now create the anchoredges list (holds the invariant number of anchor edges for each point).  This is needed to prevent popping off too many edge weights.  This treats the default case of a full initial mesh/foam
        if ptpairs is None and rbands is None:
            for i in range(0,self.ptnum):
                self.anchoredges.append(self.pointlist[i].PointWeightSum(i))

        #Now we assign each edge an index (will do this for all options for consistency).  This goes through each simplex object, and assigns an id to each edge (an the same id to the corresponding edge in the adjacent simplex) if it has not already been assigned.  The index is just taken from an incremental counter.
        edgecounter = 0
        for i in range(0,len(self.simplist)):
            for j in range(0,3):
                tempsimp = self.simplist[i]
                if tempsimp.edgeids[j] is None:
                    tempsimp.edgeids[j] = edgecounter
                    if not tempsimp.simplices[j] is None:
                        pt = tempsimp.points[(j+1)%3]
                        Lid = (tempsimp.simplices[j].LocalID(pt)+1)%3
                        tempsimp.simplices[j].edgeids[Lid] = edgecounter
                    edgecounter += 1  
        self.totalnumedges = edgecounter
        #Set the area variable in each simplex object in our list
        #self.SetAreaSimpList()   #not using right now
        
        
        #Finally, we initialize the weight operator list (collects the operators that will act on the Weight transition matrix)
        self.WeightOperators = []
        #this list will be sorted according to a causal order, but since different groups within each time-step are treated sequentially, despite the fact that events between the groups might have any time order, the list is not strictly sorted by time.  Each WeightOperator has a time-stamp that can be used for sorting if needed.

        
    #End of the constructor for triangulation2D ****************************************************************************************************************************************************************************************************************************************************************
    
    
    #Begin the methods for triangulation2D ****************************************************************************************************************************************************************************************************************************************************************
    #we eventually need something like a copy constructor for triangulations ...
    
    
    
    #Evolve method.  This assumes that the starting triangulation is good (no negative areas).  It takes in the new time-slice data in ptlist -- the list of [x,y] positions for the points at the next time-step.
    def Evolve(self, ptlist):
        #for i in range(0,len(self.simplist)):
        #    print("Initial Simplex Points: ", self.simplist[i].points,"\n")
        #print("At evolution step number ", self.atstep)
        self.LoadNewPos(ptlist)  #putting the new point positions in pointposfuture
        #next get all the Events (using the initial and final areas), both outer triangle and regular simplex
        EventListSimp = self.GetRegularEvents()
        EventListOtri = self.GetOTriEvents()
        #print("Size of original Event List: ", len(EventList))
        #now sort this list by time
        EventListSimp.sort(key=itemgetter(1), reverse=True)
        EventListOtri.sort(key=itemgetter(1), reverse=True) #this is in decending order so that removing from the end(smallest times first) inccurs the smallest computational cost
        #next Evolve the eventlist
        self.GEvolve(EventListSimp,EventListOtri)
        #next we need to push the future positions to the current positions
        self.UpdatePtPos()
        self.atstep += 1  #increment this internal counter ... useful for trigering a print message (for debugging) at a particular time interval       

        
    #this returns a list of current simpices (each element is [simplex, first time for A = 0]) whose area goes through zero sometime between their current and future positions.
    def GetRegularEvents(self):
        badsimplist = []
        for i in range(0,len(self.simplist)):
            AZT = self.AreaZeroTime2(self.simplist[i].points)
            if AZT[0]:
                badsimplist.append([self.simplist[i],AZT[1]])
        #print("Length of badsimplist: ", len(badsimplist))
        return badsimplist
        
    #this returns a list of current outer simplices (each element is [outer simplex, first time for A = 0]) whose area goes through zero sometime between their current and future positions.
    def GetOTriEvents(self):
        badotrilist = [] 
        for i in range(0,len(self.otrilist)):
            if not self.otrilist[i] is None:
                AZT = self.AreaZeroTime2(self.otrilist[i].points)
                if AZT[0]:
                    if self.atstep in self.printstep:
                        print("Considering Outer Triangle for bad OTrilist: ", self.otrilist[i].points)
                    if not (self.otrilist[i].IsFolded() or self.WillFold(self.otrilist[i].points,AZT[1])):
                        badotrilist.append([self.otrilist[i],AZT[1]])
        #print("Length of badotrilist: ", len(badotrilist))            
        return badotrilist
        
    
    #The main method for evolving the Event List (group of simplices and outer triangles that need fixing)
    #remember that the list is sorted in decending order, and we deal with the last element first
    def GEvolve(self,EventListSimp,EventListOtri):
        delta = 1e-10
        
        while (len(EventListSimp)+len(EventListOtri))> 0:
            #print("Size of Event list: ", len(EventList))
            #print([EventList[-1][0].points,EventList[-1][1], type(EventList[-1][0])],", ")
            #if self.atstep in self.printstep:
            #    print("The current Event List: ")
            #    for i in range(0,len(EventList)):
            #        print([EventList[i][0].points,EventList[i][1], type(EventList[i][0])],", ")
            isconpair = False
            neweventssimp = []  #new simpices to check
            neweventsotri = []  #and new/modified outer triangles to check
            dellistsimp = []    #simplices to delete from the simplex event list if they exist in it
            neweventsotriaux = []
            currenttime = 0
            eventtype = None
            if (len(EventListSimp) > 0) and (len(EventListOtri) > 0):
                isconpair = self.AreConcurrentEvents(EventListSimp[-1],EventListOtri[-1])
                if isconpair:
                    currenttime = EventListSimp[-1][1]  #the time of collapse of the currently considered event
                else:
                    if EventListSimp[-1][1] < EventListOtri[-1][1]:
                        eventtype = simplex2D
                        currenttime = EventListSimp[-1][1]
                    else:
                        eventtype = outertriangle
                        currenttime = EventListOtri[-1][1]
            else:
                if len(EventListSimp) > 0:
                    eventtype = simplex2D
                    currenttime = EventListSimp[-1][1]
                else:
                    eventtype = outertriangle
                    currenttime = EventListOtri[-1][1]
                    
            if isconpair:
                #deal with the concurrent events here.  pass in the simplex event and delete the outer triangle event
                del EventListOtri[-1]
                modlist = self.CombFix(EventListSimp[-1],currenttime + delta)    #returns [goodlist, badlist, trimodlist, trimodlistaux]
                neweventssimp = modlist[0]
                neweventsotri = modlist[2]
                dellistsimp = modlist[1]
                neweventsotriaux = modlist[3]
                del EventListSimp[-1]  #get rid of the evaluated event
            elif eventtype is simplex2D:
                #deal with simplex collapse events here
                modlist = self.SFix(EventListSimp[-1],currenttime + delta)    #returns [[lsimp,rsimp],[Topsimp],trimodlist,trimodlistaux]  #(n3/18)
                neweventssimp = modlist[0]
                neweventsotri = modlist[2]
                dellistsimp = modlist[1]
                neweventsotriaux = modlist[3]
                del EventListSimp[-1]  #get rid of the evaluated event
            elif eventtype is outertriangle:
                #deal with outer triangle collapse events here
                modlist = self.OTriFix(EventListOtri[-1],currenttime + delta)   #returns  [goodlist,Botsimplist,trimodlist,trimodlistaux]  #(n3/18)
                neweventssimp = modlist[0]
                neweventsotri = modlist[2]
                dellistsimp = modlist[1]
                neweventsotriaux = modlist[3]
                del EventListOtri[-1]  #get rid of the evaluated event

            #first find the time of zero area for core simplex events (outer triangle events must be treated separately since they have changed their point list, and a time of zero area will be different from their former value, and not be helpful in locating the event in EventList).
            delsimplist = []
            for j in range(0,len(dellistsimp)):
                AZT = self.AreaZeroTime2(dellistsimp[j].points,currenttime + delta)
                if AZT[0]:
                     delsimplist.append([dellistsimp[j],AZT[1]])
  
            #Now we locate the elements of delsimplist in EventList through a binary search, and delete them
            for i in range(0,len(delsimplist)):
                BinarySearchDel(EventListSimp, delsimplist[i])            
            
            for i in range(0,len(neweventsotriaux)):
                if neweventsotriaux[i][0][0]:
                    if not (neweventsotri[i].IsFolded() or self.WillFold(neweventsotriaux[i][0][2],neweventsotriaux[i][0][1])):
                    #need to search for this neweventsotri in EventListOtri
                        otrieventdeltemp = [neweventsotri[i],neweventsotriaux[i][0][1]]  #this gives the old time (so it can be identified in the EventListOtri)
                        BinarySearchDel(EventListOtri, otrieventdeltemp)
                #now let's put this event back in if warranted
                if neweventsotriaux[i][1][0]:
                    if not (neweventsotri[i].IsFolded() or self.WillFold(neweventsotri[i].points,neweventsotriaux[i][1][1])):
                        otrieventinstemp = [neweventsotri[i],neweventsotriaux[i][1][1]]
                        BinarySearchIns(EventListOtri, otrieventinstemp)
                                                    
            #now run through the newevents list and see if each object has an area less than zero (if so, add to EventList with the calulated time to zero area)
            for i in range(0,len(neweventssimp)):
                AZT = self.AreaZeroTime2(neweventssimp[i].points,currenttime + delta,Verbose = True)
                if AZT[0]:
                    #insert in the event list at the correct spot
                    ItemIn = [neweventssimp[i],AZT[1]]
                    BinarySearchIns(EventListSimp,ItemIn)
         

    #Fixing a simplex and the surrounding effected simplices/outer triangles.  SimpIn is actually a list [simplex,area zero time]
    #This returns the two new simplices, so that they can be possibly added to the local event list, also the bad simplices so they can be removed (if needed from the local event list), and the modified outer triangles
    #refer to EventTypes.pdf for a geometric picture of this collapse event, and what the major variables refer to.
    def SFix(self,SimpIn,timein):
        Simp = SimpIn[0]
        colind = self.CollapsePt(Simp.points,SimpIn[1])  #this is the local index of the offending point during the area collapse
        Topsimp = Simp.simplices[colind]
        if self.atstep in self.printstep:
            print("Simp points: ", Simp.points, ", and weights: ", Simp.weights)
            print("Topsimp points: ", Topsimp.points, ", and weights: ", Topsimp.weights)
        cpt = Simp.points[colind]
        rptlid = (colind+1)%3
        lptlid = (colind+2)%3
        rpt = Simp.points[rptlid]
        lpt = Simp.points[lptlid]
        WeightIDs = [Simp.edgeids[rptlid],Simp.edgeids[lptlid],Simp.edgeids[colind]]  #the weight ids [alpha, beta, gamma] ... used to create the weight operator, which is then stored in the weight operator list
        rptuid = Topsimp.LocalID(rpt)
        lptuid = Topsimp.LocalID(lpt)
        tpt = Topsimp.points[(rptuid+1)%3]
        rslist = [cpt,rpt,tpt]
        lslist = [cpt,tpt,lpt]
        rsimp = simplex2D(rslist)  #new right simplex
        lsimp = simplex2D(lslist)  #new left simplex
        #now create the links these simplices have to other simplices as well as their weights
        collidingweight = Simp.weights[colind]
        SimpLink(rsimp,lsimp,True,0)
        SimpLink(rsimp,Topsimp.simplices[lptuid],True,Topsimp.weights[lptuid])
        SimpLink(lsimp,Topsimp.simplices[rptuid],True,Topsimp.weights[rptuid])
        SimpLink(rsimp,Simp.simplices[lptlid],True,Simp.weights[lptlid]+collidingweight)
        SimpLink(lsimp,Simp.simplices[rptlid],True,Simp.weights[rptlid]+collidingweight)
        #also need to reassign the weight ids
        rsimp.edgeids[0] = Topsimp.edgeids[lptuid]  #for all of these, we know which points the local ids correspond to
        rsimp.edgeids[1] = WeightIDs[2]
        rsimp.edgeids[2] = WeightIDs[1]
        lsimp.edgeids[0] = Topsimp.edgeids[rptuid]
        lsimp.edgeids[1] = WeightIDs[0]
        lsimp.edgeids[2] = WeightIDs[2]
        
        if self.atstep in self.printstep:
            print("rsimp points, ", rsimp.points, ", and weights, ", rsimp.weights)
            print("lsimp points, ", lsimp.points, ", and weights, ", lsimp.weights)
            
        #Now create and add the appropriate Weight Operator the the weight operator list
        self.WeightOperators.append(WeightOperator(0, WeightIDs, self.atstep + SimpIn[1]))  #The time is the integer step the evolution is currently on plus the fraction of the interval [0,1] at which the event occurs.
        #now correct any outer triangles associated with the left and right points that are trivially affected (just a change in one of the edges)
        trimodlist = []
        trimodlistaux = []
        rttrimod = False
        lttrimod = False
        rttri = self.otriptlist[rpt]
        lttri = self.otriptlist[lpt]
        if not (rttri is None):
            if self.atstep in self.printstep:
                print("rttri outer triangle points before: ", rttri.points)
                print("rttri Rsimplex is: ", rttri.Rsimplex.points)
                print("rttri Lsimplex is: ", rttri.Lsimplex.points)
            if rttri.Rsimplex is Simp:
                rttri.Rsimplex = Simp.simplices[lptlid]
                rttrimod = True
            elif rttri.Rsimplex is Topsimp:
                rttri.Rsimplex = rsimp
            if rttri.Lsimplex is Topsimp:
                rttri.Lsimplex = rsimp
                rttrimod = True
            elif rttri.Lsimplex is Simp:
                rttri.Lsimplex = rsimp
        if not (lttri is None):
            if self.atstep in self.printstep:
                print("lttri outer triangle points before: ", lttri.points)
                print("lttri Rsimplex is: ", lttri.Rsimplex.points)
                print("lttri Lsimplex is: ", lttri.Lsimplex.points)
            if lttri.Rsimplex is Simp:
                lttri.Rsimplex = lsimp
            elif lttri.Rsimplex is Topsimp:
                lttri.Rsimplex = lsimp
                lttrimod = True
            if lttri.Lsimplex is Topsimp:
                lttri.Lsimplex = lsimp
            elif lttri.Lsimplex is Simp:
                lttri.Lsimplex = Simp.simplices[rptlid]
                lttrimod = True
        if rttrimod:
            trimodlistaux.append([])
            AZT = self.AreaZeroTime2(rttri.points,timein)
            if AZT[0]:
                trimodlistaux[-1].append([AZT[0],AZT[1],rttri.points])
            else:
                trimodlistaux[-1].append([AZT[0],None,rttri.points])                    
            trimodlist.append(rttri)
            rttri.SetPoints()
            AZT = self.AreaZeroTime2(rttri.points,timein) #again after the setpoint()
            if AZT[0]:
                trimodlistaux[-1].append([AZT[0],AZT[1]])
            else:
                trimodlistaux[-1].append([AZT[0],None])
                
            if self.atstep in self.printstep:
                print("rttri outer triangle points after: ", rttri.points)
                print("rttri Rsimplex is: ", rttri.Rsimplex.points)
                print("rttri Lsimplex is: ", rttri.Lsimplex.points)
        if lttrimod:
            trimodlistaux.append([])
            AZT = self.AreaZeroTime2(lttri.points,timein)
            if AZT[0]:
                trimodlistaux[-1].append([AZT[0],AZT[1],lttri.points])
            else:
                trimodlistaux[-1].append([AZT[0],None,lttri.points])                    
            trimodlist.append(lttri)
            lttri.SetPoints()
            AZT = self.AreaZeroTime2(lttri.points,timein) #again after the setpoint()
            if AZT[0]:
                trimodlistaux[-1].append([AZT[0],AZT[1]])
            else:
                trimodlistaux[-1].append([AZT[0],None])
            

            if self.atstep in self.printstep:
                print("lttri outer triangle points after: ", lttri.points)
                print("lttri Rsimplex is: ", lttri.Rsimplex.points)
                print("lttri Lsimplex is: ", lttri.Lsimplex.points)
                
        #deal with possible top outer triangles ... note that even if we must change the defining simplices, the points remain unchanged, so we don't need to include this in the trimodlist
        toptri = self.otriptlist[tpt]
        if not (toptri is None):
            if toptri.Rsimplex is Topsimp:
                toptri.Rsimplex = lsimp
            if toptri.Lsimplex is Topsimp:
                toptri.Lsimplex = rsimp
        #also bottom outer triangles that are not the double collapse configurations configuration (this situation has been detected before this method, and dealt with separately)
        bottri = self.otriptlist[cpt]
        if collidingweight > 0:
            trimodlistaux.append([])
            if bottri is None:  #Create the outer triangle if it does not exist yet
                self.otriptlist[cpt] = outertriangle(cpt,Simp.simplices[rptlid],Simp.simplices[lptlid])
                if len(self.otriauxlist) == 0:  #created this whole if-else clause (3/18)
                    self.otrilist.append(self.otriptlist[cpt])
                    self.otriptlist[cpt].OTLindex = len(self.otrilist)-1   #store the otrilist index (3/18)
                else:
                    self.otrilist[self.otriauxlist[-1]] = self.otriptlist[cpt]
                    self.otriptlist[cpt].OTLindex = self.otriauxlist[-1]
                    del self.otriauxlist[-1]
                trimodlistaux[-1].append([False,None,None])    
                if self.atstep in self.printstep:
                    print("creating bottom outer triangle: ", self.otriptlist[cpt].points)            
            else:   #modify the exisiting outer triangle
                if self.atstep in self.printstep:
                    print("modifying bottom outer triangle from: ", bottri.points)
                bottri.Rsimplex = Simp.simplices[rptlid]
                bottri.Lsimplex = Simp.simplices[lptlid]
                AZT = self.AreaZeroTime2(bottri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1],bottri.points])
                else:
                    trimodlistaux[-1].append([AZT[0],None,bottri.points])
                bottri.SetPoints()
                if self.atstep in self.printstep:
                    print("to: ", bottri.points)
            AZT = self.AreaZeroTime2(self.otriptlist[cpt].points,timein)
            if AZT[0]:
                trimodlistaux[-1].append([AZT[0],AZT[1]])
            else:
                trimodlistaux[-1].append([AZT[0],None])        
            trimodlist.append(self.otriptlist[cpt])
        elif collidingweight == 0:   #if the colliding weight is zero, we still need to modify folded outer triangles
            if not bottri is None:            
                if bottri.Rsimplex is Simp:
                    bottri.Rsimplex = rsimp
                if bottri.Lsimplex is Simp:
                    bottri.Lsimplex = lsimp
                if self.atstep in self.printstep:
                    print("to: ", bottri.points)

        #replace the two bad simplices in the simplex list with the two new ones
        Simpindex = Simp.SLindex
        self.simplist[Simpindex] = lsimp
        lsimp.SLindex = Simpindex
        
        Topsimpindex = Topsimp.SLindex
        self.simplist[Topsimpindex] = rsimp
        rsimp.SLindex = Topsimpindex
                
        #look through the simplex point list to see if either of the bad simplices were there and replace if so
        if self.pointlist[cpt] is Simp:
            self.pointlist[cpt] = Simp.simplices[lptlid]
        if (self.pointlist[rpt] is Simp) or (self.pointlist[rpt] is Topsimp):
            self.pointlist[rpt] = Topsimp.simplices[lptuid]
        if self.pointlist[tpt] is Topsimp:
            self.pointlist[tpt] = Topsimp.simplices[rptuid]
        if (self.pointlist[lpt] is Simp) or (self.pointlist[lpt] is Topsimp):
            self.pointlist[lpt] = Simp.simplices[rptlid]
        
        #Next, delete all the references to simplices in both of the bad simplices
        for i in range(0,3):
            Simp.simplices[i] = None
            Topsimp.simplices[i] = None    
            
        #The two bad simplices should have no references to them, (also ones in Ngroup list) except the local ones in this method, which will dissapear at the end of this method.  So, these two simplices should be given to garbage collection after this method.
        #finally, return the two new simplices, so that they can be checked to see if they need to be included in any update to the local event list. Also return the two bad simplices to remove any instances from the event list. also return the accumulated list of modified outer triangles ... these will need to be checked for inclusion in the eventlist
        #return [[lsimp,rsimp],[Simp,Topsimp],trimodlist]
        return [[lsimp,rsimp],[Topsimp],trimodlist,trimodlistaux]
    
    
    #Fixing an outer triangle
    #refer to EventTypes.pdf for a geometric picture of this collapse event, and what the major variables refer to.
    def OTriFix(self, OTriIn,timein):
        tri = OTriIn[0]
        timecoll = OTriIn[1]
        cpt = tri.point
        #first get the sum of the weights around the central point
        Wsum = tri.PointWeightSum()  #the sum of the weights about the central point, cpt
        Anum = self.anchoredges[cpt]  #The number of anchor edges
        Nloop = int((Wsum-Anum)/2)  #The number of bands looped over the point
        if self.atstep in self.printstep:
            print("Wsum: ",Wsum,", Anum: ",Anum,", Nloop: ",Nloop)
        #now generate the two initial simplex lists: one above the initial outer triangle (above the acute angle) and the other below it.  Both lists start from the left and work their way to the right
        Topsimplist = []
        Topsimplist.append(tri.Lsimplex)
        while not Topsimplist[-1] is tri.Rsimplex:
            Topsimplist.append(Topsimplist[-1].RightSimp(cpt))
        if self.atstep in self.printstep:
            print("Topsimplist:")
            for i in range(0,len(Topsimplist)):
                print(i, Topsimplist[i].points,Topsimplist[i].weights)
        Botsimplist = []
        Botsimplist.append(tri.Lsimplex.LeftSimp(cpt))
        tempsimp = tri.Rsimplex.RightSimp(cpt)
        while not Botsimplist[-1] is tempsimp:
            Botsimplist.append(Botsimplist[-1].LeftSimp(cpt))
        if self.atstep in self.printstep:
            print("Botsimplist:")
            for i in range(0,len(Botsimplist)):
                print(i, Botsimplist[i].points,Botsimplist[i].weights)
        #now for the left and right points
        lpt = tri.points[2]
        rpt = tri.points[1]
        #the weight ids [alpha, beta, gamma^0] ... used to create the weight operator
        WeightIDs = [Topsimplist[0].edgeids[(Topsimplist[0].LocalID(cpt)+1)%3],Topsimplist[-1].edgeids[(Topsimplist[-1].LocalID(cpt)+2)%3],Botsimplist[0].edgeids[(Botsimplist[0].LocalID(cpt)+1)%3]]
        BottomWeightIDs = []   #the weight ids that will be redistributed among the internal edges of nlowersimps
        for i in range(1,len(Botsimplist)-1):
            BottomWeightIDs.append(Botsimplist[i].edgeids[(Botsimplist[i].LocalID(cpt)+1)%3])
        
        #the weight along the left and right 
        Wleft = tri.GetLeftWeight()
        Wright = tri.GetRightWeight()
        Wstar = min(Wleft,Wright,Nloop)  #This is the weight of the bottom of the central simplex
        Wleftnew = Wleft - Wstar     #the new weight of the left top edge of the central simplex
        Wrightnew = Wright - Wstar   #the new weight of the right top edge of the central simplex 
        if self.atstep in self.printstep:
            print("Wleft:", Wleft, ", Wright: ", Wright, ", Wstar: ", Wstar, ", Wleftnew: ", Wleftnew, ", Wrightnew: ",Wrightnew)
        #create the WeightOperator object and append it to the list
        if Wleft == Wstar:
            st = 0
        elif Wright == Wstar:
            st = 1
        else:
            st = 2
        self.WeightOperators.append(WeightOperator(1, WeightIDs, self.atstep + timecoll, st))
    
        #First create the new central simplex
        csimp = simplex2D([cpt,lpt,rpt])
        #can set the edge IDs for csimp
        csimp.edgeids[0] = WeightIDs[2]
        csimp.edgeids[1] = WeightIDs[1]
        csimp.edgeids[2] = WeightIDs[0]
        
        #Next make the new lower simplices
        lpoints = [Botsimplist[i].points[(Botsimplist[i].LocalID(cpt)+1)%3] for i in range(0,len(Botsimplist))] #make list of perimeter points
        lpoints.append(rpt)  #add last point on
        if self.atstep in self.printstep:
            print("lpoints positions: ",self.PtPosPart(lpoints,timecoll))
        lowertri = self.OtriEventTrangulation(lpoints,timecoll)  #Get the list of triangulation triples
        nlowersimps = []
        for i in range(0,len(lowertri)):
            nlowersimps.append(simplex2D([lpoints[lowertri[i][j]] for j in range(0,3)]))  #create the list of new simplices with the triples of points given by the indices stored in lowertri (these are triples of indices for points in lpoints)
        if self.atstep in self.printstep:
            for i in range(0,len(nlowersimps)):
                print("nlowersimps[",i,"]: ", nlowersimps[i].points)
        #Set all the links (ingoing and outgoing) for these new simplices
        #first for the central simplex
        SimpLink(csimp,Topsimplist[-1],True,Wrightnew)  #this sets the links in both input simplices and the proper weights
        SimpLink(csimp,Topsimplist[0],True,Wleftnew)
        smatch = DblInTrpl([0,len(lpoints)-1],lowertri,True)
        SimpLink(csimp,nlowersimps[smatch[0]],True,Wstar)
        nlowersimps[smatch[0]].edgeids[(nlowersimps[smatch[0]].LocalID(lpt)+1)%3] = WeightIDs[2]  #edge id for the lower simplex that shares an edge with teh central simplex
        #now fix the links and weights associated with perimeter edges
        for i in range(0,len(lpoints)-1):
            smatch = DblInTrpl([i,i+1],lowertri,True)
            locid = Botsimplist[i].LocalID(cpt)
            SimpLink(nlowersimps[smatch[0]],Botsimplist[i].simplices[locid],True,Botsimplist[i].weights[locid])
            nlowersimps[smatch[0]].edgeids[(nlowersimps[smatch[0]].LocalID(lpoints[i+1])+1)%3] = Botsimplist[i].edgeids[locid]
        #now for the links and weights internal to the lower simplices
        for i in range(0,len(nlowersimps)):
            for j in range(0,3):
                if nlowersimps[i].simplices[j] is None:
                    ptdbl = [nlowersimps[i].points[(j+1)%3],nlowersimps[i].points[(j+2)%3]]
                    inddbl = [lpoints.index(ptdbl[0]),lpoints.index(ptdbl[1])]
                    smatch = DblInTrpl(inddbl,lowertri)  #should give two matches (including the one we already know), however if the nlowersimp is one of the boundary simplices, then we will have only one match (in which case, we do nothing)
                    if len(smatch) == 2:
                        SimpLink(nlowersimps[smatch[0]],nlowersimps[smatch[1]],True,0)  #notice all these weights are zero
                        SimpEdgeIDSet(nlowersimps[smatch[0]],nlowersimps[smatch[1]],BottomWeightIDs[-1])  #set the edge ids
                        del BottomWeightIDs[-1]
                    else:
                        print("There are not two matches as there should be")
                        
        #replace bad simplices with new ones (there are same number) in the simplist (simplex list)
        goodlist = [csimp]
        for i in range(0,len(nlowersimps)):
            goodlist.append(nlowersimps[i])
        for i in range(0,len(Botsimplist)):
            BSimpindex = Botsimplist[i].SLindex
            self.simplist[BSimpindex] = goodlist[i]
            goodlist[i].SLindex = BSimpindex

        #go through the point list (with references to adjacent simplices), and replace any bad simplices if needed
        if self.pointlist[lpoints[0]] is Botsimplist[0]:
            self.pointlist[lpoints[0]] = Topsimplist[0]
        for i in range(1,len(lpoints)-1):
            if self.pointlist[lpoints[i]] in [Botsimplist[i-1],Botsimplist[i]]:
                self.pointlist[lpoints[i]] = Botsimplist[i].simplices[Botsimplist[i].LocalID(cpt)]
        if self.pointlist[lpoints[-1]] is Botsimplist[-1]:   #deal with the last of lpoints and cpt on their own
            self.pointlist[lpoints[-1]] = Topsimplist[-1]
        if self.pointlist[cpt] in Botsimplist:
            self.pointlist[cpt] = Topsimplist[0]
        
        #update/remove the outer triangle that has collapsed (if remove, set out triangle point list element to None, and search and remove from the Otrilist)
        trimodlistaux = []
        trimodlist = []
        if Wstar == Nloop:  #all of the bands are coming off ... delete this outer triangle
            otind = self.otriptlist[cpt].OTLindex  #(3/18)
            self.otriptlist[cpt] = None
            self.otrilist[otind] = None
            self.otriauxlist.append(otind)
            tri.Lsimplex = None  #remove the references to neighboring simplices
            tri.Rsimplex = None        
        else:  #There are still bands looped over our point, and we must update the outer triangle.
            #Br = Nloop - Wstar  #the number of bands remaining (Don't really use this)
            foundleftside = False
            isimp = csimp
            while not foundleftside:
                cid = isimp.LocalID(cpt)
                if isimp.weights[(cid+2)%3] > 0:
                    tri.Lsimplex = isimp.simplices[(cid+2)%3]
                    foundleftside = True
                else:
                    isimp = isimp.simplices[(cid+2)%3]
            foundrightside = False
            isimp = csimp
            while not foundrightside:
                cid = isimp.LocalID(cpt)
                if isimp.weights[(cid+1)%3] > 0:
                    tri.Rsimplex = isimp.simplices[(cid+1)%3]
                    foundrightside = True
                else:
                    isimp = isimp.simplices[(cid+1)%3]
            
            trimodlistaux.append([])
            trimodlistaux[-1].append([False,None,None])  #marked as false, since this is the main event outer triangle.  This will be deleted regardless.
            tri.SetPoints()
            AZT = self.AreaZeroTime2(tri.points,timein)
            if AZT[0]:
                trimodlistaux[-1].append([AZT[0],AZT[1]])
            else:
                trimodlistaux[-1].append([AZT[0],None])            
            trimodlist.append(tri)
    
        if self.atstep in self.printstep:
            if Wstar == Nloop:
                print("Outer Triangle Removed")
            else:
                print("Outer Triangle Modified: ", tri.points, tri.Rsimplex.points, tri.Lsimplex.points)
    
        #Fix any adjacent outer triangles that are effected (i.e. any whose simplices have been deleted or changed)
        lefttri = self.otriptlist[lpt]
        if not lefttri is None:
            lefttrichange = False
            if lefttri.Rsimplex is Topsimplist[0]:
                lefttri.Rsimplex = csimp
                lefttrichange = True
            elif lefttri.Rsimplex is Botsimplist[0]:
                smatch = DblInTrpl([0,1],lowertri,True)
                lefttri.Rsimplex = nlowersimps[smatch[0]]
                lefttrichange = True
            if lefttri.Lsimplex is Botsimplist[0]:
                lefttrichange = True
                if Wleftnew == 0:
                    lefttri.Lsimplex = csimp.simplices[csimp.LocalID(cpt)]
                else:
                    lefttri.Lsimplex = csimp
            if lefttrichange:
                trimodlistaux.append([])
                AZT = self.AreaZeroTime2(lefttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1],lefttri.points])
                else:
                    trimodlistaux[-1].append([AZT[0],None,lefttri.points])
                lefttri.SetPoints()
                AZT = self.AreaZeroTime2(lefttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1]])
                else:
                    trimodlistaux[-1].append([AZT[0],None])
                
                trimodlist.append(lefttri)
        righttri = self.otriptlist[rpt]
        if not righttri is None:
            righttrichange = False
            if righttri.Lsimplex is Topsimplist[-1]:
                righttri.Lsimplex = csimp
                righttrichange = True
            elif righttri.Lsimplex is Botsimplist[-1]:
                smatch = DblInTrpl([len(lpoints)-2,len(lpoints)-1],lowertri,True)
                righttri.Lsimplex = nlowersimps[smatch[0]]
                righttrichange = True
            if righttri.Rsimplex is Botsimplist[-1]:
                righttrichange = True
                if Wrightnew == 0:
                    righttri.Rsimplex = csimp.simplices[csimp.LocalID(cpt)]
                else:
                    righttri.Rsimplex = csimp
            if righttrichange:
                trimodlistaux.append([])
                AZT = self.AreaZeroTime2(righttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1],righttri.points])
                else:
                    trimodlistaux[-1].append([AZT[0],None,righttri.points])
                righttri.SetPoints()
                AZT = self.AreaZeroTime2(righttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1]])
                else:
                    trimodlistaux[-1].append([AZT[0],None])                
                
                trimodlist.append(righttri)
            
        #need to fix any outer triangles in the bottom point set
        for i in range(1,len(lpoints)-1):
            tritemp = self.otriptlist[lpoints[i]]
            if not tritemp is None:
                if tritemp.Lsimplex is Botsimplist[i-1]:
                    smatch = DblInTrpl([i-1,i],lowertri,True)
                    tritemp.Lsimplex = nlowersimps[smatch[0]]
                if tritemp.Rsimplex is Botsimplist[i]:
                    smatch = DblInTrpl([i,i+1],lowertri,True)
                    tritemp.Rsimplex = nlowersimps[smatch[0]]
                #don't need to re-compute right and left points for this outer triangle (simplices changed, but not the points)
        
        #remove all references to other simplices within each bad simplex
        for i in range(0,len(Botsimplist)):
            for j in range(0,3):
                Botsimplist[i].simplices[j] = None
        
        #return the list of new simplices and list of bad simplices (latter are needed for possible removal from the event list ... if they happened to be part of a later event, which is now not needed).  The new simplices will be checked to see if they must be added to the event list.  Also return a list of the modified outer triangles (which must be checked for inclusion in the event list) ... also the main outer triangle if it is deleted
        return [goodlist,Botsimplist,trimodlist,trimodlistaux]
        #Note on above code: some of the outer triangle fixes are mutually exclusive, and the code would be more efficent if this is reflected in the code (not currently, but could have them arranged in proper if-blocks)
        
        
    #Fixing a concurrent combination of outer triangle and simplex at the same time (the outer triangle event is deleted (though not the outer triangle event itself), and the simplex event is passed in)
    #refer to EventTypes.pdf for a geometric picture of this collapse event, and what the major variables refer to.
    def CombFix(self, SimpIn, timein):
        Simp = SimpIn[0]
        timecoll = SimpIn[1]
        colind = self.CollapsePt(Simp.points,SimpIn[1])  #this is the local index of the offending point during the area collapse
        Topsimp = Simp.simplices[colind]
        cpt = Simp.points[colind]
        rptlid = (colind+1)%3
        lptlid = (colind+2)%3
        rpt = Simp.points[rptlid]
        lpt = Simp.points[lptlid]
        rptuid = Topsimp.LocalID(rpt)
        lptuid = Topsimp.LocalID(lpt)
        tpt = Topsimp.points[(rptuid+1)%3]

        tri = self.otriptlist[cpt]
        #first get the sum of the weights around the central point
        Wsum = tri.PointWeightSum()  #the sum of the weights about the central point, cpt
        Anum = self.anchoredges[cpt]  #The number of anchor edges
        Nloop = int((Wsum-Anum)/2)  #The number of bands looped over the point
        #now generate an intial simplex list below the initial outer triangle (acute angle is up).  This list starts from the left and works its way to the right
        Botsimplist = []
        Botsimplist.append(tri.Lsimplex.LeftSimp(cpt))
        tempsimp = tri.Rsimplex.RightSimp(cpt)
        while not Botsimplist[-1] is tempsimp:
            Botsimplist.append(Botsimplist[-1].LeftSimp(cpt))
        
        #the weight ids [alpha, beta, gamma^0,delta] ... used to create the weight operator
        WeightIDs = [Simp.edgeids[rptlid],Simp.edgeids[lptlid], Botsimplist[0].edgeids[Botsimplist[0].LocalID(lpt)], Simp.edgeids[colind]]
        BottomWeightIDs = []   #the weight ids that will be redistributed among the internal edges of nlowersimps
        for i in range(1,len(Botsimplist)-1):
            BottomWeightIDs.append(Botsimplist[i].edgeids[(Botsimplist[i].LocalID(cpt)+1)%3])
        
        #the weight along the left and right 
        Wleft = tri.GetLeftWeight()
        Wright = tri.GetRightWeight()
        Wstar = min(Wleft,Wright,Nloop)  #This is the weight of the bottom of the new central simplex
        Wpretop = Simp.weights[colind]  #The weight of the edge involved in the collision before the collision
        Wleftnew = Wleft - Wstar + Wpretop     #the new weight of the left top edge of the central simplex (Wpretop is new)
        Wrightnew = Wright - Wstar + Wpretop  #the new weight of the right top edge of the central simplex (Wpretop is new)
        #create the WeightOperator object and append it to the list
        if Wleft == Wstar:
            st = 0
        elif Wright == Wstar:
            st = 1
        else:
            st = 2
        self.WeightOperators.append(WeightOperator(2, WeightIDs, self.atstep + timecoll, st))
        
        #create the new upper simplices.
        rslist = [cpt,rpt,tpt]
        lslist = [cpt,tpt,lpt]
        clist = [cpt,lpt,rpt]
        rsimp = simplex2D(rslist)  #new right simplex
        lsimp = simplex2D(lslist)  #new left simplex
        csimp = simplex2D(clist)   #new central simplex
        #now create the new bottom simplices
        lpoints = [Botsimplist[i].points[(Botsimplist[i].LocalID(cpt)+1)%3] for i in range(0,len(Botsimplist))] #make list of perimeter points
        lpoints.append(rpt)  #add last point on
        lowertri = self.OtriEventTrangulation(lpoints,timecoll)  #Get the list of triangulation triples
        nlowersimps = []
        for i in range(0,len(lowertri)):
            nlowersimps.append(simplex2D([lpoints[lowertri[i][j]] for j in range(0,3)]))  #create the list of new simplices with the triples of points given by the indices stored in lowertri (these are triples of indices for points in lpoints)

        #Now that we have the three new upper simplices and new lower simplices, we need to link them and set their weights
        #Top ones first
        SimpLink(lsimp,Topsimp.simplices[rptuid],True,Topsimp.weights[rptuid])
        SimpLink(lsimp,rsimp,True,0)
        SimpLink(rsimp,Topsimp.simplices[lptuid],True,Topsimp.weights[lptuid])
        SimpLink(lsimp,csimp,True,Wleftnew)
        SimpLink(rsimp,csimp,True,Wrightnew)
        smatch = DblInTrpl([0,len(lpoints)-1],lowertri,True)
        SimpLink(csimp,nlowersimps[smatch[0]],True,Wstar)
        nlowersimps[smatch[0]].edgeids[(nlowersimps[smatch[0]].LocalID(lpt)+1)%3] = WeightIDs[2]
        #and edge id updating
        lsimp.edgeids[0] = Topsimp.edgeids[rptuid]
        lsimp.edgeids[1] = WeightIDs[0]
        lsimp.edgeids[2] = WeightIDs[3]
        rsimp.edgeids[0] = Topsimp.edgeids[lptuid]
        rsimp.edgeids[1] = WeightIDs[3]
        rsimp.edgeids[2] = WeightIDs[1]
        csimp.edgeids[0] = WeightIDs[2]
        csimp.edgeids[1] = WeightIDs[1]
        csimp.edgeids[2] = WeightIDs[0]
        #now fix the links and weights associated with perimeter edges
        for i in range(0,len(lpoints)-1):
            smatch = DblInTrpl([i,i+1],lowertri,True)
            locid = Botsimplist[i].LocalID(cpt)
            SimpLink(nlowersimps[smatch[0]],Botsimplist[i].simplices[locid],True,Botsimplist[i].weights[locid])
            nlowersimps[smatch[0]].edgeids[(nlowersimps[smatch[0]].LocalID(lpoints[i+1])+1)%3] = Botsimplist[i].edgeids[locid]
        #now for the links and weights internal to the lower simplices
        for i in range(0,len(nlowersimps)):
            for j in range(0,3):
                if nlowersimps[i].simplices[j] is None:
                    ptdbl = [nlowersimps[i].points[(j+1)%3],nlowersimps[i].points[(j+2)%3]]
                    inddbl = [lpoints.index(ptdbl[0]),lpoints.index(ptdbl[1])]
                    smatch = DblInTrpl(inddbl,lowertri)  #should give two matches (including the one we already know), however if the nlowersimp is one of the boundary simplices, then we will have only one match (in which case, we do nothing)
                    if len(smatch) == 2:
                        SimpLink(nlowersimps[smatch[0]],nlowersimps[smatch[1]],True,0)  #notice all these weights are zero
                        SimpEdgeIDSet(nlowersimps[smatch[0]],nlowersimps[smatch[1]],BottomWeightIDs[-1])  #set the edge ids
                        del BottomWeightIDs[-1]
        
        #replace bad simplices with new ones (there are the same number) in the simplist (simplex list) 
        goodlist = [csimp,lsimp,rsimp]
        for i in range(0,len(nlowersimps)):
            goodlist.append(nlowersimps[i])
        #badlist = [Simp,Topsimp]
        badlist = [Simp,Topsimp]
        for i in range(0,len(Botsimplist)):
            badlist.append(Botsimplist[i])
            
        for i in range(0,len(badlist)):
            bindex = badlist[i].SLindex
            self.simplist[bindex] = goodlist[i]
            goodlist[i].SLindex = bindex    
        
        #go through the point list (with references to adjacent simplices), and replace any bad simplices if needed
        if self.pointlist[lpoints[0]] in [Botsimplist[0],Simp,Topsimp]:
            self.pointlist[lpoints[0]] = csimp
        for i in range(1,len(lpoints)-1):
            if self.pointlist[lpoints[i]] in [Botsimplist[i-1],Botsimplist[i]]:
                self.pointlist[lpoints[i]] = Botsimplist[i].simplices[Botsimplist[i].LocalID(cpt)]
        if self.pointlist[lpoints[-1]] in [Botsimplist[-1],Simp,Topsimp]:   #deal with the last of lpoints and cpt on their own
            self.pointlist[lpoints[-1]] = csimp
        if self.pointlist[cpt] in Botsimplist or self.pointlist[cpt] is Simp:
            self.pointlist[cpt] = csimp
        if self.pointlist[tpt] is Topsimp:
            self.pointlist[tpt] = lsimp    
        
        #update the outer triangle that has collapsed (if remove, set out triangle point list element to None, and search and remove from the Otrilist)
        trimodlistaux = []
        trimodlist = []
        #tridellist = []
        if Wpretop == 0:  #the colliding edge has no weight
            if Wstar == Nloop:  #only the anchor weights remain, and the outer triangle should be deleted                
                otind = self.otriptlist[cpt].OTLindex  #(3/18)
                self.otriptlist[cpt] = None
                self.otrilist[otind] = None
                self.otriauxlist.append(otind)
                tri.Lsimplex = None  #remove references to neighboring simplices
                tri.Rsimplex = None
            else:   #the outer triangle persists as a collapsed/folded over band
                if Wleftnew == 0:  #the collapsed outer triangle is on the right
                    tri.Lsimplex = csimp
                    tri.Rsimplex = rsimp
                else:   #it has to be that Wrightnew == 0 and the collapsed outer triangle is on the left
                    tri.Lsimplex = lsimp
                    tri.Rsimplex = csimp
                tri.SetPoints()
                #don't need to add to the modify list, because it (an event) is not triggered directly (the area remains zero).  It is triggered as part of a neighboring outer triangle collapse
        else:  #The outer triangle reforms, and we must update it.
            tri.Rsimplex = csimp
            tri.Lsimplex = csimp
            trimodlistaux.append([])
            trimodlistaux[-1].append([False,None,None])
            tri.SetPoints()   #even thought the points don't change, their ordering does, so this is important to get the area sign correct
            AZT = self.AreaZeroTime2(tri.points,timein)
            if AZT[0]:
                trimodlistaux[-1].append([AZT[0],AZT[1]])
            else:
                trimodlistaux[-1].append([AZT[0],None])
            trimodlist.append(tri)
            
        #Fix any adjacent outer triangles that are effected (i.e. any whose simplices have been deleted or changed)
        lefttri = self.otriptlist[lpt]
        if not lefttri is None:
            lefttrichange = False
            if lefttri.Rsimplex is Topsimp:
                lefttri.Rsimplex = lsimp
                lefttrichange = True
            elif lefttri.Rsimplex is Simp:
                lefttri.Rsimplex = csimp
                lefttrichange = True
            elif lefttri.Rsimplex is Botsimplist[0]:
                smatch = DblInTrpl([0,1],lowertri,True)
                lefttri.Rsimplex = nlowersimps[smatch[0]]
            if lefttri.Lsimplex is Topsimp:
                lefttri.Lsimplex = lsimp
            elif lefttri.Lsimplex is Simp:
                lefttri.Lsimplex = csimp
                lefttrichange = True
            elif lefttri.Lsimplex is Botsimplist[0]:
                if Wleftnew == 0:
                    lefttrichange = True
                    lefttri.Lsimplex = csimp.simplices[csimp.LocalID(cpt)]
                else:
                    lefttri.Lsimplex = csimp
            if lefttrichange:
                trimodlistaux.append([])
                AZT = self.AreaZeroTime2(lefttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1],lefttri.points])
                else:
                    trimodlistaux[-1].append([AZT[0],None,lefttri.points])
                lefttri.SetPoints()
                AZT = self.AreaZeroTime2(lefttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1]])
                else:
                    trimodlistaux[-1].append([AZT[0],None])                
                trimodlist.append(lefttri)
        righttri = self.otriptlist[rpt]
        if not righttri is None:
            righttrichange = False
            if righttri.Lsimplex is Topsimp:
                righttri.Lsimplex = rsimp
                righttrichange = True
            elif righttri.Lsimplex is Simp:
                righttri.Lsimplex = csimp
                righttrichange = True
            elif righttri.Lsimplex is Botsimplist[-1]:
                smatch = DblInTrpl([len(lpoints)-2,len(lpoints)-1],lowertri,True)
                righttri.Lsimplex = nlowersimps[smatch[0]]
            if righttri.Rsimplex is Topsimp:
                righttri.Rsimplex = rsimp
            elif righttri.Rsimplex is Simp:
                righttri.Rsimplex = csimp
                righttrichange = True
            elif righttri.Rsimplex is Botsimplist[-1]:
                if Wrightnew == 0:
                    righttrichange = True
                    righttri.Rsimplex = csimp.simplices[csimp.LocalID(cpt)]
                else:
                    righttri.Rsimplex = csimp
            if righttrichange:
                trimodlistaux.append([])
                AZT = self.AreaZeroTime2(righttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1],righttri.points])
                else:
                    trimodlistaux[-1].append([AZT[0],None,righttri.points])
                righttri.SetPoints()
                AZT = self.AreaZeroTime2(righttri.points,timein)
                if AZT[0]:
                    trimodlistaux[-1].append([AZT[0],AZT[1]])
                else:
                    trimodlistaux[-1].append([AZT[0],None])
                trimodlist.append(righttri)
        
        #deal with possible top outer triangles ... note that even if we must change the defining simplices, the points remain unchanged, so we don't need to include this in the trimodlist
        toptri = self.otriptlist[tpt]
        if not (toptri is None):
            if toptri.Rsimplex is Topsimp:
                toptri.Rsimplex = lsimp
            if toptri.Lsimplex is Topsimp:
                toptri.Lsimplex = rsimp
        #need to fix any outer triangles in the bottom point set
        for i in range(1,len(lpoints)-1):
            tritemp = self.otriptlist[lpoints[i]]
            if not tritemp is None:
                if tritemp.Lsimplex is Botsimplist[i-1]:
                    smatch = DblInTrpl([i-1,i],lowertri,True)
                    tritemp.Lsimplex = nlowersimps[smatch[0]]
                if tritemp.Rsimplex is Botsimplist[i]:
                    smatch = DblInTrpl([i,i+1],lowertri,True)
                    tritemp.Rsimplex = nlowersimps[smatch[0]]                    
                #don't need to re-compute right and left points for this outer triangle (simplices changed, but not the points
        
        #remove all references to other simplices within each bad simplex
        for i in range(0,len(badlist)):
            for j in range(0,3):
                badlist[i].simplices[j] = None
        
        #now return the goodlist, badlist, the list of modified triangles, and any deleted triangles
        #don't need Simp in badlist ... will be deleted automatically (n3/18)
        del badlist[0] #Simp is the first element of badlist 
        return [goodlist, badlist, trimodlist,trimodlistaux]
    
    
    #This returns a list of the positions for each point of the given simplex (in the same order)
    def GetSimpPtPos(self,SimpIn):
        SimpPtPos = []
        for i in range(0,len(SimpIn.points)):
            SimpPtPos.append(self.pointpos[SimpIn.points[i]])
        return SimpPtPos
    
    
    #This returns a list of the positions for each point of the given outer triangle (in the same order)
    def GetOuterTrianglePtPos(self,OTriIn):
        OTriPtPos = []
        for i in range(0,len(OTriIn.points)):
            OTriPtPos.append(self.pointpos[OTriIn.points[i]])
        return OTriPtPos
   

    #This returns a list of positions (in the future/next iteratation configuration of the points) for each point of the given simplex (in the same order)
    def GetSimpPtPosFuture(self,SimpIn):
        SimpPtPos = []
        for i in range(0,len(SimpIn.points)):
            SimpPtPos.append(self.pointposfuture[SimpIn.points[i]])
        return SimpPtPos
    
    #This returns a list of positions (in the future/next iteratation configuration of the points) for each point of the given outer triangle (in the same order)
    def GetOTriPtPosFuture(self,OTriIn):
        OTriPtPos = []
        for i in range(0,len(OTriIn.points)):
            OTriPtPos.append(self.pointposfuture[OTriIn.points[i]])
        return OTriPtPos
    
    
    #triangulation2D method to calculate the area of a given simplex (SimpIn)
    def SimpArea(self,SimpIn):
        ptlist = self.GetSimpPtPos(SimpIn)
        return self.TriArea(ptlist)
    
    
    #This calculates the area of the given outer triangle
    def OuterTriArea(self,OTriIn):
        ptlist = self.GetOuterTrianglePtPos(OTriIn)
        return self.TriArea(ptlist)
    

    #returns the signed area of the given pointlist (3 points) ... need to pass points instead of simplices to be able to deal with outer triangles
    def TriArea(self,ptlist):
        return 0.5*((ptlist[1][0]-ptlist[0][0])*(ptlist[2][1]-ptlist[0][1]) - (ptlist[1][1]-ptlist[0][1])*(ptlist[2][0]-ptlist[0][0]))
    
    
    #returns the angle (radians) between the two edges defined by the ordered three points.  If the area is negative, then the angle is given as 2*pi-angle.
    def TriAngle(self,ptlist):
        dp = (ptlist[0][0]-ptlist[1][0])*(ptlist[2][0]-ptlist[1][0]) + (ptlist[0][1]-ptlist[1][1])*(ptlist[2][1]-ptlist[1][1])
        mag1 = math.sqrt((ptlist[0][0]-ptlist[1][0])*(ptlist[0][0]-ptlist[1][0]) + (ptlist[0][1]-ptlist[1][1])*(ptlist[0][1]-ptlist[1][1]))
        mag2 = math.sqrt((ptlist[2][0]-ptlist[1][0])*(ptlist[2][0]-ptlist[1][0]) + (ptlist[2][1]-ptlist[1][1])*(ptlist[2][1]-ptlist[1][1]))
        angle = math.acos(dp/(mag1*mag2))
        if self.TriArea(ptlist) < 0:
            angle = 2*math.pi - angle
        return angle
    
    
    #This function outputs true if the given simplex has the correct orientation (i.e. correct permutation ... signed area is positive)
    def OrderCorrect(self,SimpIn):
        Ocorrect = False
        if self.SimpArea(SimpIn) >= 0:
            Ocorrect = True
        return Ocorrect
    
    
    #This function will load new point positions into pointposfuture
    def LoadNewPos(self,ptlist):
        self.pointposfuture = ptlist + self.extrapoints
    
    
    #This updates the current point positions to those stored in pointposfuture.  This will be called at the end of the evolve method
    def UpdatePtPos(self):
        self.pointpos = self.pointposfuture
        
        
    #This plots the triangulation (points and lines)
    #remember to exclude the three points of the large simplex, and each simplex adjacent to them
    #Eventually want to output to file, or to some other code as part of a movie
    def TriangulationPlot(self):
        xpoints = [x[0] for x in self.pointpos[:len(self.pointpos)-self.extranum]]  #note that we exclude the bounding points
        ypoints = [x[1] for x in self.pointpos[:len(self.pointpos)-self.extranum]]
        triangles = [x.points for x in self.simplist if (len(set(x.points).intersection([(len(self.pointpos)-y) for y in range(1,self.extranum+1)])) == 0)]  #make sure that the list of triangles (triplets of points) do not include the excluded large triangle points
        plt.figure(figsize=(14,14))
        plt.gca().set_aspect('equal')
        plt.triplot(xpoints, ypoints, triangles, 'g-', lw=0.5)
        plt.show()
        #will eventually want to set some constant bounds to make a video possible

    #this will plot the triangulation with non-zero weights in red
    #If Wcutoff is specified, then only those weights >= cutoff are printed in red
    def TriangulationPlotWeights(self,filename = None,Wcutoff = None, xyrange = None):
        xpoints = [x[0] for x in self.pointpos[:len(self.pointpos)-self.extranum]]  #note that we exclude the bounding points
        ypoints = [x[1] for x in self.pointpos[:len(self.pointpos)-self.extranum]]
        triangles = [x.points for x in self.simplist if (len(set(x.points).intersection([(len(self.pointpos)-y) for y in range(1,self.extranum+1)])) == 0)]  #make sure that the list of triangles (triplets of points) do not include the excluded large triangle points
        baselen = 14
        heightlen = 14
        if not xyrange is None:
            xlen = abs(xyrange[0][1] - xyrange[0][0])
            ylen = abs(xyrange[1][1] - xyrange[1][0])
            aspectratio = ylen/xlen
            heightlen = baselen*aspectratio

        plt.figure(figsize=(baselen,heightlen))
        axes = plt.gca()
        axes.set_aspect('equal')
        if not xyrange is None: 
            axes.set_xlim(xyrange[0])
            axes.set_ylim(xyrange[1])
            
        plt.triplot(xpoints, ypoints, triangles, 'k-', lw=0.5)
        gtn = 1
        if not Wcutoff is None:
            gtn = Wcutoff
        #now add on plots of each line-segment that has a non-zero weight
        for i in range(0,len(self.simplist)):
            for j in range(0,3):
                if self.simplist[i].weights[j] >= gtn:
                    p1 = self.simplist[i].points[(j+1)%3]
                    p2 = self.simplist[i].points[(j+2)%3]
                    xpts = [self.pointpos[p1][0],self.pointpos[p2][0]]
                    ypts = [self.pointpos[p1][1],self.pointpos[p2][1]]
                    plt.plot(xpts,ypts,'r-', lw=2.0)
        
        #now add on plots of each line-segment that is part of an outer triangle
        #for i in range(0,len(self.otrilist[i])):
        #    ptid0 = self.otrilist[i].points[0]
        #    ptid1 = self.otrilist[i].points[1]
        #    ptid2 = self.otrilist[i].points[2]
        #    xpts = [self.pointpos[ptid1][0],self.pointpos[ptid0][0],self.pointpos[ptid2][0]]
        #    ypts = [self.pointpos[ptid1][1],self.pointpos[ptid0][1],self.pointpos[ptid2][1]]
        #    plt.plot(xpts,ypts)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close() 
        #will eventually want to set some constant bounds to make a video possible
        
    #Same as above, but also plotting the auxillary points ... mostly for trouble-shooting    
    def TriangulationPlotWhole(self):
        xpoints = [x[0] for x in self.pointpos[:len(self.pointpos)]]
        ypoints = [x[1] for x in self.pointpos[:len(self.pointpos)]]
        triangles = [x.points for x in self.simplist]
        plt.figure(figsize=(14,14))
        plt.gca().set_aspect('equal')
        plt.triplot(xpoints, ypoints, triangles, 'g-', lw=0.5)
        plt.show()
        plt.close()
            
    
    #function that takes in a point list, gets the simplices around each point, then returns the merged list
    #can't be stand-alone, since it accesses the pointlist
    def PtListToSimpList(self,PtListIn):
        #slist = [self.pointlist[PtListIn[i]].SimpNeighbors(PtListIn[i]) for i in range(0,len(PtListIn)) if not self.pointlist[PtListIn[i]].SimpNeighbors(PtListIn[i]) is None]
        slist = []
        for i in range(0,len(PtListIn)):
            #if PtListIn[i] == 502:
            #    print("looking at point: ", PtListIn[i], ", with simplex attached: ", self.pointlist[PtListIn[i]], self.pointlist[PtListIn[i]].points, "at timestep: ", self.atstep)
            slist.append(self.pointlist[PtListIn[i]].SimpNeighbors(PtListIn[i]))      
        outlist = slist[0]
        for i in range(1,len(slist)):
            outlist = ListMerge(outlist,slist[i])
        return outlist
    
    
    #This takes in a list of points (just their id) interpereted as the ccw ordering of perimeter points about some region.  This has the added propertery that there exists some point or interval on the line connecting the first and last points, such that the set of lines from each list point to this point or a point in this range will not intersect.  This is the scenario for the points associated with simplices below (acute angle is up) an outer triangle that is collapsing.  This function will output a list of triplets (the id of points making up new simplices ... the id is for position on the input point list) defining a new triangulation of this region.  The algorithm proceed by looking at all contiguous triples preserving the ccw order and sorting them by angle.  The triplet associated with the most acute angle with then constitute a new simplex.  The central point is then removed from the list, and the proceess repeats until the space is completely triangulated.  The positions of each point are evaluated at time teval.
    def OtriEventTrangulation(self,ListIn,teval = 0):
        poslst = self.PtPosPart(ListIn,teval)
        outlist = self.GetTriRec([[poslst[i],i] for i in range(0,len(poslst))])
        if outlist[0]:
            #print("outlist:", outlist)
            return outlist[1]
        else:
            print("No possible re-triangulation of the lower simplices was found!??")
            return None
        
    
    #recursive function:  returns the triangulation of the given perimeter points (in given in CCW order), halts at three points, and returns False if the area of one of the triangles is negative.  PosIn[i][0] is the position, and PosIn[i][1] is the index. return structure is [True/False,list of triangulation triples (the indices)]
    def GetTriRec(self,PosIn):
        outstruct = []
        sz = len(PosIn)
        if sz == 3:
            #halting condition
            if self.TriArea([PosIn[0][0],PosIn[1][0],PosIn[2][0]]) > 0:
                outstruct = [True,[[PosIn[0][1],PosIn[1][1],PosIn[2][1]]]]
            else:
                outstruct = [False, None]
        else:
            again = True
            counter = 0
            while again:
                if self.TriArea([PosIn[counter][0],PosIn[(counter+1)%sz][0],PosIn[(counter+2)%sz][0]]) > 0:
                    PosInRed = [PosIn[i] for i in range(0,sz) if not i == (counter+1)%sz]
                    outstruct = self.GetTriRec(PosInRed)
                    if outstruct[0]:
                        again = False
                        outstruct[1].append([PosIn[counter][1],PosIn[(counter+1)%sz][1],PosIn[(counter+2)%sz][1]])
                    else:
                        again = True
                        counter += 1
                else:
                    again = True
                    counter += 1
                        
                if counter >= sz:
                    again = False
                    outstruct = [False,None]
        return outstruct
        
    
    #This function returns True if the pair of events passed in are concurrent simplex/outer triangle events (these are handled differently)
    #Some regular events are outer triangle events as well (when the outer triangle coincides with a simplex, and the central outer triangle point is the collapsing point).  When this happens, the time to zero is identical (within numerical precision), and the two events should be treated together.
    def AreConcurrentEvents(self,E1,E2):
        delta = 1e-8     #the time range that two events need to be within to be considered here
        ispair = False
        if abs(E1[1]-E2[1]) < delta:
            #now check to see if the two events share the same vertices (by comparing simplices)
            if (E2[0].Rsimplex is E2[0].Lsimplex) and (E2[0].Rsimplex is E1[0]):
                ispair = True
        return ispair
                        
    
    #this function returns the time (between 0 and 1) when the area of the given simplex/triangle first goes through zero, assuming a linear interpolation of the points.  In the case of an outer triangle or simplex that is created part way through a time interval, the time of creation Tin is inputed (time between 0 and 1) and acts as the initial time in the linear interpolation (really, we just look for the zero area solution with t > Tin and < 1).  The input is a point list instead of a simplex object so that it can work equally well for outer triangles
#this was modified (3/18) to return a pair [IsSoln, timeOut], where IsSoln is a boolian that is true if the first time at which the area goes through zero is between Tin and 1, and false if not.  For IsSoln == True, timeOut gives this time.
    def AreaZeroTime(self,ptlist,Tin = 0,Verbose = False):
        #first get the beginning and end x,y coordinate for each of the three points
        ai = self.pointpos[ptlist[0]]
        bi = self.pointpos[ptlist[1]]
        ci = self.pointpos[ptlist[2]]
        af = self.pointposfuture[ptlist[0]]
        bf = self.pointposfuture[ptlist[1]]
        cf = self.pointposfuture[ptlist[2]]
        #print("Input area zero points: ", [ai,bi,ci], [af,bf,cf])
        #now claculate the coefficients for the quadratic (a*t^2+b*t+c=0)
        a = (cf[0]-ci[0]-af[0]+ai[0])*(bf[1]-bi[1]-af[1]+ai[1]) - (cf[1]-ci[1]-af[1]+ai[1])*(bf[0]-bi[0]-af[0]+ai[0])
        b = (ci[0]-ai[0])*(bf[1]-bi[1]-af[1]+ai[1]) + (cf[0]-ci[0]-af[0]+ai[0])*(bi[1]-ai[1]) - (ci[1]-ai[1])*(bf[0]-bi[0]-af[0]+ai[0]) - (cf[1]-ci[1]-af[1]+ai[1])*(bi[0]-ai[0])
        c = (ci[0]-ai[0])*(bi[1]-ai[1]) - (ci[1]-ai[1])*(bi[0]-ai[0])
        
        roots = np.roots([a,b,c])  #get the two solutions
        #first, make sure that there are two solution (only one if a = 0)
        Tout = Tin
        IsSoln = False  #just declaring this variable
        
        if roots.size == 2:
            t1 = roots[0]
            t2 = roots[1]
            if np.isreal(t1):
                t1ok = False
                t2ok = False
                if t1 >= Tin and t1 <= 1:
                    Tout = t1
                    IsSoln = True
                    t1ok = True
                if t2 >= Tin and t2 <= 1:
                    IsSoln = True
                    t2ok = True
                    if t1ok:
                        if t2 < t1:
                            Tout = t2
                    else:
                        Tout = t2
                if (not t1ok) and (not t2ok) and Verbose:
                    if self.TriArea([af,bf,cf]) < 0:
                        print("times are not in the interval", Tin, " to 1 ... t1 = ", t1, ", and t2 = ", t2)
                        print("The offending points start at: ",[ai,bi,ci], ", and end up at: ", [af,bf,cf])
                        if self.TriArea([ai,bi,ci]) < 0:
                            print("the simplex has an initial configuration with a negative area!")
        elif roots.size == 1:
            t = roots[0]  #real valued
            if t > Tin and t < 1:
                IsSoln = True
                Tout = t

        return [IsSoln, Tout]
    
    #this function is similar to the above function, but returns a pair [IsSoln, timeOut], where IsSoln is a boolian that is true if the first time at which the area goes through zero is between Tin and 1, and false if not.  For IsSoln == True, timeOut gives this time.
    def AreaZeroTime2(self,ptlist,Tin = 0,Verbose = False):
        #first get the beginning and end x,y coordinate for each of the three points
        ai = self.pointpos[ptlist[0]]
        bi = self.pointpos[ptlist[1]]
        ci = self.pointpos[ptlist[2]]
        af = self.pointposfuture[ptlist[0]]
        bf = self.pointposfuture[ptlist[1]]
        cf = self.pointposfuture[ptlist[2]]
        #print("Input area zero points: ", [ai,bi,ci], [af,bf,cf])
        #now claculate the coefficients for the quadratic (a*t^2+b*t+c=0)
        a = (cf[0]-ci[0]-af[0]+ai[0])*(bf[1]-bi[1]-af[1]+ai[1]) - (cf[1]-ci[1]-af[1]+ai[1])*(bf[0]-bi[0]-af[0]+ai[0])
        b = (ci[0]-ai[0])*(bf[1]-bi[1]-af[1]+ai[1]) + (cf[0]-ci[0]-af[0]+ai[0])*(bi[1]-ai[1]) - (ci[1]-ai[1])*(bf[0]-bi[0]-af[0]+ai[0]) - (cf[1]-ci[1]-af[1]+ai[1])*(bi[0]-ai[0])
        c = (ci[0]-ai[0])*(bi[1]-ai[1]) - (ci[1]-ai[1])*(bi[0]-ai[0])
        #print("a = ", a, ", b = ", b , ", c = ", c)
        
        IsSoln = False  #just declaring this variable
        Tout = Tin
        
        numzero = 0
        if a == 0:
            numzero += 1
        if b == 0:
            numzero += 1            
        if c == 0:
            numzero += 1        
        
        #roots = (-b +- sqrt(b**2 - 4*a*c))/(2*a)
        
        if numzero == 0:
            q = b**2 - 4*a*c
            if q > 0:
                #two real roots
                t1 = 2*c/(-b-math.sqrt(q))
                t2 = 2*c/(-b+math.sqrt(q))                
                #t1 = (-b-math.sqrt(q))/(2*a)
                #t2 = (-b+math.sqrt(q))/(2*a)
                t1ok = False
                t2ok = False
                if t1 > Tin and t1 < 1:
                    Tout = t1
                    IsSoln = True
                    t1ok = True
                if t2 > Tin and t2 < 1:
                    IsSoln = True
                    t2ok = True
                    if t1ok:
                        if t2 < t1:
                            Tout = t2
                    else:
                        Tout = t2
                if (not t1ok) and (not t2ok) and Verbose:
                    if self.TriArea([af,bf,cf]) < 0:
                        print("times are not in the interval", Tin, " to 1 ... t1 = ", t1, ", and t2 = ", t2)
                        print("The offending points start at: ",[ai,bi,ci], ", and end up at: ", [af,bf,cf])
            elif q == 0:
                #one real root
                t = -b/(2*a)
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
            #else:   #two complex solutions, nothing to do
        elif numzero == 1:
            if a == 0:
                t = -c/b
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
            elif b == 0 and -c/a > 0:
                t = math.sqrt(-c/a)
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
            else:
                #c = 0 , discarding the t = 0 soln
                t = -b/a
                if t > Tin and t < 1:
                    Tout = t
                    IsSoln = True
        #else:   #don't need to treat the cases of 2 and 3 of a,b,c being zero ... 

        return [IsSoln, Tout]
       
    
    #this returns the linearly interpolated positions of each point in ptlist (usually 3, but can handle other lengths) at the time 0 < teval < 1.
    def PtPosPart(self,ptlist,teval):
        posi = []
        posf = []
        for i in range(0,len(ptlist)):
            posi.append(self.pointpos[ptlist[i]])
            posf.append(self.pointposfuture[ptlist[i]])
        posout = []
        for i in range(0,len(ptlist)):
            posout.append([(posf[i][0]-posi[i][0])*teval + posi[i][0], (posf[i][1]-posi[i][1])*teval + posi[i][1]])
        return posout   
         
        
    #This returns the point (internal id) that passes through its opposite edge during an area collapse event known to occur at t = tcol
    def CollapsePt(self,ptlist,tcol):
        #first get the positions of the 3 points at the time of collapse
        colpos = self.PtPosPart(ptlist,tcol)
        whichpoint = 0
        d0 = (colpos[2][0] - colpos[0][0])*(colpos[1][0]-colpos[0][0]) + (colpos[2][1] - colpos[0][1])*(colpos[1][1]-colpos[0][1])  #This is the dot product of (z2-z0) and (z1-z0) ... < 0 if 0 is the middle point
        if d0 < 0:
            whichpoint = 0
        else:
            d1 = (colpos[2][0] - colpos[1][0])*(colpos[0][0]-colpos[1][0]) + (colpos[2][1] - colpos[1][1])*(colpos[0][1]-colpos[1][1])
            if d1 < 0:
                whichpoint = 1
            else:
                whichpoint = 2   #don't need to calculate the last dot product.  If the first two are >0, this must be <0
        return whichpoint
        
           
    #This methods sums up all of the weights (excluding those attached to an auxilliary point)
    def GetWeightTotal(self):
        bigsum = 0
        extrapts = [(self.ptnum-1-i) for i in range(0,self.extranum)]
        for i in range(0,len(self.simplist)):
            localbadpts = []
            for j in range(0,3):
                if self.simplist[i].points[j] in extrapts:
                    localbadpts.append(j)
            if len(localbadpts) == 0:
                for j in range(0,3):
                    bigsum += self.simplist[i].weights[j]
            elif len(localbadpts) == 1:
                bigsum += self.simplist[i].weights[localbadpts[0]]
            #if len(localbadpts) == 2, then we neglect all edge weights
        return bigsum/2
    
    
    #This is similar to the above function, but it multiplies each weight by the actual length between the two points
    def GetLengthTotal(self):
        bigsum = 0
        extrapts = [(self.ptnum-1-i) for i in range(0,self.extranum)]
        for i in range(0,len(self.simplist)):
            localbadpts = []
            for j in range(0,3):
                if self.simplist[i].points[j] in extrapts:
                    localbadpts.append(j)
            if len(localbadpts) == 0:
                pos = [self.pointpos[self.simplist[i].points[j]] for j in range(3)]
                ptlen = [math.sqrt((pos[(j+1)%3][0]-pos[(j+2)%3][0])**2+(pos[(j+1)%3][1]-pos[(j+2)%3][1])**2) for j in range(3)]
                for j in range(0,3):            
                    bigsum += self.simplist[i].weights[j]*ptlen[j]
            elif len(localbadpts) == 1:
                pos = []
                pos.append(self.pointpos[self.simplist[i].points[(localbadpts[0]+1)%3]])
                pos.append(self.pointpos[self.simplist[i].points[(localbadpts[0]+2)%3]])
                ptlen = math.sqrt((pos[0][0]-pos[1][0])**2+(pos[0][1]-pos[1][1])**2)
                bigsum += self.simplist[i].weights[localbadpts[0]]*ptlen
            #if len(localbadpts) == 2, then we neglect all edge weights
        return bigsum/2     
    
    
    #This returns true if the current outer triangle, that is being evaluated because it will go through an area collapse, is folded over at the time of collapse (and therefore does not need to be evaluated as a collapse event)
    def WillFold(self,TriptIn,tcoll):
        Foldlater = False
        #first we need the time of collapse
        #now look at the allignment of the points
        configpos = self.PtPosPart(TriptIn,tcoll)
        d0 = (configpos[2][0] - configpos[0][0])*(configpos[1][0]-configpos[0][0]) + (configpos[2][1] - configpos[0][1])*(configpos[1][1]-configpos[0][1])  #this is the dot product (z1-z0)*(z2-z0), where z0 is the center outer triangle point.
        #if d0 > 0 then it is folded over, if the triangle is in a regular collapse configuration, then the d0 < 0
        if self.atstep in self.printstep:
            print("Looking at a potential folded outer triangle, time of collapse = ", tcoll, ", and dot product d0 = ", d0)   
        if d0 > 0:
            #this is the case where the triangle is folded over
            Foldlater = True
        return Foldlater
    
    #This returns the two simplices that share the edge between the two points (point indices) in PairIn.  If the two points are not adjacent in the current triangulation or the edge is a boundary edge, then this returns None
    def GetEdgePairSimp(self,PairIn):
        SL = self.pointlist[PairIn[0]].SimpNeighbors(PairIn[0])
        STM = []
        for k in range(0,len(SL)):
            if PairIn[1] in SL[k].points:
                STM.append(SL[k])
            if len(STM) == 2:
                break
        if len(STM) == 2:
            return STM
        else:
            return None
        
    #This is a function that prints out the current state of the triangulation (the points from the simplices and outer 
    #triangles, and their weights ... enough info to reconstruct the current state of the triangulation)
    def PrintState(self):
        #first print a list of the simplices (points) and their weight
        for i in range(0,len(self.simplist)):
            print("simplex ",self.simplist[i].points, ", with weights ",self.simplist[i].weights,", and area = ",self.SimpArea(self.simplist[i]))
        for i in range(0,len(self.otrilist)):
            if not self.otrilist[i] is None:
                print("outer triangle ", self.otrilist[i].points, ", and area = ", self.OuterTriArea(self.otrilist[i]))        
     
    
    #These triangulation2D methods are currently not used ****************************************************************************************************************************************************************************************************************************************************************
    
    #sets the area data of the given simplex to its calculated area
    def SetSimpArea(self,SimpIn):
        SimpIn.area = self.SimpArea(SimpIn)

        
    #sets the area data of the given outer triangle to its calculated area
    def SetOuterTriArea(self,OTriIn):
        OTriIn.area = self.OuterTriArea(OTriIn)
    
    
        #this calculates the area for the first numset simplices in simplist and stores them in the proper simplex
    #default will update the area of all simplices
    def SetAreaSimpList(self,numset = None):
        if numset is None:
            numset = len(self.simplist)
        numuse = numset
        if numset > len(self.simplist) or numset < 0:
            numuse = len(self.simplist)
        for i in range(0,numuse):
            self.SetSimpArea(self.simplist[i])
            
    #this calculates the area for the first numset outer triangle in otrilist and stores them in the proper outer triangle
    #default will update the area of all outer triangles
    def SetAreaOTriList(self,numset = None):
        if numset is None:
            numset = len(self.otrilist)
        numuse = numset
        if numset > len(self.otrilist) or numset < 0:
            numuse = len(self.otrilist)
        for i in range(0,numuse):
            self.SetOuterTriArea(self.otrilist[i])        
            
            
    #This calculates the "arearate" for a given simplex based off of the current areas and those implied by the positions in pointposfuture.  This will be called in Evolve method, after updating the pointposfuture and before putting pointposfuture in pointpos
    def SimpAreaRate(self,SimpIn):
        ptlist = self.GetSimpPtPosFuture(SimpIn)
        A1 = self.TriArea(ptlist)
        A0 = SimpIn.area
        return (A1/A0-1)
    
    #sets the arearate data of the given simplex to its calculated arearate
    def SetSimpAreaRate(self,SimpIn):
        SimpIn.arearate = self.SimpAreaRate(SimpIn)
 
    #this calculates the arearate for the first numset simplices in simplist and stores them in the proper simplex
    #default will update the arearate of all simplices
    def SetAreaRateSimpList(self,numset = None):
        if numset is None:
            numset = len(self.simplist)
        numuse = numset
        if numset > len(self.simplist) or numset < 0:
            numuse = len(self.simplist)
        for i in range(0,numuse):
            self.SetSimpAreaRate(self.simplist[i])
            
    #This calculates the "arearate" for a given outer triangle based off of the current areas and those implied by the positions in pointposfuture.  This will be called in Evolve method, after updating the pointposfuture and before putting pointposfuture in pointpos
    def OTriAreaRate(self,OTriIn):
        ptlist = self.GetOTriPtPosFuture(OTriIn)
        A1 = self.TriArea(ptlist)
        A0 = OTriIn.area
        return (A1/A0-1)
    
    
    #sets the arearate data of the given outer triangle to its calculated arearate
    def SetOTriAreaRate(self,OTriIn):
        OTriIn.arearate = self.OTriAreaRate(OTriIn)
        
        
    #this calculates the arearate for the first numset outer triangle in otrilist and stores them in the proper 
    #default will update the arearate of all simplices
    def SetAreaRateOTriList(self,numset = None):
        if numset is None:
            numset = len(self.otrilist)
        numuse = numset
        if numset > len(self.otrilist) or numset < 0:
            numuse = len(self.otrilist)
        for i in range(0,numuse):
            self.SetOTriAreaRate(self.otrilist[i])    
    
    
    #Sorting the whole simplex list based on the area of each simplex.  This is an in-place sort. Or if endInd is specified:
    #Sorts the first endInd elements in simplist ... not in-place.
    def SimpAreaSort(self, endInd = None):
        if endInd is None:
            self.simplist.sort(key=attrgetter('area'))
        else:
            if endInd <= len(self.simplist) or endInd < 0:
                endInd = len(self.simplist)
            self.simplist[0:endInd] = sorted(self.simplist[0:endInd],key=attrgetter('area'))
        
        
    #Sorting the whole simplex list based on the area normalized rate of change of the area. Or if endInd is specified:
    #Sorts the first endInd elements in simplist ... not in-place.
    def SimpAreaRateSort(self,endInd = None):
        if endInd is None:
            self.simplist.sort(key=attrgetter('arearate'))
        else:
            if endInd <= len(self.simplist) or endInd < 0:
                endInd = len(self.simplist)
            self.simplist[0:endInd] = sorted(self.simplist[0:endInd],key=attrgetter('arearate'))
    
    
    #Sorting the whole outer triangle list based on the area of each outer triangle.
    #This is an in-place sort. Or if endInd is specified: Sorts the first endInd elements in otrilist ... not in-place.
    def OTriAreaSort(self, endInd = None):
        if endInd is None:
            self.otrilist.sort(key=attrgetter('area'))
        else:
            if endInd <= len(self.otrilist) or endInd < 0:
                endInd = len(self.otrilist)
            self.otrilist[0:endInd] = sorted(self.otrilist[0:endInd],key=attrgetter('area'))
    
    
    #Sorting the whole outer triangle list based on the area normalized rate of change of the area.
    #This is an in-place sort. Or if endInd is specified: Sorts the first endInd elements in otrilist ... not in-place.
    def OTriAreaRateSort(self, endInd = None):
        if endInd is None:
            self.otrilist.sort(key=attrgetter('arearate'))
        else:
            if endInd <= len(self.otrilist) or endInd < 0:
                endInd = len(self.otrilist)
            self.otrilist[0:endInd] = sorted(self.otrilist[0:endInd],key=attrgetter('arearate'))
    
    
    
    

#End of triangulation2D class ****************************************************************************************************************************************************************************************************************************************************************




#Set of general use stand-alone functions ****************************************************************************************************************************************************************************************************************************************************************
    
#Stand-alone function to input one of my trajectory files and output a list of points in the proper format for this
#Nan wrote this, modfied just a bit
def OpenTrajectoryFile(fileName):
    #a list of time with evolving x,y coordinates ... time is evenly spaced, and we don't need the actual value right now (though for normalizing the topological entropy the total time elapsed will be useful)
    wholeList = []
    times = []
    #open and record
    with open(fileName,"r") as f:
        #for each time
        for line in f:
            #coordinates
            listXY = []
            a = line.split(" ")
            #delete the first element which is the time
            words = a[1:]
            times.append(float(a[0]))
            m = round((len(words))/2)
            for x in range(0, m):
                listXY.append([float(words[2*x]),float(words[2*x+1])])
            wholeList.append(listXY)
    return [times,wholeList]  #return the times too


#this helper function takes in a pair of points and a list of triples, and returns the index or indices of the triples that include these two points.  To make this more efficient, the function stops after two matches (can only be two simplices that share an edge), or one match if IsSide = True (the double represents an edge on the side of this simplex set)
def DblInTrpl(dblin,trpllistin,IsSide = False):
    catchcount = 0
    outlist = []
    for i in range(0,len(trpllistin)):
        if (dblin[0] in trpllistin[i]) and (dblin[1] in trpllistin[i]):
            outlist.append(i)
            catchcount += 1
            if IsSide:
                break
            elif catchcount == 2:
                break
    return outlist
           
    
#this takes two simplices that share an edge and links them (puts eachother in the appropriate neighbor simplex list)
#this assumes that the two simplices do share an edge.  If they don't this can create erronious links
#This also takes the given weight and puts it in the edge (for both S1 and S2) shared by S1 and S2 
def SimpLink(S1,S2,wadd = False, wnew = 0):
    #need to deal with the case of either S1 or S2 being None
    if not ((S1 is None) or (S2 is None)):  #Note that we don't need to do anything for linking a new simplex to None ... it starts out that way
        #first find out which points they share
        locid = 0
        for i in range(0,3):
            if not S1.points[i] in S2.points:
                S1.simplices[i] = S2
                if wadd:
                    S1.weights[i] = wnew
                locid = i
                break
        smset = [S1.points[(locid+1)%3],S1.points[(locid+2)%3]]
        for i in range(0,3):
            if not S2.points[i] in smset:
                S2.simplices[i] = S1
                if wadd:
                    S2.weights[i] = wnew
                break    

#this takes two simplices that share an edge and are already linked, and sets their shared weight (or adds to it)
#If add is False (default), then the shared weight is set to wnew, if it is True, then wnew is added to the current weight
def SimpWeightSet(S1,S2,wnew, add = False):
    #need to deal with the case of either S1 or S2 being None
    if not ((S1 is None) or (S2 is None)):  #Just a check for bad simplices
        #first find out which points they share
        locid = 0
        for i in range(0,3):
            if not S1.points[i] in S2.points:
                if add:
                    S1.weights[i] += wnew
                else:
                    S1.weights[i] = wnew
                locid = i
                break
        smset = [S1.points[(locid+1)%3],S1.points[(locid+2)%3]]
        for i in range(0,3):
            if not S2.points[i] in smset:
                if add:
                    S2.weights[i] += wnew
                else:
                    S2.weights[i] = wnew
                break   
#*********can make the above slighly more efficient by copying the method below
                
                
#This takes two simplices that share an edge and are already linked, and sets their shared edge ID.
def SimpEdgeIDSet(S1,S2,edgeIDnew):
    #need to deal with the case of either S1 or S2 being None
    if not ((S1 is None) or (S2 is None)):  #Just a check for bad simplices
        #first find out which points they share
        locid = 0
        for i in range(0,3):
            if not S1.points[i] in S2.points:
                S1.edgeids[i] = edgeIDnew
                locid = i
                break
        spt1 = S1.points[(locid+1)%3]
        Lid = (S2.LocalID(spt1)+1)%3
        S2.edgeids[Lid] = edgeIDnew    
                
    
#Stand-alone function that takes a single list of simplices and returns a reduced version that excludes any duplicates
def SimpListCompact(SimpListIn):
    newList = []
    goodList = [True for i in range(0,len(SimpListIn))]
    for i in range(0,len(SimpListIn)):
        if goodList[i]:
            newList.append(SimpList[i])
        for j in range(i+1,len(SimpListIn)):
            if goodList[j]:
                if SimplistIn[i] is SimplistIn[j]:
                    goodList[j] = False
    return newList


#Stand-alone function that takes two lists of simplices, and merges them, avoiding duplicates
#can also accomodate lists of anything that are being compared by the is operator and have overlap
def ListMerge(SList1,SList2):
    #assume that each list is already compact (no duplicates)
    newList = [SList1[i] for i in range(0,len(SList1))]
    refList = [SList1[i] for i in range(0,len(SList1))]
    for i in range(0,len(SList2)):
        isgood = True
        for j in range(0,len(refList)):
            if refList[j] is SList2[i]:
                isgood = False
                del refList[j]
                break
        if isgood:
            newList.append(SList2[i])
    return newList


#Stand-alone function that returns True if the two lists have any common elements (used for simplices but can be use for any lists with common element types that can be compared with the is operator)
def ListIntersect(SList1, SList2):
    #assume that each list is already compact (no duplicates)
    Intersection = False
    for i in range(0,len(SList2)):
        for j in range(0,len(SList1)):
            if SList1[j] is SList2[i]:
                Intersection = True
                break
        if Intersection:
            break
    return Intersection
     
    
#Stand-alone function which takes a list of simplices and returns a list of point IDs with no duplicates
#can accomodate outer tringles directly ... so the input list can be a mix of simplices and outer triangles
def SimpListToPtList(SListIn):
    ptlist = SListIn[0].points
    for i in range(1,len(SListIn)):
        ptlist = ListMerge(ptlist,SListIn[i].points)
    return ptlist

#Stand-alone function which does a binary search on a given sorted list (each element is a double, were the second item is the ordering parameter).  The list is assumed to be in decending order.  The item that is searched for is also a double [event,time to zero area].  The search is over the time variable, but the event variable is used for direct comparison.  If a match is found, then it is deleted from the list.
def BinarySearchDel(ListIn, ItemIn):
    Lindex = 0
    Rindex = len(ListIn) - 1
    success = False
    matchindex = 0
    while Rindex >= Lindex and not success:
        Mindex = (Rindex+Lindex)//2
        if ListIn[Mindex][0] == ItemIn[0]:
            #have a match
            success = True
            matchindex = Mindex
        else:
            if ItemIn[1] < ListIn[Mindex][1]:
                Lindex = Mindex + 1
            else:
                Rindex = Mindex - 1
    if success:
        del ListIn[matchindex]
    else:
        print("did not delete item from EventList, event was not found")
            

#Stand-alone function which does a binary search on a given sorted list (each element is a double, were the second item is the ordering parameter).  The list is assumed to be in decending order.  The item that is searched for is also a double [event,time to zero area].  The binary search finds the adjacent pair of elements inbetween which the input item's time fits.  If such a pair is found, then the ItemIn is inserted into this position.
def BinarySearchIns(ListIn, ItemIn):
    Lindex = 0
    Rindex = len(ListIn) - 1
    if len(ListIn) == 0:
        ListIn.append(ItemIn)
    elif ItemIn[1] < ListIn[Rindex][1]:
        ListIn.append(ItemIn)
    elif ItemIn[1] > ListIn[Lindex][1]:
        ListIn.insert(0,ItemIn)
    else:
        while Rindex - Lindex > 1:
            Mindex = (Rindex+Lindex)//2
            if ItemIn[1] < ListIn[Mindex][1]:
                Lindex = Mindex
            else:
                Rindex = Mindex
        if Rindex - Lindex == 1:
            ListIn.insert(Rindex,ItemIn)
        else:
            #right an left indices are concurrent.  This can happen when ItemIn has an identical time to one of the
            #items in ListIn.  These are either the same object (in which case we don't insert), or we have a future combined event (with a simplex collapse and a triangle collapse being concurrent ... in which case we add in the new object)
            if not type(ItemIn[0]) == type(ListIn[Rindex][0]):
                ListIn.insert(Rindex,ItemIn)

#This is a stand-alone function that takes two Trajectory Time-slices, and outputs the trajectory time-slice that is a fraction of the way (halfway is the default) between the two in time (as a linear interpolation)
#It is assumed that the two time-slices are of the same size (and order), and that frac is in [0,1]
#it is also assumed that TS1 comes before TS2 in time
def FracTraj(TS1, TS2, frac = 0.5):
    OutTraj = []
    for i in range(0,len(TS1)):
        OutTraj.append([(TS2[i][0]-TS1[i][0])*frac + TS1[i][0], (TS2[i][1]-TS1[i][1])*frac + TS1[i][1]])
    return OutTraj
      
#this is a stand-alone function that takes a trajectory set, and outputs a trajectory set as a copy of the first, but with additional linearly interpolated time-slices between each adjacent time slice in the original (equally spaced in time)
def TrajInFill(Traj):
    TrajOut = []
    for i in range(0,len(Traj)-1):
        TrajOut.append(Traj[i])
        TrajOut.append(FracTraj(Traj[i],Traj[i+1]))
    TrajOut.append(Traj[-1])
    return TrajOut                
        
#Notes:  
#The cycling around a point to get anchor weights or surrounding simplices has been fixed to accomodate the case of the edge points.  However, it wouldn't be an issue if we instead of the large bounding triangle, we used a (symbolic) point at infinity (making this a sphere).  Of course, that approach would need some special treatment for simplex areas.