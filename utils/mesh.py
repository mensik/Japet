import re
import numpy
import h5py
from Tkinter import *

class Mesh:
    def loadH5F(self, fileName):
        file = h5py.File(fileName, 'r')
        
        mesh = file['ENS_MAA'].values()[0] #Obtain the first mesh in the file
        coords = numpy.reshape(mesh['NOE']['COO'][:], (-1,2),'F')
        ids = mesh['NOE']['NUM'][:]
        family = mesh['NOE']['FAM'][:]
        
        self.vetrices = {}
        for i in range(ids.size):
            self.vetrices[ids[i]] = tuple([family[i], tuple(2*x + 250 for x in coords[i])])
        
        elements = numpy.reshape(mesh['MAI']['TR3']['NOD'], (-1,3),'F')
        family = mesh['MAI']['TR3']['FAM'][:]
        
        self.elements = {}
        for i in range(family.size):
            self.elements[i] = tuple([family[i], tuple(elements[i])])
        
        
        print self.elements
    
    def load(self, fileName):
        file = open(fileName, 'r')
        
        #Vetrices
        line = file.readline()
        while not re.match("#Vetrices *", line) :
            line = file.readline()
        line = file.readline() #info with number of elements
        
        self.vetrices = {}
        line = file.readline() 
        while not re.match("#Elements *", line) :
            data = re.split("[: ]", line.strip())
            self.vetrices[data[0]] = tuple([data[4], tuple(float(x) + 20 for x in data[1:4])])
            line = file.readline() 

        self.elements = {}
        line = file.readline() 
        line = file.readline() 
        while line :
            data = re.split("[: ]", line.strip())
            self.elements[data[0]] = tuple([data[1], tuple(data[3:])])
            line = file.readline()
            
NODE_SIZE = 1

class App:
    def __init__(self, master, mesh):
        self.mesh = mesh
        self.master = master
        frame = Frame(master, padx=5, pady=5)
        frame.pack()
        self.mainCanvas = Canvas(frame, width=850, height=750, bg="white")
        self.mainCanvas.grid(row=0, column=0, sticky=N+W)
        
        infoPane = LabelFrame(frame, text="Info", padx=3, pady=3)
        infoPane.grid(row=0, column=1, sticky=W+E)
        
        self.ip = infoPane
        
        infoPane.labelId = Label(infoPane, text="ID:", width=10)
        infoPane.labelId.grid(row=0, column=0)
        
        infoPane.idVariable=StringVar()
        infoPane.entryId = Entry(infoPane, width = 5, textvariable=infoPane.idVariable)
        infoPane.entryId.grid(row=0, column=1)
        
        
        self.elements = {}
        self.invElements = {}
        for k in self.mesh.elements.keys():
            nodes = self.mesh.elements[k][1]
            coords = [self.mesh.vetrices[node][1][:2] for node in nodes]
            self.elements[k] = self.mainCanvas.create_polygon(coords, fill='', outline='black')
            self.invElements[self.elements[k]] = k
            
        self.vetrices = {}
        self.invVetrices = {}
        for k in self.mesh.vetrices.keys():
            coords = self.mesh.vetrices[k][1][:2]
            self.vetrices[k] = self.mainCanvas.create_oval(coords[0] - NODE_SIZE, coords[1] - NODE_SIZE, coords[0] + NODE_SIZE, coords[1] + NODE_SIZE, fill="blue")
            self.invVetrices[self.vetrices[k]] = k
            

        
        self.mainCanvas.bind("<Button-1>", self.popup)

        self.alteredCanvasWidgets = []
        
        self.popupMenu = Menu(master, tearoff=0)
        self.popupMenu.add_command(label="TEST")
    
    def cleanCanvas(self):
        for it in self.alteredCanvasWidgets :
            if it in self.invElements.keys():
                self.mainCanvas.itemconfig(it, fill='white')
            elif it in self.invVetrices.keys():
                pass
        self.alteredCanvasWidgets = []
        
    def markDomain(self, domainId):
        for id, elm in self.mesh.elements.iteritems():
            if elm[0] == domainId: 
                self.mainCanvas.itemconfigure(self.elements[id], fill='green')
                self.alteredCanvasWidgets.append(self.elements[id])
                
    def popup(self,event):
 
        x = self.mainCanvas.canvasx(event.x)
        y = self.mainCanvas.canvasy(event.y)
        list = self.mainCanvas.find_overlapping(x - 1, y - 1, x + 1, y + 1)
        
        elemCmd = []
        elemText = []
        
        for it in list:
            if it in self.invElements.keys():
                elemCmd.append(lambda: self.elementAction(self.invElements[it]))
                elemText.append("Element: " + self.invElements[it])
            elif it in self.invVetrices.keys():
                elemCmd.append(lambda: self.nodeAction(self.invVetrices[it]))
                elemText.append("Node: " + self.invVetrices[it])
                
        if len(elemCmd) == 1 :
            elemCmd[0]()
            return
        elif len(elemCmd) == 0 :
            return
        
        self.popupMenu.destroy()
        self.popupMenu = Menu(self.master, tearoff=0)
        
        for mIt in zip(elemText, elemCmd):
            self.popupMenu.add_command(label=mIt[0], command=mIt[1])
        
        self.popupMenu.post(event.x_root, event.y_root)
    
    def elementAction(self, idElement):
        self.cleanCanvas()
        itId = self.elements[idElement]
        
        self.ip.idVariable.set(str(idElement))
        self.markDomain(self.mesh.elements[idElement][0])
        
        self.mainCanvas.itemconfig(itId, fill='red')
        self.alteredCanvasWidgets.append(itId)
    
    def nodeAction(self, idNode):
        self.cleanCanvas()

        
        

def main():
    m = Mesh()
    m.loadH5F("../deploy/test.med")
    #m.load("outMesh.msh")
    
    root = Tk()
    App(root, m)
    root.mainloop()

if __name__ == "__main__":
    main()
