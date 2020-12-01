import snap
import sys
import csv
import numpy as np

argList = sys.argv
#sub_graph = argList[0]

#file = open(sub_graph,'r')
#csv_reader = csv.reader(file,delimiter='\t')

# create a graph PNGraph
Graph = snap.TNGraph.New()
Graph.AddNode(1)
Graph.AddNode(2)
Graph.AddNode(3)
if(not Graph.IsNode(2)):
  Graph.AddNode(2)

for node in Graph.Nodes():
	print(node.GetId())