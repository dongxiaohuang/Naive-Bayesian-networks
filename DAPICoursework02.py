#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python
from DAPICourseworkLibrary import *
from numpy import *
from math import log
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for i in theData[:,root]:
         prior[i] += 1
    prior /= len(theData[:,root]) # normalization
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
    counterP = zeros((noStates[varP])) # the number of states of P
    for c,p in theData[:, [varC,varP]]:
        counterP[p] += 1
        cPT[c,p] += 1
    cPT /= counterP # normalization
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here
    for row,col in theData[:,[varRow, varCol]]:
        jPT[row, col] += 1
    jPT /= len(theData[:,varRow]); # normalization
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
    prior = zeros(len(aJPT[0])) # column length
    for row in range(len(aJPT[:,0])):
        for col in range(len(aJPT[0])):
            prior[col] += aJPT[row,col]
    prior /= sum(prior) # normalization
    aJPT /= prior # Bayesian Formula
# coursework 1 taks 4 ends here
    return aJPT

# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    for i in range(len(rootPdf)):
        rootPdf[i] = prior[i] # initial
        for j in range(len(theQuery)):
            rootPdf[i] *= naiveBayes[j+1][theQuery[j],i]
    rootPdf /= sum(rootPdf) # normalization
# end of coursework 1 task 5
    return rootPdf

# End of Coursework 1

# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables

def MutualInformation(jP):
    mi=0.0
    BASE = 2
# Coursework 2 task 1 should be inserted here
# marginalize the joint probability table
    jP = jP/ jP.sum()
    pRow = numpy.sum(jP, axis = 1)
    pCol = numpy.sum(jP, axis = 0)
    for i in range(len(pRow)):
        for j in range(len(pCol)):
            if(jP[i,j] * pRow[i] * pCol[j] == 0):
                continue
            jPIJ = jP[i,j]
            mi += jPIJ * log(jPIJ/(pRow[i]*pCol[j]),BASE)
# end of coursework 2 task 1
    return mi

#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for i in range(noVariables):
        for j in range(noVariables):
            if(j <= i):
                jP = JPT(theData, i, j, noStates)
                MIMatrix[i,j] = MutualInformation(jP)
    MIMatrix += MIMatrix.T - numpy.diag(MIMatrix.diagonal())
# end of coursework 2 task 2
    return MIMatrix

# Function to compute an ordered list of dependencies
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    upperTriangular =  depMatrix[numpy.triu_indices(len(depMatrix[0]))]
    matrixLen = len(depMatrix)
    for node1 in range(matrixLen):
        for node2 in range(matrixLen):
            if(node2 > node1):
                dependency = upperTriangular[node1*matrixLen + node2 - node1*(node1+1)/2]
                # print dependency
                depList.append([dependency, node1, node2])
    depList2 = sorted(depList, reverse = True)
# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    addedNode = []

    depList = sorted(depList, key=lambda depList: depList[0], reverse = True)
    # initial
    addedNode.append(depList[0][1])
    addedNode.append(depList[0][2])
    spanningTree.append(depList[0])
    depList.remove(depList[0])
    depList = [elem.tolist() for elem in depList]
    spanningTree = buildSpanningTree(depList, addedNode,spanningTree)
    return array(spanningTree)

def buildSpanningTree(depList, addedNode, spanningTree):
    i = 0
    if len(depList) == 0:
        return []
    while (len(depList) != 0):
        node1 = depList[i][1]
        node2 = depList[i][2]
        if(node1 not in addedNode and node2 not in addedNode):
            i+=1
            continue
        if(node1 in addedNode and node2 in addedNode):
            depList.remove(depList[i])
            continue
        if(node1 in addedNode or node2 in addedNode):
            spanningTree.append(depList[i])
            addedNode.append(node1)
            addedNode.append(node2)
            depList.remove(depList[i])
            buildSpanningTree(depList, addedNode, spanningTree)
    return spanningTree

# End of coursework 2




# main program part for Coursework 1
# # Part1 Q1
# noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
# theData = array(datain)
# AppendString("results.txt","Coursework One Results by Dongxiao Huang(dh4317)")
# AppendString("results.txt","") #blank line
# AppendString("results.txt","The prior probability of node 0")
# prior = Prior(theData, 0, noStates)
# AppendList("results.txt", prior)
# # Part1 Q2
# AppendString("results.txt","") #blank line
# AppendString("results.txt","The conditional probability matrix P(2|0)")
# cPT = CPT(theData, 2, 0, noStates)
# AppendArray("results.txt", cPT)
# # Part1 Q3
# AppendString("results.txt","") #blank line
# AppendString("results.txt","The joint probability matrix P(2&0)")
# jPT = JPT(theData, 2, 0, noStates)
# AppendArray("results.txt", jPT)
#
# # Part1 Q4
# AppendString("results.txt","") #blank line
# AppendString("results.txt","The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0)")
# aJPT = JPT2CPT(jPT)
# AppendArray("results.txt", aJPT)
# # Part1 Q5
# AppendString("results.txt","") #blank line

# cpt1 = CPT(theData, 1, 0, noStates)
# cpt2 = CPT(theData, 2, 0, noStates)
# cpt3 = CPT(theData, 3, 0, noStates)
# cpt4 = CPT(theData, 4, 0, noStates)
# cpt5 = CPT(theData, 5, 0, noStates)
# naiveBayes = [prior,cpt1,cpt2,cpt3,cpt4,cpt5]
# AppendString("results.txt","The results of queries[4,0,0,0,5] on the naive network")
# theQuery = [4,0,0,0,5]
# rootPdf = Query(theQuery, naiveBayes)
# AppendList("results.txt", rootPdf)
# AppendString("results.txt","The results of queries[6,5,2,5,5] on the naive network")
# theQuery = [6,5,2,5,5]
# rootPdf = Query(theQuery, naiveBayes)
# AppendList("results.txt", rootPdf)

# Coursework02
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("DAPIResults02.txt","Coursework Two Results by Dongxiao Huang(dh4317)")
AppendString("DAPIResults02.txt","") #blank line

#Part2 Q2
AppendString("DAPIResults02.txt","The dependency matrix for HepatitisC data set")
MIMatrix = DependencyMatrix(theData, noVariables, noStates)
AppendArray("DAPIResults02.txt", MIMatrix)
#Part2 Q3
AppendString("DAPIResults02.txt","The dependency list for HepatitisC data set")
dependecyList = DependencyList(MIMatrix)
AppendArray("DAPIResults02.txt", dependecyList)
#Part2 Q4
AppendString("DAPIResults02.txt","The spanning tree found for HepatitisC data set")
spanningTree = SpanningTreeAlgorithm(dependecyList, noVariables)
AppendArray("DAPIResults02.txt", spanningTree)

#
