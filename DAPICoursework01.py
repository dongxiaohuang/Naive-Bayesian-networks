#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python
from DAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for i in theData[:,root]:
         prior[i] += 1
    prior /= len(theData[:,root])
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
    cPT /= counterP
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here
    for row,col in theData[:,[varRow, varCol]]:
        jPT[row, col] += 1
    jPT /= len(theData[:,varRow]);
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
    prior /= sum(prior)
    aJPT /= prior
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
    rootPdf /= sum(rootPdf)
# end of coursework 1 task 5
    return rootPdf

# End of Coursework 1

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
AppendString("results.txt","Coursework One Results by Dongxiao Huang(dh4317)")
AppendString("results.txt","") #blank line
AppendString("results.txt","The prior probability of node 0")
prior = Prior(theData, 0, noStates)
AppendList("results.txt", prior)
# Part1 Q2
AppendString("results.txt","") #blank line
AppendString("results.txt","The conditional probability matrix P(2|0)")
cPT = CPT(theData, 2, 0, noStates)
AppendArray("results.txt", cPT)
# Part1 Q3
AppendString("results.txt","") #blank line
AppendString("results.txt","The joint probability matrix P(2&0)")
jPT = JPT(theData, 2, 0, noStates)
AppendArray("results.txt", jPT)

# Part1 Q4
AppendString("results.txt","") #blank line
AppendString("results.txt","The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0)")
aJPT = JPT2CPT(jPT)
AppendArray("results.txt", aJPT)
# Part1 Q5
AppendString("results.txt","") #blank line

cpt1 = CPT(theData, 1, 0, noStates)
cpt2 = CPT(theData, 2, 0, noStates)
cpt3 = CPT(theData, 3, 0, noStates)
cpt4 = CPT(theData, 4, 0, noStates)
cpt5 = CPT(theData, 5, 0, noStates)
naiveBayes = [prior,cpt1,cpt2,cpt3,cpt4,cpt5]
AppendString("results.txt","The results of queries[4,0,0,0,5] on the naive network")
theQuery = [4,0,0,0,5]
rootPdf = Query(theQuery, naiveBayes)
AppendList("results.txt", rootPdf)
AppendString("results.txt","The results of queries[6,5,2,5,5] on the naive network")
theQuery = [6,5,2,5,5]
rootPdf = Query(theQuery, naiveBayes)
AppendList("results.txt", rootPdf)


# continue as described
#
