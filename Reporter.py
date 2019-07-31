## Ben Ryan C15507277 
## Machine Learning Assignment 1 Data Quality Report

## This program reads in the provided dataset and outputs 2 .csv files.
## One file containing an analysis of the continuous data, the other 
## containing an analysis of the categorical data

import pandas as pd
import numpy as np
import warnings

## This function reads in the headers from the feature names file
## Next it converts it to a list so it can be used when reading in the data
## I have also selected which features are continuous and which are categorical here
def getHeaders():
	headersDF = pd.read_csv('./data/feature_names.txt', header=None, nrows=16)
	
	headersList = [""]*len(headersDF)
	for row in range(0,len(headersDF)):
		headersList[row] = headersDF[0][row]
	
	conFeat = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
	catFeat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'target']
	
	return headersList, conFeat, catFeat

## This function takes the list of continuos features and the full data set.
## With this it selects just the continuous columns for processing
## It forms a new dataframe of appropriate size then applies the required
## calculations to these columns and places the results in the new dataframe
def processContinuous(conFeat, data):
	conHead = ['Count', 'Miss %', 'Card.', 'Min', '1st Qrt.', 'Mean', 'Median', '3rd Qrt', 'Max', 'Std. Dev.']
	
	conOutDF = pd.DataFrame(index=conFeat, columns=conHead)
	conOutDF.index.name = 'FEATURENAME'
	columns = data[conFeat]

	#COUNT
	count = columns.count()
	conOutDF[conHead[0]] = count
	
	#MISS % - no continuous features have missing data
	percents = ['']*len(conFeat)
	for col in columns:
		percents[conFeat.index(col)] = 0.00

	conOutDF[conHead[1]] = percents
	
	#CARDINALITY
	conOutDF[conHead[2]] = columns.nunique()
	
	#MINIMUM
	conOutDF[conHead[3]] = columns.min()
	
	#1ST QUARTILE
	conOutDF[conHead[4]] = columns.quantile(0.25)
	
	#MEAN
	conOutDF[conHead[5]] = round(columns.mean(), 2)
	
	#MEDIAN
	conOutDF[conHead[6]] = columns.median()
	
	#3rd QUARTILE
	conOutDF[conHead[7]] = columns.quantile(0.75)
	
	#MAX
	conOutDF[conHead[8]] = columns.max()
	
	#STANDARD DEVIATION
	conOutDF[conHead[9]] = round(columns.std(),2)
	
	return conOutDF
	
## This function takes the list of categorical features and the full data set
## It will then extract just the categorical columns for processing
## It forms a new dataframe of appropriate size then processes the data
## and stores it in the new dataframe
def processCategorical(catFeat, data):
	catHead = ['Count', 'Miss %', 'Card.', 'Mode', 'Mode Freq', 'Mode %', '2nd Mode', '2nd Mode Freq', '2nd Mode %']

	catOutDF = pd.DataFrame(index=catFeat, columns=catHead)
	catOutDF.index.name = 'FEATURENAME'
	columns = data[catFeat]
	
	#COUNT
	count = columns.count()
	catOutDF[catHead[0]] = count
	
	#CARDINALITY
	catOutDF[catHead[2]] = columns.nunique()

	#preparing arrays for storing data
	amt = len(catFeat)
	missPercents = ['']*amt
	modeFreqs = ['']*amt
	modes = ['']*amt
	modes2 = ['']*amt
	modePercents = ['']*amt
	modeFreqs2 = ['']*amt
	modePercents2 = ['']*amt

	for col in columns:
		values = columns[col].value_counts()
		index = catFeat.index(col)
		
		#MISS %
		try:
			qMarksCount = values.loc[' ?']
			percent = (qMarksCount/count[index]) * 100
			missPercents[index] = round(percent, 2)
			
			#adjust cardinality to account for ? being counted as unique value
			catOutDF['Card.'][index] -= 1
		except Exception as e:
			missPercents[index] = 0.00
		
		#MODES
		mode = values.index[0]
		mode2 = values.index[1]
		modes[index] = mode
		modes2[index] = mode2
		
		#MODE FREQ
		modeCount = values.loc[mode]
		modeCount2 = values.loc[mode2]
		modeFreqs[index] = modeCount
		modeFreqs2[index] = modeCount2

		#MODE %
		miss = missPercents[index]
		
		modePer = (modeCount/(count[index]*((100-miss)/100)))*100
		modePercents[index] = round(modePer, 2)
		
		modePer2 = (modeCount2/(count[index]*((100-miss)/100)))*100
		modePercents2[index] = round(modePer2, 2)

	catOutDF[catHead[1]] = missPercents
	catOutDF[catHead[3]] = modes
	catOutDF[catHead[4]] = modeFreqs
	catOutDF[catHead[5]] = modePercents
	catOutDF[catHead[6]] = modes2
	catOutDF[catHead[7]] = modeFreqs2
	catOutDF[catHead[8]] = modePercents2
	
	return catOutDF
	
## controls the flow of the program
def main():
	#syntax of certain things will be changing in the future of pandas
	#this just disables the warning about it.
	warnings.simplefilter(action='ignore', category=Warning)

	allHead, conFeat, catFeat = getHeaders()
	
	#READ DATA, with headers joined on
	data = pd.read_csv('./data/dataset.txt', header=None, nrows=30940, names=allHead)

	#PROCESS DATA
	conOutDF = processContinuous(conFeat, data)
	catOutDF = processCategorical(catFeat, data)

	#WRITE TO FILES
	conOutDF.to_csv("./data/ContinuousReport.csv")
	catOutDF.to_csv("./data/CategoricalReport.csv")

if __name__ == '__main__':
	main()