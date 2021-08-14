# Script to analyze data for Neuropsychological evidence of multi-domain network hubs in the human thalamus
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nibabel.processing import resample_from_to
from nilearn.image import resample_to_img
import nilearn
import scipy
import os
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nilearn import masking
from nilearn import plotting
import matplotlib.pyplot as plt
from scipy.stats import zscore
from mlxtend.evaluate import permutation_test
from scipy import stats, linalg
from sklearn.linear_model import LinearRegression
sns.set_context("paper")

def print_demographic(df):
	try:
		print('Thalamus group lesion size')
		df.loc[df.Group=='Thalamus']['Lesion Size'].mean()
		df.loc[df.Group=='Thalamus']['Lesion Size'].std()
		print('Thalamus group age')
		print(df.loc[df.Group=='Thalamus']['Age'].mean())
		print(df.loc[df.Group=='Thalamus']['Age'].std())
		print('Thalamus group educ')
		print(df.loc[df.Group=='Thalamus']['Educ'].mean())
		print(df.loc[df.Group=='Thalamus']['Educ'].std())
	except:
		return

	try:
		print('Comparison group lesion size')
		df.loc[df.Group=='Comparison']['Lesion Size'].mean()
		df.loc[df.Group=='Comparison']['Lesion Size'].std()
		print('Comparison group age')
		print(df.loc[df.Group=='Comparison']['Age'].mean())
		print(df.loc[df.Group=='Comparison']['Age'].std())
		print('Comparison group educ')
		print(df.loc[df.Group=='Comparison']['Educ'].mean())
		print(df.loc[df.Group=='Comparison']['Educ'].std())
	except:
		return

	try:
		print('Ex Comparison group lesion size')
		df.loc[df.Group=='Expanded Comparison']['Lesion Size'].mean()
		df.loc[df.Group=='Expanded Comparison']['Lesion Size'].std()
		print('Expanded Comparison group age')
		print(df.loc[df.Group=='Expanded Comparison']['Age'].mean())
		print(df.loc[df.Group=='Expanded Comparison']['Age'].std())
		print('Expanded Comparison group educ')
		print(df.loc[df.Group=='Expanded Comparison']['Educ'].mean())
		print(df.loc[df.Group=='Expanded Comparison']['Educ'].std())
	except:
		return


def load_and_normalize_neuropsych_data(df):
	''' normalize neuropsych data using popluation norm'''

	# load data
	#df = pd.read_csv('data/data.csv')
	# remove acute patietns (chronicity < 3 months)
	#df = df.loc[df['Chronicity']>2].reset_index()

	#calculate several composite scores of RAVLT
	df['RAVLT_Immediate_Recall'] = df['RAVLT_T1']
	df['RAVLT_Learning'] =  df['RAVLT_T5'] +  df['RAVLT_T4'] +  df['RAVLT_T3'] +  df['RAVLT_T2'] +  df['RAVLT_T1']- 5* df['RAVLT_T1']

	# Norms for TMT
	# TMT norm data from Tombaugh, T. N. (2004). Trail Making Test A and B: normative data stratified by age and education. Archives of Clinical Neuropsychology: The Official Journal of the National Academy of Neuropsychologists, 19(2), 203–214.
	TMTA_norm ={
	'24': {'mean': 22.93, 'sd': 6.87},
	'34': {'mean': 24.40, 'sd': 8.71},
	'44': {'mean': 28.54, 'sd': 10.09},
	'54': {'mean': 31.78, 'sd': 9.93},
	'59+': {'mean': 31.72, 'sd': 10.14},
	'59-': {'mean': 35.10, 'sd': 10.94},
	'64+': {'mean': 31.32, 'sd': 6.96},
	'64-': {'mean': 33.22, 'sd': 9.10},
	'69+': {'mean': 33.84, 'sd': 6.69},
	'69-': {'mean': 39.14, 'sd': 11.84},
	'74+': {'mean': 40.13, 'sd': 14.48},
	'74-': {'mean': 42.47, 'sd': 15.15},
	'79+': {'mean': 41.74, 'sd': 15.32},
	'79-': {'mean': 50.81, 'sd': 17.44},
	'84+': {'mean': 55.32, 'sd': 21.28},
	'84-': {'mean': 58.19, 'sd': 23.31},
	'89+': {'mean': 63.46, 'sd': 29.22},
	'89-': {'mean': 57.56, 'sd': 21.54},
	}

	TMTB_norm ={
	'24': {'mean': 48.97, 'sd': 12.69},
	'34': {'mean': 50.68 , 'sd': 12.36},
	'44': {'mean': 58.46, 'sd': 16.41},
	'54': {'mean': 63.76, 'sd': 14.42},
	'59+': {'mean': 68.74 , 'sd': 21.02},
	'59-': {'mean': 78.84 , 'sd': 19.09},
	'64+': {'mean': 64.58, 'sd': 18.59},
	'64-': {'mean': 74.55, 'sd': 19.55},
	'69+': {'mean': 67.12, 'sd': 9.31},
	'69-': {'mean': 91.32, 'sd': 28.8},
	'74+': {'mean': 86.27, 'sd': 24.07},
	'74-': {'mean': 109.95, 'sd': 35.15},
	'79+': {'mean': 100.68, 'sd': 44.16},
	'79-': {'mean': 130.61, 'sd': 45.74},
	'84+': {'mean': 132.15, 'sd': 42.95},
	'84-': {'mean': 152.74, 'sd': 65.68},
	'89+': {'mean': 140.54, 'sd': 75.38},
	'89-': {'mean': 167.69, 'sd': 78.50},
	}

	#BNT norms
	# Norm from: Tombaugh, T. N., & Hubiey, A. M. (1997). The 60-item Boston Naming Test: Norms for cognitively intact adults aged 25 to 88 years. Journal of Clinical and Experimental Neuropsychology, 19(6), 922-932.
	# 18-39 55.8 (3.8)
	# 40-49 56.8 (3)
	# 50-59 55.2 (4)
	# 60-69 53.3 (4.6)
	# 70-79 48.9 (6.3)

	BNT_norm ={
	'39': {'mean': 55.8, 'sd': 3.8},
	'49': {'mean': 56.8 , 'sd': 3},
	'59': {'mean': 55.2, 'sd': 4},
	'69': {'mean': 53.3, 'sd': 4.6},
	'79': {'mean': 48.9 , 'sd': 6.3},
	}

	#COWA norms: In years of educ
	#M, 12  36.9 (9.8)
	#   13-15   40.5 (9.4)
	#   16  41 (9.8)
	#F, 12  35.9 (9.6)
	#   13-15   39.4 (10.1)
	#   16  46.5(11.2)

	COWA_norm = {
	'M12': {'mean': 36.9, 'sd': 9.8},
	'M15': {'mean': 40.5 , 'sd': 9.4},
	'M16': {'mean': 41, 'sd': 9.8},
	'F12': {'mean': 35.9, 'sd': 9.6},
	'F15': {'mean': 39.4 , 'sd': 10.1},
	'F16': {'mean': 46.5 , 'sd': 11.2},
	}

	# RAVLT norms from : Ivnik, R. J., Malec, J. F., Tangalos, E. G., Petersen, R. C., Kokmen, E., & Kurland, L. T. (1990). The Auditory-Verbal Learning Test (AVLT): norms for ages 55 years and older. Psychological Assessment: A Journal of Consulting and Clinical Psychology, 2(3), 304.
	# RVLT Delay recall
	#   55-59   10.4 (3.1)
	#   60-64   9.9 (3.1)
	#   65-69   8.3 (3.5)
	#   70-74   7.4 (3.1)
	#   75-79   6.9 (2.9)
	#   80-84   5.5 (3.3)
	#   85-     5.4 (2.7)

	RAVLT_Delayed_Recall_norm = {
	'59': {'mean': 10.4, 'sd': 3.1},
	'64': {'mean': 9.9 , 'sd': 3.1},
	'69': {'mean': 8.3, 'sd': 3.5},
	'74': {'mean': 7.4, 'sd': 3.1},
	'79': {'mean': 6.9 , 'sd': 2.9},
	'84': {'mean': 5.5 , 'sd': 3.3},
	'85': {'mean': 5.4 , 'sd': 2.7},
	}

	# RAVLT  Recognition
	#   55-59   14.1 (1.3)
	#   60-64   13.9 (1.5)
	#   65-69   13.3 (2)
	#   70-74   12.7 (2.1)
	#   75-79   12.5 (2.4)
	#   80-84   12.3 (2.4)
	#   85-     12.3 (2.3)

	RAVLT_Recognition_norm = {
	'59': {'mean': 14.1, 'sd': 1.3},
	'64': {'mean': 13.9 , 'sd': 1.5},
	'69': {'mean': 13.3, 'sd': 2},
	'74': {'mean': 12.7, 'sd': 2.1},
	'79': {'mean': 12.5 , 'sd': 2.4},
	'84': {'mean': 12.3 , 'sd': 2.4},
	'85': {'mean': 12.3 , 'sd': 2.3},
	}

	#RAVLT learning Norms, from trial 1 to trial 5

	#trial 1
	RAVLT_T1_norm = {
	'59': {'mean': 6.8, 'sd': 1.6},
	'64': {'mean': 6.4 , 'sd': 1.9},
	'69': {'mean': 5.7, 'sd': 1.6},
	'74': {'mean': 5.5, 'sd': 1.5},
	'79': {'mean': 5.0 , 'sd': 1.5},
	'84': {'mean': 4.4 , 'sd': 1.5},
	'85': {'mean': 4.0 , 'sd': 1.8},
	}

	#trial 1 is also immediate learning
	RAVLT_Immediate_Recall_norm = {
	'59': {'mean': 6.8, 'sd': 1.6},
	'64': {'mean': 6.4 , 'sd': 1.9},
	'69': {'mean': 5.7, 'sd': 1.6},
	'74': {'mean': 5.5, 'sd': 1.5},
	'79': {'mean': 5.0 , 'sd': 1.5},
	'84': {'mean': 4.4 , 'sd': 1.5},
	'85': {'mean': 4.0 , 'sd': 1.8},
	}

	#trial 2
	RAVLT_T2_norm = {
	'59': {'mean': 9.5, 'sd': 2.2},
	'64': {'mean': 9.0 , 'sd': 2.3},
	'69': {'mean': 8.6, 'sd': 2.1},
	'74': {'mean': 7.8, 'sd': 1.8},
	'79': {'mean': 7.0 , 'sd': 1.9},
	'84': {'mean': 6.5 , 'sd': 1.5},
	'85': {'mean': 6.0 , 'sd': 1.8},
	}

	#trial 3
	RAVLT_T3_norm = {
	'59': {'mean': 11.4, 'sd': 2.0},
	'64': {'mean': 10.6 , 'sd': 2.3},
	'69': {'mean': 9.7, 'sd': 2.3},
	'74': {'mean': 9.1, 'sd': 2.1},
	'79': {'mean': 8.2 , 'sd': 2.2},
	'84': {'mean': 7.7 , 'sd': 2.1},
	'85': {'mean': 7.4 , 'sd': 2.2},
	}

	#trial 4
	RAVLT_T4_norm = {
	'59': {'mean': 12.4, 'sd': 1.9},
	'64': {'mean': 11.7 , 'sd': 2.7},
	'69': {'mean': 10.6, 'sd': 2.4},
	'74': {'mean': 10.2, 'sd': 2.4},
	'79': {'mean': 9.2 , 'sd': 2.2},
	'84': {'mean': 8.6 , 'sd': 2.5},
	'85': {'mean': 7.9 , 'sd': 2.4},
	}

	#trial 5
	RAVLT_T5_norm = {
	'59': {'mean': 13.1, 'sd': 1.9},
	'64': {'mean': 11.9 , 'sd': 2.0},
	'69': {'mean': 11.2, 'sd': 2.4},
	'74': {'mean': 10.5, 'sd': 2.6},
	'79': {'mean': 10.1 , 'sd': 2.2},
	'84': {'mean': 9.0 , 'sd': 2.5},
	'85': {'mean': 9.1 , 'sd': 2.3},
	}

	# Learning, which is sum of T1 to T5 minus 5* T1
	RAVLT_Learning_norm = {
	'59': {'mean': 19.3, 'sd': 5.8},
	'64': {'mean': 17.8 , 'sd': 7.0},
	'69': {'mean': 17.2, 'sd': 6.1},
	'74': {'mean': 15.6, 'sd': 6.9},
	'79': {'mean': 14.5 , 'sd': 6.6},
	'84': {'mean': 14.3 , 'sd': 7.3},
	'85': {'mean': 14.4 , 'sd': 6.6},
	}


	# rey o complex figure construction
	# From: Fastenau, P. S., Denburg, N. L., & Hufford, B. J. (1999). Adult norms for the Rey-Osterrieth Complex Figure Test and for supplemental recognition and matching trials from the Extended Complex Figure Test. The Clinical Neuropsychologist, 13(1), 30-47.
	#   40  32.83 (3.1)
	#   50  31.79 (4.55)
	#   60  31.94 (3.37)
	#   70  31.76 (3.64)
	#   80  30.14 (4.52)

	Complex_Figure_Copy_norm = {
	'40': {'mean': 32.83, 'sd': 3.1},
	'50': {'mean': 31.79 , 'sd': 4.55}, #40 - 55
	'60': {'mean': 31.62, 'sd': 4.61}, #45-60
	'70': {'mean': 31.76, 'sd': 3.64},	#60-75
	'80': {'mean': 30.14 , 'sd': 4.52}, #75-85
	}

	# rey o complex figure delayed recall.
	# Note the recall is 30 minute delay
	#   40  19.28 (7.29)
	#   50  17.13 (7.14)
	#   60  16.55 (6.08)
	#   70  15.18 (5.57)
	#   80  13.17 (5.32)
	Complex_Figure_Recall_norm = {
	'40': {'mean': 19.28, 'sd': 7.29},
	'50': {'mean': 17.13 , 'sd': 7.14}, #40 - 55
	'60': {'mean': 17.48, 'sd': 7.11},	#45-60
	'70': {'mean': 15.29, 'sd': 5.57},  #60-75
	'80': {'mean': 13.17 , 'sd': 5.32}, #75-85
	}

	# Convert raw scores to norm adjusted z score
	for i in df.index:

		#Normalize TMT
		if df.loc[i, 'Age'] <= 34:
			df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['34']['mean']) / TMTA_norm['34']['sd']
		elif 34 < df.loc[i, 'Age'] <= 44:
			df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['44']['mean']) / TMTA_norm['44']['sd']
		elif 44 < df.loc[i, 'Age'] <= 54:
			df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['54']['mean']) / TMTA_norm['54']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] <= 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['59-']['mean']) / TMTA_norm['59-']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['64-']['mean']) / TMTA_norm['64-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69-']['mean']) / TMTA_norm['69-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69-']['mean']) / TMTA_norm['69-']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['74-']['mean']) / TMTA_norm['74-']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['79-']['mean']) / TMTA_norm['79-']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['84-']['mean']) / TMTA_norm['84-']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['89-']['mean']) / TMTA_norm['89-']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] > 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['59+']['mean']) / TMTA_norm['59+']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['64+']['mean']) / TMTA_norm['64+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69+']['mean']) / TMTA_norm['69+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69+']['mean']) / TMTA_norm['69+']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['74+']['mean']) / TMTA_norm['74+']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['79+']['mean']) / TMTA_norm['79+']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['84+']['mean']) / TMTA_norm['84+']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['89+']['mean']) / TMTA_norm['89+']['sd']


		if df.loc[i, 'Age'] <= 34:
			df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['34']['mean']) / TMTB_norm['34']['sd']
		elif 34 < df.loc[i, 'Age'] <= 44:
			df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['44']['mean']) / TMTB_norm['44']['sd']
		elif 44 < df.loc[i, 'Age'] <= 54:
			df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['54']['mean']) / TMTB_norm['54']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] > 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['59+']['mean']) / TMTB_norm['59+']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['64+']['mean']) / TMTB_norm['64+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69+']['mean']) / TMTB_norm['69+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69+']['mean']) / TMTB_norm['69+']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['74+']['mean']) / TMTB_norm['74+']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['79+']['mean']) / TMTB_norm['79+']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['84+']['mean']) / TMTB_norm['84+']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['89+']['mean']) / TMTB_norm['89+']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] <= 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['59-']['mean']) / TMTB_norm['59-']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['64-']['mean']) / TMTB_norm['64-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69-']['mean']) / TMTB_norm['69-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69-']['mean']) / TMTB_norm['69-']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['74-']['mean']) / TMTB_norm['74-']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['79-']['mean']) / TMTB_norm['79-']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['84-']['mean']) / TMTB_norm['84-']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['89-']['mean']) / TMTB_norm['89-']['sd']

		#Normalize BNT
		if df.loc[i, 'Age'] <= 39:
			df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['39']['mean']) / BNT_norm['39']['sd']
		elif 39 < df.loc[i, 'Age'] <= 49:
			df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['49']['mean']) / BNT_norm['49']['sd']
		elif 49 < df.loc[i, 'Age'] <= 59:
			df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['59']['mean']) / BNT_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['69']['mean']) / BNT_norm['69']['sd']
		elif 69 < df.loc[i, 'Age']:
			df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['79']['mean']) / BNT_norm['79']['sd']

		#Cowa
		if (df.loc[i, 'Sex'] == 'M') & (df.loc[i, 'Educ'] <= 12):
			df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['M12']['mean']) / COWA_norm['M12']['sd']
		elif (df.loc[i, 'Sex'] == 'M') & (12 < df.loc[i, 'Educ'] <= 15):
			df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['M15']['mean']) / COWA_norm['M15']['sd']
		elif (df.loc[i, 'Sex'] == 'M') & (15 < df.loc[i, 'Educ']):
			df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['M16']['mean']) / COWA_norm['M16']['sd']
		elif (df.loc[i, 'Sex'] == 'F') & (df.loc[i, 'Educ'] <= 12):
			df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['F12']['mean']) / COWA_norm['F12']['sd']
		elif (df.loc[i, 'Sex'] == 'F') & (12 < df.loc[i, 'Educ'] <= 15):
			df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['F15']['mean']) / COWA_norm['F15']['sd']
		elif (df.loc[i, 'Sex'] == 'F') & (15 < df.loc[i, 'Educ']):
			df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['F16']['mean']) / COWA_norm['F16']['sd']

		#RAVLT
		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['59']['mean']) / RAVLT_Delayed_Recall_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['64']['mean']) / RAVLT_Delayed_Recall_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['69']['mean']) / RAVLT_Delayed_Recall_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['74']['mean']) / RAVLT_Delayed_Recall_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['79']['mean']) / RAVLT_Delayed_Recall_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['84']['mean']) / RAVLT_Delayed_Recall_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['85']['mean']) / RAVLT_Delayed_Recall_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Delayed_Recognition'] - RAVLT_Recognition_norm['59']['mean']) / RAVLT_Recognition_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Delayed_Recognition'] - RAVLT_Recognition_norm['64']['mean']) / RAVLT_Recognition_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Delayed_Recognition'] - RAVLT_Recognition_norm['69']['mean']) / RAVLT_Recognition_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Delayed_Recognition'] - RAVLT_Recognition_norm['74']['mean']) / RAVLT_Recognition_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Delayed_Recognition'] - RAVLT_Recognition_norm['79']['mean']) / RAVLT_Recognition_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Delayed_Recognition'] - RAVLT_Recognition_norm['84']['mean']) / RAVLT_Recognition_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Delayed_Recognition'] - RAVLT_Recognition_norm['85']['mean']) / RAVLT_Recognition_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['59']['mean']) / RAVLT_T1_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['64']['mean']) / RAVLT_T1_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['69']['mean']) / RAVLT_T1_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['74']['mean']) / RAVLT_T1_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['79']['mean']) / RAVLT_T1_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['84']['mean']) / RAVLT_T1_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['85']['mean']) / RAVLT_T1_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['59']['mean']) / RAVLT_T2_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['64']['mean']) / RAVLT_T2_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['69']['mean']) / RAVLT_T2_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['74']['mean']) / RAVLT_T2_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['79']['mean']) / RAVLT_T2_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['84']['mean']) / RAVLT_T2_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['85']['mean']) / RAVLT_T2_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['59']['mean']) / RAVLT_T3_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['64']['mean']) / RAVLT_T3_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['69']['mean']) / RAVLT_T3_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['74']['mean']) / RAVLT_T3_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['79']['mean']) / RAVLT_T3_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['84']['mean']) / RAVLT_T3_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['85']['mean']) / RAVLT_T3_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['59']['mean']) / RAVLT_T4_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['64']['mean']) / RAVLT_T4_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['69']['mean']) / RAVLT_T4_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['74']['mean']) / RAVLT_T4_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['79']['mean']) / RAVLT_T4_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['84']['mean']) / RAVLT_T4_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['85']['mean']) / RAVLT_T4_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['59']['mean']) / RAVLT_T5_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['64']['mean']) / RAVLT_T5_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['69']['mean']) / RAVLT_T5_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['74']['mean']) / RAVLT_T5_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['79']['mean']) / RAVLT_T5_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['84']['mean']) / RAVLT_T5_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['85']['mean']) / RAVLT_T5_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_Learning_z'] = (df.loc[i, 'RAVLT_Learning'] - RAVLT_Learning_norm['59']['mean']) / RAVLT_Learning_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_Learning_z'] = (df.loc[i, 'RAVLT_Learning'] - RAVLT_Learning_norm['64']['mean']) / RAVLT_Learning_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_Learning_z'] = (df.loc[i, 'RAVLT_Learning'] - RAVLT_Learning_norm['69']['mean']) / RAVLT_Learning_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_Learning_z'] = (df.loc[i, 'RAVLT_Learning'] - RAVLT_Learning_norm['74']['mean']) / RAVLT_Learning_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_Learning_z'] = (df.loc[i, 'RAVLT_Learning'] - RAVLT_Learning_norm['79']['mean']) / RAVLT_Learning_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_Learning_z'] = (df.loc[i, 'RAVLT_Learning'] - RAVLT_Learning_norm['84']['mean']) / RAVLT_Learning_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_Learning_z'] = (df.loc[i, 'RAVLT_Learning'] - RAVLT_Learning_norm['85']['mean']) / RAVLT_Learning_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_Immediate_Recall_z'] = (df.loc[i, 'RAVLT_Immediate_Recall'] - RAVLT_Immediate_Recall_norm['59']['mean']) / RAVLT_Immediate_Recall_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_Immediate_Recall_z'] = (df.loc[i, 'RAVLT_Immediate_Recall'] - RAVLT_Immediate_Recall_norm['64']['mean']) / RAVLT_Immediate_Recall_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_Immediate_Recall_z'] = (df.loc[i, 'RAVLT_Immediate_Recall'] - RAVLT_Immediate_Recall_norm['69']['mean']) / RAVLT_Immediate_Recall_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_Immediate_Recall_z'] = (df.loc[i, 'RAVLT_Immediate_Recall'] - RAVLT_Immediate_Recall_norm['74']['mean']) / RAVLT_Immediate_Recall_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_Immediate_Recall_z'] = (df.loc[i, 'RAVLT_Immediate_Recall'] - RAVLT_Immediate_Recall_norm['79']['mean']) / RAVLT_Immediate_Recall_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_Immediate_Recall_z'] = (df.loc[i, 'RAVLT_Immediate_Recall'] - RAVLT_Immediate_Recall_norm['84']['mean']) / RAVLT_Immediate_Recall_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_Immediate_Recall_z'] = (df.loc[i, 'RAVLT_Immediate_Recall'] - RAVLT_Immediate_Recall_norm['85']['mean']) / RAVLT_Immediate_Recall_norm['85']['sd']


		if df.loc[i, 'Age'] <= 40:
			df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['40']['mean']) / Complex_Figure_Copy_norm['40']['sd']
		elif 40 < df.loc[i, 'Age'] <= 55:
			df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['50']['mean']) / Complex_Figure_Copy_norm['50']['sd']
		elif 55 < df.loc[i, 'Age'] <= 60:
			df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['60']['mean']) / Complex_Figure_Copy_norm['60']['sd']
		elif 60 < df.loc[i, 'Age'] <= 75:
			df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['70']['mean']) / Complex_Figure_Copy_norm['70']['sd']
		elif 75 < df.loc[i, 'Age']:
			df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['80']['mean']) / Complex_Figure_Copy_norm['80']['sd']

		if df.loc[i, 'Age'] <= 40:
			df.loc[i, 'Complex_Figure_Delayed_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['40']['mean']) / Complex_Figure_Recall_norm['40']['sd']
		elif 40 < df.loc[i, 'Age'] <= 55:
			df.loc[i, 'Complex_Figure_Delayed_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['50']['mean']) / Complex_Figure_Recall_norm['50']['sd']
		elif 55 < df.loc[i, 'Age'] <= 60:
			df.loc[i, 'Complex_Figure_Delayed_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['60']['mean']) / Complex_Figure_Recall_norm['60']['sd']
		elif 60 < df.loc[i, 'Age'] <= 75:
			df.loc[i, 'Complex_Figure_Delayed_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['70']['mean']) / Complex_Figure_Recall_norm['70']['sd']
		elif 75 < df.loc[i, 'Age']:
			df.loc[i, 'Complex_Figure_Delayed_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['80']['mean']) / Complex_Figure_Recall_norm['80']['sd']

	#df.to_csv('data/data_z.csv')
	return df


def cal_lesion_size(df):
	''' Calculate lesion size '''

	#df = pd.read_csv('data/data_z.csv')
	for i in df.index:
		# load mask and get size
		s = df.loc[i, 'Sub']

		# annoying zeros
		if s == '902':
			s = '0902'
		if s == '802':
			s = '0802'
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = nib.load(fn)
			df.loc[i, 'Lesion Size'] = np.sum(m.get_fdata()) * 0.125 #0.5 isotropic voxels, cubic = 0.125
		except:
			df.loc[i, 'Lesion Size'] = np.nan

		### now get GM and WM volume and ratio
		try:
			WM_mask = nib.load('/home/kahwang/0.5mm/WM_mask.nii.gz')
			rs_m = resample_to_img(m, WM_mask)
			df.loc[i, 'WM ratio'] = np.sum(rs_m.get_fdata() * WM_mask.get_fdata()) / np.sum(rs_m.get_fdata())
			df.loc[i, 'GM ratio'] = 1 - df.loc[i, 'WM ratio']
		except:
			df.loc[i, 'WM ratio'] = np.nan
			df.loc[i, 'GM ratio'] = np.nan

	#df.to_csv('data/data_z.csv')
	return df


def cal_lesion_PC(df):
	''' get mean voxel-wise PC within lesion mask'''

	PC = nib.load('data/Voxelwise_4mm_MGH_PC.nii')
	for i in df.index:
		# load mask and get size
		s = df.loc[i, 'Sub']

		# annoying zeros
		if s == '902':
			s = '0902'
		if s == '802':
			s = '0802'
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = nib.load(fn)
			rs_m = resample_to_img(m, PC)
			pcs = PC.get_fdata()[rs_m.get_fdata()>0]
			df.loc[i, 'mean PC'] = np.nanmean(pcs[pcs!=0])
		except:
			df.loc[i, 'mean PC'] = np.nan

	return df


def neuropsych_zscore(zthreshold, df):
	'''determine if a task is imparied with a zscore threshold, default =1.645'''

	#df = pd.read_csv('data/data_z.csv')

	df['TMTA_z_Impaired'] = df['TMTA_z'] >-1*zthreshold
	df['TMTB_z_Impaired'] = df['TMTB_z'] >-1*zthreshold
	df['BNT_z_Impaired'] = df['BNT_z'] <zthreshold
	df['COWA_z_Impaired'] = df['COWA_z'] <zthreshold
	df['RAVLT_Delayed_Recall_z_Impaired'] = df['RAVLT_Delayed_Recall_z'] <zthreshold
	df['RAVLT_Recognition_z_Impaired'] = df['RAVLT_Recognition_z'] <zthreshold
	df['Complex_Figure_Copy_z_Impaired'] = df['Complex_Figure_Copy_z'] <zthreshold
	df['Complex_Figure_Delayed_Recall_z_Impaired'] = df['Complex_Figure_Delayed_Recall_z'] <zthreshold
	df['RAVLT_T1_z_Impaired'] = df['RAVLT_T1_z'] <zthreshold
	df['RAVLT_T2_z_Impaired'] = df['RAVLT_T2_z'] <zthreshold
	df['RAVLT_T3_z_Impaired'] = df['RAVLT_T3_z'] <zthreshold
	df['RAVLT_T4_z_Impaired'] = df['RAVLT_T4_z'] <zthreshold
	df['RAVLT_T5_z_Impaired'] = df['RAVLT_T5_z'] <zthreshold
	df['RAVLT_Learning_z_Impaired'] = df['RAVLT_Learning_z'] <zthreshold
	df['RAVLT_Immediate_Recall_z_Impaired'] = df['RAVLT_Immediate_Recall_z'] <zthreshold

	## 10 Domains:
	# Visual motor, psychomotor: TMTA_z_Impaired
	# EF: TMTB_z_Impaired
	# Verbal Memory, immediate recall: RAVLT_Immediate_Recall_z_Impaired
	# Memory, learning: RAVLT_Learning_z_Impaired
	# Memory, delayed recall: RAVLT_Delayed_Recall_z_Impaired
	# Memory, recognition: RAVLT_Recognition_z_Impaired
	# Language, verbal fluency: COWA_z_Impaired
	# Language, naming: BNT_z_Impaired
	# Construction: Complex_Figure_Copy_z_Impaired
	# Visual memory: Complex_Figure_Recall_z_Impaired


	df['MM_impaired'] = sum([df['TMTA_z_Impaired'],
	df['TMTB_z_Impaired'],
	df['RAVLT_Immediate_Recall_z_Impaired'],
	df['RAVLT_Learning_z_Impaired'],
	df['RAVLT_Delayed_Recall_z_Impaired'],
	df['RAVLT_Recognition_z_Impaired'],
	df['COWA_z_Impaired'],
	df['BNT_z_Impaired'],
	df['Complex_Figure_Copy_z_Impaired'],
	df['Complex_Figure_Delayed_Recall_z_Impaired'],
	])

	#df.to_csv('data/data_z.csv')
	return df


def compare_tests(df):
	# visual-motor
	print('TMTA')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['TMTA_z'].values, df.loc[df['Site']=='Th']['TMTA_z'].values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['TMTA_z'].dropna().values, df.loc[df['Site']=='Th']['TMTA_z'].dropna().values, method='approximate', num_rounds=1000))

	# executive function
	print('TMTB')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['TMTB_z'].dropna().values, df.loc[df['Site']=='Th']['TMTB_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['TMTB_z'].dropna().values, df.loc[df['Site']=='Th']['TMTB_z'].dropna().values, func= 'x_mean < y_mean', method='approximate', num_rounds=4000))

	# Language
	print('BNT')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['BNT_z'].dropna().values, df.loc[df['Site']=='Th']['BNT_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['BNT_z'].dropna().values, df.loc[df['Site']=='Th']['BNT_z'].dropna().values, method='approximate', num_rounds=1000))

	print('COWA')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['COWA_z'].dropna().values, df.loc[df['Site']=='Th']['COWA_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['COWA_z'].dropna().values, df.loc[df['Site']=='Th']['COWA_z'].dropna().values, method='approximate', num_rounds=1000))

	# learning, long-term memory recall
	print('RAVLT recall')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Delayed_Recall_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Delayed_Recall_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['RAVLT_Delayed_Recall_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Delayed_Recall_z'].dropna().values, method='approximate', num_rounds=1000))

	print('RAVLT, recog')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Recognition_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Recognition_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['RAVLT_Recognition_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Recognition_z'].dropna().values, method='approximate', num_rounds=1000))

	print('RAVLT, learning trials')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Learning_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Learning_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['RAVLT_Learning_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Learning_z'].dropna().values, method='approximate', num_rounds=1000))

	print('RAVLT, immediate recall')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Immediate_Recall_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Immediate_Recall_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['RAVLT_Immediate_Recall_z'].dropna().values, df.loc[df['Site']=='Th']['RAVLT_Immediate_Recall_z'].dropna().values, method='approximate', num_rounds=1000))

	# complex figure
	print('complex figure, copy and recall')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['Complex_Figure_Delayed_Recall_z'].dropna().values, df.loc[df['Site']=='Th']['Complex_Figure_Delayed_Recall_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['Complex_Figure_Delayed_Recall_z'].dropna().values, df.loc[df['Site']=='Th']['Complex_Figure_Delayed_Recall_z'].dropna().values, method='approximate', num_rounds=10000))

	print('Complex_Figure_Copy_Comparison')
	#print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['Complex_Figure_Copy_z'].dropna().values, df.loc[df['Site']=='Th']['Complex_Figure_Copy_z'].dropna().values))
	print(permutation_test(df.loc[(df['Site']=='ctx')]['Complex_Figure_Copy_z'].dropna().values, df.loc[df['Site']=='Th']['Complex_Figure_Copy_z'].dropna().values, method='approximate', num_rounds=10000))

	# lesion size and demographics
	print('lesion size')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['Lesion Size'].values, df.loc[df['Site']=='Th']['Lesion Size'].values))
	print('age')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['Age'].values, df.loc[df['Site']=='Th']['Age'].values))
	#df['MM_impaired']
	print('MM')
	print(permutation_test(df.loc[(df['Site']=='ctx')]['MM_impaired'].dropna().values, df.loc[df['Site']=='Th']['MM_impaired'].dropna().values, method='approximate', num_rounds=10000))
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['MM_impaired'].values, df.loc[df['Site']=='Th']['MM_impaired'].values))
	print('MM')
	print(permutation_test(wdf.loc[(wdf['Site']=='ctx')]['MM_impaired'].dropna().values, wdf.loc[wdf['Site']=='Th']['MM_impaired'].dropna().values, method='approximate', num_rounds=10000))

	# do patients with more multi-domain impairment had larger lesions?
	print(permutation_test(df.loc[(df['MM_impaired']>1) & (df['Site']=='Th') ]['Lesion Size'].dropna().values, df.loc[(df['MM_impaired']<2) & (df['Site']=='Th')]['Lesion Size'].dropna().values, method='approximate', num_rounds=10000))
	print(np.mean(df.loc[(df['MM_impaired']>=2) & (df['Site']=='Th') ]['Lesion Size'].dropna().values))
	print(np.std(df.loc[(df['MM_impaired']>=2) & (df['Site']=='Th') ]['Lesion Size'].dropna().values))
	print(np.mean(df.loc[(df['MM_impaired']<=1) & (df['Site']=='Th') ]['Lesion Size'].dropna().values))
	print(np.std(df.loc[(df['MM_impaired']<=1) & (df['Site']=='Th') ]['Lesion Size'].dropna().values))


def print_desc_stats(df, testname):
	print('th mean')
	print(np.mean(df.loc[(df['Group']=='Thalamus')][testname].dropna().values))
	print('th std')
	print(np.std(df.loc[(df['Group']=='Thalamus')][testname].dropna().values))
	print('ctx mean')
	print(np.mean(df.loc[(df['Group']=='Comparison')][testname].dropna().values))
	print('ctx std')
	print(np.std(df.loc[(df['Group']=='Comparison')][testname].dropna().values))


def plot_neuropsy_indiv_comparisons():
	'''plot neuropsych scores and compare between groups'''

	list_of_neuropsych = ['TMT Part A', 'TMT Part B', 'Boston Naming', 'COWA', 'RAVLT Delayed Recall', 'RAVLT Recognition', 'RAVLT Trial 1', ' RAVLT Trial 2',
	'RAVLT Trial 3', 'RAVLT Trial 4', 'RAVLT Trial 5', 'Complex Figure Copy', 'Complex Figure Recall']

	list_of_neuropsych_z = ['TMTA_z', 'TMTB_z', 'BNT_z', 'COWA_z', 'RAVLT_Delayed_Recall_z', 'RAVLT_Recognition_z', 'RAVLT_T1_z',
	'RAVLT_T2_z', 'RAVLT_T3_z', 'RAVLT_T4_z', 'RAVLT_T5_z', 'Complex_Figure_Copy_z', 'Complex_Figure_Recall_z']

	list_of_neuropsych_comp = ['TMTA_Comparison', 'TMTB_Comparison', 'BNT_Comparison', 'COWA_Comparison', 'RAVLT_Delayed_Recall_Comparison', 'RAVLT_Recognition_Comparison', 'RAVLT_T1_Comparison',
	'RAVLT_T2_Comparison', 'RAVLT_T3_Comparison', 'RAVLT_T4_Comparison', 'RAVLT_T5_Comparison', 'Complex_Figure_Copy_Comparison', 'Complex_Figure_Recall_Comparison']

	#Need to only plot included comparison patients
	for i, test in enumerate(list_of_neuropsych_z):
		plt.figure(figsize=[2.4,3])
		sns.set_context('paper', font_scale=1)
		sns.set_style('white')
		sns.set_palette("Set1")

		tdf = df.loc[df[list_of_neuropsych_comp[i]]==True]
		fig1=sns.pointplot(x="Site", y=test, join=False, hue='Site', dodge=False,
					  data=tdf)
		fig1=sns.stripplot(x="Site", y=test,
					  data=tdf, alpha = .4)

		fig1.set_xticklabels(['Thalamic \nPatients', 'Comparison \nPatients'])
		#fig1.set_ylim([-3, 7])
		#fig1.set_aspect(.15)
		fig1.legend_.remove()
		plt.xlabel('')
		plt.ylabel(list_of_neuropsych[i])
		plt.tight_layout()
		fn = '/home/kahwang/RDSS/tmp/fig_%s.pdf' %test
		plt.savefig(fn)
		plt.close()


def plot_neuropsy_comparisons(df):
	'''plot neuropsych scores and compare between groups'''
	#df = pd.read_csv('data/data_z.csv')

	#invert TMT
	df['TMTA_z'] = df['TMTA_z']*-1
	df['TMTB_z'] = df['TMTB_z']*-1

	#need to melt df
	tdf = pd.melt(df, id_vars = ['Sub', 'Site'],
		value_vars = ['TMTA_z', 'TMTB_z',  'BNT_z', 'COWA_z',
		'RAVLT_Learning_z', 'RAVLT_Immediate_Recall_z', 'RAVLT_Delayed_Recall_z', 'RAVLT_Recognition_z',
		'Complex_Figure_Delayed_Recall_z', 'Complex_Figure_Copy_z'] , value_name = 'Z Score', var_name ='Task' )

	plt.close()
	plt.figure(figsize=[6,4])
	sns.set_context('paper', font_scale=1)
	sns.set_style('white')
	sns.set_palette("Set1")

	fig1=sns.pointplot(x="Task", y="Z Score", hue="Site",
				  data=tdf, dodge=.42, join=False)

	fig1=sns.stripplot(x="Task", y="Z Score", hue="Site",
				  data=tdf, dodge=True, alpha=.25)
	fig1.legend_.remove()
	fig1.set_ylim([-5.5, 5.5])
	fig1.set_xticklabels(['TMT \nPart A',  'TMT \nPart B', 'Boston \nNaming', 'COWA',
	'RAVLT Learning', 'RAVLT \nFirst Trial', 'RAVLT \nDelayed Recall', 'RAVLT \nDelayed Recognition',
	'Comeplex Figure \nDelayed Recall', 'Complex Figure \nConstruction'], rotation=90)

	plt.xlabel('')
	#plt.show()
	plt.tight_layout()
	fn = 'images/neuropych.pdf'
	plt.savefig(fn)


def plot_neuropsych_table(df):
	''' table of task impairment'''
	#df = pd.read_csv('data/data_z.csv')
	ddf = df.loc[:,['SubID','TMTA_z', 'TMTB_z',  'BNT_z', 'COWA_z',
	'RAVLT_Learning_z', 'RAVLT_Immediate_Recall_z', 'RAVLT_Delayed_Recall_z', 'RAVLT_Recognition_z',
	'Complex_Figure_Delayed_Recall_z', 'Complex_Figure_Copy_z', 'MM_impaired']]
	plt.close()
	x = df['Site']=='Th'
	tddf = ddf.loc[x]

	#invert tmtbz ### be careful and check if it is already inverted!!
	tddf.loc[:,'TMTB_z'] = tddf.loc[:,'TMTB_z']*-1
	tddf.loc[:,'TMTA_z'] = tddf.loc[:,'TMTA_z']*-1

	tddf = tddf.set_index('SubID')
	figt = sns.heatmap(tddf.sort_values('MM_impaired'), vmin = -5, vmax=5, center=0, cmap="coolwarm")
	figt.set_xticklabels(['TMT \nPart A',  'TMT \nPart B', 'Boston \nNaming', 'COWA',
	'RAVLT Learning', 'RAVLT \nFirst Trial', 'RAVLT \nDelayed Recall', 'RAVLT \nDelayed Recognition',
	'Comeplex Figure \nDelayed Recall', 'Complex Figure \nConstruction'], rotation=90)
	plt.xlabel('')
	plt.ylabel('Patient')
	plt.tight_layout()
	fn = 'images/tasktable.png'
	plt.savefig(fn)


def draw_lesion_overlap(group, df, fname):
	''' draw lesion overlap, group ='Thalamus' or 'Comparison' '''

	m=0
	for s in df.loc[df['Group']==group]['Sub']:
		# load mask and get size

		# annoying zeros in file path.....
		if s == '902':
			s = '0902'
		if s == '802':
			s = '0802'
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = m + nib.load(fn).get_fdata()
		except:
			continue

	h = nib.load('/home/kahwang/0.5mm/0902.nii.gz')
	lesion_overlap_nii = nilearn.image.new_img_like(h, m)
	lesion_overlap_nii.to_filename(('images/%s') %fname)

	return lesion_overlap_nii


def map_lesion_unique_masks(df):
	''' map each neuropsych's unique lesion mask'''

	thalamus_mask_data = nib.load('/home/kahwang/0.5mm/tha_0.5_mask.nii.gz').get_fdata()
	thalamus_mask_data = thalamus_mask_data>0
	list_of_neuropsych_z = ['TMTA', 'TMTB', 'BNT', 'COWA', 'RAVLT_Delayed_Recall', 'RAVLT_Recognition',
		'RAVLT_Immediate_Recall', 'RAVLT_Learning', 'Complex_Figure_Copy', 'Complex_Figure_Delayed_Recall']
	list_of_neuropsych_var = ['TMT part A', 'TMT part B', 'Boston Naming', 'COWA', 'RAVLT Recall', 'RAVLT Recognition',
		'RAVLT Immediate Recall', 'RAVLT Learning', 'Complex Figure Copy', 'Complex Figure Delayed Recall']
	h = nib.load('/home/kahwang/0.5mm/0902.nii.gz')

	mask_niis = {}
	for j, neuropsych in enumerate(list_of_neuropsych_z):
		strc = neuropsych + '_z_Impaired'
		tdf = df.loc[df['Site']=='Th'].loc[df['MNI_Mask']=='T'].loc[df[strc]==True]
		num_patient = len(tdf) # number of patients
		if num_patient ==0:
			continue # break loop if no impairment
		tdf.loc[tdf['Sub']=='902','Sub'] = '0902'

		#lesion overlap of patients with impariment
		true_m=0
		for i in tdf.index:
			s = tdf.loc[i, 'Sub']
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			true_m = true_m + nib.load(fn).get_fdata()

		true_m = true_m * thalamus_mask_data
		impairment_overlap_nii = nilearn.image.new_img_like(h, true_m)

		fn = 'images/' + neuropsych + '_lesionmask_pcount.nii.gz'
		impairment_overlap_nii.to_filename(fn)

		mask_niis[list_of_neuropsych_var[j]] = impairment_overlap_nii

	# count number of task lesion mask overlap
	Num_task_mask = np.zeros(np.shape(mask_niis[list(mask_niis.keys())[0]].get_fdata()))
	for task in mask_niis.keys():
		 Num_task_mask = Num_task_mask + 1.0*(mask_niis[task].get_fdata()>0)

	#Num_task_mask = 1.0*(RAVLT_mask.get_fdata()>0) + 1.0*(Verbal_mask.get_fdata()>0) + 1.0*(Memory_mask.get_fdata()>0) + 1.0*(TMTB_mask.get_fdata()>0) #1.0*(TMTA_mask.get_fdata()>0)
	Num_task_mask = nilearn.image.new_img_like(h, Num_task_mask)
	Num_task_mask.to_filename('images/Num_of_task_impaired_overlap.nii.gz')

	return Num_task_mask


def load_PC(dset):
	''' load PC calculations, dset = 'MGH' or 'NKI'''
	thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
	thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_fdata()
	thalamus_mask_data = thalamus_mask_data>0
	thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data)
	fn = 'data/%s_pc_vectors_corr.npy' %dset
	pc_vectors = np.load(fn)

	# average across subjects
	pcs = np.nanmean(np.nanmean(pc_vectors, axis =2), axis=1)
	pc_img = masking.unmask(pcs, thalamus_mask)
	diff_nii_img = nib.load('images/mm_unique.nii.gz')
	rsfc_pc05 = resample_to_img(pc_img, diff_nii_img, interpolation='nearest') #this is the PC variable for kde and point plots

	#because......... AFNI can't "floor" a colorbar, need to manipuluat the image a bit before writing it out for plotting. Display pc .45 to .65
	vpc = pcs.copy()
	vpc = vpc-0.3
	vpc[vpc<=0] = 0.0001
	vpc_img = masking.unmask(vpc, thalamus_mask)
	fn = 'images/%s_pc.nii.gz' %dset
	vpc_img.to_filename(fn)

	return rsfc_pc05


def cal_ave_num_impaired_task_pervoxel():
	'''calculate average number of impaired task per voxel'''

	## multimodal lesion voxels' ave impaired task
	thalamus_mask_data = nib.load('/home/kahwang/0.5mm/tha_0.5_mask.nii.gz').get_fdata()
	thalamus_mask_data = thalamus_mask_data>0
	m = np.zeros(np.shape(nib.load('images/Num_of_task_impaired_overlap.nii.gz').get_fdata()))
	N_m = m.copy()
	for i, s in enumerate(df.loc[(df['MM_impaired']>=2) & (df['Site'] =='Th')]['Sub']):  #df.loc[(df['MM_impaired']>4) &
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			N_m = N_m + nib.load(fn).get_fdata()
			m = m + nib.load(fn).get_fdata()*df.loc[df['Sub']==s]['MM_impaired'].values[0]
		except:
			continue
	m = m / N_m
	m = m * thalamus_mask_data
	h = nib.load('images/Num_of_task_impaired_overlap.nii.gz')
	ave_of_task_impaired = nilearn.image.new_img_like(h, 1.0*m)
	ave_of_task_impaired.to_filename('images/ave_of_task_impaired.nii.gz')

	## single modal lesion voxels' ave impaired task
	m=np.zeros(np.shape(nib.load('images/Num_of_task_impaired_overlap.nii.gz').get_fdata()))
	N_m = m.copy()
	for i, s in enumerate(df.loc[(df['MM_impaired']<2) & (df['Site'] =='Th')]['Sub']):  #df.loc[(df['MM_impaired']>4) &
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			N_m = N_m + nib.load(fn).get_fdata()
			m = m + nib.load(fn).get_fdata()*df.loc[df['Sub']==s]['MM_impaired'].values[0]
		except:
			continue
	m = m / N_m
	m = m * thalamus_mask_data
	h = nib.load('images/Num_of_task_impaired_overlap.nii.gz')
	ave_of_task_impaired = nilearn.image.new_img_like(h, 1.0*m)
	ave_of_task_impaired.to_filename('images/ave_of_task_impaired_SM.nii.gz')


def plt_MM_SM_lesion_mask(df):
	thalamus_mask_data = nib.load('/home/kahwang/0.5mm/tha_0.5_mask.nii.gz').get_fdata()
	thalamus_mask_data = thalamus_mask_data>0
	m=np.zeros(np.shape(nib.load('images/Num_of_task_impaired_overlap.nii.gz').get_fdata))
	for s in df.loc[(df['MM_impaired']>=2) & (df['Site'] =='Th')]['Sub']:
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = m + nib.load(fn).get_fdata()
		except:
			continue
	m = m * thalamus_mask_data
	h = nib.load('images/Num_of_task_impaired_overlap.nii.gz')
	mmlesion_overlap_nii = nilearn.image.new_img_like(h, 1.0*m)
	#mmlesion_overlap_nii.to_filename('images/mmlesion_overlap.nii.gz')

	m=np.zeros(np.shape(nib.load('images/Num_of_task_impaired_overlap.nii.gz').get_fdata))
	for s in df.loc[(df['MM_impaired']<=1) & (df['Site'] =='Th')]['Sub']:
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = m + nib.load(fn).get_fdata()
		except:
			continue

	m = m * thalamus_mask_data
	h = nib.load('images/Num_of_task_impaired_overlap.nii.gz')
	smlesion_overlap_nii = nilearn.image.new_img_like(h, 1.0*m)
	#smlesion_overlap_nii.to_filename('images/smlesion_overlap.nii.gz')

	diff_nii = 1.0*(mmlesion_overlap_nii.get_fdata()>0) - 1.0*(smlesion_overlap_nii.get_fdata()>0)
	diff_nii_img = nilearn.image.new_img_like(h, 1.0*diff_nii)
	#diff_nii_img.to_filename('images/mm_v_sm_overlap.nii.gz')

	mm_unique = nilearn.image.new_img_like(h, 1.0*(diff_nii ==1))
	sm_unique = nilearn.image.new_img_like(h, 1.0*(diff_nii ==-1))

	return diff_nii_img, mm_unique, sm_unique


def dice(im1, im2):
    """
    Computes the Dice coefficient
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def write_indiv_subj_PC(diff_nii_img, dset):
	''' load the PC value within lesion masks for *EACH* individual normative subject'''

	#load PC vectors and tha mask to put voxel values back to nii object
	fn = 'data/%s_pc_vectors_corr.npy' %(dset)
	pc_vectors = np.load(fn) #resting state FC's PC. dimension 236 (sub) by 2xxx (tha voxel)
	thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
	thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_fdata()
	thalamus_mask_data = thalamus_mask_data>0
	thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data)
	diff_nii_img_2mm = resample_to_img(diff_nii_img, thalamus_mask, interpolation='nearest')

	#for threshold in [0,1,2,3,4,5,6,7,8]:
	# create DF
	pcdf = pd.DataFrame()
	i=0
	for p in [-1, 1]:  #np.unique(Num_task_mask_2mm.get_fdata())[np.unique(Num_task_mask_2mm.get_fdata())>0]

		for s in np.arange(0, pc_vectors.shape[1]):
			pc = np.nanmean(pc_vectors[:,s,:], axis = 1)
			fcpc_image = masking.unmask(pc, thalamus_mask).get_fdata()
			#rsfc_pc05 = resample_to_img(fcpc_image, Num_task_mask)
			#np.mean(rsfc_pc05.get_fdata()[Num_task_mask.get_fdata()==1])

			pcdf.loc[i, 'Subject'] = s
			pcdf.loc[i, 'Cluster'] = p
			pcdf.loc[i, 'PC'] = np.nanmean(fcpc_image[(diff_nii_img_2mm.get_fdata()==p) & thalamus_mask_data])
			#pcdf.loc[i, 'MM_impaired_num'] = df.loc[df['Sub'] == p]['MM_impaired'].values[0]
			#pcdf.loc[i, 'Size'] = df.loc[df['Sub'] == p]['Lesion Size'].values[0]
			i = i+1

	pcdf = pcdf.dropna()

	#model, random intercept
	md = smf.mixedlm("PC ~ Cluster ", data = pcdf ,re_formula = '1', groups=pcdf['Subject']).fit() #re_formula = 'Cluster'
	#print(threshold)
	print(md.summary())

	avePC = np.nanmean(np.nanmean(pc_vectors, axis=2), axis=1)
	avePC_image = masking.unmask(avePC, thalamus_mask)

	return avePC_image, pcdf


def cal_impairment_scores(pdf):
	''' calculate severity and globalness scores while accounting for lesion size'''

	tests=['TMTA_z', 'TMTB_z', 'BNT_z',
		   'COWA_z', 'RAVLT_Delayed_Recall_z', 'RAVLT_Recognition_z',
		   'RAVLT_Learning_z', 'RAVLT_Immediate_Recall_z', 'Complex_Figure_Copy_z',
		   'Complex_Figure_Delayed_Recall_z']
	pdf['num_of_test'] = 10 - np.sum(pdf[tests].isnull(), axis=1)
	pdf['ave_impairment'] = np.nansum(pdf[tests], axis=1) / pdf['num_of_test']

	# regression of lesion size, create scores
	regression_model = LinearRegression()
	x = pdf['Lesion Size']
	x = np.expand_dims(x, axis=1)
	regression_model.fit(x, pdf['ave_impairment'])
	pdf['Average Impairment Score'] = pdf['ave_impairment'] - regression_model.predict(x)

	regression_model = LinearRegression()
	x = pdf['Lesion Size']
	x = np.expand_dims(x, axis=1)
	regression_model.fit(x, pdf['MM_impaired'])
	pdf['Multi-Domain Impairment Score'] = pdf['MM_impaired'] - regression_model.predict(x) +2

	return pdf


if __name__ == "__main__":

	### Note that some functions below were "commented out" just so we dont repeat it.
	########################################################################
	# Compare test scores between patient groups
	########################################################################

	### Prep dataframe through steps:
	df = pd.read_csv('data/data_z.csv')
	#df = load_and_normalize_neuropsych_data(df)
	#df = cal_lesion_size(df)
	#df = neuropsych_zscore(-1.645, df)

	# extended comparison patients
	cdf = pd.read_csv('data/ecdf.csv')
	all_df = df.append(cdf)
	all_df = all_df.reset_index()

	########################################################################
	# Get some basic descriptive stats
	########################################################################
	print_demographic(all_df)


	###################################################################################################################
	# plot lesion overlap (Figure 1A and Figure 6A)
	####################################################################################################################
	#lesion_overlap_nii = draw_lesion_overlap('Thalamus', all_df, 'thalamus_lesion_overlap.nii.gz')
	#comaprison_lesion_overlap_nii = draw_lesion_overlap('Thalamus', all_df, 'comparison_lesion_overlap.nii.gz')
	#extended_comaprison_lesion_overlap_nii = draw_lesion_overlap('Thalamus', all_df, 'extended_comparison_lesion_overlap.nii.gz')

	## note we ended up not using nilearn plotting because we couldn't "zoom" into the thalamus
	#plotting.plot_stat_map(lesion_overlap_nii, bg_img = mni_template, display_mode='z', cut_coords=5, colorbar = False, black_bg=False, cmap='gist_ncar')
	#plotting.show()


	###################################################################################################################
	# compare test scores, print descriptives (Figure 1B)
	###################################################################################################################
	#compare_tests(df
	#print_desc_stats(df, 'COWA_z')

	#plot_neuropsy_indiv_comparisons()
	#plot_neuropsy_comparisons()


	################################################################################################
	# Plot table of z scores to show mutlimodal impairment (Figure 2B)
	################################################################################################
	#plot_neuropsych_table()


	################################################################################################
	### Plot lesion sites associate with each task, and its overlap (Figure 2A)
	################################################################################################
	#Num_task_mask = map_lesion_unique_masks(df)
	#cal_ave_num_impaired_task_pervoxel()


	################################################################################################
	### Now draw lesions overlap for patients with and without multimodal impairment (Figure 2C)
	### plot lesion masks for these subjects. Identify Multi-domain (MM) and single domain(SM) sites
	################################################################################################
	#diff_nii_img, mm_unique, sm_unique = plt_MM_SM_lesion_mask(df)
	#mm_unique.to_filename('images/mm_unique.nii.gz')


	############################################################
	#### Compare specificty/globality vs averaged severity (Figure 3)
	############################################################
	## calculate impairment and global scores
	tdf = all_df.loc[all_df['Group']=='Thalamus']
	cdf = all_df.loc[all_df['Group'] == 'Expanded Comparison']
	pdf = pd.concat([tdf, cdf])
	pdf = cal_impairment_scores(pdf)

	# plot
	def plot_severity_v_globality():
		plt.figure(figsize=[4,3])
		fig = sns.regplot(x='Average Impairment Score', y='Multi-Domain Impairment Score', data=pdf.loc[pdf['Group']=='Expanded Comparison'], ci = 95, scatter = False, order = 1, color = 'grey')
		fig = sns.scatterplot(x='Average Impairment Score', y='Multi-Domain Impairment Score', data=pdf.loc[pdf['Group']=='Expanded Comparison'], alpha = .5, color = 'grey')
		#fig = sns.regplot(x='Average Impairment Score', y='Multi-Domain Impairment Score', data=pdf.loc[(pdf['Group']=='Thalamus') & (pdf['MM_impaired']>=2)], ci = 95, scatter = False, order = 1, color = 'r')
		fig = sns.scatterplot(x='Average Impairment Score', y='Multi-Domain Impairment Score', data=pdf.loc[(pdf['Group']=='Thalamus') & (pdf['MM_impaired']>=2)], alpha = 1, color = 'r')
		#fig = sns.regplot(x='Average Impairment Score', y='Multi-Domain Impairment Score', data=pdf.loc[(pdf['Group']=='Thalamus') & (pdf['MM_impaired']<=1)], ci = 95, scatter = False, order = 1, color = 'b')
		fig = sns.scatterplot(x='Average Impairment Score', y='Multi-Domain Impairment Score', data=pdf.loc[(pdf['Group']=='Thalamus') & (pdf['MM_impaired']<=1)], alpha = 1, color = 'b')
		fig.set_xlim(-1.7, 1.4)
		fig.set_ylim(0,6)
		plt.tight_layout()
		fn = 'images/sparse_v_global.pdf'
		plt.savefig(fn)
	#plot_severity_v_globality()


	############################################################
	#### Neurosynth analyses (Figure 4)
	############################################################
	def plot_neurosynth_map():
		terms = ['executive', 'naming', 'fluency', 'recall', 'recognition']
		for term in terms:
			fn = 'data/%s_association-test_z_FDR_0.01.nii.gz' %term
			nsnii = nib.load(fn)
			fn = 'images/%s.png' %term
			plotting.plot_stat_map(nsnii, display_mode='z', cut_coords=5, title=term, output_file = fn)
			plt.close()
		nsnii = nib.load('data/overlap.nii.gz')
		plotting.plot_stat_map(nsnii, display_mode='z', cut_coords=5, title='overlap', cmap = 'spring', output_file = 'images/term_overlap.png')

	#### Neurosynth FC selectivity analysis
	def plot_neurosynth_selectivity():
		fc = np.load('data/MGH_term_fc_fcorr.npy')
		fc[fc<0] = 0
		fc = np.mean(fc, axis=2)
		tfcdf = pd.DataFrame()
		fc_total = 0

		for ci in [0,1,2,3,4]:
			fc_total = fc_total + fc[:,ci]
		i = 0
		Terms = ['executive', 'naming', 'fluency', 'recall', 'recognition']
		for ic, ci in enumerate([0,1,2,3,4]):
			for v in np.arange(0, fc.shape[0]):
				tfcdf.loc[i, 'FC weight ratio'] = fc[v,ci] / fc_total[v]
				tfcdf.loc[i, 'Term'] = Terms[ic]
				i=i+1

		plt.close()
		sns.set_context("paper")
		plt.figure(figsize=[4.2,4])
		sns.kdeplot(x='FC weight ratio', data=tfcdf, hue='Term', common_norm = True, legend = True, fill=False, linewidth=2, alpha = .5, palette='Paired')
		#sns.histplot(data=vfdf, x="FC weight ratio", hue='Network')
		#plt.show()
		plt.tight_layout()
		#plt.show()
		fn = 'images/Terms_FC_kde.pdf'
		plt.savefig(fn)

	#plot_neurosynth_map()
	#plot_neurosynth_selectivity()


	################################################################################################
	# Compare thalamic lesion sites' PC values (Figures 5 B-D)
	################################################################################################
	### calculate PC vectors
	### use cal_PC() function in cal_FC_PC
	#import cal_FC_PC
	#cal_FC_PC.main()

	##### Look at diff in PC between SM and MM lesion sites
	#rsfc_pc = nib.load('images/RSFC_PC.nii.gz')
	#from nilearn.image import resample_to_img
	#rsfc_pc05 = resample_to_img(rsfc_pc, Num_task_mask)
	#rsfc_pc05.to_filename('images/PC.5.nii.gz')
	#rsfc_pc05 = nib.load('images/PC.5.nii.gz')

	#### load PC, ave across subjects and thresholds
	def plot_MMvSM_PC():
		diff_nii_img = nib.load('images/mm_v_sm_overlap.nii.gz')
		mm_unique = nib.load('images/mm_unique.nii.gz')
		sm_unique = nib.load('images/sm_unique.nii.gz')
		#rsfc_pc05 = load_PC('NKI')
		rsfc_pc05 = load_PC('MGH')

		## compile df for kde plot
		PCs={}
		PCs['MM']= masking.apply_mask(rsfc_pc05, mm_unique)
		PCs['SM']= masking.apply_mask(rsfc_pc05, sm_unique)
		print(np.mean(PCs['MM']))
		print(np.mean(PCs['SM']))

		i=0
		pcdf = pd.DataFrame()
		for t in ['MM', 'SM']:
			pdf = pd.DataFrame()
			pdf['PC'] = PCs[t]
			pdf['#Impairment'] = t

			pcdf =pd.concat([pcdf, pdf])
		A=pcdf.loc[pcdf['#Impairment']=='MM']['PC']
		B=pcdf.loc[pcdf['#Impairment']=='SM']['PC']
		scipy.stats.ks_2samp(A,B)

		####kde plot for Fig 5C
		plt.close()
		sns.set_context("paper")
		plt.figure(figsize=[4,3])
		sns.kdeplot(x='PC', data=pcdf, hue='#Impairment', common_norm = False, legend = False, fill=True, linewidth=3, alpha = .5, palette=['r', '#0269FE'])
		#plt.show()
		fn = 'images/MM_SM_kde.pdf'
		plt.savefig(fn)
	#plot_MMvSM_kde_PC()

	#### plot comparison between SM and MM PC values (Figure 5D)
	_, pcdf = write_indiv_subj_PC(diff_nii_img, 'MGH')

	def plot_MMvSM_pointplot_PC():
		print(stats.ttest_rel(pcdf.loc[pcdf['Cluster']==-1]['PC'], pcdf.loc[pcdf['Cluster']==1]['PC']))
		print(pcdf.groupby(['Cluster']).mean())

		plt.close()
		plt.figure(figsize=[4,3])
		fig = sns.pointplot(x="Cluster", y='PC', join=False, dodge=False, data=pcdf, hue='Cluster', palette=['#0269FE', 'r'])
		fig = sns.stripplot(x="Cluster", y="PC",
						  data=pcdf, dodge=False, jitter = False, alpha=.03, palette=['#0269FE', 'r'])
		fig.legend_.remove()

		fn = 'images/PC_MMvSM.pdf'
		plt.savefig(fn)
	#plot_MMvSM_pointplot_PC()


	################################################################################################
	# Compare thalamic patients' versus comaprison patients' PC values (Figures 6 B)
	################################################################################################
	#all_df = cal_lesion_PC(all_df)
	#cdf = cal_lesion_PC(cdf)

	print(permutation_test(all_df.loc[(all_df['Group']=='Expanded Comparison') & (all_df['MM_impaired']>1)]['mean PC'].dropna().values, all_df.loc[(all_df['Group']=='Expanded Comparison') & (all_df['MM_impaired']<=1)]['mean PC'].dropna().values, method='approximate', num_rounds=1000))
	print(permutation_test(all_df.loc[(all_df['Group']=='Thalamus') & (all_df['MM_impaired']>1)]['mean PC'].dropna().values, all_df.loc[(all_df['Group']=='Expanded Comparison') & (all_df['MM_impaired']>1)]['mean PC'].dropna().values, method='approximate', num_rounds=1000))

	PC = nib.load('data/Voxelwise_4mm_MGH_PC.nii')
	mmpcs = np.array([])
	smpcs = np.array([])
	for s in cdf.Sub.values:
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = nib.load(fn)
			rs_m = resample_to_img(m, PC)
			if cdf.loc[(cdf['Sub']==s) & (cdf['MM_impaired']>1)]:
				tpc = masking.apply_mask(PC, rs_m)
				tpc = tpc[tpc!=0]
				mmpcs = np.append(mmpcs,tpc)
			if cdf.loc[(cdf['Sub']==s) & (cdf['MM_impaired']<=1)]:
				tpc = masking.apply_mask(PC, rs_m)
				tpc = tpc[tpc!=0]
				smpcs = np.append(smpcs,tpc)
		except:
			continue

	PCs={}
	PCs['MM']= masking.apply_mask(rsfc_pc05, mm_unique)
	PCs['SM']= masking.apply_mask(rsfc_pc05, sm_unique)

	print(stats.ttest_ind(cdf.loc[cdf['MM_impaired']>1]['mean PC'], cdf.loc[cdf['MM_impaired']<=1]['mean PC']))
	#print(pcdf.groupby(['Cluster']).mean())
	cdf.loc[cdf['MM_impaired']>1, 'Cluster'] = 'MM'
	cdf.loc[cdf['MM_impaired']<=1, 'Cluster'] = 'SM'
	cdf['PC'] = cdf['mean PC']
	plt.close()
	plt.figure(figsize=[4,3])
	fig = sns.pointplot(x="Cluster", y='PC', join=False, dodge=False, data=cdf, hue='Cluster', palette=['#0269FE', 'r'])
	fig = sns.stripplot(x="Cluster", y="PC",
					  data=cdf, dodge=False, jitter = False, alpha=.03, palette=['#0269FE', 'r'])
	fig.set_xticklabels(['SM',  'MM'])
	fig.legend_.remove()
	#fig4.set_xticklabels(['SM',  'MM'])
	#fn = 'images/PC_MMvSM.pdf'
	plt.show()


	############################################################################################################################
	##### FC weight and selectivity analysis (Figure 7A-B)
	############################################################################################################################
	Schaeffer_CI = np.loadtxt('data/Schaeffer400_7network_CI')
	fc = np.load('data/MGH_mmmask_fc_fcorr.npy')
	fc = np.mean(fc, axis=2)
	zfc = zscore(fc, axis=1)
	fndf = pd.DataFrame()
	Networks = ['V', 'SM', 'DA', 'CO', 'Lm', 'FP', 'DF']
	for i, ci in enumerate([1,2,3,4,5,6,7]):
		fndf.loc[i, 'FC (z-score'] = np.mean(zfc[:,Schaeffer_CI==ci])
		fndf.loc[i, 'Network'] = Networks[i]

	vfdf = pd.DataFrame()
	fc[fc<0] = 0

	fc_total = 0
	for ci in [1,2,3,4,5,6,7]:
		fc_total = fc_total + np.mean(fc[:,Schaeffer_CI==ci], axis=1)

	i = 0
	for ic, ci in enumerate([1,2,3,4,5,6,7]):
		for v in np.arange(0, fc.shape[0]):
			vfdf.loc[i, 'FC weight ratio'] = np.mean(fc[v,Schaeffer_CI==ci]) / fc_total[v]
			vfdf.loc[i, 'Network'] = Networks[ic]
			i=i+1

	plt.close()
	sns.set_context("paper")
	plt.figure(figsize=[4.2,4])
	sns.kdeplot(x='FC weight ratio', data=vfdf, hue='Network', common_norm = True, legend = True, fill=False, linewidth=2, alpha = .5, palette=['#9856A7', '#7F9ABD', '#589741', '#D16CF7', '#F7F45F', '#E7BC5A', '#CB777F'])
	#sns.histplot(data=vfdf, x="FC weight ratio", hue='Network')
	#plt.show()
	plt.tight_layout()
	fn = 'images/MM_FC_kde.pdf'
	plt.savefig(fn)


	################################################################################################
	###### Nuclei analysis (Figure 7C)
	######################################################################################################
	morel = nib.load('images/Thalamus_Morel_consolidated_mask_v3.nii.gz')
	#diff_nii_img_2mm
	#diff_nii_img = nib.load('images/mm_unique.nii.gz')
	#mm_unique_2mm = resample_to_img(diff_nii_img, morel, interpolation = 'nearest')
	mm_unique = nib.load('images/mm_unique_2mm.nii.gz')  #mRNA maps are in 2mm
	sm_unique = nib.load('images/sm_unique_2mm.nii.gz')

	mm_nuclei_vec = morel.get_fdata()[mm_unique.get_fdata()==1]
	sm_nuclei_vec = morel.get_fdata()[sm_unique.get_fdata()==1]
	#sm_unique_2mm = resample_to_img(sm_unique, morel, interpolation = 'nearest')
	#sm_nuclei_vec = morel.get_fdata()[sm_unique_2mm.get_fdata()==1]

	morel_list={
	'1': 'AN',
	'2':'VM',
	'3':'VL',
	'4':'MGN',
	'5':'MD',
	'6':'PuA',
	'7':'LP',
	'8':'IL',
	'9':'VA',
	'10':'Po',
	'11':'LGN',
	'12':'PuM',
	'13':'PuI',
	'14':'PuL',
	'17':'VP'}

	mmdf = pd.DataFrame()
	for i, n in enumerate(np.unique(mm_nuclei_vec)):
		if n>0:
			mmdf.loc[i, 'Percentage overlap'] = (np.sum(mm_nuclei_vec == n)*8 / np.sum(8.0*(morel.get_fdata()==n))) * 100
			mmdf.loc[i, 'Nuclei'] = morel_list[str(int(n))]

	smdf = pd.DataFrame()
	for i, n in enumerate(np.unique(sm_nuclei_vec)):
		if n >0:
			smdf.loc[i, 'Percentage overlap'] = (np.sum(sm_nuclei_vec == n)*8 / np.sum(8.0*(morel.get_fdata()==n))) * 100
			smdf.loc[i, 'Nuclei'] = morel_list[str(int(n))]

	ndf = pd.concat([mmdf,smdf])
	# ndf = pd.DataFrame()
	# for i, n in enumerate(np.unique(sm_nuclei_vec)):
	# 	ndf.loc[i, 'Size'] = np.sum(sm_nuclei_vec == n)*8
	# 	ndf.loc[i, 'Nuclei'] = morel_list[str(int(n))]

	#yeo = nib.load('images/1000subjects_TightThalamus_clusters007_ref.nii.gz')
	plt.close()
	sns.set_context("paper")
	plt.figure(figsize=[4,4])
	fign = sns.barplot(x="Nuclei", y="Percentage overlap", data=ndf)
	plt.tight_layout()
	#figw.set_ylim([0, 30])
	#plt.show()
	fn = 'images/modrel_bar.pdf'
	plt.savefig(fn)
	#plt.show()


	#########################################################################################################################################################
	#### compare CALB v PVALB values within the lesion mask (Figure 8)
	#########################################################################################################################################################

	mm_unique = nib.load('images/mm_unique_2mm.nii.gz')  #mRNA maps are in 2mm
	sm_unique = nib.load('images/sm_unique_2mm.nii.gz')
	pvalb_sm = masking.apply_mask('images/pvalb_std.nii.gz', sm_unique)
	pvalb_mm = masking.apply_mask('images/pvalb_std.nii.gz', mm_unique)
	calb_mm = masking.apply_mask('images/calb_std.nii.gz', mm_unique)
	calb_sm = masking.apply_mask('images/calb_std.nii.gz', sm_unique)
	pvalb_sm[pvalb_sm == 0] = np.nan
	pvalb_mm[pvalb_mm == 0] = np.nan
	calb_mm[calb_mm == 0] = np.nan
	calb_sm[calb_sm == 0] = np.nan

	#stats
	print('pvalb')
	print(scipy.stats.ks_2samp(pvalb_mm,pvalb_sm))
	print('calb')
	print(scipy.stats.ks_2samp(calb_mm,calb_sm))
	pv = np.hstack((pvalb_mm, pvalb_sm))
	ca = np.hstack((calb_mm, calb_sm))

	mdf = pd.DataFrame()
	mdf['Normalized PVALB'] = pvalb_mm
	mdf['Lesion Site'] = 'MM'
	sdf = pd.DataFrame()
	sdf['Normalized PVALB'] = pvalb_sm
	sdf['Lesion Site'] = 'SM'
	pdf = pd.concat([mdf, sdf])

	plt.close()
	sns.set_context("paper")
	plt.figure(figsize=[4.2,4])
	sns.kdeplot(x='Normalized PVALB', data=pdf, hue='Lesion Site', common_norm = True, legend = True, fill=False, linewidth=2, alpha = .5, palette=['red', 'blue'])
	#sns.histplot(data=vfdf, x="FC weight ratio", hue='Network')
	#plt.show()
	plt.tight_layout()
	#plt.show()
	fn = '/home/kahwang/RDSS/tmp/PV_kde.pdf'
	plt.savefig(fn)


	mdf = pd.DataFrame()
	mdf['Normalized CALB'] = calb_mm
	mdf['Lesion Site'] = 'MM'
	sdf = pd.DataFrame()
	sdf['Normalized CALB'] = calb_sm
	sdf['Lesion Site'] = 'SM'
	cdf = pd.concat([mdf, sdf])

	plt.close()
	sns.set_context("paper")
	plt.figure(figsize=[4.2,4])
	sns.kdeplot(x='Normalized CALB', data=cdf, hue='Lesion Site', common_norm = True, legend = True, fill=False, linewidth=2, alpha = .5, palette=['red', 'blue'])
	#sns.histplot(data=vfdf, x="FC weight ratio", hue='Network')
	#plt.show()
	plt.tight_layout()
	#plt.show()
	fn = '/home/kahwang/RDSS/tmp/CA_kde.pdf'
	plt.savefig(fn)




#end of line
