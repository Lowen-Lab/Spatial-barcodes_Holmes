import os
import sys
import networkx as nx
import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import savgol_filter
import multiprocessing
from joblib import Parallel, delayed
###################################   USER DEFINED VARIABLES   ###################################
project_dir = "./"

count_table_name = './barcode_info/all_barcode_count.table.txt'
freq_table_name = './barcode_info/all_barcode_freq.table.txt'

subsample_level = int(3e3)
iterations = 1000
interval_method = '95CI'
parallel_max_cpu = 50
info_suffix = '.iter'+str(iterations)+'.'+str(subsample_level)+'.'+interval_method
minplot_prop = 1e-4

if parallel_max_cpu == 0:
	num_cores = multiprocessing.cpu_count()
elif parallel_max_cpu > 0:
	num_cores = min(parallel_max_cpu,multiprocessing.cpu_count())
elif parallel_max_cpu < 0:
	num_cores = multiprocessing.cpu_count()-parallel_max_cpu
def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h
def quantile_interval(data,low_quantile=0.25,high_quantile=0.75):
	m = np.median(data)
	l = np.quantile(data,low_quantile)
	u = np.quantile(data,high_quantile)
	return m, l, u


sample_info_dict = {}
plate_dict = {}
plate_exp_dict = {}
exp_list = {}
plate_slipID_dict = {}
site_dict = {}
plate_condition_dict = {}
infile = open('sample_key.txt',"r")
for line in infile:
	line = line.strip()
	if len(line)>0:
		if line[0] !="#":
			line = line.split('\t')
			sampleID = line[2]
			if sampleID != 'NA':
				coverslip_no = int(line[1])
				try:
					ct_val = float(line[5])
					ct_val = (round(np.log10(1e12*math.exp(ct_val*-0.6931)),2)/6.0)
				except:
					ct_val = 0.1
				exp_val = line[0]
				MC_val = line[3]
				plate_no = int(line[4])
				if MC_val == '0.2MC' or MC_val == '0.3MC' or MC_val == '0.4MC' or MC_val == 'SH':
					exp_val = 1
				else:
					exp_val = 2
				plate_slipID = str(plate_no)+"-"+str(coverslip_no)
				plate_slipID_dict[plate_slipID] = sampleID
				if coverslip_no != 17:
					try:
						plate_dict[plate_no].append(sampleID)
					except:
						plate_dict[plate_no] = [sampleID]

					site_dict[sampleID] = coverslip_no
					sample_info_dict[sampleID] = (plate_no,ct_val,MC_val,exp_val)
					plate_condition_dict[plate_no] = MC_val
					plate_exp_dict[plate_no] = exp_val
					exp_list[exp_val] = ''
infile.close()
exp_list = list(exp_list.keys())
exp_list = sorted(exp_list)



def subsample_counts(barcode_count_dict_in,num_to_pick):
	rand_list = []
	for barcodeID in barcode_count_dict_in:
		count = barcode_count_dict_in[barcodeID]
		for i in range(0,count):
			rand_list.append(barcodeID)
	if len(rand_list)>=num_to_pick:
		random.shuffle(rand_list)
		subset_list = rand_list[0:int(num_to_pick)]
	else:
		return None
	count_dict_out = {}
	freq_dict_out = {}
	subsampled_reads = len(subset_list)
	for i in range(0,len(subset_list)):
		barcodeID = subset_list[i]
		try:
			count_dict_out[barcodeID] += 1
		except:
			count_dict_out[barcodeID] = 1
	local_barcodeID_list = list(count_dict_out.keys())
	float_num_to_pick = float(num_to_pick)
	for i in range(0,len(local_barcodeID_list)):
		barcodeID = local_barcodeID_list[i]
		count = count_dict_out[barcodeID]
		barocde_freq = float(count)/float_num_to_pick
		freq_dict_out[barcodeID] = barocde_freq
	return count_dict_out


count_file = open(count_table_name,"r")
first_line = True
barcodeID_count_dict = {}
full_barcode_list = {}
for line in count_file:
	line = line.strip().split("\t")
	if first_line == True:
		first_line = False
		sample_list = line
	else:
		barcodeID = line[0]
		full_barcode_list[barcodeID] = ''
		for number in range(1,len(line)):
			count = int(line[number])
			sampleID = sample_list[number-1]
			if count > 0:
				try:
					barcodeID_count_dict[sampleID][barcodeID] = count
				except:
					barcodeID_count_dict[sampleID] = {}
					barcodeID_count_dict[sampleID][barcodeID] = count
count_file.close()
full_barcode_list = list(full_barcode_list.keys())
full_barcode_list = sorted(full_barcode_list)

plate_list = list(plate_dict.keys())
# plate_list = plate_list[0:5]
plate_list = sorted(plate_list)


def hist_barcode_distribution(plate_no):
	max_count = 0
	count_distribution_dict = {}
	sub_count_dict = {}
	for iter_num in range(0,iterations):
		# print(iter_num)
		count_distribution_dict[iter_num] = {}
		sub_count_dict[iter_num] = {}
		local_samplelist = plate_dict[plate_no]

		local_barcode_count_dict = {}
		local_sub_count_dict = {}
		for s in range(0,len(local_samplelist)):
			sampleID = local_samplelist[s]
			try:
				sample_barcode_count_dict = barcodeID_count_dict[sampleID]
			except:
				sample_barcode_count_dict = {}
			if sample_barcode_count_dict != {}:
				subsampled_count_dict = subsample_counts(sample_barcode_count_dict,subsample_level)
				if subsampled_count_dict != None:
					local_barcode_count_dict[sampleID] = subsampled_count_dict
		local_samplelist = list(local_barcode_count_dict.keys())

		if len(local_samplelist) >0:
			for b in range(0,len(full_barcode_list)):
				barcodeID = full_barcode_list[b]
				sample_count = 0
				for s in range(0,len(local_samplelist)):
					sampleID = local_samplelist[s]
					try:
						count = local_barcode_count_dict[sampleID][barcodeID]
					except:
						count = 0
					if count >= 1:
						sample_count += 1
				if sample_count >0:
					try:
						count_distribution_dict[iter_num][sample_count] += 1
					except:
						count_distribution_dict[iter_num][sample_count] = 1
					try:
						sub_count_dict[iter_num][sampleID] += 1
					except:
						sub_count_dict[iter_num][sampleID] = 1
		count_list = list(count_distribution_dict[iter_num].keys())
		if len(count_list) >0:
			local_max_count = max(count_list)
			if local_max_count > max_count:
				max_count = local_max_count

	avg_count_distribution_dict = {}
	for sample_count in range(1,max_count):
		count_prop_list = []
		for iter_num in range(0,iterations):
			sub_count = sub_count_dict[iter_num][sampleID]
			try:
				count = count_distribution_dict[iter_num][sample_count]
				prop = count/sub_count
			except:
				count = 0
				prop = 0.0
			if prop > 0:
				count_prop_list.append(prop)
		if len(count_prop_list)>=3:
			if interval_method == '95CI':
				avg_count_prop, CI_low, CI_high = mean_confidence_interval(count_prop_list)
			elif interval_method == 'IQR':
				avg_count_prop, CI_low, CI_high = quantile_interval(count_prop_list)
			count_tup = (avg_count_prop,CI_low,CI_high)
			avg_count_distribution_dict[sample_count] = count_tup
	return (plate_no,avg_count_distribution_dict)


processed_list = Parallel(n_jobs=num_cores)(delayed(hist_barcode_distribution)(plate_no) for plate_no in plate_list)


average_count_distribution_dict = {}
max_count = 0
for sample_tup in processed_list:
	plate_no = sample_tup[0]
	average_count_distribution_dict[plate_no] = sample_tup[1]
	local_samplecounts = list(average_count_distribution_dict[plate_no].keys())
	if len(local_samplecounts)>0:
		if max(local_samplecounts)>max_count:
			max_count = max(local_samplecounts)
outlines = ''
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	outlines+= '\t'+plate_condition_dict[plate_no]
outlines+= '\n'
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	outlines+= '\t'+str(plate_no)
outlines+= '\n'
for sample_count in range(1,max_count):
	outlines += str(sample_count)
	for p in range(0,len(plate_list)):
		plate_no = plate_list[p]
		try:
			count_tup = average_count_distribution_dict[plate_no][sample_count]
		except:
			count_tup = (0.0,0.0,0.0)
		outlines +='\t'+str(count_tup[0])+','+str(count_tup[1])+','+str(count_tup[2])
	outlines+= '\n'

outfile = open('barcode_count_distribution.'+info_suffix+'.txt','w')
outfile.write(outlines)
outfile.close()

color_key = {'0.2MC':'#00aeef','0.3MC':'#776cb5','0.4MC':'#ee2a7b','KD':'#8dc63f','WT':'#be1e2d','SH':'#414042'}
num_row = 1
num_col = len(exp_list)
panel_width = 3

fig, ax = plt.subplots(num_row,num_col, sharex=True, sharey=True)
fig.set_size_inches(num_col*panel_width, num_row*panel_width)

col_dict = {}
for e in range(0,len(exp_list)):
	exp_val = exp_list[e]
	col_dict[exp_val] = e
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	m_list = []
	l_list = []
	u_list = []
	x_list = []
	conditionID = plate_condition_dict[plate_no]
	exp_val = plate_exp_dict[plate_no]
	col = col_dict[exp_val]
	color_val = color_key[conditionID]
	for sample_count in range(1,max_count):
		try:
			count_tup = average_count_distribution_dict[plate_no][sample_count]
		except:
			count_tup = (0.0,0.0,0.0)
		if count_tup[0] >0 and count_tup[1]>0 and count_tup[2]>0:
			m_list.append(count_tup[0])
			l_list.append(count_tup[1])
			u_list.append(count_tup[2])
			x_list.append(sample_count)
	if len(x_list)>5:
		fit_window = 5
		poly_order = 3
		m_hat = savgol_filter(m_list, fit_window, poly_order)
		l_hat = savgol_filter(l_list, fit_window, poly_order)
		u_hat = savgol_filter(u_list, fit_window, poly_order)
		ax[col].plot(x_list, m_hat,color=color_val, linewidth=0.2)
		ax[col].fill_between(x_list,l_hat,u_hat, color=color_val, alpha=0.4)
	elif len(x_list)>=4:
		fit_window = 4
		poly_order = 2
		m_hat = savgol_filter(m_list, fit_window, poly_order)
		l_hat = savgol_filter(l_list, fit_window, poly_order)
		u_hat = savgol_filter(u_list, fit_window, poly_order)
		ax[col].plot(x_list, m_hat,color=color_val, linewidth=0.2)
		ax[col].fill_between(x_list,l_hat,u_hat, color=color_val, alpha=0.4)
	else:
		ax[col].plot(x_list, m_list,color=color_val, linewidth=0.2)
		ax[col].fill_between(x_list,l_list,u_list, color=color_val, alpha=0.4)
	ax[col].set_ylim(minplot_prop,1.0)
	ax[col].set_yscale('log')
plt.tight_layout()
plt.savefig('barcode_range_distribution'+info_suffix+'.pdf')
