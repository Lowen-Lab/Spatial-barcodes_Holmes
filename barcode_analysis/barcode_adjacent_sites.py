import os
import sys
import networkx as nx
import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import multiprocessing
from joblib import Parallel, delayed
###################################   USER DEFINED VARIABLES   ###################################
project_dir = "./"
count_table_name = './barcode_info/all_barcode_count.table.txt'
freq_table_name = './barcode_info/all_barcode_freq.table.txt'

subsample_level = int(3e3)
iterations = 1000
interval_method = '95CI'#'IQR'#
parallel_max_cpu = 120
max_ct = 35.0
max_coverslip_dist_nearest = 20
min_coverslip_dist_furthest = 21
num_neighbors_to_compare = 4
directions_to_use = ['up','down','right','left']

info_suffix = '.count_pos.iter'+str(iterations)+'.'+str(subsample_level)+'.max-ct'+str(max_ct)+'.coverslips'+str(num_neighbors_to_compare)

if parallel_max_cpu == 0:
	num_cores = multiprocessing.cpu_count()
elif parallel_max_cpu > 0:
	num_cores = min(parallel_max_cpu,multiprocessing.cpu_count())
elif parallel_max_cpu < 0:
	num_cores = multiprocessing.cpu_count()-parallel_max_cpu
####################################


import scipy.stats
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

location_dict = {}
infile = open('location_key.txt',"r")
for line in infile:
	line = line.strip()#.split("\t")
	if len(line)>0:
		if line[0] !="#":
			line = line.split('\t')
			coverslip_no = int(line[0])
			loc_x = int(line[1])
			loc_y = int(line[2])
			location_dict[coverslip_no] = (loc_x,loc_y)
infile.close()

def simple_euc_dist(ax,ay,bx,by):
	return math.sqrt((ax - bx)**2 + (ay - by)**2)

location_dist_dict = {}
furthest_dict = {}
for coverslip1 in location_dict:
	location_dist_dict[coverslip1] = []
	loc1 = location_dict[coverslip1]
	x1,y1=loc1[0],loc1[1]
	for coverslip2 in location_dict:
		if coverslip1 != coverslip2:
			loc2 = location_dict[coverslip2]
			x2,y2=loc2[0],loc2[1]
			dist = simple_euc_dist(x1,y1,x2,y2)
			rand_val = random.uniform(-1*dist/100.,dist/100.)
			dist_tup = (dist+rand_val,coverslip2)
			location_dist_dict[coverslip1].append(dist_tup)
	location_dist_dict[coverslip1] = sorted(location_dist_dict[coverslip1],reverse=False)

row_col_dict = {}
row_col_to_coverslip_dict = {}
rowID_list = {}
colID_list = {}
infile = open('plate_row_col_key.txt',"r")
for line in infile:
	line = line.strip()#.split("\t")
	if len(line)>0:
		if line[0] !="#":
			line = line.split('\t')
			loc_xy = line[0]
			rowcol = line[1]
			coverslip_no = line[2]
			row = int(rowcol.split(',')[0])
			col = int(rowcol.split(',')[1])
			if coverslip_no != 'NA' and coverslip_no != '17':
				# coverslip_no = coverslip_no
				row = int(row/2)
				col = int(col/2)
				loc_x = int(loc_xy.split(',')[0])
				loc_y = int(loc_xy.split(',')[1])
				rowcol = str(row)+'__'+str(col)
				row_col_to_coverslip_dict[rowcol] = coverslip_no
				row_col_dict[coverslip_no] = (col,row)
				rowID_list[row] = ''
				colID_list[col] = ''
infile.close()
rowID_list = list(rowID_list.keys())
colID_list = list(colID_list.keys())
coverslip_list = list(row_col_dict.keys())

sample_info_dict = {}
plate_dict = {}
plate_exp_dict = {}
exp_list = {}
condition_list = {}
plate_slipID_dict = {}
site_dict = {}
plate_condition_dict = {}
infile = open('sample_key.txt',"r")
for line in infile:
	line = line.strip()#.split("\t")
	if len(line)>0:
		if line[0] !="#":
			line = line.split('\t')
			sampleID = line[2]
			if sampleID != 'NA':
				coverslip_no = int(line[1])
				try:
					ct_val = float(line[5])
				except:
					ct_val = 0.
				exp_val = line[0]
				MC_val = line[3]
				plate_no = int(line[4])
				try:
					plate_rep_no = int(line[7])
				except:
					plate_rep_no = 'na'
				if plate_rep_no != 'na':
					exp_val = plate_rep_no
					plate_slipID = str(plate_no)+"-"+str(coverslip_no)
					plate_slipID_dict[plate_slipID] = sampleID
					if coverslip_no != 17:
						try:
							plate_dict[plate_no].append(sampleID)
						except:
							plate_dict[plate_no] = [sampleID]

						site_dict[sampleID] = coverslip_no
						sample_info_dict[sampleID] = (plate_no,ct_val,MC_val,exp_val)
						plate_condition_dict[plate_no] = MC_val#+" "+exp_val
						plate_exp_dict[plate_no] = exp_val
						exp_list[exp_val] = ''
						condition_list[MC_val] = ''
infile.close()

count_file = open(count_table_name,"r")
first_line = True
barcodeID_count_dict = {}
full_barcode_list = {}
read_count_dict = {}
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
				exp_val = sampleID.split('_')[0]
				try:
					barcodeID_count_dict[sampleID][barcodeID] = count
				except:
					barcodeID_count_dict[sampleID] = {}
					barcodeID_count_dict[sampleID][barcodeID] = count
				try:
					read_count_dict[sampleID]+=count
				except:
					read_count_dict[sampleID]=count
count_file.close()
full_barcode_list = list(full_barcode_list.keys())
full_barcode_list = sorted(full_barcode_list)


sample_specific_neighbor_dict = {}
sample_specific_furthest_neighbor_dict = {}
exp_list = list(exp_list.keys())
exp_list = sorted(exp_list)
condition_list = list(condition_list.keys())
condition_list = sorted(condition_list)
plate_list = list(plate_dict.keys())
# plate_list = plate_list[0:5]
plate_list = sorted(plate_list)
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	local_samplelist = plate_dict[plate_no]
	for i in range(0,len(local_samplelist)):
		focal_sampleID = local_samplelist[i]
		focal_coverslip_no = site_dict[focal_sampleID]
		try:
			focal_ct = sample_info_dict[sampleID][1]
		except:
			focal_ct =0.
		try:
			focal_read_count = read_count_dict[focal_sampleID]
		except:
			focal_read_count = 0
		if focal_ct <= max_ct and focal_read_count >= subsample_level:
			local_dist_list = location_dist_dict[focal_coverslip_no]
			local_dist_list = sorted(local_dist_list,reverse=False)
			num_added = 0
			for j in range(0,len(local_dist_list)):
				other_coverslip_dist = local_dist_list[j][0]
				other_coverslip_no = local_dist_list[j][1]
				other_plate_slipID = str(plate_no)+"-"+str(other_coverslip_no)
				other_sampleID = plate_slipID_dict[other_plate_slipID]
				try:
					other_ct = sample_info_dict[other_sampleID][1]
				except:
					other_ct = 0.
				try:
					other_read_count = read_count_dict[other_sampleID]
				except:
					other_read_count = 0
				if other_sampleID != focal_sampleID and other_coverslip_no != 17 and other_ct <= max_ct and other_read_count >= subsample_level*1.1 and other_coverslip_dist < max_coverslip_dist_nearest: # and num_added<num_neighbors_to_compare
					num_added += 1
					try:
						sample_specific_neighbor_dict[focal_sampleID].append(other_coverslip_no)
					except:
						sample_specific_neighbor_dict[focal_sampleID] = [other_coverslip_no]
			local_dist_list = sorted(local_dist_list,reverse=True)
			num_added = 0
			for j in range(0,len(local_dist_list)):
				other_coverslip_dist = local_dist_list[j][0]
				other_coverslip_no = local_dist_list[j][1]
				other_plate_slipID = str(plate_no)+"-"+str(other_coverslip_no)
				other_sampleID = plate_slipID_dict[other_plate_slipID]
				try:
					other_ct = sample_info_dict[other_sampleID][1]
				except:
					other_ct = 0.
				try:
					other_read_count = read_count_dict[other_sampleID]
				except:
					other_read_count = 0
				try:
					local_nearest_sample_list = sample_specific_neighbor_dict[focal_sampleID]
				except:
					local_nearest_sample_list = []
				if other_sampleID != focal_sampleID and other_coverslip_no != 17 and other_ct <= max_ct and other_read_count >= subsample_level*1.1 and other_sampleID not in local_nearest_sample_list and other_coverslip_dist > min_coverslip_dist_furthest: # and num_added<num_neighbors_to_compare
					num_added += 1
					try:
						sample_specific_furthest_neighbor_dict[focal_sampleID].append(other_coverslip_no)
					except:
						sample_specific_furthest_neighbor_dict[focal_sampleID] = [other_coverslip_no]

import scipy.stats
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
	return count_dict_out#,subsampled_reads#,freq_dict_out


def hist_barcode_distribution(plate_no):
	max_count = 0
	near_count_distribution_dict = {}
	far_count_distribution_dict = {}
	sub_count_dict = {}
	for iter_num in range(0,iterations):
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
				subsampled_count_dict = subsample_counts(sample_barcode_count_dict,subsample_level)#,sub_count_dict[iter_num][sampleID]
				if subsampled_count_dict != None:
					local_barcode_count_dict[sampleID] = subsampled_count_dict
		local_samplelist = list(local_barcode_count_dict.keys())
		if len(local_samplelist) >0:
			for s in range(0,len(local_samplelist)):
				sampleID = local_samplelist[s]
				focal_coverslip_no = site_dict[sampleID]
				try:
					neighbor_slips = sample_specific_neighbor_dict[sampleID]
					neighbor_slips = neighbor_slips[0:min(len(neighbor_slips),num_neighbors_to_compare+2)]
					random.shuffle(neighbor_slips)
					neighbor_slips = neighbor_slips[0:num_neighbors_to_compare]
				except:
					neighbor_slips = []
				try:
					furthest_slips = sample_specific_furthest_neighbor_dict[sampleID]
					furthest_slips = furthest_slips[0:min(len(furthest_slips),num_neighbors_to_compare+2)]
					random.shuffle(furthest_slips)
					furthest_slips = furthest_slips[0:num_neighbors_to_compare]
				except:
					furthest_slips = []
				if len(neighbor_slips)==num_neighbors_to_compare and len(furthest_slips)==num_neighbors_to_compare:
					all_other_slips = plate_dict[plate_no]
					near_sample_count = 0
					far_sample_count = 0
					nonzero_barcodes = 0
					for b in range(0,len(full_barcode_list)):
						barcodeID = full_barcode_list[b]
						focal_near_count = 0
						focal_far_count = 0
						local_nonzero_barcodes = 0
						try:
							count = local_barcode_count_dict[sampleID][barcodeID]
						except:
							count = 0
						if count >= 1:
							local_nonzero_barcodes += 1
							for c in range(0,len(neighbor_slips)):
								other_coverslip = neighbor_slips[c]
								plate_slipID = str(plate_no)+"-"+str(other_coverslip)
								other_sampleID = plate_slipID_dict[plate_slipID]
								try:
									other_count = local_barcode_count_dict[other_sampleID][barcodeID]
								except:
									other_count = 0
								if other_count >= 1:
									focal_near_count += 1
							for c in range(0,len(furthest_slips)):
								other_coverslip = furthest_slips[c]
								plate_slipID = str(plate_no)+"-"+str(other_coverslip)
								other_sampleID = plate_slipID_dict[plate_slipID]
								try:
									other_count = local_barcode_count_dict[other_sampleID][barcodeID]
								except:
									other_count = 0
								if other_count >= 1:
									focal_far_count += 1
						if focal_near_count >0:
							near_sample_count+=1
						if focal_far_count >0:
							far_sample_count+=1
						if local_nonzero_barcodes >0:
							nonzero_barcodes += 1
					if nonzero_barcodes>0:
						near_sample_count = (near_sample_count/nonzero_barcodes)
						far_sample_count = (far_sample_count/nonzero_barcodes)
						try:
							near_count_distribution_dict.append(near_sample_count)
						except:
							near_count_distribution_dict = [near_sample_count]
						try:
							far_count_distribution_dict.append(far_sample_count)
						except:
							far_count_distribution_dict = [far_sample_count]
	return (plate_no,near_count_distribution_dict,far_count_distribution_dict)


processed_list = Parallel(n_jobs=num_cores)(delayed(hist_barcode_distribution)(plate_no) for plate_no in plate_list)


average_near_count_distribution_dict = {}
average_far_count_distribution_dict = {}

# max_count = 0
max_obs = 0
for sample_tup in processed_list:
	plate_no = sample_tup[0]
	if len(sample_tup[1])>0 and len(sample_tup[2])>0:
		average_near_count_distribution_dict[plate_no] = sample_tup[1]
		average_far_count_distribution_dict[plate_no] = sample_tup[2]
		local_max_obs = max(average_near_count_distribution_dict[plate_no])
		if local_max_obs > max_obs:
			max_obs = local_max_obs
		local_max_obs = max(average_far_count_distribution_dict[plate_no])
		if local_max_obs > max_obs:
			max_obs = local_max_obs
if max_obs >0.9 and max_obs < 1.0:
	max_obs = 1.0

plate_list = list(average_near_count_distribution_dict.keys())
num_row = len(condition_list)
num_col = len(exp_list)
panel_width = 3

num_bins = 100
bin_vals = np.linspace(0.,1.,num_bins+1)
bin_dict_near = {}
bin_dict_far = {}
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	bin_dict_near[plate_no],bin_edges = np.histogram(average_near_count_distribution_dict[plate_no],bins=bin_vals)
	bin_dict_far[plate_no],bin_edges = np.histogram(average_far_count_distribution_dict[plate_no],bins=bin_vals)
outlines = '\t'
for b in range(0,len(bin_vals)-1):
	val = round(bin_vals[b],3)
	outlines+= '\t'+str(val)
outlines+= '\n'
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	outlines+= str(plate_condition_dict[plate_no])+'\t'+str(plate_no)
	for b in range(0,len(bin_vals)-1):
		near_count = bin_dict_near[plate_no][b]
		far_count = bin_dict_far[plate_no][b]
		outlines += '\t'+str(near_count)+','+str(far_count)
	outlines+= '\n'
outfile = open('adjacent_barcode_distribution'+info_suffix+'.bin'+str(num_bins)+'.txt','w')
outfile.write(outlines)
outfile.close()

num_bins = 20
bin_vals = np.linspace(0.,1.,num_bins+1)

bin_dict_near = {}
bin_dict_far = {}
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	bin_dict_near[plate_no],bin_edges = np.histogram(average_near_count_distribution_dict[plate_no],bins=bin_vals)
	bin_dict_far[plate_no],bin_edges = np.histogram(average_far_count_distribution_dict[plate_no],bins=bin_vals)

outlines = '\t'
for b in range(0,len(bin_vals)-1):
	val = round(bin_vals[b],3)
	outlines+= '\t'+str(val)
outlines+= '\n'
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	outlines+= str(plate_condition_dict[plate_no])+'\t'+str(plate_no)
	for b in range(0,len(bin_vals)-1):
		near_count = bin_dict_near[plate_no][b]
		far_count = bin_dict_far[plate_no][b]
		outlines += '\t'+str(near_count)+','+str(far_count)
	outlines+= '\n'

outfile = open('adjacent_barcode_distribution'+info_suffix+'.bin'+str(num_bins)+'.txt','w')
outfile.write(outlines)
outfile.close()

color_key = {'0.2MC':'#00aeef','0.3MC':'#776cb5','0.4MC':'#ee2a7b','KD':'#8dc63f','WT':'#be1e2d','SH':'#414042'}

fig, ax = plt.subplots(num_row,num_col, sharex=True, sharey=True)
fig.set_size_inches(num_col*panel_width, num_row*panel_width)

col_dict = {}
for e in range(0,len(exp_list)):
	exp_val = exp_list[e]
	col_dict[exp_val] = e
row_dict = {}
for r in range(0,len(condition_list)):
	conditionID = condition_list[r]
	row_dict[conditionID] = r

plot_dict = {}
plot_color_dict = {}
plot_label_dict = {}
for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	conditionID = plate_condition_dict[plate_no]
	exp_val = plate_exp_dict[plate_no]
	col = col_dict[exp_val]
	color_val = color_key[conditionID]
	plate_label = str(plate_no)+'-'+conditionID
	try:
		plot_dict[col].append(average_near_count_distribution_dict[plate_no])
	except:
		plot_dict[col] = []
		plot_dict[col].append(average_near_count_distribution_dict[plate_no])
	try:
		plot_color_dict[col].append(color_val)
		plot_label_dict[col].append(plate_label)
	except:
		plot_color_dict[col]=[color_val]
		plot_label_dict[col]=[plate_label]

for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	conditionID = plate_condition_dict[plate_no]
	exp_val = plate_exp_dict[plate_no]
	col = col_dict[exp_val]
	row = row_dict[conditionID]
	color_val = color_key[conditionID]
	x[row,col].hist([average_near_count_distribution_dict[plate_no],average_far_count_distribution_dict[plate_no]],bins=bin_vals, density=True, histtype='bar', color=['dodgerblue','crimson'])
	ax[row,col].set_title(conditionID+' - '+str(plate_no))
plt.tight_layout()
plt.savefig('adjacent_barcode_distribution'+info_suffix+'.pdf')
