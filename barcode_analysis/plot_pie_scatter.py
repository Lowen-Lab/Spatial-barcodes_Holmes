import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
###################################   USER DEFINED VARIABLES   ###################################
project_dir = "./"
scale_radius_to_ct = False

count_table_name = './barcode_info/all_barcode_count.table.txt'
# freq_table_name = './barcode_info/all_barcode_freq.table.txt'
barcode_color_filename = 'barcode_color_map.txt'
ouput_directory = project_dir+'pie_charts/'

##################################################################################################
sample_info_dict = {}
plate_dict = {}
plate_slipID_dict = {}
site_dict = {}
plate_title_dict = {}
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
					ct_val = (round(np.log10(1e12*math.exp(ct_val*-0.6931)),2)/6.0) ## Transformed value to scale size of pie chart, optionally
				except:
					ct_val = 0.1
				exp_val = line[0]
				MC_val = line[3]
				plate_no = int(line[4])
				plate_slipID = str(plate_no)+"-"+str(coverslip_no)
				plate_slipID_dict[plate_slipID] = sampleID
				try:
					plate_dict[plate_no].append(sampleID)
				except:
					plate_dict[plate_no] = [sampleID]

				site_dict[sampleID] = coverslip_no
				sample_info_dict[sampleID] = (plate_no,ct_val,MC_val)
				plate_title_dict[plate_no] = MC_val+" "+exp_val
infile.close()
plate_list = list(plate_dict.keys())

#load x,y coordinates for coverslips for scatterplot
location_dict = {}
infile = open('location_key.txt',"r")
for line in infile:
	line = line.strip()
	if len(line)>0:
		if line[0] !="#":
			line = line.split('\t')
			coverslip_no = int(line[0])
			loc_x = int(line[1])
			loc_y = int(line[2])
			location_dict[coverslip_no] = (loc_x,loc_y)
infile.close()
plot_row_col_dict = {}
rowID_list = {}
colID_list = {}
#load x,y coordinates for coverslips for scatterplot
infile = open('plate_row_col_key.txt',"r")
for line in infile:
	line = line.strip()
	if len(line)>0:
		if line[0] !="#":
			line = line.split('\t')
			loc_xy = line[0]
			rowcol = line[1]
			coverslip_no = line[2]
			row = int(rowcol.split(',')[0])
			col = int(rowcol.split(',')[1])
			if coverslip_no != 'NA':
				coverslip_no = int(coverslip_no)
				loc_x = int(loc_xy.split(',')[0])
				loc_y = int(loc_xy.split(',')[1])
			plot_row_col_dict[coverslip_no] = (row,col)
			rowID_list[row] = ''
			colID_list[col] = ''
infile.close()
rowID_list = list(rowID_list.keys())
colID_list = list(colID_list.keys())

color_dict = {}
infile = open('color_key.txt',"r")
for line in infile:
	line = line.strip()
	if len(line)>0:
		if line[0] !="#":
			line = line.split('\t')
			coverslip_no = int(line[0])
			hex_val = line[1]
			color_dict[coverslip_no] = hex_val
infile.close()

### Load barcode color values
colorblind_friendly_palette = ['#E8ECFB', '#D9CCE3', '#D1BBD7', '#CAACCB', '#BA8DB4', '#AE76A3', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#437DBF', '#5289C7', '#6195CF', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F6C141', '#F4A736', '#F1932D', '#EE8026', '#E8601C', '#E65518', '#DC050C', '#A5170E', '#72190E', '#42150A']
barcode_color_dict = {}
outlines = ''
if os.path.isfile(barcode_color_filename):
	hex_list = {}
	infile = open(barcode_color_filename,'r')
	for line in infile:
		line = line.strip().split("\t")
		barcodeID = line[0]
		hex_val = line[1]
		barcode_color_dict[barcodeID] = hex_val
else:
	print('No barcode color map file provided. Using alphabetical order of barcodes to generate one.')
	for num in range(0,len(barcode_list)):
		barcodeID = barcode_list[num]
		color_index = num%len(colorblind_friendly_palette)
		color_hex_value = colorblind_friendly_palette[color_index]
		barcode_color_dict[barcodeID] = color_hex_value

#load barcode counts and generate barocde frequency dictionary
countnfile = open(count_table_name,"r")
first_line = True
barcodeID_count_dict = {}
barcodeID_temp_dict = {}
for line in countnfile:
	line = line.strip().split("\t")
	if first_line == True:
		first_line = False
		sample_list = line
	else:
		barcodeID = line[0]
		barcodeID_temp_dict[barcodeID] = ''
		for number in range(1,len(line)):
			count = int(line[number])
			sampleID = sample_list[number-1]
			if count > 1:
				try:
					barcodeID_count_dict[sampleID][barcodeID] = count
				except:
					barcodeID_count_dict[sampleID] = {}
					barcodeID_count_dict[sampleID][barcodeID] = count
countnfile.close()
barcode_sum_dict = {}
barcode_freq_dict = {}
for sampleID in barcodeID_count_dict:
	sample_sum = 0
	for barcodeID in barcodeID_count_dict[sampleID]:
		sample_sum += barcodeID_count_dict[sampleID][barcodeID]
	barcode_sum_dict[sampleID] = sample_sum
	barcode_freq_dict[sampleID] = {}
	sample_sum = float(sample_sum)
	for barcodeID in barcodeID_count_dict[sampleID]:
		count = barcodeID_count_dict[sampleID][barcodeID]
		if count == 0:
			freq = 0.
		else:
			freq = count/sample_sum
		barcode_freq_dict[sampleID][barcodeID] = freq
sample_list = list(barcode_freq_dict.keys())
sample_list = sorted(sample_list)

full_barcode_list = list(barcodeID_temp_dict.keys())
del barcodeID_temp_dict


freq_dict = {}
sample_color_dict = {}
sample_to_column_num_dict = {}
barcode_list = []
barcode_sum_order_dict = {}
for b in range(0,len(full_barcode_list)):
	barcodeID = full_barcode_list[b]
	barcode_sum_order_dict[barcodeID] = 0.
for s in range(0,len(sample_list)):
	sampleID = sample_list[s]
	for b in range(0,len(full_barcode_list)):
		barcodeID = full_barcode_list[b]
		barcode_color = barcode_color_dict[barcodeID]
		try:
			freq = barcode_freq_dict[sampleID][barcodeID]
		except:
			freq = 0.0
		if freq > 0.0:
			try:
				plate_no = sample_info_dict[sampleID][0]
			except:
				plate_no = ''
			if plate_no != '':
				coverslip_no = site_dict[sampleID]
				plate_slipID = str(plate_no)+"-"+str(coverslip_no)
				try:
					freq_dict[plate_slipID].append(freq)
				except:
					freq_dict[plate_slipID] = []
					freq_dict[plate_slipID].append(freq)
				try:
					sample_color_dict[plate_slipID].append(barcode_color)
				except:
					sample_color_dict[plate_slipID] = []
					sample_color_dict[plate_slipID].append(barcode_color)
				if freq > 0.:
					barcode_sum_order_dict[barcodeID]+=freq
sum_order_list = []
for barcodeID in barcode_sum_order_dict:
	freq_sum = barcode_sum_order_dict[barcodeID]
	tup = (freq_sum,barcodeID)
	sum_order_list.append(tup)
sum_order_list = sorted(sum_order_list,reverse=True)
barcode_order_list = []
for i in range(0,len(sum_order_list)):
	barcodeID = sum_order_list[i][1]
	barcode_order_list.append(barcodeID)
coverslip_list = list(freq_dict.keys())
full_barcode_list = sorted(full_barcode_list)

num_row = 11
num_col = 11
panel_width = 1.25

def reduce_invisible_complexity(freq_array_in,color_array,min_visible_freq=5e-3):
	merged = 0
	freq_array_out = []
	color_array_out = []
	current_invisible_sum = 0.0
	current_color_num = 0
	for i in range(0,len(freq_array_in)):
		freq = freq_array_in[i]
		if freq >= min_visible_freq or (current_invisible_sum+freq) >min_visible_freq:
			if current_invisible_sum >0.0:
				freq_array_out.append(current_invisible_sum)
				color_val = color_array[current_color_num]
				color_array_out.append(color_val)
			
			current_invisible_sum = 0.0
			current_color_num = i
			freq_array_out.append(freq)
			color_val = color_array[current_color_num]
			color_array_out.append(color_val)
		else:
			if current_invisible_sum == 0.0:
				current_color_num = i
			current_invisible_sum += freq
			merged += 1
	return freq_array_out,color_array_out

if not os.path.exists(ouput_directory):
	os.makedirs(ouput_directory)

for p in range(0,len(plate_list)):
	plate_no = plate_list[p]
	fig, ax = plt.subplots(num_row, num_col, sharex=True, sharey=True)
	fig.set_size_inches(num_col*panel_width, num_row*panel_width)
	for coverslip_no in range(1,34):
		row = plot_row_col_dict[coverslip_no][1]
		col = plot_row_col_dict[coverslip_no][0]
		plate_slipID = str(plate_no)+"-"+str(coverslip_no)
		try:
			sampleID = plate_slipID_dict[plate_slipID]
		except:
			sampleID = ''
		if sampleID == '':
			ax[row,col].axis('off')
		else:
			try:
				freq_array = freq_dict[plate_slipID]
				color_array = sample_color_dict[plate_slipID]
			except:
				freq_array = []
				color_array = []
			if scale_radius_to_ct==True:
				try:
					ct_val = sample_info_dict[sampleID][1]
					radius = max(0.01,ct_val*1.7)
				except:
					radius = 0.01
			else:
				radius = 2.
			if freq_array != []:
				plot_freq_array,plot_color_array = reduce_invisible_complexity(freq_array,color_array)
				ax[row,col].pie(plot_freq_array,colors=plot_color_array,radius=radius)
			else:
				ax[row,col].pie([1.0],colors=['grey'],radius=radius)
	for row in range(0,num_row):
		for col in range(0,num_col):
			ax[row,col].axis('off')
	condition = sample_info_dict[plate_slipID_dict[str(plate_no)+"-17"]][2]
	if scale_radius_to_ct == False:
		plt.savefig(ouput_directory+condition+'_'+str(plate_no)+".piechart.pdf")
		plt.savefig(ouput_directory+condition+'_'+str(plate_no)+".piechart.png")
	else:
		plt.savefig(ouput_directory+condition+'_'+str(plate_no)+".piechart.Ct-scaled.pdf")
		plt.savefig(ouput_directory+condition+'_'+str(plate_no)+".piechart.Ct-scaled.png")
	plt.close()
	print(str(plate_no) + ' done')
