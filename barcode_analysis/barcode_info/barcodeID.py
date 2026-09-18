'''
barcoded amplicon processing pipeline - v1.4
###################################################################################################
#####################################    About this script    #####################################
###################################################################################################

This script is designed to take in amplicon sequences that have nucleotide barcodes at specific 
sites. There is no minimum or maximum number of barcodes, but this script currently only permits 
two possible nucleotides per site (e.g. a barcode at site 33 can be either A or T, but not C or G).
This script process the illumina sequence files to generate tables that summarize the frequency of
all barcodes detected in each sample, as well as reporting alpha and beta diversity indicies for
those samples.

Barcodes are identified in reads where the alleles you detect are: (1) from the list of 
possible nucleotides at barcode sites, (2) match all the expected nucleotide at all non-barcode site,
and (3) meet the base quality thresholds set by the user.

Features of this pipeline:
 -	Input sequences can be compressed or uncompressed .fastq files
 -	It has a parallel processing mode that takes advantage of multi-core processors
 -	A key feature is that you may interrupt the script running at any time, then resume , and will not attempt
	to redo steps that have already been completed, unless requested by the user.
		- Safeguards are in place to prevent incompletely generated output files from being saved
		- Additional samples can be added to a folder at any time, and re-running BarcodeID will process
		  only the additional samples, before generating combined outputs
 -	All information about reads discarded because they contain unexpected bases are saved in separate
	tables to help identify positively selected mutants.
 -	The script has the ability to complete its analysis where the only required user input is the path to
	the directory where reads are located (default: "raw_data")

########################################    Requirements    #######################################
python3
bbtools (available in path)
Java (for running bbtools)
bash (for calling bbtools) (this is available by default on Mac and Linus operating systems)

For any questions, please contact the author of this script, Dave VanInsberghe, through the Lowen Lab
GitHub, or directly via dvanins@emory.edu.
If you find this script helpful, please cite the original publication to which it was published.

###################################################################################################
###################################   HOW TO RUN THIS SCRIPT    ###################################
###################################################################################################

In terminal, enter:
cd /location/of/script/
python barcodeID.py

The first time you run the script, it will pick a random sample and attempt to predict the barcode 
site locations and possible nucleotides. Answer the text prompts to confirm the prediction is 
correct, or enter the correct values. All remaining samples will then be processed.

###################################################################################################
#######################################    UPDATE NOTES    ########################################
###################################################################################################
03/16/2033 - v0.2 -	Added ability to process reads that have already been cleaned/merged
05/17/2023 - v0.3 -	Added auto-detection of barcode sites and expected amplicon length if not available
05/31/2023 - v0.4 -	Added auto-detection of forward and reverse read file extension and file pairs
					and steps to calculate alpha and beta diversity indicies
07/05/2023 - v0.5 - Added automatic unzip feature to check if new files need unzipping first
10/06/2023 - v0.6 - Added function to trim ends of reads to skip low quality ends
11/17/2023 - v1.0 - First publicly available version
04/25/2024 - v1.1 - Added generation of stacked bar plot figures and improved file suffix prediction
10/07/2024 - v1.2 - Added barcode quality correction and agglomeration (similar to DADA2 for 16S)
06/04/2025 - v1.3 - Bug fixes and optimized quality correction runtime
03/14/2026 - v1.4 - Improved error correction, user prompts, and summary statistic figures
'''
###################################### Load required modules ######################################
import os
import sys
import math
import scipy
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import time
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

###################################### Establish variables #######################################
project_dir = "./"
output_dir = project_dir+"barcode_info/"
trim_dir = project_dir + 'trim/'
temp_dir = trim_dir+"temp/"
command_prefix = "bash "
###################################################################################################
##################################  USER DEFINED VARIABLES  #######################################
###################################################################################################
input_read_dir = project_dir+"raw_data/"

amplicon_info_infile_name = 'amplicon_info.txt'
titer_filename = 'sample_titers.txt' #this file is not necessary, but can be added to scale stacked box plots to log10 transformed viral titers. Tab separated file, one sample per line: sampleID \t titer \n

min_Q_score = 20
F_edge_ignore = 0 #these values are used to exclude the edge of reads if their quality is low and needlessly discarding excessive data due to sequencing quality issues
R_edge_ignore = 0 #these values are used to exclude the edge of reads if their quality is low and needlessly discarding excessive data due to sequencing quality issues
force_trim_pre_merge = 0 #use this value to force trimming the first x nucleotides from input reads, which is helpful when the beginning of reads are low quality
truncate_length_post_merge = 0 #use this value to truncate reads post-merging if amplicons are not always equal lengths - use only as a last resort to make processing complete, or for troubleshooting when no barcodes are detected in some samples

overwrite_existing_files = False #when False, pipeline will avoid re-running samples that have already been processed 
generate_stacked_bar_plots = False #when True, stacked bar plots will be generated from observed barcodes frequencies

reads_already_screened_for_quality = False # 'True' or 'False' --- If reads were already merged and screened for average quality and length (such as during a demultiplexing step), the script will not attempt to filter or merge the input reads
raw_read_input_type = "paired" # 'paired' or 'single' --- When 'single' the pipeline will look for barcodes in one file, when 'paired' the pipeline will look for forward and reverse reads. Single core use has not been extensively tested - recommend using "paired" and setting "parallel_max_cpu" to 1 if you only wish to use one core at a time
forward_read_suffix = "" #if empty, the script will attempt to predict these values, but it only works under specific assumptions
reverse_read_suffix = "" #this value is ignored if 'reads_already_screened_for_quality' is 'True' or 'raw_read_input_type' is 'single' - NOTE: this variable cannot be removed without causing missing value errors

parallel_process = True #'True' or 'False' --- when True, the script uses the joblib package to run multiple samples simultaneously. Currently not stable on windows
parallel_max_cpu = 180 # when parallel process true, this sets the max CPUs the script is allowed to use. If you want to use all available, set to 0. If you want to use all but one, set to -1 (or -2 to use all but 2 CPUs)

global verbose
verbose = True #when this value is 'True', the script will print status updates

num_subsamples = 20 #number of times barcode counts should be subsampled to calculate diversity indicies
num_beta_subsamples = 10
min_barcodes_per_sample = 1000 #non-unique barcode counts that pass processing to be included in final output tables and statistics
max_prop_mismatch = 0.25 #the max proportion of reads that can be discarded from a sample due to mismatches from expected nucleotide before the sample is not included in downstream analyses
############################################ FUNCTIONS ############################################
np.seterr(divide='ignore')

def run_command(command,mode="quiet"):
# def run_command(command,mode="verbose"):
	run_status = False
	if mode == "verbose":
		return_code = os.system(command)
	elif mode == "quiet":
		return_code = os.system(command+" >/dev/null 2>&1")
	if return_code == 0:
		run_status = True
	return run_status

def unzip_gz(filename,path_to_file):
	command = "gzip -d "+path_to_file+filename
	out = run_command(command)
	if out == False:
		sys.exit()
	return filename
def unzip_tar_gz(filename,path_to_file):
	command = "tar -xzf "+path_to_file+filename
	out = run_command(command)
	if out == False:
		sys.exit()
	return filename
def zip_gz(filename,path_to_file):
	command = "gzip -9 "+path_to_file+filename
	out = run_command(command)
	if out == False:
		sys.exit()
	return filename
def zip_tar_gz(filename,path_to_file):
	command = "tar -czf "+path_to_file+filename+".tar.gz "+path_to_file+filename
	out = run_command(command)
	if out == False:
		sys.exit()
	return filename

def load_barcode_info(amplicon_info_infile_name):
	expected_amplicon_seq = ''
	barcode_sites = {}
	barcode_class = {}
	library_alleles = {}
	library_names = {}
	amplicon_info_infile = open(amplicon_info_infile_name,"r")
	for line in amplicon_info_infile:
		line = line.strip()
		if len(line) >0:
			if line[0] != "#":
				line = line.split("\t")
				if line[0] == "amplicon_seq":
					expected_amplicon_seq = line[1]
				elif line[0] == "library_name":
					lib_num = int(line[1])
					lib_name = line[2]
					library_names[lib_num] = lib_name
				else:
					site = int(line[0])
					barcode_sites[site] = line[1].split(",")
					try:
						barcode_class[site] = line[2]
					except:
						barcode_class[site] = 'DS'
					if barcode_class[site] == 'CS':
						library_alleles[site] = tuple(line[3].split(","))

	amplicon_info_infile.close()
	return expected_amplicon_seq,barcode_sites,barcode_class,library_alleles,library_names

def check_if_input_zipped(input_dir,parallel_process,num_cores):
	files_found = False
	#Unzip any files that need unzipping
	tar_gz_inputs = [f for f in os.listdir(input_dir) if f.endswith(".tar.gz")]
	if len(tar_gz_inputs) > 0:  #unzip .tar.gz files
		unzip_files = input("Unzip '.tar.gz' files in input directory? (y/n)")
		if unzip_files == "y" or unzip_files == "yes" or unzip_files == "Y":
			if parallel_process == True:
				processed_list = Parallel(n_jobs=num_cores)(delayed(unzip_tar_gz)(i,input_dir) for i in tar_gz_inputs)
			else:
				for targzfile in tar_gz_inputs:
					unzip_tar_gz(gzfile,input_dir)
	if os.path.isfile(temp_dir+"file_extensions.txt") == False:
		gz_inputs = [f for f in os.listdir(input_dir) if f.endswith(".gz")]
		if len(gz_inputs) >0:  #unzip .gz files
			print("Unziping '.gz' files in input directory")
			if parallel_process == True:
				processed_list = Parallel(n_jobs=num_cores)(delayed(unzip_gz)(i,input_dir) for i in gz_inputs)
			else:
				for gzfile in gz_inputs:
					if os.path.isfile(input_dir+gzfile.replace('.gz','')) == True:
						os.remove(input_dir+gzfile.replace('.gz',''))
					unzip_gz(gzfile,input_dir)
	fq_inputs = [f for f in os.listdir(input_dir) if f.endswith(".fastq") or f.endswith(".fq")]
	gz_inputs = [f for f in os.listdir(input_dir) if f.endswith(".gz")]
	if len(fq_inputs) >0 or len(gz_inputs)>0:
		files_found = True
	return files_found

def predict_paired_file_extensions(input_dir):
	exten_filelist = [f for f in os.listdir(input_dir) if f.endswith(".fastq") or f.endswith(".fq")]
	shortest_filename = exten_filelist[0]
	for f in range(0,len(exten_filelist)):
		filename = exten_filelist[f]
		if len(filename) < len(shortest_filename):
			shortest_filename = filename
	string_dict = {}
	for i in range(-1,(-1*len(shortest_filename)),-1):
		string_dict[i] = {}
		for f in range(0,len(exten_filelist)):
			filename = exten_filelist[f]
			string = filename[i:]
			try:
				string_dict[i][string] += 1
			except:
				string_dict[i][string] = 1
	first_dichotomous = 0
	last_dichotomous = 0
	for i in range(-1,(-1*len(shortest_filename)),-1):
		if len(string_dict[i]) == 2:
			if first_dichotomous == 0:
				first_dichotomous = i
			else:
				last_dichotomous = i
	exten_count = {}
	for f in range(0,len(exten_filelist)):
		filename = exten_filelist[f]
		try:
			exten_count[filename[last_dichotomous:]] += 1
		except:
			exten_count[filename[last_dichotomous:]] = 1
	exten_pair = {1:'',2:''}
	for exten in exten_count:
		try:
			exten_pair[int(exten[first_dichotomous])] = exten
		except:
			pass
	successful_prediction = False
	if len([f for f in os.listdir(input_dir) if f.endswith(exten_pair[1])]) == len([f for f in os.listdir(input_dir) if f.endswith(exten_pair[2])]):
		if len(exten_filelist)/2 == len([f for f in os.listdir(input_dir) if f.endswith(exten_pair[1])]):
			successful_prediction = True
			return exten_pair[1],exten_pair[2]
	if successful_prediction == False:
		sys.exit('Unable to predict forward and reverse read suffix values. Enter manually in "user defined variables" section in BarcodeID script to proceed.\nExiting.')

def predict_single_file_extension(input_dir):
	exten_filelist = [f for f in os.listdir(input_dir) if f.endswith(".fastq") or f.endswith(".fq")]
	shortest_filename = exten_filelist[0]
	for f in range(0,len(exten_filelist)):
		filename = exten_filelist[f]
		if len(filename) < len(shortest_filename):
			shortest_filename = filename
	string_dict = {}
	for i in range(-1,(-1*len(shortest_filename)),-1):
		string_dict[i] = {}
		for f in range(0,len(exten_filelist)):
			filename = exten_filelist[f]
			string = filename[i:]
			try:
				string_dict[i][string] += 1
			except:
				string_dict[i][string] = 1
	last_monomorphic = 0
	searching_for_last_monomorphic = True
	for i in range(-1,(-1*len(exten_filelist[0])),-1):
		if len(string_dict[i]) == 1 and searching_for_last_monomorphic == True:
			last_monomorphic = i
		elif len(string_dict[i]) > 1 and searching_for_last_monomorphic == True:
			searching_for_last_monomorphic = False
	exten_count = {}
	for f in range(0,len(exten_filelist)):
		filename = exten_filelist[f]
		try:
			exten_count[filename[last_monomorphic:]] += 1
		except:
			exten_count[filename[last_monomorphic:]] = 1
	successful_prediction = False
	if exten_count[filename[last_monomorphic:]] == len(exten_filelist):
		successful_prediction = True
		return filename[last_monomorphic:]
	if successful_prediction == False:
		sys.exit("Unable to predict read suffix value. Enter manually to proceed.\nExiting.")

def detect_barcode_content(seq_dict,amplicon_length):
	amplicon_expect = ''
	barcode_sites = {}
	nts = ['A','T','C','G']
	prop_lineout = 'loc\tA\tT\tC\tG\n'
	minor_freq_list = []
	minor_freq_loc_list = []
	for col in range(0,amplicon_length):
		prop_lineout += str(col)
		nt_count_dict = {'A':0,'T':0,'C':0,'G':0}
		for seq in seq_dict:
			if len(seq) == amplicon_length:
				try:
					cur_nt = seq[col]
					nt_count_dict[cur_nt] += 1
				except:
					if seq[col] != "N":
						sys.exit('Unexpected character found: "'+cur_nt+'" at site '+str(col))
		nt_list = []
		for n in range(0,len(nts)):
			nt = nts[n]
			nt_counted = nt_count_dict[nt]
			nt_prop = float(nt_counted)/float(len(seq_dict))
			prop_lineout += '\t'+str(nt_prop)
			temp_tup = (nt_count_dict[nt],nt)
			nt_list.append(temp_tup)
		nt_list = sorted(nt_list, reverse=True)
		major_allele = nt_list[0][1]
		minor_allele = nt_list[1][1]
		major_freq = float(nt_list[0][0])/float(len(seq_dict))
		minor_freq = float(nt_list[1][0])/float(len(seq_dict))
		minor_freq_list.append(minor_freq)
		loc_tup = (minor_freq,col,(major_allele,minor_allele))
		minor_freq_loc_list.append(loc_tup)
		prop_lineout += '\n'
	outfile = open(project_dir+"barcode_predict_info.all_allele_freq.txt","w")
	outfile.write(prop_lineout)
	outfile.close()
	
	plot_minor_freq = []
	plot_location = []
	plot_color = []
	amplicon_expect = "N"*amplicon_length
	minor_freq_loc_list = sorted(minor_freq_loc_list,reverse=True)
	med_freq = np.median(minor_freq_list)
	for i in range(0,len(minor_freq_loc_list)):
		local_freq = minor_freq_loc_list[i][0]
		loc = minor_freq_loc_list[i][1]
		allele_tup = minor_freq_loc_list[i][2]
		amplicon_expect = amplicon_expect[0:loc]+allele_tup[0]+amplicon_expect[loc+1:amplicon_length]
		if (local_freq/med_freq)>=50 and local_freq >= 0.03:
			barcode_sites[loc] = [allele_tup[0],allele_tup[1]]
			plot_color.append('crimson')
		else:
			plot_color.append('dodgerblue')
		plot_minor_freq.append(local_freq)
		plot_location.append(loc)

	fig,ax = plt.subplots()
	plt.scatter(plot_location,plot_minor_freq,s=8,c=plot_color)
	ax.set_ylim(0.0,0.55)
	ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
	ax.set_ylabel("Frequency of minor allele")
	ax.set_xlabel("Location in amplicon")
	plt.savefig(project_dir+"barcode_predict.minor_allele_freq.pdf")
	return amplicon_expect, barcode_sites

def subset_fastq(filename,unique_seqs_to_pull=100):	
	read_len_list = []
	trim_seq_subset_dict = {}
	num_added = 0
	num_seq_passed = 0
	line_counter = -1
	avg_read_len = 0
	infile = open(filename,"r")
	for line in infile:
		line = line.strip()
		line_counter += 1
		if line_counter == 0:
			header = line
			writing = False
		elif line_counter == 1:
			num_seq_passed += 1
			if num_seq_passed <= 5000:
				read_len_list.append(len(line))
				if num_seq_passed == 5000:
					avg_read_len = int(np.median(read_len_list))
					# print(avg_read_len)
			elif len(line) == avg_read_len:
				try:
					trim_seq_subset_dict[line]
				except:
					trim_seq_subset_dict[line] = ''
					num_added += 1
					if num_added >= unique_seqs_to_pull:
						break
		elif line_counter == 3:
			line_counter = -1
	infile.close()
	return trim_seq_subset_dict,avg_read_len

def filter_fastq(filename_in,filename_out,length_min,length_max):
	line_counter = -1
	infile = open(filename_in,"r")
	outfile = open(filename_out,"w")
	for line in infile:
		line = line.strip()
		line_counter += 1
		if line_counter == 0:
			header = line
			writing = False
		elif line_counter == 1:
			if len(line) >= length_min and len(line) <= length_max:
				outfile.write(header+"\n"+line+"\n")
				writing = True
			else:
				writing = False
		else:
			if line_counter == 3:
				line_counter = -1
			if writing == True:
				outfile.write(line+"\n")
	infile.close()
	outfile.close()

def calc_expect_read_len(filename_in):
	len_list = []
	line_counter = -1
	infile = open(filename_in,"r")
	for line in infile:
		line = line.strip()
		line_counter += 1
		if line_counter == 0:
			header = line
			writing = False
		elif line_counter == 1:
			len_list.append(len(line))
		elif line_counter == 3:
			line_counter = -1
			if len(len_list) >= 5000:
				break
	infile.close()
	try:
		return int(np.nanmedian(len_list))
	except:
		return 0

def check_adapter(filename_in):
	infile = open(filename_in,'r')
	adapter_pass = False
	for line in infile:
		line = line.strip()
		if len(line) >0:
			if line[0] == ">":
				header = line
			else:
				seq = line
				if seq != "N":
					adapter_pass = True
	return adapter_pass

def fastq_to_fasta(filename_in,filename_out):
	line_counter = -1
	infile = open(filename_in,"r")
	outfile = open(filename_out,"w")
	for line in infile:
		line = line.strip()
		line_counter += 1
		if line_counter == 0:
			outfile.write(line.split(" ")[0]+"\n")
		elif line_counter == 1:
			outfile.write(line+"\n")
		elif line_counter == 3:
			line_counter = -1
	infile.close()
	outfile.close()
	return filename_out

def pull_qual_info(input_seq_filename,max_read_count=1e6,min_qual_count=20):
	flank_ignore = 0
	infile = open(input_seq_filename,"r")
	line_counter = -1
	read_count = 0
	qual_sub_dict = {}
	loc_sub_dict = {}
	sub_per_read_dict = {}
	keep_reading = True
	for line in infile:
		line = line.strip()
		line_counter += 1
		if line_counter == 0:
			head = line
			read_count += 1
			if read_count == max_read_count:
				keep_reading = False
		elif line_counter == 1:
			nt_string = line
			nt_string = nt_string[0:len(nt_string)-flank_ignore]
		elif line_counter == 3:
			if keep_reading == True:
				line_counter = -1
			phreds = line
			phreds = phreds[0:len(phreds)-flank_ignore]
			read_sub_info = []
			mismatch_count = 0
			lowQ_count = 0
			if len(nt_string)==len(expected_amplicon_seq):
				for loc in range(0,len(nt_string)):
					backbone_nt = ''
					expected_nts = {}
					site_type = ''
					try:
						expected_nts = barcode_sites[loc]
						site_type = "barcode"
					except:
						expected_nts[expected_amplicon_seq[loc]] = ''
						backbone_nt = expected_amplicon_seq[loc]
						site_type = "backbone"
					if site_type == "backbone":
						nt = nt_string[loc]
						qu = ord(phreds[loc])-33
						sub_tup = (backbone_nt,nt,qu,loc)
						if qu >= min_qual_count:
							read_sub_info.append(sub_tup)
							if nt != backbone_nt:
								mismatch_count += 1
						else:
							lowQ_count += 1
				try:
					sub_per_read_dict[mismatch_count] += 1
				except:
					sub_per_read_dict[mismatch_count] = 1
				if mismatch_count <= 2 and lowQ_count == 0:
					for sub_tup in read_sub_info:
						backbone_nt = sub_tup[0]
						sub_nt = sub_tup[1]
						qu = sub_tup[2]
						loc = sub_tup[3]
						try:
							qual_sub_dict[backbone_nt][sub_nt][qu] += 1
						except:
							try:
								qual_sub_dict[backbone_nt][sub_nt][qu] = 1
							except:
								try:
									qual_sub_dict[backbone_nt][sub_nt] = {}
									qual_sub_dict[backbone_nt][sub_nt][qu] = 1
								except:
									qual_sub_dict[backbone_nt] = {}
									qual_sub_dict[backbone_nt][sub_nt] = {}
									qual_sub_dict[backbone_nt][sub_nt][qu] = 1
	output_tup = (input_seq_filename.split(".f")[0],qual_sub_dict,sub_per_read_dict)
	return output_tup

def read_in_barcodes(accession,barcode_filename_in,mismatch_filename_in,nonbarcode_filename_in):
	barcode_count_dict = {}
	barcode_list = []
	infile = open(barcode_filename_in)
	for line in infile:
		line = line.strip().split("\t")
		if line[0] != "barcode":
			barcode_string = line[0]
			count = int(line[1])
			barcode_count_dict[barcode_string] = count
			barcode_list.append(barcode_string)
	infile.close()
	barcode_list = list(set(barcode_list))
	mismatch_by_site = {}
	infile = open(mismatch_filename_in)
	for line in infile:
		line = line.strip().split("\t")
		if line[0] != "Site_number":
			site = int(line[0])
			prop = float(line[1])
			mismatch_by_site[site] = prop
	infile.close()

	infile = open(output_dir+"samples/"+accession+".total_read_count.txt","r")
	for line in infile:
		line = line.strip().split('\t')
		if line[0]=='total reads screened':
			seq_count = int(line[1])
		elif line[0] == 'pass barcode screen':
			pass_seq_count = int(line[1])
	infile.close()
	post_merge_readcount = 0
	infile = open(output_dir+"samples/"+accession+".read_count_postmerge.txt",'r')
	for line in infile:
		line = line.strip()
		post_merge_readcount = int(line)
	infile.close()
	raw_readcount = 0
	infile = open(output_dir+"samples/"+accession+".read_count_raw.txt",'r')
	for line in infile:
		line = line.strip()
		raw_readcount = int(line)
	infile.close()
	read_count_tup = (raw_readcount,post_merge_readcount,seq_count,pass_seq_count)
	trimed_reads = accession+'.fq'
	if os.path.isfile(trim_dir+trimed_reads) == True:
		command = 'cp '+trim_dir+trimed_reads+' '+temp_dir
		run_command(command)
		zip_gz(trimed_reads,temp_dir)
		command = 'mv '+temp_dir+trimed_reads+'.gz '+trim_dir
		out = run_command(command)
		if out == False:
			print("zip gz failed: "+trimed_reads)
			sys.exit()
		else:
			os.remove(trim_dir+trimed_reads)
	raw_reads_forward = accession+forward_read_suffix
	raw_reads_reverse = accession+reverse_read_suffix
	if os.path.isfile(input_read_dir+raw_reads_forward) == True:
		command = 'cp '+input_read_dir+raw_reads_forward+' '+temp_dir
		run_command(command)
		zip_gz(raw_reads_forward,temp_dir)
		command = 'mv '+temp_dir+raw_reads_forward+'.gz '+input_read_dir
		out = run_command(command)
		if out == False:
			print("zip gz failed: "+raw_reads_forward)
			sys.exit()
		else:
			os.remove(input_read_dir+raw_reads_forward)
	if os.path.isfile(input_read_dir+raw_reads_reverse) == True:
		command = 'cp '+input_read_dir+raw_reads_reverse+' '+temp_dir
		run_command(command)
		zip_gz(raw_reads_reverse,temp_dir)
		command = 'mv '+temp_dir+raw_reads_reverse+'.gz '+input_read_dir
		out = run_command(command)
		if out == False:
			print("zip gz failed: "+raw_reads_reverse)
			sys.exit()
		else:
			os.remove(input_read_dir+raw_reads_reverse)
	return barcode_count_dict, barcode_list,mismatch_by_site,read_count_tup


def shannon_alpha(freq_array):
	H = 0
	S_obs = 0
	for freq in freq_array:
		if freq >0.0:
			H_i = -1*freq*np.log(freq)
			H += H_i
			S_obs += 1
	if S_obs >=1:
		H_max = np.log(S_obs)
		E = H/H_max
		H = round(H,3)
		E = float(round_to_n_sig_figs(E,3))
	else:
		H,E = 0,0
	return H,E

def simpson_alpha(freq_array):
	p_sum = 0
	S = 0
	for freq in freq_array:
		if freq >0.0:
			S += 1
			p_sum += freq**2
	H = p_sum
	return float(round_to_n_sig_figs(H,3))

# def bray_curtis_dissimilarity_log(array1,array2):
# 	placeholder = 10
# 	array1 = np.array(array1)
# 	array2 = np.array(array2)
# 	array1_nozero = np.where(array1==0.,placeholder,array1)
# 	array2_nozero = np.where(array2==0.,placeholder,array2)
# 	min_array = np.where(array1_nozero<=array2_nozero,array1_nozero,array2_nozero)
# 	min_array = np.log(min_array)*-1
# 	placeholder = np.log(placeholder)*-1
# 	array1_nozero = -1*np.log(array1_nozero)
# 	array2_nozero = -1*np.log(array2_nozero)
# 	array1_nozero = np.where(array1_nozero==placeholder,0,array1_nozero)
# 	array2_nozero = np.where(array2_nozero==placeholder,0,array2_nozero)
# 	min_array = np.where(min_array==placeholder,0,min_array)
# 	S1 = 1./np.sum(np.where(array1_nozero>0.,1,0))
# 	S2 = 1./np.sum(np.where(array2_nozero>0.,1,0))
# 	C = 1./np.sum(min_array)
# 	B = 1.-((2.*C)/(S1+S2))
# 	return float(round_to_n_sig_figs(B,3))
# def bray_curtis_dissimilarity(array1,array2):
# 	placeholder = 10
# 	array1 = np.array(array1)
# 	array2 = np.array(array2)
# 	array1_nozero = np.where(array1==0.,placeholder,array1)
# 	array2_nozero = np.where(array2==0.,placeholder,array2)
# 	min_array = np.where(array1_nozero<=array2_nozero,array1_nozero,array2_nozero)
# 	min_array = np.where(min_array==placeholder,0,min_array)
# 	S1 = np.sum(np.where(array1>0.,1,0))
# 	S2 = np.sum(np.where(array2>0.,1,0))
# 	C = np.sum(min_array)
# 	B = 1.-((2.*C)/(S1+S2))
# 	return float(round_to_n_sig_figs(B,3))

def jaccard_dissimilarity(array1,array2,min_count=1.0):
	array1 = np.array(array1)
	array2 = np.array(array2)
	A = np.where(array1>=min_count,1,0)
	B = np.where(array2>=min_count,1,0)
	AB_sum = A+B
	AB = np.sum(np.where(AB_sum==2,1,0))
	C = np.sum(np.where(AB_sum>0,1,0))
	A = np.sum(A)
	B = np.sum(B)
	if C == 0:
		j = 1.
	else:
		J = 1. - (AB/C)
	return float(round_to_n_sig_figs(J,3))

def bray_curtis_dissimilarity_log(array1,array2):
	C,S1,S2 = 0.,0.,0.
	for i in range(0,len(array1)):
		if array1[i]>0:
			val1 = np.log(array1[i])*-1
		else:
			val1 = 0.
		if array2[i]>0:
			val2 = np.log(array2[i])*-1
		else:
			val2 = 0.
		if val1 > 0.0 and val2 > 0.0:
			minval = min(val1,val2)
			C += 1./min(val1,val2)
		if val1 > 0.:
			S1 += 1./val1
		if val2 > 0.:
			S2 += 1./val2
	B = 1-((2*C)/(S1+S2))
	return float(round_to_n_sig_figs(B,3))

def bray_curtis_dissimilarity(array1,array2):
	C,S1,S2 = 0.,0.,0.
	for i in range(0,len(array1)): #assumes both arrays are same size
		val1 = array1[i]
		val2 = array2[i]
		if val1 > 0.0 and val2 > 0.0:
			C += min(val1,val2)
		if val1 > 0.0:
			S1 += val1
		if val2 > 0.0:
			S2 += val2
	B = 1.-((2.*C)/(S1+S2))
	return float(round_to_n_sig_figs(B,3))

# def jaccard_dissimilarity(array1,array2,min_count=1.0):
# 	AB,A,B,C = 0,0,0,0
# 	for i in range(0,len(array1)):
# 		val1 = array1[i]
# 		val2 = array2[i]
# 		if val1 >= min_count or val2 >= min_count:
# 			C += 1
# 		if val1 >= min_count and val2 >= min_count:
# 			AB += 1
# 	if C == 0:
# 		j = 1.
# 	else:
# 		J = 1. - (AB/C)
# 	return float(round_to_n_sig_figs(J,3))

def chao_1_richness(count_array,min_count=1):
	S_obs = 0
	S_single = 0
	S_double = 0
	for count in count_array:
		if count >= min_count:
			S_obs += 1
			if count == 1:
				S_single += 1
			elif count == 2:
				S_double += 1
	S_unobs = (S_single*(S_single-1))/(2*S_double+1)
	R = S_obs + S_unobs
	return round(R,1)

def true_richness(count_array,min_count=1):
	t_rich = 0
	for i in range(0,len(count_array)):
		count = count_array[i]
		if count >= min_count:
			t_rich += 1
	return t_rich

def is_number(n):
	try:
		float(n)
	except ValueError:
		return False
	return True

def round_to_n_sig_figs(val,num_sig_figs):
	if is_number(val):
		val = float(val)
		if val == 0.0:
			return '0.0'
		elif is_number(num_sig_figs):
			if val<0:
				multiplier = -1
			else:
				multiplier = 1
			val = val*multiplier
			num_sig_figs = float(num_sig_figs)
			if num_sig_figs.is_integer:
				num_sig_figs = int(num_sig_figs)
				if num_sig_figs == 0:
					num_sig_figs = 1
				sci_val = "{:.10e}".format(val)
				split_sci_val = sci_val.split("e")
				if len(split_sci_val) == 2:
					rounded_base_number = round(float(split_sci_val[0]),num_sig_figs-1)
					exponent = int(split_sci_val[1])
					if str(rounded_base_number) == '10.0':
						val_out = str(round(float(val),num_sig_figs))
					elif exponent == 0:
						val_out = str(rounded_base_number) + ((num_sig_figs)-1)*'0'
					elif exponent < 0:
						exponent*=-1
						val_out = '0.' + (exponent-1)*'0' + str(rounded_base_number).replace(".","")
						if exponent >3:
							val_out = str(float(val_out))
						elif len(val_out) >7:
							val_out = str(float(val_out))
					elif exponent > 0:
						val_out = str(rounded_base_number) +'e'+ (str(exponent))
					else:
						sys.exit("Unexpected error while rounding: "+str(val))
					if multiplier == -1:
						val_out = '-'+val_out
					return val_out
			else:
				sys.exit("Non-integer value for 'num_sig_figs' provided: "+str(num_sig_figs))
		else:
			sys.exit("Unable to round: '"+str(val) + "' to: '"+str(num_sig_figs)+"' decimals")
	else:
		sys.exit("Unable to round: '"+str(val) + "' to: '"+str(num_sig_figs)+"' decimals")

def rsquared(x, y):
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	return r_value**2

def subsample_counts(count_array,barcodeID_array,num_to_pick):
	num_to_pick = int(num_to_pick)
	rand_list = []
	for i in range(0,len(count_array)):
		count = count_array[i]
		barcodeID = barcodeID_array[i]
		if count >0:
			for j in range(0,count):
				rand_list.append(barcodeID)
	random.shuffle(rand_list)
	subset_list = rand_list[0:num_to_pick]
	count_dict_out = {}
	freq_dict_out = {}
	for i in range(0,len(subset_list)):
		barcodeID = subset_list[i]
		try:
			count_dict_out[barcodeID] += 1.
		except:
			count_dict_out[barcodeID] = 1.
	local_barcodeID_list = list(count_dict_out.keys())
	float_num_to_pick = float(num_to_pick)
	for i in range(0,len(local_barcodeID_list)):
		barcodeID = local_barcodeID_list[i]
		count = count_dict_out[barcodeID]
		barocde_freq = count/float_num_to_pick
		freq_dict_out[barcodeID] = barocde_freq
	return count_dict_out,freq_dict_out

def bootstrap_counts(count_array,barcodeID_array,num_to_pick):
	num_to_pick = int(num_to_pick)
	rand_list = []
	for i in range(0,len(count_array)):
		count = count_array[i]
		barcodeID = barcodeID_array[i]
		if count >0:
			for j in range(0,count):
				rand_list.append(barcodeID)
	random.shuffle(rand_list)
	# subset_list = rand_list[0:num_to_pick]
	subset_list = []
	subset_num = np.random.randint(0,len(rand_list),num_to_pick)
	for n in range(0,num_to_pick):
		subset_list.append(rand_list[subset_num[n]])
	count_dict_out = {}
	freq_dict_out = {}
	for i in range(0,len(subset_list)):
		barcodeID = subset_list[i]
		try:
			count_dict_out[barcodeID] += 1.
		except:
			count_dict_out[barcodeID] = 1.
	local_barcodeID_list = list(count_dict_out.keys())
	float_num_to_pick = float(num_to_pick)
	for i in range(0,len(local_barcodeID_list)):
		barcodeID = local_barcodeID_list[i]
		count = count_dict_out[barcodeID]
		barocde_freq = count/float_num_to_pick
		freq_dict_out[barcodeID] = barocde_freq
	return count_dict_out,freq_dict_out


def subsample_alpha_func(input_tup,full_barcode_list,subsample_iterations=3):
	sampleID = input_tup[0]
	counts = input_tup[1]
	sample_total = np.sum(counts)
	# pick_set_list = [3.e3,5.e3,7.e3,1e4,1.5e4,2.e4,2.5e4,3e4,3.5e4,4.e4,5e4,6e4,7e4,8e4,9e4,1e5,2e5,4e5,8e5,1e6,2e6]
	pick_set_list = [1.00e03,1.20e03,1.44e03,1.73e03,2.07e03,2.49e03,2.99e03,3.58e03,4.30e03,5.16e03,6.19e03,7.43e03,8.92e03,1.07e04,1.28e04,1.54e04,1.85e04,2.22e04,2.66e04,3.19e04,3.83e04,4.60e04,5.52e04,6.62e04,7.95e04,9.54e04,1.14e05,1.37e05,1.65e05,1.98e05,2.37e05,2.85e05,3.42e05,4.10e05,4.92e05,5.91e05,7.09e05,8.51e05,1.02e06]
	# pick_set_list = [1.e3,2.e3,3.e3,4.e3,5.e3,6.e3,7.e3,8.e3,1e4,2.e4,3e4,4.e4,5e4,6e4,7e4,8e4,9e4,1e5,2e5,4e5,8e5,1e6,2e6]
	diversity_dict = {}
	flat_diversity_dict = {}
	for s in range(0,len(pick_set_list)):
		subsample_size = pick_set_list[s]
		if sample_total >= (subsample_size*1.2):
			diversity_dict[subsample_size] = {}
			for iter_num in range(0,subsample_iterations):
				rand_count_dict,rand_freq_dict = subsample_counts(counts,full_barcode_list,subsample_size)
				rand_count = []
				rand_freq = []
				local_barcode_list = []
				for b in range(0,len(full_barcode_list)):
					barcodeID = full_barcode_list[b]
					try:
						count = rand_count_dict[barcodeID]
					except:
						count = 0
					try:
						freq = rand_freq_dict[barcodeID]
					except:
						freq = 0
					if count >= 1:
						local_barcode_list.append(barcodeID)
						rand_count.append(count)
						rand_freq.append(freq)
				local_shannon,local_even = shannon_alpha(rand_freq)
				local_simpson = simpson_alpha(rand_freq)
				local_chao = chao_1_richness(rand_count)
				try:
					flat_diversity_dict['subsample_size'].append(subsample_size)
					flat_diversity_dict['shannon'].append(local_shannon)
					flat_diversity_dict['simpson'].append(local_simpson)
					flat_diversity_dict['even'].append(local_even)
					flat_diversity_dict['chao'].append(local_chao)
					flat_diversity_dict['rich'].append(len(local_barcode_list))
				except:
					flat_diversity_dict['subsample_size'] = [subsample_size]
					flat_diversity_dict['shannon'] = [local_shannon]
					flat_diversity_dict['simpson'] = [local_simpson]
					flat_diversity_dict['even'] = [local_even]
					flat_diversity_dict['chao'] = [local_chao]
					flat_diversity_dict['rich'] = [len(local_barcode_list)]
				try:
					diversity_dict[subsample_size]['shannon'].append(local_shannon)
					diversity_dict[subsample_size]['simpson'].append(local_simpson)
					diversity_dict[subsample_size]['even'].append(local_even)
					diversity_dict[subsample_size]['chao'].append(local_chao)
					diversity_dict[subsample_size]['rich'].append(int(len(local_barcode_list)))
				except:
					diversity_dict[subsample_size]['shannon'] = [local_shannon]
					diversity_dict[subsample_size]['simpson'] = [local_simpson]
					diversity_dict[subsample_size]['even'] = [local_even]
					diversity_dict[subsample_size]['chao'] = [local_chao]
					diversity_dict[subsample_size]['rich'] = [int(len(local_barcode_list))]
	subsample_count_list = list(diversity_dict.keys())
	subsample_count_list = sorted(subsample_count_list)
	median_diversity_dict = {}
	# subsample_val_list = []
	for s in range(0,len(subsample_count_list)):
		subsample_size = subsample_count_list[s]
		# for v in range(0,len(diversity_dict[subsample_size]['shannon'])):
	
		shannon_med = np.average(diversity_dict[subsample_size]['shannon'])
		simpson_med = np.average(diversity_dict[subsample_size]['simpson'])
		even_med = np.average(diversity_dict[subsample_size]['even'])
		chao_med = np.average(diversity_dict[subsample_size]['chao'])
		rich_med = np.average(diversity_dict[subsample_size]['rich'])
		try:
			median_diversity_dict['shannon'].append(shannon_med)
			median_diversity_dict['simpson'].append(simpson_med)
			median_diversity_dict['even'].append(even_med)
			median_diversity_dict['chao'].append(chao_med)
			median_diversity_dict['rich'].append(rich_med)
		except:
			median_diversity_dict['shannon'] = [shannon_med]
			median_diversity_dict['simpson'] = [simpson_med]
			median_diversity_dict['even'] = [even_med]
			median_diversity_dict['chao'] = [chao_med]
			median_diversity_dict['rich'] = [rich_med]
	# try:
	# 	subsample_count_list = list(set(diversity_dict['subsample_size']))
	# 	subsample_count_list = sorted(subsample_count_list)
	# except:
	# 	subsample_count_list = []
	fit_div_dict = {}
	fit_div_summary_dict = {}
	if len(subsample_count_list) >= 3:
		fit_div_dict['shannon'],fit_div_summary_dict['shannon'] = fit_alpha_div(median_diversity_dict['shannon'],subsample_count_list)
		fit_div_dict['simpson'],fit_div_summary_dict['simpson'] = fit_alpha_div(median_diversity_dict['simpson'],subsample_count_list)
		fit_div_dict['even'],fit_div_summary_dict['even'] = fit_alpha_div(median_diversity_dict['even'],subsample_count_list)
		fit_div_dict['chao'],fit_div_summary_dict['chao'] = fit_alpha_richness(median_diversity_dict['chao'],subsample_count_list)
		fit_div_dict['rich'],fit_div_summary_dict['rich'] = fit_alpha_richness(median_diversity_dict['rich'],subsample_count_list)
		# fit_div_dict['shannon'] = fit_alpha_div(median_diversity_dict['shannon'],subsample_count_list)
		# fit_div_dict['simpson'] = fit_alpha_div(median_diversity_dict['simpson'],subsample_count_list)
		# fit_div_dict['even'] = fit_alpha_div(median_diversity_dict['even'],subsample_count_list)
		# fit_div_dict['chao'] = fit_alpha_richness(median_diversity_dict['chao'],subsample_count_list)
		# fit_div_dict['rich'] = fit_alpha_richness(median_diversity_dict['rich'],subsample_count_list)
	out_tup = (sampleID,flat_diversity_dict,fit_div_dict,fit_div_summary_dict)
	return out_tup

def fit_alpha_div(alpha_vals,sub_num_list,subsample_fit_level=1e4):
	init_vals = (1., 1.,1.)
	x_real = np.array(sub_num_list)
	y_real = np.array(alpha_vals)
	def alpha_div_func(x, a, b, c):
		output = a * x**b + c
		return output
	def alpha_loss(theta):
		a, b, c = theta
		y_pred = alpha_div_func(x_real, a, b,c)
		loss_val = np.sum(np.subtract(y_real,y_pred)**2)
		# loss_sub = np.subtract(y_real,y_pred)
		# loss_sq = loss_sub**2
		# loss_val = np.nansum(loss_sq)
		return loss_val
	res = minimize(alpha_loss, init_vals)
	loss_val = alpha_loss(res.x)
	alpha_infer = max(1e-3,alpha_div_func(int(subsample_fit_level),res.x[0], res.x[1], res.x[2]))
	fit_summary_string = str(alpha_infer)+'\t'+str(loss_val)+'\t'+str(len(sub_num_list)) +'\t'+str(res.x[0])+'\t'+str(res.x[1])+'\t'+str(res.x[2])
	return alpha_infer,fit_summary_string

def fit_beta_div(beta_vals,sub_num_list,subsample_fit_level=1e4):
	init_vals = (1., 1.,0.)
	# init_vals = (0.5, 4.,1e-8)
	x_real = np.array(sub_num_list)
	y_real = np.array(beta_vals)
	def beta_div_func(x, a, b, c):
		output = a * x**b + c
		# output = ((np.abs(a) * x) / (np.abs(b) + x)) + x*np.abs(c)
		# output = a * x**2 + b*x + c
		return output
	def beta_loss(theta):
		a, b, c = theta
		y_pred = beta_div_func(x_real, a, b, c)
		scalar = np.log10(x_real)
		scalar = scalar/min(scalar)
		loss_val = np.sum((np.subtract(y_real,y_pred)**2)*scalar)
		return loss_val
	res = minimize(beta_loss, init_vals)#,method='Powell')
	loss_val = beta_loss(res.x)
	alpha_infer = max(1e-3,beta_div_func(int(subsample_fit_level),res.x[0], res.x[1], res.x[2]))
	fit_summary_string = 'power'+'\t'+str(alpha_infer)+'\t'+str(loss_val)+'\t'+str(len(sub_num_list)) +'\t'+str(res.x[0])+'\t'+str(res.x[1])+'\t'+str(res.x[2])
	return alpha_infer,fit_summary_string,loss_val

def fit_beta_div_poly(beta_vals,sub_num_list,subsample_fit_level=1e4):
	# init_vals = (1e-8, 1e-4,0.)
	init_vals = (1., 1.,0.)
	x_real = np.array(sub_num_list)
	y_real = np.array(beta_vals)
	def beta_div_func(x, a, b, c):
		# output = a * x**2 + b*x + c
		output = a/(x**b)+c
		# output = a * x**2 + b*x + c
		return output
	def beta_loss(theta):
		a, b, c = theta
		y_pred = beta_div_func(x_real, a, b, c)
		scalar = np.log10(x_real)
		scalar = scalar/min(scalar)
		loss_val = np.sum((np.subtract(y_real,y_pred)**2)*scalar)
		return loss_val
	res = minimize(beta_loss, init_vals)#,method='Powell')
	loss_val = beta_loss(res.x)
	alpha_infer = max(1e-3,beta_div_func(int(subsample_fit_level),res.x[0], res.x[1], res.x[2]))
	fit_summary_string = 'poly'+'\t'+str(alpha_infer)+'\t'+str(loss_val)+'\t'+str(len(sub_num_list)) +'\t'+str(res.x[0])+'\t'+str(res.x[1])+'\t'+str(res.x[2])
	return alpha_infer,fit_summary_string,loss_val

def fit_alpha_richness(alpha_vals,sub_num_list,subsample_fit_level=1e4):
	x_real = np.array(sub_num_list)
	y_real = np.array(alpha_vals)
	max_y_real = max(y_real)
	init_vals = (max_y_real*0.5, max_y_real*4.,max_y_real*1e-8)
	def alpha_rich_func(x, a, b, c): ### Michaelis Menten
		output = ((np.abs(a) * x) / (np.abs(b) + x)) + x*np.abs(c)
		return output
	def richness_loss(theta):
		a, b, c = theta
		y_pred = alpha_rich_func(x_real, a, b, c)
		loss_val = np.nansum(np.subtract(y_real,y_pred)**2)
		return loss_val
	res = minimize(richness_loss, init_vals)#,method='BFGS')
	loss_val = richness_loss(res.x)
	alpha_infer = max(1e-3,alpha_rich_func(int(subsample_fit_level),res.x[0], res.x[1], res.x[2]))
	fit_summary_string = str(alpha_infer)+'\t'+str(loss_val) +'\t'+str(len(sub_num_list)) +'\t'+str(res.x[0])+'\t'+str(res.x[1])+'\t'+str(res.x[2])
	return alpha_infer,fit_summary_string


# def fit_alpha(alpha_vals,sub_num_list,poly_num=2,subsample_fit_level=1e4):
# 	sub_num_list = np.log(sub_num_list)
# 	poly_coeff = np.polyfit(sub_num_list,alpha_vals,2)
# 	poly_func = np.poly1d(poly_coeff)
# 	subsample_fit = poly_func(np.log(subsample_fit_level))
# 	return subsample_fit

def subsample_beta_func(input_tup,full_barcode_list,subsample_iterations=3,subsample_fit_depth=5e4):
	sample_tup_1 = input_tup[0]
	sample_tup_2 = input_tup[1]
	sample_1,counts_1 = sample_tup_1[0],sample_tup_1[1]
	sample_2,counts_2 = sample_tup_2[0],sample_tup_2[1]
	sample_total_1 = np.sum(counts_1)
	sample_total_2 = np.sum(counts_2)
	
	max_sample = min(int(subsample_fit_depth/2),max(4000,int(min(sample_total_1,sample_total_2)*0.8)))
	min_sample = max(2500,int(max_sample*0.25))
	max_magnitude = int(str(max_sample)[0])
	temp_pick_set = np.linspace(min_sample,max_sample,max(6,max_magnitude*2))
	# temp_pick_set = [2.00e3,2.30e3,2.65e3,3.04e3,3.50e3,4.02e3,4.63e3,5.32e3,6.12e3,7.04e3,8.09e3,9.30e3,1.07e4,1.23e4,1.42e4,1.63e4,1.87e4,2.15e4,2.48e4,2.85e4,3.27e4,3.76e4,4.33e4,4.98e4,5.73e4,6.58e4,7.57e4,8.71e4,1.00e5,1.15e5,1.32e5,1.52e5,1.75e5,2.01e5,2.32e5,2.66e5,3.06e5,3.52e5,4.05e5,4.66e5,5.36e5,6.16e5,7.08e5,8.15e5,9.37e5,1.08e6]
	pick_set_list = []
	for i in range(0,len(temp_pick_set)):
		val=int(temp_pick_set[i])
		# val=int(round((temp_pick_set[i])/10,0)*10)
		pick_set_list.append(val)
	# max_sample = round(np.log10(min(sample_total_1,sample_total_2)*0.8),2)
	# min_sample = 3.3#max(3.3,max_sample-3)
	# max_magnitude = int(str(max_sample)[0])
	# temp_pick_set = np.linspace(min_sample,max_sample,min(6,max_magnitude+3))
	# pick_set_list = []
	# for i in range(0,len(temp_pick_set)):
	# 	val=int(round((10**temp_pick_set[i])/10,0)*10)
	# 	pick_set_list.append(val)
	
	# pick_set_list = [5e2,1e3,2e3,4e3,8e3,1.6e4,3.2e4,6.4e4,1.28e5]
	# pick_set_list = [4e3,6e3,8e3,1e4,2e4,3e4,4e4,5e4,6e4,8e4]
	# pick_set_list = [2e3,2.5e3,3e3,4e3,5e3,6e3,7e3,8e3,1e4,2e4,4e4]
	count1_array = np.array(counts_1)
	count2_array = np.array(counts_2)
	freq_1 = list(count1_array/np.sum(count1_array))
	freq_2 = list(count2_array/np.sum(count2_array))
	del count1_array
	del count2_array
	full_bray_dist = bray_curtis_dissimilarity(freq_1,freq_2)
	full_log_bray_dist = bray_curtis_dissimilarity_log(freq_1,freq_2)
	full_jaccard_dist = jaccard_dissimilarity(counts_1,counts_2)

	dissimilarty_dict = {}
	flat_dissimilarty_dict = {}
	for s in range(0,len(pick_set_list)):
		subsample_size = pick_set_list[s]
		if sample_total_1 >= (subsample_size*1.2) and sample_total_2 >= (subsample_size*1.2):
			dissimilarty_dict[subsample_size] = {}
			for iter_num in range(0,subsample_iterations):
				rand_count_dict_1,rand_freq_dict_1 = subsample_counts(counts_1,full_barcode_list,subsample_size)
				rand_count_dict_2,rand_freq_dict_2 = subsample_counts(counts_2,full_barcode_list,subsample_size)
				# rand_count_dict_1,rand_freq_dict_1 = subsample_topN_counts(counts_1,full_barcode_list,subsample_size)
				# rand_count_dict_2,rand_freq_dict_2 = subsample_topN_counts(counts_2,full_barcode_list,subsample_size)
				rand_count_1 = []
				rand_count_2 = []
				rand_freq_1 = []
				rand_freq_2 = []
				local_barcode_list = []
				for b in range(0,len(full_barcode_list)):
					barcodeID = full_barcode_list[b]
					try:
						count1 = rand_count_dict_1[barcodeID]
					except:
						count1 = 0
					try:
						count2 = rand_count_dict_2[barcodeID]
					except:
						count2 = 0
					try:
						freq1 = rand_freq_dict_1[barcodeID]
					except:
						freq1 = 0.
					try:
						freq2 = rand_freq_dict_2[barcodeID]
					except:
						freq2 = 0.
					if count1 >0 or count2 >0:
						local_barcode_list.append(barcodeID)
						rand_count_1.append(count1)
						rand_count_2.append(count2)
						rand_freq_1.append(freq1)
						rand_freq_2.append(freq2)
				local_bray_dist = bray_curtis_dissimilarity(rand_freq_1,rand_freq_2)
				local_log_bray_dist = bray_curtis_dissimilarity_log(rand_freq_1,rand_freq_2)
				# print(sample_1+' '+sample_2+' '+str(subsample_size)+' '+str(s)+' '+str(local_bray_dist))
				local_jaccard_dist = jaccard_dissimilarity(rand_count_1,rand_count_2)
				try:
					flat_dissimilarty_dict['subsample_size'].append(subsample_size)
					flat_dissimilarty_dict['bray'].append(local_bray_dist)
					flat_dissimilarty_dict['bray_log'].append(local_log_bray_dist)
					flat_dissimilarty_dict['jaccard'].append(local_jaccard_dist)
				except:
					flat_dissimilarty_dict['subsample_size'] = [subsample_size]
					flat_dissimilarty_dict['bray'] = [local_bray_dist]
					flat_dissimilarty_dict['bray_log'] = [local_log_bray_dist]
					flat_dissimilarty_dict['jaccard'] = [local_jaccard_dist]
				try:
					dissimilarty_dict
					dissimilarty_dict[subsample_size]['bray'].append(local_bray_dist)
					dissimilarty_dict[subsample_size]['bray_log'].append(local_log_bray_dist)
					dissimilarty_dict[subsample_size]['jaccard'].append(local_jaccard_dist)
				except:
					dissimilarty_dict[subsample_size]['bray'] = [local_bray_dist]
					dissimilarty_dict[subsample_size]['bray_log'] = [local_log_bray_dist]
					dissimilarty_dict[subsample_size]['jaccard'] = [local_jaccard_dist]
	try:
		subsample_count_list = list(set((flat_dissimilarty_dict['subsample_size'])))
		subsample_count_list = sorted(subsample_count_list)
	except:
		subsample_count_list = []
	median_dissimilarty_dict = {}
	for s in range(0,len(subsample_count_list)):
		subsample_size = subsample_count_list[s]
		bray_med = np.average(dissimilarty_dict[subsample_size]['bray'])
		bray_log_med = np.average(dissimilarty_dict[subsample_size]['bray_log'])
		jaccard_med = np.average(dissimilarty_dict[subsample_size]['jaccard'])
		try:
			median_dissimilarty_dict['bray'].append(bray_med)
			median_dissimilarty_dict['bray_log'].append(bray_log_med)
			median_dissimilarty_dict['jaccard'].append(jaccard_med)
		except:
			median_dissimilarty_dict['bray'] = [bray_med]
			median_dissimilarty_dict['bray_log'] = [bray_log_med]
			median_dissimilarty_dict['jaccard'] = [jaccard_med]
	fit_diss_dict = {}
	fit_div_summary_dict = {}
	# try:
	# 	num_med = len(median_dissimilarty_dict['bray'])
	# except:
	# 	num_med = 0
	if len(subsample_count_list) >=3:
		# local_subsample_count_list = dissimilarty_dict['subsample_size']#[max(0,len(median_dissimilarty_dict['bray'])-8):len(median_dissimilarty_dict['bray'])]
		# bray_list = dissimilarty_dict['bray']#[max(0,len(median_dissimilarty_dict['bray'])-8):len(median_dissimilarty_dict['bray'])]
		# bray_log_list = dissimilarty_dict['bray_log']#[max(0,len(median_dissimilarty_dict['bray'])-8):len(median_dissimilarty_dict['bray'])]
		# jaccard_list = dissimilarty_dict['jaccard']#[max(0,len(median_dissimilarty_dict['jaccard'])-8):len(median_dissimilarty_dict['jaccard'])]
		# jaccard_count_list = subsample_count_list
		# fit_diss_dict['bray'],fit_div_summary_dict['bray'] = fit_alpha_div(dissimilarty_dict['bray'],dissimilarty_dict['subsample_size'])
		# fit_diss_dict['bray_log'],fit_div_summary_dict['bray_log'] = fit_alpha_div(dissimilarty_dict['bray_log'],dissimilarty_dict['subsample_size'])
		# fit_diss_dict['jaccard'],fit_div_summary_dict['jaccard'] = fit_alpha_div(dissimilarty_dict['jaccard'],dissimilarty_dict['subsample_size'])
		bray_fit,bray_stat,bray_loss = fit_beta_div(median_dissimilarty_dict['bray'],subsample_count_list,subsample_fit_depth)
		bray_log_fit,bray_log_stat,bray_log_loss = fit_beta_div(median_dissimilarty_dict['bray_log'],subsample_count_list,subsample_fit_depth)
		jaccard_fit,jaccard_stat,jaccard_loss = fit_beta_div(median_dissimilarty_dict['jaccard'],subsample_count_list,subsample_fit_depth)

		bray_fit_poly,bray_stat_poly,bray_loss_poly = fit_beta_div_poly(median_dissimilarty_dict['bray'],subsample_count_list,subsample_fit_depth)
		bray_log_fit_poly,bray_log_stat_poly,bray_log_loss_poly = fit_beta_div_poly(median_dissimilarty_dict['bray_log'],subsample_count_list,subsample_fit_depth)
		jaccard_fit_poly,jaccard_stat_poly,jaccard_loss_poly = fit_beta_div_poly(median_dissimilarty_dict['jaccard'],subsample_count_list,subsample_fit_depth)

		if bray_loss_poly <= bray_loss*0.2 and bray_fit_poly>0 and bray_fit_poly<1. and bray_fit_poly <= (full_bray_dist*1.1):
			min_bray_fit = bray_fit_poly
			min_bray_fit_stat = bray_stat_poly+'\t'+bray_stat
		elif bray_fit>0 and bray_fit<1. and bray_fit <= (full_bray_dist*1.1):
			min_bray_fit = bray_fit
			min_bray_fit_stat = bray_stat+'\t'+bray_stat_poly
		else:
			min_bray_fit = full_bray_dist
			min_bray_fit_stat = 'direct\t'+str(full_bray_dist)+'\tna\tna\tna\tna\tna'

		if bray_log_loss_poly <= bray_log_loss*0.2 and bray_log_fit_poly>0 and bray_log_fit_poly<1. and bray_log_fit_poly <= (full_log_bray_dist*1.1):
			min_bray_log_fit = bray_log_fit_poly
			min_bray_log_fit_stat = bray_log_stat_poly+'\t'+bray_log_stat
		elif bray_log_fit>0 and bray_log_fit<1. and bray_log_fit <= (full_log_bray_dist*1.1):
			min_bray_log_fit = bray_log_fit
			min_bray_log_fit_stat = bray_log_stat+'\t'+bray_log_stat_poly
		else:
			min_bray_log_fit = full_log_bray_dist
			min_bray_log_fit_stat = 'direct\t'+str(full_log_bray_dist)+'\tna\tna\tna\tna\tna'

		if jaccard_loss_poly <= jaccard_loss*0.2 and jaccard_fit_poly>0 and jaccard_fit_poly<1. and jaccard_fit_poly <= (full_jaccard_dist*1.1):
			min_jaccard_fit = jaccard_fit_poly
			min_jaccard_fit_stat = jaccard_stat_poly+'\t'+jaccard_stat
		elif jaccard_fit>0 and jaccard_fit<1. and jaccard_fit <= (full_jaccard_dist*1.1):
			min_jaccard_fit = jaccard_fit
			min_jaccard_fit_stat = jaccard_stat+'\t'+jaccard_stat_poly
		else:
			min_jaccard_fit = full_jaccard_dist
			min_jaccard_fit_stat = 'direct\t'+str(full_jaccard_dist)+'\tna\tna\tna\tna\tna'

		fit_diss_dict['bray'],fit_div_summary_dict['bray'] = min_bray_fit,min_bray_fit_stat
		fit_diss_dict['bray_log'],fit_div_summary_dict['bray_log'] = min_bray_log_fit,min_bray_log_fit_stat
		fit_diss_dict['jaccard'],fit_div_summary_dict['jaccard'] = min_jaccard_fit,min_jaccard_fit_stat

		# fit_diss_dict['bray'],fit_div_summary_dict['bray'] = fit_beta_div(median_dissimilarty_dict['bray'],subsample_count_list,subsample_fit_depth)
		# fit_diss_dict['bray_log'],fit_div_summary_dict['bray_log'] = fit_beta_div(median_dissimilarty_dict['bray_log'],subsample_count_list,subsample_fit_depth)
		# fit_diss_dict['jaccard'],fit_div_summary_dict['jaccard'] = fit_beta_div(median_dissimilarty_dict['jaccard'],subsample_count_list,subsample_fit_depth)


		# fit_diss_dict['bray'] = float(round_to_n_sig_figs(fit_alpha(bray_list,bray_count_list,1,subsample_fit_level),4))
		# fit_diss_dict['bray_log'] = float(round_to_n_sig_figs(fit_alpha(bray_log_list,bray_count_list,1,subsample_fit_level),4))
		# fit_diss_dict['jaccard'] = float(round_to_n_sig_figs(fit_alpha(jaccard_list,jaccard_count_list,1,subsample_fit_level),4))
	else:
		fit_diss_dict['bray'],fit_div_summary_dict['bray'] = full_bray_dist,'direct\t'+str(full_bray_dist)+'\tna\tna\tna\tna\tna'
		fit_diss_dict['bray_log'],fit_div_summary_dict['bray_log'] = full_log_bray_dist,'direct\t'+str(full_log_bray_dist)+'\tna\tna\tna\tna\tna'
		fit_diss_dict['jaccard'],fit_div_summary_dict['jaccard'] = full_jaccard_dist,'direct\t'+str(full_jaccard_dist)+'\tna\tna\tna\tna\tna'
	out_tup = (sample_1,sample_2,flat_dissimilarty_dict,fit_diss_dict,fit_div_summary_dict)
	return out_tup


def fit_qual_model(prop_sub_infile_name,count_sub_infile_name,fit_prop_sub_outfile_name,min_qual_val_to_fit=20,poly_num=1):
	fit_sub_dict = {}
	min_qual_obs = 999
	max_qual_obs = -1
	num_col = 2
	num_row = 2
	col = -1 #column
	row = -1 #row
	# fig_num = 0
	focal_nt_obs_dict = {}
	nt_count_dict = {}
	count_sub_infile = open(count_sub_infile_name,"r")
	first_line = True
	for line in count_sub_infile:
		line = line.rstrip('\n').split("\t")
		if first_line == True:
			qual_list = line
			first_line = False
		else:
			base_nt = line[0]
			sub_nt = line[1]
			for a in range(2,len(line)):
				qual_val = int(qual_list[a])
				count_val = int(line[a])
				try:
					focal_nt_obs_dict[base_nt+str(qual_val)] += count_val
				except:
					focal_nt_obs_dict[base_nt+str(qual_val)] = count_val

				try:
					nt_count_dict[base_nt+sub_nt+str(qual_val)] += count_val
				except:
					nt_count_dict[base_nt+sub_nt+str(qual_val)] = count_val
	count_sub_infile.close()

	bases = ['A','T','C','G']
	qual_val_list = {}

	for p in range(0,len(bases)):
		focal_nt = bases[p]
		col+=1
		if col%num_col==0:
			row+=1
			col = 0
		for h in range(0,len(bases)):
			focal_sub_nt = bases[h]
			prop_sub_infile = open(prop_sub_infile_name,"r")
			first_line = True
			for line in prop_sub_infile:
				line = line.rstrip('\n').split("\t")
				if first_line == True:
					qual_list = line
					first_line = False
				else:
					base_nt = line[0]
					sub_nt = line[1]
					if base_nt== focal_nt and sub_nt == focal_sub_nt:
						for a in range(2,len(line)):
							qual_val = int(qual_list[a])
							prop_val = float(round_to_n_sig_figs(float(line[a]),4))
							focal_nt_count = focal_nt_obs_dict[base_nt+str(qual_val)]
							sub_nt_count = nt_count_dict[base_nt+sub_nt+str(qual_val)]
							if prop_val > 0. and qual_val >= min_qual_val_to_fit and focal_nt_count >= 1e4 and sub_nt_count>=10:
								fit_sub_dict[focal_nt+focal_sub_nt+'-'+str(qual_val)] = prop_val
								qual_val_list[qual_val] = ''
			prop_sub_infile.close()
	qual_val_list = list(qual_val_list.keys())
	qual_val_list = sorted(qual_val_list)

	for qual_val in range(min_qual_val_to_fit,max(41,max(qual_val_list)+1)):
		for p in range(0,len(bases)):
			focal_nt = bases[p]
			for h in range(0,len(bases)):
				focal_sub_nt = bases[h]
				try:
					prop = fit_sub_dict[focal_nt+focal_sub_nt+'-'+str(qual_val)]
				except:
					fit_sub_dict[focal_nt+focal_sub_nt+'-'+str(qual_val)] = 0.0
			# prop_total = 0.
				# prop_total += prop
			# if prop_total > 0.:
			# 	remainder = 1.0 - prop_total
			# 	fit_sub_dict[focal_nt+focal_nt+'-'+str(qual_val)] = remainder
		# qual_val = qual_val_list[q]
	outlines = '\t'
	for qual_val in range(min_qual_val_to_fit,max(41,max(qual_val_list)+1)):
		outlines += '\t'+str(qual_val)
	outlines += '\n'
	for p in range(0,len(bases)):
		focal_nt = bases[p]
		prop_total = 0.
		for h in range(0,len(bases)):
			focal_sub_nt = bases[h]
			outlines += focal_nt+'\t'+focal_sub_nt
			for qual_val in range(min_qual_val_to_fit,max(41,max(qual_val_list)+1)):
			# for q in range(0,len(qual_val_list)):
			# 	qual_val = qual_val_list[q]
				try:
					prop = fit_sub_dict[focal_nt+focal_sub_nt+'-'+str(qual_val)]
				except:
					prop = 0.0
				outlines += '\t'+str(prop)
			outlines += '\n'
	fit_sub_outfile = open(fit_prop_sub_outfile_name,'w')
	fit_sub_outfile.write(outlines)
	fit_sub_outfile.close()
	return fit_sub_dict


def load_fit_qual(prop_infile_name):
	sub_dict = {}
	sub_infile = open(prop_infile_name,"r")
	first_line = True
	for line in sub_infile:
		line = line.rstrip('\n').split("\t")
		if first_line == True:
			qual_list = line
			first_line = False
		else:
			base_nt = line[0]
			sub_nt = line[1]
			for a in range(2,len(line)):
				qual_val = int(qual_list[a])
				prop_val = float(line[a])
				sub_dict[base_nt+sub_nt+'-'+str(qual_val)] = prop_val
	sub_infile.close()
	return sub_dict


def barcode_from_fullseq(barcode_dict,seq_in):
	barcode = ''
	site_list = []
	for barcode_site in barcode_dict:
		site_list.append(barcode_site)
	site_list = sorted(site_list)
	for s in range(0,len(site_list)):
		barcode_site = site_list[s]
		barcode += seq_in[barcode_site]
	return barcode

def find_all_possible_barcode_combinations(barcode_sites,barcode_site_class_dict,site_categorical_allele_dict={},category_name_dict={}):
	full_barcode_list = []
	class_allele_dict = {}
	cat_barcode_list_dict = {}
	reverse_class_allele_dict = {}
	diversity_class_sites = []
	categorical_class_sites = []
	categorical_class_nums = []
	barcode_site_list = sorted(list(set(list(barcode_sites.keys()))))
	
	for b in range(0,len(barcode_site_list)):
		site = barcode_site_list[b]
		site_class = barcode_site_class_dict[site]
		if site_class == 'DS':
			diversity_class_sites.append(site)
		elif site_class == 'CS':
			categorical_class_sites.append(site)
			categorical_class_nums.append(b)
	for b in range(0,len(barcode_site_list)):
		site = barcode_site_list[b]
		site_class = barcode_site_class_dict[site]
		full_barcode_list = list(set(full_barcode_list))
		nt_list = barcode_sites[site]
		building_barcodes = []
		if full_barcode_list == []:
			for num in range(0,len(nt_list)): 
				building_barcodes.append(nt_list[num])
		else:
			for bar_num in range(0,len(full_barcode_list)):
				for nt_num in range(0,len(nt_list)):
					growing_nt_string = full_barcode_list[bar_num]+nt_list[nt_num]
					building_barcodes.append(growing_nt_string)
		full_barcode_list = building_barcodes
	full_barcode_list = sorted(list(set(full_barcode_list)))
	# print(category_name_dict)
	if len(categorical_class_sites)>0:
		category_name_list = sorted(list(category_name_dict.keys()))
		# print(category_name_list)
		for c in range(0,len(category_name_list)):
			cat = category_name_dict[c]
			# print(str(c)+' '+cat)
			cat_allele_type = ''
			for b in range(0,len(categorical_class_sites)):
				site = categorical_class_sites[b]
				cat_nt = site_categorical_allele_dict[site][c]
				cat_allele_type += cat_nt
			class_allele_dict[c] = cat_allele_type
			reverse_class_allele_dict[cat_allele_type] = cat
		# print(reverse_class_allele_dict)
		temp_barcode_list = sorted(list(set(full_barcode_list)))
		full_barcode_list = []
		for i in range(0,len(temp_barcode_list)):
			barcodeID = temp_barcode_list[i]
			cat_allele_type = ''
			for s in range(0,len(categorical_class_sites)):
				num = categorical_class_nums[s]
				cat_allele_type += barcodeID[num]

			try:
				cat = reverse_class_allele_dict[cat_allele_type]
			except:
				cat = 'null'
			if cat != 'null':
				try:
					cat_name = category_name_dict[cat]
				except:
					cat_name = cat
				full_barcode_list.append(barcodeID)
				try:
					cat_barcode_list_dict[cat_name].append(barcodeID)
				except:
					cat_barcode_list_dict[cat_name] = [barcodeID]
		return full_barcode_list,cat_barcode_list_dict
	else:
		return full_barcode_list,cat_barcode_list_dict

def screen_barcode_site_qual(seq_in,qual_in,seq_expect,barcode_dict,F_edge_ignore,R_edge_ignore):
	barcode_mismatches = 0
	backbone_mismatches = 0
	# high_qual_mismatches = 0
	mismatch_sites = []
	expected_barcode_nt_list = []
	barcode_string = ''
	qual_list = []
	for site in range(0+F_edge_ignore,len(seq_expect)-R_edge_ignore):
		try:
			expected_barcode_nt_list = barcode_dict[site]
			barcode_site = True
		except:
			expected_barcode_nt_list = []
			barcode_site = False
		site_nt = seq_in[site]
		site_qual = ord(qual_in[site])-33
		if barcode_site == False:
			if site_nt != seq_expect[site]:
				backbone_mismatches += 1
				mismatch_sites.append(site)
		elif barcode_site == True:
			barcode_string += site_nt
			qual_list.append(site_qual)
			if site_nt != expected_barcode_nt_list[0] and site_nt != expected_barcode_nt_list[1]:
				barcode_mismatches += 1
				mismatch_sites.append(site)
	qual_list = tuple(qual_list)
	out_tup = (barcode_string,qual_list)
	# total_mismatches = barcode_mismatches+backbone_mismatches
	mismatch_sites = list(set(mismatch_sites))
	return out_tup,barcode_mismatches,backbone_mismatches,mismatch_sites#,high_qual_mismatches

# def read_in_barcode_qual_data(filepath,minQ_val=15):
# 	output_list = []
# 	infile = open(filepath)
# 	for line in infile:
# 		line = line.strip().split('\t')
# 		B = line[0]
# 		Qlist = line[1].split(',')
# 		Q = []
# 		for q in range(0,len(Qlist)):
# 			Q.append(int(Qlist[q]))
# 		Q = tuple(Q)
# 		min_Q = min(Q)
# 		max_Q = max(Q)
# 		Qsum = np.sum(Q)
# 		C = int(line[2])
# 		if min_Q >= minQ_val:
# 			out_tup = (C,min_Q,Qsum,B,Q)
# 			output_list.append(out_tup)
# 	return output_list

def cal_hamming_dist(seq1,seq2):
	mismatch_count = 0
	if len(seq1)!=len(seq2):
		sys.exit("Cannot calculate hamming distance between strings with different lengths: "+seq1+' '+seq2)
	for i in range(0,len(seq1)):
		if seq1[i] != seq2[i]:
			mismatch_count+=1
	return mismatch_count

def abundance_pval(intup_i, intup_j): #i is the focal barcode, j is the higher abundance barcode
	barcode_i, qual_tup_i, count_i = intup_i[3],intup_i[4],intup_i[0]
	barcode_j, count_j = intup_j[0],intup_j[1]
	count_i = count_i
	p_list = []
	for l in range(0,len(barcode_i)):
		nt_i = barcode_i[l]
		nt_j = barcode_j[l]
		q_i = qual_tup_i[l]
		p = sub_prob_dict[nt_j+nt_i+'-'+str(q_i)]
		p_list.append(p)
	y_ij = np.prod(p_list)
	mu_val = y_ij*count_j
	# val1 = scipy.stats.poisson.pmf(k=0,mu=mu_val)
	# val2 = scipy.stats.poisson.cdf(k=count_i,mu=mu_val)
	# if val1 == 1.0:
	# 	pA = 0.0
	# else:
	# 	pA = (1.0/(1.0-val1)) * (1.0-val2)
	val1 = scipy.stats.poisson.pmf(k=0,mu=mu_val)
	val2 = scipy.stats.poisson.sf(count_i-1,mu=mu_val)
	if val1 == 1.0:
		pA = 0.0
	else:
		pA = val2 / (1.0-val1)
	if str(pA) == 'nan':
		pA = 1.0
	# else:
	# 	pA = float(round_to_n_sig_figs(pA,4))
	return pA

def divisive_partition(input_barcode_list, full_barcode_dict, accession,pval_thresh=1e-40,max_hamming_dist=2):
	new_seed_counter = 0
	ranked_barcode_list = sorted(input_barcode_list,reverse=True)
	partition_dict = {}
	partition_count_dict = {}
	consolodated_dict = {}
	invalid_barcode_list = {}
	debug_lines = ''
	first_seed_barcode = ''
	active_ranked_barcode_list = []
	debug_outlines = ''

	#initiate first seed barcode
	for i in range(0,len(ranked_barcode_list)):
		query_tup = ranked_barcode_list[i]
		query_C = query_tup[0]
		query_Qsum = query_tup[2]
		query_B = query_tup[3]
		query_Q = query_tup[4]
		valid_barcodeID = False
		try:
			full_barcode_dict[query_B]
			valid_barcodeID = True
		except:
			pass
		if valid_barcodeID == True:
			if partition_dict == {} and query_C > 1:
				partition_dict[query_B] = [query_tup]
				partition_count_dict[query_B] = query_C
				consolodated_dict[query_tup] = 'first-seed'
				first_seed_barcode = query_B
				new_seed_counter += 1
			else:
				if query_B == first_seed_barcode:
					partition_dict[query_B].append(query_tup)
					partition_count_dict[query_B] += query_C
					consolodated_dict[query_tup] = 'first-seed-collapse'
				elif query_C >= 1:
					active_ranked_barcode_list.append(query_tup)
			try:
				status_val = consolodated_dict[query_tup]
			except:
				status_val = 'na'
			if status_val == 'first-seed':
				debug_outlines += query_B+'\t'+str(query_C)+'\t'+status_val+'\n'

	active_ranked_barcode_list = sorted(active_ranked_barcode_list,reverse=True)

	#Find subsequent valid barcodes and initiate seeds
	hamming_dist_dict = {}
	abundance_pval_dict = {}
	if len(active_ranked_barcode_list)==0:
		return partition_count_dict
	still_iterating = True
	iteration_count = 0
	while still_iterating == True:
		iteration_count +=1
		seed_barcode_list = list(partition_dict.keys())
		pA_rank_list = []
		new_seed_B = ''
		if len(active_ranked_barcode_list) >0:
			minimum_pA_threshold = pval_thresh/(float(len(active_ranked_barcode_list)))
			for i in range(0,len(active_ranked_barcode_list)):
				query_tup = active_ranked_barcode_list[i]
				query_C = query_tup[0]
				query_Qsum = query_tup[2]
				query_B = query_tup[3]
				query_Q = query_tup[4]
				if query_C > 1:
					query_pA_list = []
					for b in range(0,len(seed_barcode_list)):
						seed_B = seed_barcode_list[b]
						seed_C = partition_count_dict[seed_B]
						seed_tup = (seed_B,seed_C)
						if seed_B != query_B:
							try:
								hdist = hamming_dist_dict[query_B][seed_B]
							except:
								hdist = cal_hamming_dist(query_B,seed_B)
								try:
									hamming_dist_dict[query_B][seed_B] = hdist
								except:
									hamming_dist_dict[query_B] = {}
									hamming_dist_dict[query_B][seed_B] = hdist
							if seed_C > query_C and hdist <= max_hamming_dist:
								try:
									pA = abundance_pval_dict[query_tup][seed_tup]
								except:
									pA = abundance_pval(query_tup, seed_tup)
									try:
										abundance_pval_dict[query_tup][seed_tup] = pA
									except:
										abundance_pval_dict[query_tup] = {}
										abundance_pval_dict[query_tup][seed_tup] = pA
							else:
								pA = 0.0
							dist_val = 1.0/hdist
							count_val = 1.0/query_C
							temp_tup = (pA,count_val,dist_val,seed_B)
							query_pA_list.append(temp_tup)
					query_pA_list = sorted(query_pA_list,reverse=True)
					min_tup = query_pA_list[0]
					min_pA = min_tup[0]
					min_seed_B = min_tup[3]
					try:
						min_hdist = hamming_dist_dict[query_B][min_seed_B]
					except:
						min_hdist = cal_hamming_dist(query_B,min_seed_B)
						try:
							hamming_dist_dict[query_B][min_seed_B] = min_hdist
						except:
							hamming_dist_dict[query_B] = {}
							hamming_dist_dict[query_B][min_seed_B] = min_hdist
					min_seed_C = partition_count_dict[min_seed_B]
					temp_tup = (min_pA,1.0/query_C,1.0/min_hdist,1.0/min_seed_C,min_seed_B,query_tup)
					pA_rank_list.append(temp_tup)
		if len(pA_rank_list) > 0:
			minimum_pA_threshold = pval_thresh/float(len(pA_rank_list))
			pA_rank_list = sorted(pA_rank_list,reverse=False)
			max_query_tup = pA_rank_list[0][-1]
			max_query_seed_B = pA_rank_list[0][-2]
			max_query_seed_B_count = int(1./pA_rank_list[0][-3])
			max_query_seed_B_dist = int(1./pA_rank_list[0][2])
			max_query_pA = pA_rank_list[0][0]
			max_query_B = max_query_tup[3]
			max_query_C = max_query_tup[0]
			max_query_Q = max_query_tup[4]
			# valid_query_barcodeID = False
			# try:
			# 	full_barcode_dict[max_query_B]
			# 	valid_query_barcodeID = True
			# except:
			# 	pass
			prev_consolodated_query_barcodeID = False
			try:
				consolodated_dict[query_tup]
				prev_consolodated_query_barcodeID = True
			except:
				pass
			new_active_ranked_barcode_list = []
			if max_query_C > 1 and max_query_pA<=minimum_pA_threshold:# and valid_query_barcodeID == True:
				partition_dict[max_query_B] = [max_query_tup]
				partition_count_dict[max_query_B] = max_query_C
				consolodated_dict[max_query_tup] = 'new-seed'
				abundance_pval_dict[max_query_tup] = {}
				new_seed_B = max_query_B
				new_seed_counter += 1
				for b in range(0,len(active_ranked_barcode_list)):
					query_tup = active_ranked_barcode_list[b]
					query_C = query_tup[0]
					# query_Qsum = query_tup[2]
					query_B = query_tup[3]
					query_Q = query_tup[4]
					
					consolodated_query_barcodeID = False
					try:
						consolodated_dict[query_tup]
						consolodated_query_barcodeID = True
					except:
						pass
					if query_B == new_seed_B and query_tup != max_query_tup and consolodated_query_barcodeID == False:
						partition_dict[query_B].append(query_tup)
						partition_count_dict[query_B] += query_C
						consolodated_dict[query_tup] = 'new-seed-collapse'
						abundance_pval_dict[query_tup] = {}
					elif query_B != new_seed_B:
						new_active_ranked_barcode_list.append(query_tup)
			else:
				new_active_ranked_barcode_list = active_ranked_barcode_list
			if max_query_C > 1 and prev_consolodated_query_barcodeID == False:# and max_query_seed_B_dist <=2:
				try:
					status_val = consolodated_dict[max_query_tup]
				except:
					status_val = 'na'
				debug_outlines += max_query_B+'\t'+str(iteration_count)+'\t'+str(max_query_C)+'\t'+max_query_seed_B+'\t'+str(max_query_seed_B_dist)+'\t'+str(max_query_seed_B_count)+'\t'+str(max_query_pA)+'\t'+str(minimum_pA_threshold)+'\t'+status_val+'\n'
			new_active_ranked_barcode_list = list(set(new_active_ranked_barcode_list))
			if len(new_active_ranked_barcode_list) == len(active_ranked_barcode_list):
				still_iterating = False
			active_ranked_barcode_list = new_active_ranked_barcode_list
			active_ranked_barcode_list = sorted(active_ranked_barcode_list,reverse=True)
		else:
			active_ranked_barcode_list = []
			still_iterating = False
	for i in range(0,len(ranked_barcode_list)):
		query_tup = ranked_barcode_list[i]
		query_B = query_tup[3]
		if len(query_B) == len(query_B.replace('N','')):
			try:
				consolodated_dict[query_tup]
			except:
				active_ranked_barcode_list.append(query_tup)
	# active_ranked_barcode_list = list(set(active_ranked_barcode_list))
	# active_ranked_barcode_list = sorted(active_ranked_barcode_list,reverse=True)

	#agglomerate remaining valid barcodes
	abundance_pval_dict = {}
	seed_barcode_list = list(partition_dict.keys())
	try:
		minimum_pA_threshold = pval_thresh/(float(len(active_ranked_barcode_list)))
	except:
		minimum_pA_threshold = 1.
	if len(active_ranked_barcode_list)>0:
		for i in range(0,len(active_ranked_barcode_list)):
			query_tup = active_ranked_barcode_list[i]
			query_C = query_tup[0]
			query_minQ = query_tup[1]
			query_Qsum = query_tup[2]
			query_B = query_tup[3]
			query_Q = query_tup[4]
			try:
				full_barcode_dict[query_B]
			except:
				pass
			consolodated_barcodeID = False
			try:
				consolodated_dict[query_tup]
				consolodated_barcodeID = True
			except:
				pass
			valid_barcodeID = False
			try:
				full_barcode_dict[query_B]
				valid_barcodeID = True
			except:
				pass
			if consolodated_barcodeID == False:
				query_pA_list = []
				if query_C > 1:
					for b in range(0,len(seed_barcode_list)):
						seed_B = seed_barcode_list[b]
						seed_C = partition_count_dict[seed_B]
						seed_tup = (seed_B,seed_C)
						hdist = cal_hamming_dist(query_B,seed_B)
						try:
							hdist = hamming_dist_dict[query_B][seed_B]
						except:
							hdist = cal_hamming_dist(query_B,seed_B)
							try:
								hamming_dist_dict[query_B][seed_B] = hdist
							except:
								hamming_dist_dict[query_B] = {}
								hamming_dist_dict[query_B][seed_B] = hdist
						if seed_B != query_B:
							if query_C < seed_C and hdist <= max_hamming_dist:
								# pA = abundance_pval(query_tup, seed_tup)
								try:
									pA = abundance_pval_dict[query_tup][seed_tup]
								except:
									pA = abundance_pval(query_tup, seed_tup)
									try:
										abundance_pval_dict[query_tup][seed_tup] = pA
									except:
										abundance_pval_dict[query_tup] = {}
										abundance_pval_dict[query_tup][seed_tup] = pA
							else:
								pA = 0.0
							temp_tup = (pA,1.0/hdist,0,seed_B)
							query_pA_list.append(temp_tup)
				if len(query_pA_list) >= 1:
					query_pA_list = sorted(query_pA_list,reverse=True)
					min_tup = query_pA_list[0]
					min_pA = min_tup[0]
					min_seed_B = min_tup[3]
					min_hdist = cal_hamming_dist(query_B,min_seed_B)
					min_seed_C = partition_count_dict[min_seed_B]
					minimum_pA_threshold = pval_thresh/(float(len(seed_barcode_list)))
					if min_pA>=minimum_pA_threshold and min_hdist <= max_hamming_dist: # and query_C > 1
						partition_dict[min_seed_B].append(query_tup)
						count_before = partition_count_dict[min_seed_B]
						partition_count_dict[min_seed_B] += query_C
						consolodated_dict[query_tup] = 'seed-collapse'
						abundance_pval_dict[query_tup] = {}
					elif valid_barcodeID == True:
						try:
							partition_dict[query_B].append(query_tup)
							count_before = partition_count_dict[query_B]
							partition_count_dict[query_B] += query_C
							consolodated_dict[query_tup] = 'remain-add'
						except:
							partition_dict[query_B] = [query_tup]
							partition_count_dict[query_B] = query_C
							consolodated_dict[query_tup] = 'remain'
					try:
						status_val = consolodated_dict[query_tup]
					except:
						status_val = 'na'
					debug_outlines += query_B+'\t'+min_seed_B+'\t'+str(query_C)+'\t'+str(min_seed_C)+'\t'+str(min_hdist)+'\t'+str(min_pA)+'\t'+str(minimum_pA_threshold)+'\t'+status_val+'\n'
				elif valid_barcodeID == True and query_minQ>=40:
					try:
						partition_dict[query_B].append(query_tup)
						count_before = partition_count_dict[query_B]
						partition_count_dict[query_B] += query_C
						consolodated_dict[query_tup] = 'last-add'
					except:
						partition_dict[query_B] = [query_tup]
						partition_count_dict[query_B] = query_C
						consolodated_dict[query_tup] = 'last'
	debug_outfile = open(temp_dir+accession+'.agglom_info.txt','w')
	debug_outfile.write(debug_outlines)
	debug_outfile.close()
	return partition_count_dict

def raw_read_processing_single(accession,command_prefix,raw_reads_forward,trim_dir,temp_dir,detect_barcode_content_first,amplicon_length,force_trim_pre_merge,truncate_length_post_merge):
	global verbose
	if verbose == True:
		update_print_line("Processing input reads",accession)
	trim_fastq_filename = temp_dir+accession+".trim.fq"
	clean_filename_out = trim_dir+accession+".fq"
	if detect_barcode_content_first == True or amplicon_length == 0:
		read_len_expected = calc_expect_read_len(raw_reads_forward)
	else:
		read_len_expected = amplicon_length
	command = command_prefix+'bbduk.sh in="'+raw_reads_forward+'" out="'+trim_fastq_filename+'" qin=33 maq=20 minlength='+str(read_len_expected)+' maxlength='+str(read_len_expected)+' overwrite=t'
	out = run_command(command+ ' overwrite=t threads=2 -Xmx1000m')
	if out == False:
		sys.exit()
	time.sleep(1)
	os.rename(trim_fastq_filename,clean_filename_out)
	return clean_filename_out

def truncate_merged_reads(filename_in,truncate_length):
	infile = open(filename_in,"r")
	outfile = open(filename_in.replace(".fq",".temp.fq"),"w")
	line_counter = -1
	seq_count = 0
	for line in infile:
		line = line.strip()
		line_counter += 1
		if line_counter == 0:
			header = line
			outfile.write(line+"\n")
		elif line_counter == 1: #base calls
			nucleotide_line = line[0:min(truncate_length,len(line))]
			outfile.write(nucleotide_line+"\n")
		elif line_counter == 2:
			outfile.write(line+"\n")
		elif line_counter == 3: #quality scores
			qual_line = line[0:min(truncate_length,len(line))]
			outfile.write(qual_line+"\n")
			line_counter = -1
	outfile.close()
	os.remove(filename_in)
	os.rename(filename_in.replace(".fq",".temp.fq"),filename_in)

def count_seqs(filepath_in,mode):
	infile = open(filepath_in,'r')
	line_counter = -1
	seq_count = 0
	for line in infile:
		line = line.strip()
		if len(line)>0:
			if mode == 'fastq':
				line_counter += 1
				if line_counter == 0:
					seq_count += 1
				elif line_counter == 3:
					line_counter = -1
			elif mode == 'fasta':
				if line[0]=='>':
					seq_count += 1
	infile.close()
	return seq_count

def raw_read_processing_paired(accession,command_prefix,raw_reads_forward,raw_reads_reverse,trim_dir,temp_dir,detect_barcode_content_first,amplicon_length,force_trim_pre_merge,truncate_length_post_merge):
	global verbose
	if verbose == True:
		update_print_line("Processing paired input reads",accession)
	repair_fastq_filename = temp_dir+accession+".repair.fq"
	temp_adapter_filename = temp_dir+accession+".adapters.fa"
	pretrim_fastq_filename = temp_dir+accession+".merge_pretrim.fq"
	merged_trimmed_filename = temp_dir+accession+".merge.fq"
	clean_filename_out = trim_dir+accession+".fq"

	raw_reads_forward_gz = raw_reads_forward+'.gz'
	raw_reads_reverse_gz = raw_reads_reverse+'.gz'
	if os.path.isfile(raw_reads_forward) == False and os.path.isfile(raw_reads_forward_gz) == True:
		command = 'cp '+raw_reads_forward_gz+' '+temp_dir
		run_command(command)
		unzip_gz(raw_reads_forward_gz.replace(input_read_dir,''),temp_dir)
		command = 'mv '+raw_reads_forward.replace(input_read_dir,temp_dir)+' '+input_read_dir
		out = run_command(command)
		if out == False:
			print("unzip gz failed: "+raw_reads_forward)
			sys.exit()
		else:
			os.remove(raw_reads_forward_gz)
	if os.path.isfile(raw_reads_reverse) == False and os.path.isfile(raw_reads_reverse_gz) == True:
		command = 'cp '+raw_reads_reverse_gz+' '+temp_dir
		run_command(command)
		unzip_gz(raw_reads_reverse_gz.replace(input_read_dir,''),temp_dir)
		command = 'mv '+raw_reads_reverse.replace(input_read_dir,temp_dir)+' '+input_read_dir
		out = run_command(command)
		if out == False:
			print("unzip gz failed: "+raw_reads_reverse)
			sys.exit()
		else:
			os.remove(raw_reads_reverse_gz)

	if os.path.isfile(raw_reads_forward) == False or os.path.isfile(raw_reads_reverse) == False:
		sys.exit('Unable to find paired files for sample: '+accession)
	if detect_barcode_content_first == True or amplicon_length == 0:
		median_forward_read_len = calc_expect_read_len(raw_reads_forward)
		median_reverse_read_len = calc_expect_read_len(raw_reads_reverse)
		diff_median_len = np.absolute(median_forward_read_len-median_reverse_read_len)
		read_len_expected = max(median_forward_read_len,median_reverse_read_len)
	else:
		read_len_expected = amplicon_length
		diff_median_len = 0
	
	
	command = command_prefix+'repair.sh in="'+raw_reads_forward+'" in2="'+raw_reads_reverse+'" out="'+repair_fastq_filename+'" fint=t repair=t threads=1 overwrite=t -da -Xmx1000m'
	out = run_command(command)
	if out == False:
		print("repair.sh failed: "+accession)
		run_command(command,'verbose')
		sys.exit()
	
	read_count_pre_cleaning = count_seqs(repair_fastq_filename,'fastq')
	read_count_pre_cleaning = int(read_count_pre_cleaning/2) #divded by 2 because this is an interleaved file
	outline = str(read_count_pre_cleaning)+'\n'
	outfile = open(output_dir+"samples/"+accession+".read_count_raw.txt",'w')
	outfile.write(outline)
	outfile.close()

	len_diff_buffer = 50
	command = command_prefix+'bbmerge.sh in="'+repair_fastq_filename+'" outadapter="'+temp_adapter_filename+'" minlength='+str(read_len_expected-(len_diff_buffer+diff_median_len))+' maxlength='+str(read_len_expected+(len_diff_buffer+diff_median_len))+' overwrite=t' #forcetrimleft=9
	if force_trim_pre_merge >0:
		command += " forcetrimleft="+str(force_trim_pre_merge)
	out = run_command(command+" threads=1 -Xmx2000m")
	if out == False:
		print("bbmerge.sh adapter screen failed: "+accession)
		run_command(command,'verbose')
		sys.exit()
	time.sleep(1)

	adapter_pass = check_adapter(temp_adapter_filename)
	if adapter_pass == True:
		command = command_prefix+'bbmerge.sh in="'+repair_fastq_filename+'" out="'+pretrim_fastq_filename+'" qin=33 mix=f adapters="'+temp_adapter_filename+'" minlength='+str(read_len_expected-(len_diff_buffer+diff_median_len))+' maxlength='+str(read_len_expected+(len_diff_buffer+diff_median_len)) #forcetrimleft=9 
		if force_trim_pre_merge >0:
			command += " forcetrimleft="+str(force_trim_pre_merge)
		out = run_command(command+" overwrite=t threads=1 -Xmx2000m")
		if out == False:
			print("bbmerge.sh merge with adapter trim failed: "+accession)
			run_command(command,'verbose')
			sys.exit()
		time.sleep(1)
	else:
		command = command_prefix+'bbmerge.sh in="'+repair_fastq_filename+'" out="'+pretrim_fastq_filename+'" qin=33 mix=f minlength='+str(read_len_expected-(len_diff_buffer+diff_median_len))+' maxlength='+str(read_len_expected+(len_diff_buffer+diff_median_len))+' overwrite=t' #forcetrimleft=9 
		if force_trim_pre_merge >0:
			command += " forcetrimleft="+str(force_trim_pre_merge)
		out = run_command(command+" overwrite=t threads=1 -Xmx2000m")
		if out == False:
			print("bbmerge.sh merge failed")
			run_command(command,'verbose')
			sys.exit()
		time.sleep(1)

	if amplicon_length == 0:
		median_merged_len = calc_expect_read_len(pretrim_fastq_filename)
	else:
		median_merged_len = amplicon_length
	command = command_prefix+'bbduk.sh in="'+pretrim_fastq_filename+'" out="'+merged_trimmed_filename+'" qin=33 maq=20 minlength='+str(median_merged_len)+' maxlength='+str(median_merged_len)+' tpe=f tbo=f'
	out = run_command(command+ ' overwrite=t threads=1 -Xmx1000m')
	if out == False:
		print("bbduk.sh failed: "+accession)
		run_command(command,'verbose')
		sys.exit()
	time.sleep(1)

	if truncate_length_post_merge > 0:
		truncate_merged_reads(merged_trimmed_filename,truncate_length_post_merge)
	
	# if detect_barcode_content_first == False:
	read_count_cleaned = count_seqs(merged_trimmed_filename,'fastq')
	outline = str(read_count_cleaned)+'\n'
	outfile = open(output_dir+"samples/"+accession+".read_count_postmerge.txt",'w')
	outfile.write(outline)
	outfile.close()

	os.remove(repair_fastq_filename)
	os.remove(temp_adapter_filename)
	os.remove(pretrim_fastq_filename)
	os.rename(merged_trimmed_filename,clean_filename_out)
	if detect_barcode_content_first == False:
		raw_reads_forward_filename = raw_reads_forward.replace(input_read_dir,'')
		raw_reads_reverse_filename = raw_reads_reverse.replace(input_read_dir,'')
		command = 'cp '+input_read_dir+raw_reads_forward_filename+' '+temp_dir
		run_command(command)
		zip_gz(raw_reads_forward_filename,temp_dir)
		command = 'mv '+temp_dir+raw_reads_forward_filename+'.gz '+input_read_dir
		out = run_command(command)
		if out == False:
			print("zip gz failed: "+raw_reads_forward_filename)
			sys.exit()
		else:
			os.remove(input_read_dir+raw_reads_forward_filename)
		
		command = 'cp '+input_read_dir+raw_reads_reverse_filename+' '+temp_dir
		run_command(command)
		zip_gz(raw_reads_reverse_filename,temp_dir)
		command = 'mv '+temp_dir+raw_reads_reverse_filename+'.gz '+input_read_dir
		out = run_command(command)
		if out == False:
			print("zip gz failed: "+raw_reads_reverse_filename)
			sys.exit()
		else:
			os.remove(input_read_dir+raw_reads_reverse_filename)
	return clean_filename_out

def subsample_read_counts(barcode_qual_count_dict_in,subsample_count):
	rand_list = []
	for barcode_qual_tup in barcode_qual_count_dict_in:
		barcode_string = barcode_qual_tup[0]
		barcode_qual_list = barcode_qual_tup[1]
		count = barcode_qual_count_dict_in[barcode_qual_tup]
		for i in range(0,count):
			rand_list.append(barcode_qual_tup)
	random.shuffle(rand_list)
	subset_list = rand_list[0:int(subsample_count)]
	barcode_qual_count_dict_out = {}
	for i in range(0,len(subset_list)):
		barcode_qual_tup = subset_list[i]
		try:
			barcode_qual_count_dict_out[barcode_qual_tup] += 1
		except:
			barcode_qual_count_dict_out[barcode_qual_tup] = 1
	return barcode_qual_count_dict_out

def barcode_screen_trimmed_reads(accession,sequence_filename,expected_amplicon_seq,barcode_sites,min_Q_score,temp_dir,output_dir,F_edge_ignore,R_edge_ignore):
	reads_per_sample_ceiling = 1e9
	sequence_filename_gz = sequence_filename+'.gz'
	### Load in processed reads and collect sequence to quality info to screen for barcodes
	global verbose
	if verbose == True:
		update_print_line("Screening reads for barcodes",accession)
	barcode_qual_count_filename = trim_dir+accession+'.barcode_qual_counts.txt'
	if os.path.isfile(sequence_filename) == False and os.path.isfile(sequence_filename_gz) == True:
		command = 'cp '+sequence_filename_gz+' '+temp_dir
		run_command(command)
		unzip_gz(sequence_filename_gz.replace(trim_dir,''),temp_dir)
		command = 'mv '+sequence_filename.replace(trim_dir,temp_dir)+' '+trim_dir
		out = run_command(command)
		if out == False:
			print("unzip gz failed: "+sequence_filename_gz)
			sys.exit()
		else:
			os.remove(sequence_filename_gz)
	fastq_infile = open(sequence_filename,"r")
	mismatch_by_site = {}
	barcode_qual_count_dict = {}
	uncorrected_barcode_count_dict = {}
	barcode_list = []
	line_counter = -1
	seq_screen_count = 0
	seq_count = 0
	pass_seq_count = 0
	for line in fastq_infile:
		line = line.strip()
		line_counter += 1
		if line_counter == 0:
			header = line
			writing = False
		elif line_counter == 1: #base calls
			nucleotide_line = line
		elif line_counter == 3: #quality scores
			qual_line = line
			if len(nucleotide_line) == len(expected_amplicon_seq):
				seq_screen_count += 1
				if verbose == True:
					if seq_screen_count%1e5==0 and seq_screen_count > 1:
						update_print_line("Reads screened: "+str(seq_screen_count),accession)
				read_barcode_tup,barcode_mismatches,backbone_mismatches,mismatch_list = screen_barcode_site_qual(nucleotide_line,qual_line,expected_amplicon_seq,barcode_sites,F_edge_ignore,R_edge_ignore)#(seq_in,qual_in,seq_expect,barcode_dict,min_barcode_site_qual,min_other_site_qual)
				if min(read_barcode_tup[1]) >= min_Q_score:
					seq_count += 1
					total_mismatches = barcode_mismatches+backbone_mismatches
					if backbone_mismatches == 0 and barcode_mismatches <= 1:
						try:
							barcode_qual_count_dict[read_barcode_tup] += 1
						except:
							barcode_qual_count_dict[read_barcode_tup] = 1
						pass_seq_count += 1
						if barcode_mismatches == 0:
							try:
								uncorrected_barcode_count_dict[read_barcode_tup[0]] += 1
							except:
								uncorrected_barcode_count_dict[read_barcode_tup[0]] = 1
					if len(mismatch_list)>0:
						for m in range(0,len(mismatch_list)):
							try:
								mismatch_by_site[mismatch_list[m]] += 1
							except:
								mismatch_by_site[mismatch_list[m]] = 1
			line_counter = -1
	fastq_infile.close()

	if pass_seq_count > reads_per_sample_ceiling:
		barcode_qual_count_dict = subsample_read_counts(barcode_qual_count_dict,reads_per_sample_ceiling)
		if verbose == True:
			update_print_line("Subsampled read counts to "+str(int(reads_per_sample_ceiling)),accession)

	barcode_qual_list = []
	outstring = ''
	for read_barcode_tup in barcode_qual_count_dict:
		barcode_string = read_barcode_tup[0]
		barcode_qual_tup = read_barcode_tup[1]
		count = barcode_qual_count_dict[read_barcode_tup]
		min_Q = min(barcode_qual_tup)
		max_Q = max(barcode_qual_tup)
		Qsum = np.sum(barcode_qual_tup)
		if min_Q >= min_Q_score:
			out_tup = (count,min_Q,Qsum,barcode_string,barcode_qual_tup)
			barcode_qual_list.append(out_tup)
		outstring += barcode_string+'\t'+str(barcode_qual_tup[0])
		for i in range(1,len(barcode_qual_tup)):
			outstring += ','+str(barcode_qual_tup[i])
		outstring += '\t'+str(count)+'\n'
	barcode_qual_count_file = open(barcode_qual_count_filename,'w')
	barcode_qual_count_file.write(outstring)
	barcode_qual_count_file.close()

	full_barcode_list,barcode_list_by_category = find_all_possible_barcode_combinations(barcode_sites,barcode_site_class,viral_library_alleles,viral_library_name_dict)

	full_barcode_dict = {}
	for f in range(0,len(full_barcode_list)):
		barcode = full_barcode_list[f]
		try:
			full_barcode_dict[barcode] = None
		except:
			print(barcode)
			sys.exit(full_barcode_list)

	#### count unique uncorrected barcodes
	raw_barcode_list = sorted(list(uncorrected_barcode_count_dict.keys()))
	raw_barcode_outfile = open(temp_dir+accession+".raw_barcode_count.txt","w")
	raw_barcode_outfile.write("barcode\tcount\n")
	for num in range(0,len(raw_barcode_list)):
		barcode_string = raw_barcode_list[num]
		raw_barcode_count = uncorrected_barcode_count_dict[barcode_string]
		raw_barcode_outfile.write(str(barcode_string)+"\t"+str(raw_barcode_count)+"\n")
	raw_barcode_outfile.close()

	#### perform agglomerative clustering
	barcode_count_dict = divisive_partition(barcode_qual_list,full_barcode_dict,accession)

	#### count unique barcodes
	barcode_list = sorted(list(barcode_count_dict.keys()))
	barcode_outfile = open(temp_dir+accession+".barcode_count.txt","w")
	barcode_outfile.write("barcode\tcount\n")
	for num in range(0,len(barcode_list)):
		barcode_string = barcode_list[num]
		barcode_count = barcode_count_dict[barcode_string]
		barcode_outfile.write(str(barcode_string)+"\t"+str(barcode_count)+"\n")
	barcode_outfile.close()

	### summarize mismatch count
	mismatch_prop_by_site = {}
	mismatch_by_site_outfile = open(temp_dir+accession+".mismatch_info.txt","w")
	mismatch_by_site_outfile.write("Site_number\tmismatches\n")
	for col in range(0,len(expected_amplicon_seq)):
		try:
			mismatch_count = mismatch_by_site[col]
		except:
			mismatch_count = 0
		if mismatch_count > 0:
			mismatch_prop = mismatch_count/seq_count
		else:
			mismatch_prop = 0.
		mismatch_prop_by_site[col] = mismatch_prop
		mismatch_by_site_outfile.write(str(col)+"\t"+str(mismatch_prop)+"\n")
	mismatch_by_site_outfile.close()

	### write file with full read count
	read_count_outfile = open(temp_dir+accession+".total_read_count.txt","w")
	read_count_outfile.write('total reads screened\t'+str(seq_count)+'\npass barcode screen\t'+str(pass_seq_count)+'\n')
	read_count_outfile.close()
	post_merge_readcount = 0
	infile = open(output_dir+"samples/"+accession+".read_count_postmerge.txt",'r')
	for line in infile:
		line = line.strip()
		post_merge_readcount = int(line)
	infile.close()
	raw_readcount = 0
	infile = open(output_dir+"samples/"+accession+".read_count_raw.txt",'r')
	for line in infile:
		line = line.strip()
		raw_readcount = int(line)
	infile.close()
	read_count_tup = (raw_readcount,post_merge_readcount,seq_count,pass_seq_count)
	read_filename = accession+'.fq'
	command = 'cp '+trim_dir+read_filename+' '+temp_dir
	run_command(command)
	zip_gz(read_filename,temp_dir)
	command = 'mv '+temp_dir+read_filename+'.gz '+trim_dir
	out = run_command(command)
	if out == False:
		print("zip gz failed: "+read_filename)
		sys.exit()
	else:
		os.remove(trim_dir+read_filename)
	os.rename(temp_dir+accession+".mismatch_info.txt",output_dir+"samples/"+accession+".mismatch_info.txt")
	os.rename(temp_dir+accession+".barcode_count.txt",output_dir+"samples/"+accession+".barcode_count.txt")
	os.rename(temp_dir+accession+".raw_barcode_count.txt",output_dir+"samples/"+accession+".raw_barcode_count.txt")
	os.rename(temp_dir+accession+".total_read_count.txt",output_dir+"samples/"+accession+".total_read_count.txt")

	return barcode_count_dict,barcode_list,mismatch_prop_by_site,read_count_tup


def clean_files_for_one_sample(accession,output_dir,trim_dir):
	sample_barcode_info_filename = output_dir+"samples/"+accession+".barcode_count.txt"
	barcode_mismatch_info_filename = output_dir+"samples/"+accession+".mismatch_info.txt"
	sample_nonbarcode_info_filename = output_dir+"samples/"+accession+".nonbarcode_count.txt"
	processed_read_filename = trim_dir+accession+".fq"
	sample_readcount_raw_filename = output_dir+"samples/"+accession+".read_count_raw.txt"
	sample_readcount_postmerge_filename = output_dir+"samples/"+accession+".read_count_postmerge.txt"
	
	#remove any files that exist
	if os.path.isfile(sample_barcode_info_filename) == True:
		os.remove(sample_barcode_info_filename)
	if os.path.isfile(barcode_mismatch_info_filename) == True:
		os.remove(barcode_mismatch_info_filename)
	if os.path.isfile(sample_nonbarcode_info_filename) == True:
		os.remove(sample_nonbarcode_info_filename)
	if os.path.isfile(processed_read_filename) == True:
		os.remove(processed_read_filename)
	if os.path.isfile(sample_readcount_raw_filename) == True:
		os.remove(sample_readcount_raw_filename)
	if os.path.isfile(sample_readcount_postmerge_filename) == True:
		os.remove(sample_readcount_postmerge_filename)

def subset_all_samples(accession_list,input_read_dir,forward_read_suffix,reverse_read_suffix,subset_target=1e5):
	'''This function takes a small number of reads from all input files to perform barcode prediction on'''
	subset_number = max(int(int((subset_target/len(accession_list))/10)*10),int(1e3))
	all_reads_for_path = temp_dir+'subset_all_reads.R1.fastq'
	all_reads_rev_path = temp_dir+'subset_all_reads.R2.fastq'
	outfile = open(all_reads_for_path,'w')
	outfile.close()
	outfile = open(all_reads_rev_path,'w')
	outfile.close()
	for a in range(0,len(accession_list)):
		accession = accession_list[a]
		
		raw_reads_forward = input_read_dir+accession+forward_read_suffix
		raw_reads_forward_gz = raw_reads_forward+'.gz'
		if os.path.isfile(raw_reads_forward) == False and os.path.isfile(raw_reads_forward_gz) == True:
			command = 'cp '+raw_reads_forward_gz+' '+temp_dir
			run_command(command)
			unzip_gz(raw_reads_forward_gz.replace(input_read_dir,''),temp_dir)
			command = 'mv '+raw_reads_forward.replace(input_read_dir,temp_dir)+' '+input_read_dir
			out = run_command(command)
			if out == False:
				print("unzip gz failed: "+raw_reads_forward)
				sys.exit()
			else:
				os.remove(raw_reads_forward_gz)
		outlines = ''
		infile = open(raw_reads_forward,"r")
		line_counter = -1
		seq_count = 0
		include = True
		for line in infile:
			line = line.rstrip('\n')
			line_counter += 1
			if line_counter == 0:
				seq_count += 1
				if seq_count <= subset_number:
					include = True
				else:
					include = False
			elif line_counter == 3: #quality scores
				line_counter = -1
			if include == True:
				outlines += line+'\n'
			if seq_count > subset_number:
				break
		infile.close()
		outfile = open(all_reads_for_path,'a')
		outfile.write(outlines)
		outfile.close()

		if reverse_read_suffix != '':
			raw_reads_reverse = input_read_dir+accession+reverse_read_suffix
			raw_reads_reverse_gz = raw_reads_reverse+'.gz'
			if os.path.isfile(raw_reads_reverse) == False and os.path.isfile(raw_reads_reverse_gz) == True:
				# unzip_gz(raw_reads_reverse_gz)
				command = 'cp '+raw_reads_reverse_gz+' '+temp_dir
				run_command(command)
				unzip_gz(raw_reads_reverse_gz.replace(input_read_dir,''),temp_dir)
				command = 'mv '+raw_reads_reverse.replace(input_read_dir,temp_dir)+' '+input_read_dir
				out = run_command(command)
				if out == False:
					print("unzip gz failed: "+raw_reads_reverse)
					sys.exit()
				else:
					os.remove(raw_reads_reverse_gz)

			outlines = ''
			infile = open(raw_reads_reverse,"r")
			line_counter = -1
			seq_count = 0
			include = True
			for line in infile:
				line = line.rstrip('\n')
				line_counter += 1
				if line_counter == 0:
					seq_count += 1
					if seq_count <= subset_number:
						include = True
					else:
						include = False
				elif line_counter == 3: #quality scores
					line_counter = -1
				if include == True:
					outlines += line+'\n'
				if seq_count > subset_number:
					break
			infile.close()
			outfile = open(all_reads_rev_path,'a')
			outfile.write(outlines)
			outfile.close()
	return all_reads_for_path,all_reads_rev_path

def update_print_line(string_in,accession):
	global longest_accession_length
	print(accession+" "*(longest_accession_length-len(accession))+" - "+string_in)

def separate_counts_by_viral_library(accession_tup_in):
	'''
	accession_tup_in = (accession,barcode_count_dict,barcode_list,mismatch_by_site,read_count_tup)
	read_count_tup = (raw_readcount,post_merge_readcount,seq_count,pass_seq_count)
	'''
	accession  = accession_tup_in[0]
	barcode_count_dict = accession_tup_in[1]
	mismatch_by_site = accession_tup_in[3]
	read_count_tup = accession_tup_in[4]
	count_dict_out = {}
	split_read_total_dict = {}
	for barcodeID in barcode_count_dict:
		count = barcode_count_dict[barcodeID]
		try:
			cat_name = barcode_category_lookup_dict[barcodeID]
		except:
			cat_name = ''
		if cat_name != '':
			local_barcodeID = barcodeID#cat_name+'-'+barcodeID
			try:
				count_dict_out[cat_name][local_barcodeID] = count
			except:
				count_dict_out[cat_name] = {}
				count_dict_out[cat_name][local_barcodeID] = count
			try:
				split_read_total_dict[cat_name] += count
			except:
				split_read_total_dict[cat_name] = count
	total_read_count = read_count_tup[3]
	used_cat_list = list(count_dict_out.keys())
	output_list = []
	for i in range(0,len(used_cat_list)):
		cat_name = used_cat_list[i]
		cat_accession = accession+'_'+cat_name
		try:
			cat_read_count = split_read_total_dict[cat_name]
		except:
			cat_read_count = 0
		count_prop = cat_read_count/total_read_count

		cat_read_count_tup = (int(read_count_tup[0]*count_prop),int(read_count_tup[1]*count_prop),int(read_count_tup[2]*count_prop),split_read_total_dict[cat_name])
		out_tup = (cat_accession,count_dict_out[cat_name],sorted(list(count_dict_out[cat_name].keys())),mismatch_by_site,cat_read_count_tup)
		output_list.append(out_tup)
	return output_list


##########################      Core pipeline function      ##########################

# def reads_to_barcodes_one_sample(accession,command_prefix,forward_read_suffix,reverse_read_suffix,input_read_dir,trim_dir,temp_dir,output_dir,expected_amplicon_seq,barcode_sites,barcode_site_class,min_Q_score,overwrite_existing_files,raw_read_input_type,reads_already_screened_for_quality,detect_barcode_content_first,F_edge_ignore,R_edge_ignore,force_trim_pre_merge,truncate_length_post_merge,trim_all_files_for_qual_learning=False):
def reads_to_barcodes_one_sample(accession,detect_barcode_content_first,trim_all_files_for_qual_learning=False):
	# update_print_line("Sample processing start",accession)
	processed_something = False
	raw_reads_forward = input_read_dir+accession+forward_read_suffix
	raw_reads_reverse = input_read_dir+accession+reverse_read_suffix
	
	amplicon_length = len(expected_amplicon_seq)
	sample_barcode_info_filename = output_dir+"samples/"+accession+".barcode_count.txt"
	barcode_mismatch_info_filename = output_dir+"samples/"+accession+".mismatch_info.txt"
	sample_nonbarcode_info_filename = output_dir+"samples/"+accession+".nonbarcode_count.txt"
	processed_read_filename = trim_dir+accession+".fq"
	processed_read_filename_gz = processed_read_filename+".gz"
	#clear the temporary files folder from any previously interrupted runs
	temp_files_list = [f for f in os.listdir(temp_dir)]
	for files in temp_files_list:
		if accession+"." in files:
			os.remove(temp_dir+files)
	#remove any files that already exist if 'overwrite_existing_files' is set to 'True'
	if overwrite_existing_files == True:
		if os.path.isfile(sample_barcode_info_filename) == True:
			os.remove(sample_barcode_info_filename)
		if os.path.isfile(barcode_mismatch_info_filename) == True:
			os.remove(barcode_mismatch_info_filename)
		if os.path.isfile(sample_nonbarcode_info_filename) == True:
			os.remove(sample_nonbarcode_info_filename)
		if os.path.isfile(processed_read_filename) == True:
			os.remove(processed_read_filename)
	if reads_already_screened_for_quality == False:
		if os.path.isfile(processed_read_filename) == False and os.path.isfile(processed_read_filename_gz) == False:
			if raw_read_input_type == "paired":
				processed_read_filename = raw_read_processing_paired(accession,command_prefix,raw_reads_forward,raw_reads_reverse,trim_dir,temp_dir,detect_barcode_content_first,amplicon_length,force_trim_pre_merge,truncate_length_post_merge)
				processed_something = True
			elif raw_read_input_type == "single":
				processed_read_filename = raw_read_processing_single(accession,command_prefix,raw_reads_forward,trim_dir,temp_dir,detect_barcode_content_first,amplicon_length,force_trim_pre_merge,truncate_length_post_merge)
				processed_something = True
	elif reads_already_screened_for_quality == True:
		processed_read_filename = raw_reads_forward
	if trim_all_files_for_qual_learning == True:
		qual_summary_tup = pull_qual_info(processed_read_filename)
		return qual_summary_tup
	else:
		# barcode_count_dict,barcode_list,mismatch_by_site,read_count_tup = barcode_screen_trimmed_reads(accession,processed_read_filename,expected_amplicon_seq,barcode_sites,min_Q_score,temp_dir,output_dir,F_edge_ignore,R_edge_ignore) #,high_qual_nonbarcode_dict
		# processed_something = True
		if os.path.isfile(sample_barcode_info_filename) == False:
			barcode_count_dict,barcode_list,mismatch_by_site,read_count_tup = barcode_screen_trimmed_reads(accession,processed_read_filename,expected_amplicon_seq,barcode_sites,min_Q_score,temp_dir,output_dir,F_edge_ignore,R_edge_ignore) #,high_qual_nonbarcode_dict
			processed_something = True
		else:
			barcode_count_dict,barcode_list,mismatch_by_site,read_count_tup = read_in_barcodes(accession,sample_barcode_info_filename,barcode_mismatch_info_filename,sample_nonbarcode_info_filename) #,high_qual_nonbarcode_dict
		out_tup = (accession,barcode_count_dict,barcode_list,mismatch_by_site,read_count_tup)
		if processed_something == True:
			update_print_line("Completed",accession)
		if categorical_sites_present == True:
			out_tup_list = separate_counts_by_viral_library(out_tup)
			return out_tup_list
		return out_tup


#####################################   MAIN   ######################################

#Create output and temporary working directories if they do not exist already
if not os.path.exists(trim_dir):
	os.makedirs(trim_dir)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
if not os.path.exists(output_dir+"samples/"):
	os.makedirs(output_dir+"samples/")
if not os.path.exists(temp_dir):
	os.makedirs(temp_dir)

#Set up parallel processing if enabled
if parallel_process == True:
	import multiprocessing
	from joblib import Parallel, delayed
	if parallel_max_cpu == 0:
		num_cores = multiprocessing.cpu_count()
	elif parallel_max_cpu > 0:
		num_cores = min(parallel_max_cpu,multiprocessing.cpu_count())
	elif parallel_max_cpu < 0:
		num_cores = multiprocessing.cpu_count()+parallel_max_cpu
else:
	num_cores = 1

### Check if the files in the input directory need to be unzipped
files_found = check_if_input_zipped(input_read_dir,parallel_process,num_cores)
if files_found == False:
	sys.exit("No input files with '.fastq', '.fq', '.gz', or '.tar.gz' extensions found. Exiting.")

### Predict forward and reverse read suffix if left blank
continue_running = False
if forward_read_suffix == '':
	suffix_info_file_path = temp_dir+"file_extensions.txt"
	try:
		file_suffix_infile = open(suffix_info_file_path)
		for line in file_suffix_infile:
			line = line.strip().split("\t")
			if line[0] == "forward":
				forward_read_suffix = line[1]
			elif line[0] == "reverse":
				try:
					reverse_read_suffix = line[1]
				except:
					reverse_read_suffix = ''
			continue_running = True
	except:
		if raw_read_input_type == "paired" and reads_already_screened_for_quality == False:
			forward_read_suffix,reverse_read_suffix = predict_paired_file_extensions(input_read_dir)
			print('Predicted read suffix\n\tForward: "'+str(forward_read_suffix)+'"\n\tReverse: "'+str(reverse_read_suffix)+'"')
			print("Is this correct?")
			decision = input("(yes or no)\n")
			if decision == "Y" or decision == "y" or decision == "Yes" or decision == "yes" or decision == "YES":
				continue_running = True
			else:
				forward_read_suffix = input('Enter forward read suffix (ex "_L001_R1_001.fastq") or "exit" to cancel:\n')
				if forward_read_suffix == "exit":
					sys.exit()
				reverse_read_suffix	= input('Enter Reverse read suffix (ex "_L001_R2_001.fastq") or "exit" to cancel:\n')
				if reverse_read_suffix == "exit":
					sys.exit()
				if forward_read_suffix != "exit" and reverse_read_suffix != "exit":
					continue_running = True
			if continue_running == True:
				suffix_info_file = open(suffix_info_file_path,"w")
				suffix_info_file.write("forward\t"+forward_read_suffix+"\nreverse\t"+reverse_read_suffix+"\n")
				suffix_info_file.close()
		elif raw_read_input_type == "single" or reads_already_screened_for_quality == True:
			forward_read_suffix = predict_single_file_extension(input_read_dir)
			print('Predicted read suffix: "'+str(forward_read_suffix)+'"')
			print("Is this correct?")
			decision = input("(yes or no)\n")
			if decision == "Y" or decision == "y" or decision == "Yes" or decision == "yes" or decision == "YES":
				continue_running = True
			else:
				forward_read_suffix = input('Enter read suffix (ex "_L001_001.fastq") or "exit" to cancel:\n')
				if forward_read_suffix == "exit":
					sys.exit()
				else:
					continue_running = True
			if continue_running == True:
				suffix_info_file = open(suffix_info_file_path,"w")
				suffix_info_file.write("forward\t"+forward_read_suffix+"\nreverse\t"+reverse_read_suffix+"\n")
				suffix_info_file.close()

if raw_read_input_type == "paired":
	if forward_read_suffix != "" and reverse_read_suffix != "":
		continue_running = True
	else:
		continue_running = False
elif raw_read_input_type == "single":
	if forward_read_suffix != "":
		continue_running = True
	else:
		continue_running = False

if continue_running == False:
	sys.exit("No read suffix supplied, please re-run and enter information when prompted")


### Create list of samples to process
file_list = [f for f in os.listdir(input_read_dir) if f.endswith(forward_read_suffix)]
file_list_gz = [f for f in os.listdir(input_read_dir) if f.endswith(forward_read_suffix+'.gz')]
file_list.extend(file_list_gz)

accession_list = []
files_not_accounted_for = 0
for files in file_list:
	accession = files.split(forward_read_suffix)[0]
	accession_list.append(accession)
	if raw_read_input_type == "paired":
		rev_filepath = input_read_dir+accession+reverse_read_suffix
		if os.path.isfile(rev_filepath)==False:
			rev_filepath_gz = input_read_dir+accession+reverse_read_suffix+'.gz'
			if os.path.isfile(rev_filepath_gz)==False:
				print("Reverse file not found: "+accession+reverse_read_suffix)
				files_not_accounted_for += 1
if files_not_accounted_for > 0:
	sys.exit('Unable to find '+str(files_not_accounted_for)+' reverse files. Exiting.')
accession_list = list(set(accession_list))

sorted_accession_list = sorted(accession_list)
print_line_dict = {}
longest_accession_length = 0
for a in range(0,len(sorted_accession_list)):
	accession = sorted_accession_list[a]
	print_line_dict[accession] = a
	if len(accession)>longest_accession_length:
		longest_accession_length = len(accession)

### load barcode info. If unavailable, pick a random file and attempt to predict barcode info
detect_barcode_content_first = True
amplicon_info_infile_path = project_dir+amplicon_info_infile_name
if os.path.isfile(amplicon_info_infile_path):
	expected_amplicon_seq,barcode_sites,barcode_site_class,viral_library_alleles,viral_library_name_dict = load_barcode_info(project_dir+amplicon_info_infile_path)
	if expected_amplicon_seq != "" and barcode_sites != {}:
		detect_barcode_content_first = False
	else:
		if len(viral_library_name_dict) == 0:
			categorical_sites_present = False
		else:
			categorical_sites_present = True
else:
	print("\n\nCould not detect barcode info input file.")

if detect_barcode_content_first == True:
	'''	This section will attempt to predict the barcode info for your experiment if it is not already given.
	It will run through the entire pipeline with this one sample, then you need to check the output to determine
	if it correctly identified the barcode sites. If you agree with the prediction, remove ".auto" from
	the "barcode.info.auto.txt" file that was generated, then re-run the script to process all remaining samples.'''
	overwrite_existing_files = True
	if len(accession_list)==1:
		accession_entered = accession_list[0]
		print("Only one file found, using: "+str(accession_entered)+" to predict barcode info")
	else:
		decision = input('Use reads from all samples for predicting barcode info?:\n')
		if decision == "Y" or decision == "y" or decision == "Yes" or decision == "yes" or decision == "YES":
			accession_entered = 'all'
		else:
			accession_entered = input('Enter sample name for predicting barcode info, or "random" to use random sample:\n')
	temp_cont = False
	if accession_entered == 'all':
		accession = 'all'
	elif accession_entered == "random":
		accession = accession_list[random.randint(0,len(accession_list))]
	elif accession_entered in sorted_accession_list:
		accession = accession_entered
	else:
		sys.exit("sample ID not valid.")
	if accession_entered == "random" or accession_entered != 'all':
		print('Attempting to predict barcode using sample: "'+accession+'"\n')

	num_expected_barcode_sites = input('How many barcode sites should be present in these samples? (enter number below)\n')
	num_expected_barcode_sites = num_expected_barcode_sites.strip()
	if is_number(num_expected_barcode_sites)== True:
		num_expected_barcode_sites = int(num_expected_barcode_sites)
	else:
		sys.exit("Value entered is not a number")
	decision = input('\nDo any barcode sites distinguish different viruses or barcode libraries? (yes or no)\n(this does not apply to viruses that have barcoded regions in separate genomic loci)\n')
	decision = decision.strip()
	if decision == "Y" or decision == "y" or decision == "Yes" or decision == "yes" or decision == "YES":
		categorical_sites_present = True
	else:
		categorical_sites_present = False
		num_viral_libraries = 1
	if categorical_sites_present == True:
		num_viral_libraries = input('How many different viruses or barcode libraries are present?\n')
		num_viral_libraries = num_viral_libraries.strip()
		if is_number(num_viral_libraries)== True:
			num_viral_libraries = int(num_viral_libraries)
			if num_viral_libraries == 1:
				sys.exit('Error: you cannot have only 1 library in this mode. Exiting.')
		else:
			sys.exit("Value entered is not a number")
	
	if accession == 'all':
		raw_reads_forward,raw_reads_reverse = subset_all_samples(accession_list,input_read_dir,forward_read_suffix,reverse_read_suffix)
	else:
		raw_reads_forward = input_read_dir+accession+forward_read_suffix
		raw_reads_reverse = input_read_dir+accession+reverse_read_suffix
	if raw_read_input_type == "paired":
		processed_read_filename = raw_read_processing_paired(accession,command_prefix,raw_reads_forward,raw_reads_reverse,trim_dir,temp_dir,detect_barcode_content_first,0,force_trim_pre_merge,truncate_length_post_merge)
	elif raw_read_input_type == "single":
		processed_read_filename = raw_read_processing_single(accession,command_prefix,raw_reads_forward,trim_dir,temp_dir,detect_barcode_content_first,0,force_trim_pre_merge,truncate_length_post_merge)

	temp_seq_dict,temp_read_length = subset_fastq(processed_read_filename,500000)
	expected_amplicon_seq,barcode_sites = detect_barcode_content(temp_seq_dict,temp_read_length)
	continue_running = False
	temp_barcode_site_list = list(barcode_sites.keys())
	temp_barcode_site_list = sorted(temp_barcode_site_list)
	barcode_site_class = {}
	viral_library_alleles = {}
	viral_library_name_dict = {}
	for site_num in range(0,len(temp_barcode_site_list)):
		site = temp_barcode_site_list[site_num]
		barcode_site_class[site] = 'DS'
	if len(barcode_sites) == num_expected_barcode_sites:
		print('\n\nThe correct number of expected barcode sites were automatically detected.')
		continue_running = False
		print("amplicon sequence:\n\t"+expected_amplicon_seq)
		for site_num in range(0,len(temp_barcode_site_list)):
			site = temp_barcode_site_list[site_num]
			print("\t"+str(site)+"\t"+barcode_sites[site][0]+" or "+barcode_sites[site][1])
		print('\n\nAre these sites and possible nucleotides correct?')
		decision = input("(yes or no)\n")
		if decision == "Y" or decision == "y" or decision == "Yes" or decision == "yes" or decision == "YES":
			continue_running = True
			outfile = open(project_dir+"amplicon_info.txt","w")
		else:
			continue_running = False
			clean_files_for_one_sample(accession,output_dir,trim_dir)
			outfile = open(project_dir+"amplicon_info.auto.txt","w")
		
		if categorical_sites_present == True:
			print("\nEnter shorthand name for each viral library (e.g. 'WT'):")
			for v in range(0,num_viral_libraries):
				library_name = input('    Library #'+str(v+1)+'  -  ')
				viral_library_name_dict[v] = library_name
			entry = input('\nList sites that distinguish different viruses or libraries (separate numbers by commas):  -  ')
			if ',' in entry:
				entry = entry.strip().split(',')
			else:
				entry = [entry.strip()]
			for e in range(0,len(entry)):
				val = entry[e].strip()
				if is_number(val) ==False:
					sys.exit('   Invalid response, '+val+' is not a number')
				else:
					site = int(float(val))
					if site not in barcode_sites:
						sys.exit('   Error, '+str(site)+' is not a predicted barcode site')
					else:
						barcode_site_class[site] = 'CS'

			print("\nEnter nucleotides linked to each virus when prompted:")	
			for e in range(0,len(entry)):
				val = entry[e].strip()
				site = int(float(val))
				cat_alleles = []
				print('  Alleles at site '+str(site))
				for v in range(0,num_viral_libraries):
					library_name = viral_library_name_dict[v]
					allele = input('    '+library_name+'  -  ')
					if allele not in barcode_sites[site]:
						sys.exit('   '+allele+' is not one of the two alleles at site '+str(site)+'. Can be '+barcode_sites[site][0]+' or '+barcode_sites[site][1])+'. Exiting'
					else:
						cat_alleles.append(allele)
				cat_alleles = tuple(cat_alleles)
				viral_library_alleles[site] = cat_alleles

	else:
		clean_files_for_one_sample(accession,output_dir,trim_dir)
		print('\n\nThe incorrect number of expected barcode sites were predicted.\nExpected '+str(num_expected_barcode_sites)+", found: "+str(len(barcode_sites)))
		print("\namplicon sequence:\n\t"+expected_amplicon_seq)
		for site_num in range(0,len(temp_barcode_site_list)):
			site = temp_barcode_site_list[site_num]
			print("\t"+str(site)+"\t"+barcode_sites[site][0]+" or "+barcode_sites[site][1])
		print('\nThe predicted barcode sites will be written to the file: "amplicon_info.auto.txt"\nEither try re-running the script to re-predict barcode content using a different random sample, or edit this file manually, then rename it to "amplicon_info.txt" before re-running this script to proceed.')
		outfile = open(project_dir+"amplicon_info.auto.txt","w")
	outlines = "amplicon_seq\t"+expected_amplicon_seq+"\n"
	if categorical_sites_present == True:
		for v in range(0,num_viral_libraries):
			library_name = viral_library_name_dict[v]
			outlines += 'library_name\t'+str(v)+'\t'+library_name+'\n'
	temp_barcode_site_list = list(barcode_sites.keys())
	temp_barcode_site_list = sorted(temp_barcode_site_list)
	for site_num in range(0,len(temp_barcode_site_list)):
		site = temp_barcode_site_list[site_num]
		site_class = barcode_site_class[site]
		if site_class == 'DS':
			outlines += str(site)+'\t'+barcode_sites[site][0]+','+barcode_sites[site][1]+'\t'+site_class+'\n'
		elif site_class == 'CS':
			cat_alleles = str(viral_library_alleles[site]).replace('(','').replace(')','').replace(' ','').replace("'",'')
			outlines += str(site)+'\t'+barcode_sites[site][0]+','+barcode_sites[site][1]+'\t'+site_class+'\t'+cat_alleles+'\n'
	outfile.write(outlines)
	outfile.close()
	if accession == 'all':
		clean_files_for_one_sample(accession,output_dir,trim_dir)
		os.remove(temp_dir+'subset_all_reads.R1.fastq')
		os.remove(temp_dir+'subset_all_reads.R2.fastq')
	if expected_amplicon_seq == "" or barcode_sites == {} or continue_running == False:
		clean_files_for_one_sample(accession,output_dir,trim_dir)
		sys.exit("\n\nExiting.")

if len(viral_library_name_dict) == 0:
	categorical_sites_present = False
full_barcode_list,barcode_list_by_category = find_all_possible_barcode_combinations(barcode_sites,barcode_site_class,viral_library_alleles,viral_library_name_dict)
cat_name_list = sorted(list(barcode_list_by_category.keys()))
if len(viral_library_alleles)>0:
	categorical_sites_present = True
barcode_category_lookup_dict = {}
for c in range(0,len(cat_name_list)):
	cat_name = cat_name_list[c]
	for i in range(0,len(barcode_list_by_category[cat_name])):
		barcode_category_lookup_dict[barcode_list_by_category[cat_name][i]] = cat_name
	# print(str(cat_name)+' '+str(len(barcode_list_by_category[cat_name])))

### Trim read for all samples to summarize substitution frequency given quality
lane_prop_sub_qual_filename = output_dir+"sub_qual_info.prop.txt"
lane_count_sub_qual_filename = output_dir+"sub_qual_info.count.txt"
lane_prop_fit_qual_filename = output_dir+"sub_qual_fit.prop.txt"
if os.path.isfile(lane_prop_sub_qual_filename) == False:
	if parallel_max_cpu == 0:
		pre_trim_numcore = int(multiprocessing.cpu_count()/5)
	elif parallel_max_cpu > 0:
		pre_trim_numcore = min(parallel_max_cpu,int(multiprocessing.cpu_count()/5))
	elif parallel_max_cpu < 0:
		pre_trim_numcore = int(multiprocessing.cpu_count()/5)-parallel_max_cpu
	processed_list = []
	processed_list = Parallel(n_jobs=pre_trim_numcore)(delayed(reads_to_barcodes_one_sample)(accession,detect_barcode_content_first,True) for accession in accession_list)
	full_qual_sub_dict = {}
	qual_vals = {}
	for output_tup in processed_list:
		sampleID = output_tup[0]
		local_qual_sub_dict = output_tup[1]
		# sub_count_dict = output_tup[2]
		for backbone_nt in local_qual_sub_dict:
			for sub_nt in local_qual_sub_dict[backbone_nt]:
				for qual in local_qual_sub_dict[backbone_nt][sub_nt]:
					try:
						qual_vals[qual]
					except:
						qual_vals[qual] = ''
					try:
						full_qual_sub_dict[backbone_nt][sub_nt][qual] +=  local_qual_sub_dict[backbone_nt][sub_nt][qual]
					except:
						try:
							full_qual_sub_dict[backbone_nt][sub_nt][qual] =  local_qual_sub_dict[backbone_nt][sub_nt][qual]
						except:
							try:
								full_qual_sub_dict[backbone_nt][sub_nt] = {}
								full_qual_sub_dict[backbone_nt][sub_nt][qual] = local_qual_sub_dict[backbone_nt][sub_nt][qual]
							except:
								full_qual_sub_dict[backbone_nt] = {}
								full_qual_sub_dict[backbone_nt][sub_nt] = {}
								full_qual_sub_dict[backbone_nt][sub_nt][qual] = local_qual_sub_dict[backbone_nt][sub_nt][qual]
	qual_bins = sorted(list(qual_vals.keys()))
	count_output_string = '\t'
	prop_output_string = '\t'
	for q in range(0,len(qual_bins)):
		qual = qual_bins[q]
		count_output_string += '\t'+str(qual)
		prop_output_string += '\t'+str(qual)
	count_output_string += '\n'
	prop_output_string += '\n'
	bases = ['A','T','C','G']
	for b in range(0,len(bases)):
		full_count_total_dict = {}
		backbone_nt = bases[b]
		for n in range(0,len(bases)):
			sub_nt = bases[n]
			count_output_string += backbone_nt+'\t'+sub_nt
			for q in range(0,len(qual_bins)):
				qual = qual_bins[q]
				try:
					f_count = full_qual_sub_dict[backbone_nt][sub_nt][qual]
				except:
					f_count = 0
				count_output_string += '\t'+str(f_count)
				try:
					full_count_total_dict[qual] += f_count
				except:
					full_count_total_dict[qual] = f_count
			count_output_string += '\n'
		for n in range(0,len(bases)):
			sub_nt = bases[n]
			prop_output_string += backbone_nt+'\t'+sub_nt
			for q in range(0,len(qual_bins)):
				qual = qual_bins[q]
				try:
					f_count = full_qual_sub_dict[backbone_nt][sub_nt][qual]
				except:
					f_count = 0
				f_total = float(full_count_total_dict[qual])
				if f_total >0.0:
					f_prop = float(f_count)/f_total
				else:
					f_prop = 0.0
				prop_output_string += '\t'+str(f_prop)
			prop_output_string += '\n'

	outfile = open(lane_count_sub_qual_filename,"w")
	outfile.write(count_output_string)
	outfile.close()
	outfile = open(lane_prop_sub_qual_filename,"w")
	outfile.write(prop_output_string)
	outfile.close()
if os.path.isfile(lane_prop_fit_qual_filename) == False:
	sub_prob_dict = fit_qual_model(lane_prop_sub_qual_filename,lane_count_sub_qual_filename,lane_prop_fit_qual_filename,min_Q_score)
else:
	sub_prob_dict = load_fit_qual(lane_prop_fit_qual_filename)


### Fully process all samples
detect_barcode_content_first = False
processed_list = []
if parallel_process == True:
	processed_list = Parallel(n_jobs=num_cores)(delayed(reads_to_barcodes_one_sample)(accession,detect_barcode_content_first) for accession in accession_list)
elif parallel_process == False:
	for accession in accession_list:
		barcode_tup = reads_to_barcodes_one_sample(accession,detect_barcode_content_first)
		processed_list.append(barcode_tup)

##### create list of all possible barcodes
bases = ['A','T','C','G']

barcodes_to_exclude = {}
# barcodes_to_exclude = {'TCTATATGCG':'','TGGACTGCGG':''}
temp_full_barcode_list = full_barcode_list
full_barcode_list = []
for b in range(0,len(temp_full_barcode_list)):
	barcodeID = temp_full_barcode_list[b]
	skip = False
	try:
		barcodes_to_exclude[barcodeID]
		skip = True
	except:
		skip = False
	if skip == False:
		full_barcode_list.append(barcodeID)

##### merge all barcode info once all samples finish running
full_barcode_count_dict = {}
full_mismatch_dict = {}
full_nonbarcode_dict = {}
full_read_count = {}
nosingleton_read_count = {}
read_count_dict = {}
# full_nonbarcode_seq_list = []
full_processed_list = []

for accession_output in processed_list:
	if isinstance(accession_output,list):
		for val in accession_output:
			full_processed_list.append(val)
	else:
		full_processed_list.append(accession_output)
for accession_output in full_processed_list:
	accession = accession_output[0]
	accession_barcode_list = accession_output[2]
	full_mismatch_dict[accession] = accession_output[3]
	read_count_dict[accession] = accession_output[4]
	accession_barcode_count = 0
	accession_barcode_count_nosingle = 0
	# if categorical_sites_present == False:
	full_barcode_count_dict[accession] = accession_output[1]
	for barcodeID in full_barcode_count_dict[accession]:
		skip = False
		try:
			barcodes_to_exclude[barcodeID]
			skip = True
		except:
			skip = False
		if skip == False:
			local_count = full_barcode_count_dict[accession][barcodeID]
			if local_count > 0:
				accession_barcode_count += local_count
				if local_count > 1:
					accession_barcode_count_nosingle += local_count
	full_read_count[accession] = accession_barcode_count
	nosingleton_read_count[accession] = accession_barcode_count_nosingle

accession_list = sorted(list(full_read_count.keys()))


read_count_summary_outlines = '\traw_read_count\tpostmerge_read_count\treads_screened_for_barcodes\treads_with_likely_barcodes\tbarcodes_post-agglomeration\tnon-singelton_barcodes_post-agglomeration\t\n'
accession_list_passcount = []
read_count_plot_dict = {}
low_prop_labels = {}
for a in range(0,len(accession_list)):
	accession = accession_list[a]
	try:
		local_count = full_read_count[accession]
	except:
		local_count = 0
	if local_count>=min_barcodes_per_sample:
		accession_list_passcount.append(accession)

	read_count_tup = read_count_dict[accession]
	read_count_summary_outlines += accession
	prop_vals = []
	for i in range(0,len(read_count_tup)):
		read_count_summary_outlines += '\t'+str(read_count_tup[i])
		prop = read_count_tup[i]/read_count_tup[0]
		prop_vals.append(prop)
		if i >0:
			prev_prop = read_count_tup[i]/read_count_tup[i-1]
			prop_field = str(i)+'-'+str(round(prev_prop,3))
			try:
				num_nearby = len(low_prop_labels[prop_field])
			except:
				low_prop_labels[prop_field] = []
				num_nearby = 0
			if prev_prop <= 0.50:# and read_count_tup[0]>min_barcodes_per_sample*0.5:
				label_tup = (i,prop+0.001+0.008*num_nearby,accession+': '+str(read_count_tup[i-1])+' > '+str(read_count_tup[i]))
				low_prop_labels[prop_field].append(label_tup)
	read_count_summary_outlines += '\t'+str(full_read_count[accession])
	prop_vals.append(full_read_count[accession]/read_count_tup[0])
	read_count_plot_dict[accession] = prop_vals
	read_count_summary_outlines += '\t'+str(nosingleton_read_count[accession])
	read_count_summary_outlines += '\n'
outfile = open(output_dir+'read_count_summary.txt','w')
outfile.write(read_count_summary_outlines)
outfile.close()
read_count_plot_df = pd.DataFrame(read_count_plot_dict,index=['raw input','post-merge','screened','filter-pass','post-agglom.'])

read_count_lines = read_count_plot_df.plot.line(marker='o',lw=0.5,markersize=2,legend=False)
read_count_lines.set_ylabel('Proportion of reads remaining at each stage')
read_count_lines.set_xticks(list(range(0,5)))
read_count_lines.set_yticks(list(np.linspace(0.0,1.0, num=5)))
if len(low_prop_labels)>0:
	for prop_field in low_prop_labels:
		for label_tup in low_prop_labels[prop_field]:
			plt.text(label_tup[0], label_tup[1], label_tup[2],fontsize=4)
plt.savefig(output_dir+"read_count_summary.pdf")
plt.close()

num_discarded_from_low_readcount = len(accession_list)-len(accession_list_passcount)
accession_list = accession_list_passcount
del accession_list_passcount
if num_discarded_from_low_readcount>0:
	print("Number of samples discarded due to low read count: "+str(num_discarded_from_low_readcount))
else:
	print("No samples discarded due to low read count.")
accession_list_pass_mismatch = []

mismatch_plot_dict = {}
max_mismatch_dict = {}
high_mismatch_labels = {}
mismatch_table_outfile = open(output_dir+"all_mismatch_info.table.txt","w")
for a in range(0,len(accession_list)):
	accession = accession_list[a]
	max_mismatch_dict[accession] = 0
	mismatch_table_outfile.write("\t"+accession)
mismatch_table_outfile.write("\n")
for site in range(0,len(expected_amplicon_seq)):
	mismatch_table_outfile.write(str(site))
	for a in range(0,len(accession_list)):
		accession = accession_list[a]
		try:
			mismatch_prop = round(float(full_mismatch_dict[accession][site]),5)
		except:
			mismatch_prop = 0.0
		mismatch_table_outfile.write("\t"+str(mismatch_prop))
		try:
			mismatch_plot_dict[accession].append(mismatch_prop)
		except:
			mismatch_plot_dict[accession] = [mismatch_prop]
		if mismatch_prop > max_mismatch_dict[accession]:
			max_mismatch_dict[accession] = mismatch_prop
		if mismatch_prop > 0.05:
			prop_field = str(site)+'-'+str(round(mismatch_prop,2))
			try:
				num_nearby = len(high_mismatch_labels[prop_field])
			except:
				high_mismatch_labels[prop_field] = []
				num_nearby = 0
			label_tup = (site,mismatch_prop+0.001+0.005*num_nearby,accession+': '+str(site))
			high_mismatch_labels[prop_field].append(label_tup)
	mismatch_table_outfile.write("\n")
mismatch_table_outfile.close()

site_list = list(range(0,len(expected_amplicon_seq)))
mismatch_plot_df = pd.DataFrame(mismatch_plot_dict,index=site_list)
mismatch_lines = mismatch_plot_df.plot.line(marker='o',lw=0.5,markersize=2,legend=False)
mismatch_lines.set_ylabel('Proportion of reads with mismatch at each site')
mismatch_lines.set_xticks(list(range(0,len(expected_amplicon_seq),10)))
mismatch_lines.set_yticks(list(np.linspace(0.0,1.0, num=5)))
if len(high_mismatch_labels)>0:
	for prop_field in high_mismatch_labels:
		for label_tup in high_mismatch_labels[prop_field]:
			plt.text(label_tup[0], label_tup[1], label_tup[2],fontsize=4)
plt.savefig(output_dir+"read_mismatch_summary.pdf")
plt.close()

for accession in mismatch_plot_dict:
	if np.sum(mismatch_plot_dict[accession])<=max_prop_mismatch:
		accession_list_pass_mismatch.append(accession)
num_discarded_from_mismatch = len(accession_list)-len(accession_list_pass_mismatch)
accession_list = accession_list_pass_mismatch
del accession_list_pass_mismatch
if num_discarded_from_mismatch>0:
	print("Number of samples discarded due to high mismatch rate: "+str(num_discarded_from_mismatch))
else:
	print("No samples discarded due to high mismatch rate.")


### Write final summary output tables
print("Writing barcode output summary tables")
accession_list = sorted(accession_list)
barcode_count_dict = {}
barcode_freq_dict = {}
barcode_sum_freq_dict = {}
barcode_count_dict_nosingleton = {}
barcode_freq_dict_nosingleton = {}
table_outfile = open(output_dir+"all_barcode_count.table.txt","w")
freq_table_outfile = open(output_dir+"all_barcode_freq.table.txt","w")
for a in range(0,len(accession_list)):
	accession = accession_list[a]
	table_outfile.write("\t"+accession)
	freq_table_outfile.write("\t"+accession)
table_outfile.write("\n")
freq_table_outfile.write("\n")
for b in range(0,len(full_barcode_list)):
	barcodeID = full_barcode_list[b]
	table_outfile.write(barcodeID)
	freq_table_outfile.write(barcodeID)
	for a in range(0,len(accession_list)):
		accession = accession_list[a]
		read_count = full_read_count[accession]
		read_count_nosingleton = nosingleton_read_count[accession]
		try:
			barcode_count = full_barcode_count_dict[accession][barcodeID]
			barcode_freq = (float(barcode_count)/float(read_count))
			barcode_freq_nosingleton = (float(barcode_count)/float(read_count_nosingleton))
		except:
			barcode_count = 0
			barcode_freq = 0
			barcode_freq_nosingleton = 0
		table_outfile.write("\t"+str(barcode_count))
		freq_table_outfile.write("\t"+str(barcode_freq))
		try:
			barcode_freq_dict[accession].append(barcode_freq)
			barcode_count_dict[accession].append(barcode_count)
		except:
			barcode_freq_dict[accession] = [barcode_freq]
			barcode_count_dict[accession] = [barcode_count]
		try:
			barcode_sum_freq_dict[barcodeID] += barcode_freq
		except:
			barcode_sum_freq_dict[barcodeID] = barcode_freq
		if read_count > 1:
			try:
				barcode_freq_dict_nosingleton[accession].append(barcode_freq_nosingleton)
				barcode_count_dict_nosingleton[accession].append(barcode_count)
			except:
				barcode_freq_dict_nosingleton[accession] = [barcode_freq_nosingleton]
				barcode_count_dict_nosingleton[accession] = [barcode_count]
	table_outfile.write("\n")
	freq_table_outfile.write("\n")
table_outfile.close()
freq_table_outfile.close()


print("Calculating alpha-diversity indicies")
all_process_inputs = []
for i in range(0,len(accession_list)):
	sample1 = accession_list[i]
	count1 = barcode_count_dict[sample1]
	process_tup = (sample1,count1)#,full_barcode_list)
	all_process_inputs.append(process_tup)
processed_list = Parallel(n_jobs=num_cores)(delayed(subsample_alpha_func)(process_tup,full_barcode_list,num_subsamples) for process_tup in all_process_inputs)

alpha_subsample_infodict = {}
alpha_fit_dict = {}
alpha_fit_summary_dict = {}
for tup in processed_list:
	s1 = tup[0]
	alpha_subsample_infodict[s1] = tup[1]
	alpha_fit_dict[s1] = tup[2]
	alpha_fit_summary_dict[s1] = tup[3]

dist_function_name_list = ['rich','shannon','simpson','even','chao']
for d in range(0,len(dist_function_name_list)):
	dist_function_name = dist_function_name_list[d]
	regress_outfile_path = output_dir+"subsample_regress."+dist_function_name+".n"+str(num_subsamples)+".txt"
	outfile = open(regress_outfile_path,'w')
	outfile.close()
	string_out = ''
	for p in range(0,len(accession_list)):
		sampleID = accession_list[p]
		# for i in range(0,len(sub_size_list)):

		try:
			local_subsample_size_list = alpha_subsample_infodict[sampleID]['subsample_size']
			local_dist_list = alpha_subsample_infodict[sampleID][dist_function_name]
		except:
			local_subsample_size_list = []
			local_dist_list = []
		if local_dist_list != []:
			local_alpha_dict = {}
			for i in range(0,len(local_subsample_size_list)):
				subsample_val = local_subsample_size_list[i]
				alpha_val = local_dist_list[i]
				try:
					local_alpha_dict[subsample_val].append(alpha_val)
				except:
					local_alpha_dict[subsample_val] = [alpha_val]
			local_subsample_size_set = list(local_alpha_dict.keys())
			for i in range(0,len(local_subsample_size_set)):
					subsample_size = local_subsample_size_set[i]
					string_out += sampleID+'\t'+str(subsample_size)
					for j in range(0,len(local_alpha_dict[subsample_size])):
						string_out += '\t'+str(local_alpha_dict[subsample_size][j])
					string_out += '\n'
			if len(string_out)>1e4:
				outfile = open(regress_outfile_path,'a')
				outfile.write(string_out)
				outfile.close()
				string_out = ''
	if len(string_out)>0:
		outfile = open(regress_outfile_path,'a')
		outfile.write(string_out)
		outfile.close()
		string_out = ''

alpha_div_outfile = open(output_dir+"all_barcode.alpha_diversity.txt","w")
alpha_div_outfile.write("sampleID\tshannon_div\tsimpson_div\tchao_richness\tshannon_evenness\tobs_richness\n")
# alpha_div_outfile.close()
stat_summary_outlines = '#sampleID\tdiversity_function\tvalue_inferred\tloss_value\tnum_subsample_depths\tfit_a\tfit_b\tfit_c\n'
for a in range(0,len(accession_list)):
	sampleID = accession_list[a]
	try:
		sample_fit_dict = alpha_fit_dict[sampleID]
	except:
		sample_fit_dict = {}
	if sample_fit_dict != {}:
		H_shannon = sample_fit_dict['shannon']
		H_simpson = sample_fit_dict['simpson']
		E_shannon = sample_fit_dict['even']
		R_chao = sample_fit_dict['chao']
		R_rich = sample_fit_dict['rich']

		H_shannon_stat = alpha_fit_summary_dict[sampleID]['shannon']
		H_simpson_stat = alpha_fit_summary_dict[sampleID]['simpson']
		E_shannon_stat = alpha_fit_summary_dict[sampleID]['even']
		R_chao_stat = alpha_fit_summary_dict[sampleID]['chao']
		R_rich_stat = alpha_fit_summary_dict[sampleID]['rich']
		stat_summary_outlines += sampleID+'\tshannon\t'+H_shannon_stat+'\n'
		stat_summary_outlines += sampleID+'\tsimpson\t'+H_simpson_stat+'\n'
		stat_summary_outlines += sampleID+'\teven\t'+E_shannon_stat+'\n'
		stat_summary_outlines += sampleID+'\tchao\t'+R_chao_stat+'\n'
		stat_summary_outlines += sampleID+'\trich\t'+R_rich_stat+'\n'

	else:
		H_shannon,E_shannon = shannon_alpha(barcode_freq_dict_nosingleton[sampleID])
		H_simpson = simpson_alpha(barcode_freq_dict_nosingleton[sampleID])
		R_chao = chao_1_richness(barcode_count_dict_nosingleton[sampleID],1) #value of 2 to not count singletons
		R_rich = true_richness(barcode_count_dict_nosingleton[sampleID],1) #value of 2 to not count singletons
	alpha_div_outfile.write(sampleID+"\t"+str(H_shannon)+"\t"+str(H_simpson)+"\t"+str(R_chao)+"\t"+str(E_shannon)+"\t"+str(R_rich)+"\n")
alpha_div_outfile.close()
alpha_stat_outfile = open(output_dir+'alpha_fit_stat_summary.txt','w')
alpha_stat_outfile.write(stat_summary_outlines)
alpha_stat_outfile.close()

print("Calculating beta-diversity indicies")
all_process_inputs = []
for i in range(0,len(accession_list)):
	sample1 = accession_list[i]
	for j in range(i,len(accession_list)):
		sample2 = accession_list[j]
		count1 = barcode_count_dict[sample1]
		count2 = barcode_count_dict[sample2]
		process_tup = ((sample1,count1),(sample2,count2))
		all_process_inputs.append(process_tup)

processed_list = Parallel(n_jobs=num_cores)(delayed(subsample_beta_func)(process_tup,full_barcode_list,num_beta_subsamples) for process_tup in all_process_inputs)
beta_subsample_infodict = {}
beta_fit_dict = {}
beta_fit_summary_dict = {}
for tup in processed_list:
	s1 = tup[0]
	s2 = tup[1]
	beta_subsample_infodict[s1+s2] = tup[2]
	beta_fit_dict[s1+s2] = tup[3]
	beta_fit_summary_dict[s1+s2] = tup[4]

dist_function_name_list = ['bray','bray_log','jaccard']
for d in range(0,len(dist_function_name_list)):
	dist_function_name = dist_function_name_list[d]
	regress_outfile_path = output_dir+"subsample_regress."+dist_function_name+".n"+str(num_subsamples)+".txt"
	outfile = open(regress_outfile_path,'w')
	outfile.close()
	string_out = ''
	for p in range(0,len(accession_list)):
		sample1 = accession_list[p]
		for q in range(p,len(accession_list)):
			sample2 = accession_list[q]
			try:
				local_sub_size_list = beta_subsample_infodict[sample1+sample2]['subsample_size']
				local_dist_list = beta_subsample_infodict[sample1+sample2][dist_function_name]
			except:
				local_sub_size_list = []
				local_dist_list = []
			if len(local_dist_list)>0:
				local_dist_dict = {}
				for i in range(0,len(local_sub_size_list)):
					subsample_size = local_sub_size_list[i]
					dist_val = local_dist_list[i]
					try:
						local_dist_dict[subsample_size].append(dist_val)
					except:
						local_dist_dict[subsample_size] = [dist_val]
				local_subsample_size_set = list(local_dist_dict.keys())
				for i in range(0,len(local_subsample_size_set)):
					subsample_size = local_subsample_size_set[i]
					string_out += sample1+'\t'+sample2+'\t'+str(subsample_size)
					for j in range(0,len(local_dist_dict[subsample_size])):
						string_out += '\t'+str(local_dist_dict[subsample_size][j])
					string_out += '\n'
			if len(string_out)>1e4:
				outfile = open(regress_outfile_path,'a')
				outfile.write(string_out)
				outfile.close()
				string_out = ''
	if len(string_out)>0:
		outfile = open(regress_outfile_path,'a')
		outfile.write(string_out)
		outfile.close()
		string_out = ''

func_outfile_dict = {'bray':'bray-curtis','bray_log':'log_bray-curtis','jaccard':'jaccard'}
for d in range(0,len(dist_function_name_list)):
	dist_function_name = dist_function_name_list[d]
	function_outfilename = func_outfile_dict[dist_function_name]
	
	beta_outfile_name = output_dir+"all_barcode."+function_outfilename+"_dissimilarity.txt"
	beta_outfile = open(beta_outfile_name,"w")
	beta_outfile.close()
	outlines = ''
	for j in range(0,len(accession_list)):
		outlines += "\t"+accession_list[j]
	outlines += "\n"

	beta_stat_summary_outfile_name = output_dir+'fit_stat_summary.'+function_outfilename+".txt"
	beta_stat_outfile = open(beta_stat_summary_outfile_name,"w")
	beta_stat_outfile.close()
	
	stat_summary_outlines = '#sampleID_1\tsampleID_2\tvalue_inferred\tloss_value\tnum_subsample_depths\tfit_a\tfit_b\tfit_c\n'
	for i in range(0,len(accession_list)):
		outlines += accession_list[i]
		for j in range(0,len(accession_list)):
			accession1 = accession_list[i]
			accession2 = accession_list[j]
			try:
				dist = beta_fit_dict[accession1+accession2][dist_function_name]
			except:
				try:
					dist = beta_fit_dict[accession2+accession1][dist_function_name]
				except:
					dist = 'nan'
			if dist != 'nan' and dist != 'na':
				try:
					beta_fit_summary_string = beta_fit_summary_dict[accession1+accession2][dist_function_name]
				except:
					beta_fit_summary_string = ''
				if beta_fit_summary_string != '':
					stat_summary_outlines += accession1+'\t'+accession2+'\t'+beta_fit_summary_string+'\n'
			outlines += "\t"+str(dist)
		outlines += "\n"
		if len(outlines) > 1e4:
			beta_outfile = open(beta_outfile_name,"a")
			beta_outfile.write(outlines)
			beta_outfile.close()
			outlines = ''
		if len(stat_summary_outlines) > 1e4:
			beta_stat_outfile = open(beta_stat_summary_outfile_name,"a")
			beta_stat_outfile.write(stat_summary_outlines)
			beta_stat_outfile.close()
			stat_summary_outlines = ''
	if len(outlines) > 0:
		beta_outfile = open(beta_outfile_name,"a")
		beta_outfile.write(outlines)
		beta_outfile.close()
		outlines = ''
	if len(stat_summary_outlines) > 0:
		beta_stat_outfile = open(beta_stat_summary_outfile_name,"a")
		beta_stat_outfile.write(stat_summary_outlines)
		beta_stat_outfile.close()
		stat_summary_outlines = ''

####################################### Plot stacked bar plots #######################################
#make color palette
colorblind_friendly_palette = ['#E8ECFB', '#D9CCE3', '#D1BBD7', '#CAACCB', '#BA8DB4', '#AE76A3', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#437DBF', '#5289C7', '#6195CF', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F6C141', '#F4A736', '#F1932D', '#EE8026', '#E8601C', '#E65518', '#DC050C', '#A5170E', '#72190E', '#42150A']
used_barcodes = {}
barcode_freq_rank_list = []
for barcodeID in barcode_sum_freq_dict:
	freq_sum = barcode_sum_freq_dict[barcodeID]
	freq_tup = (freq_sum,barcodeID)
	barcode_freq_rank_list.append(freq_tup)
	used_barcodes[barcodeID] = ''
for barcodeID in full_barcode_list:
	try:
		used_barcodes[barcodeID]
	except:
		freq_tup = (0.,barcodeID)
		barcode_freq_rank_list.append(freq_tup)
del used_barcodes
barcode_freq_rank_list = sorted(barcode_freq_rank_list,reverse=True)
barcode_color_dict = {}
polarity = -1
for num in range(0,len(barcode_freq_rank_list)):
	if polarity == 1:
		polarity = -1
	else:
		polarity = 1
	barcodeID = barcode_freq_rank_list[num][1]
	color_index = polarity*num % len(colorblind_friendly_palette)
	color_hex_value = colorblind_friendly_palette[color_index]
	barcode_color_dict[barcodeID] = color_hex_value
full_barcode_color_list = []
barcode_outfile = open("barcode_color_map.txt","w")
for i in range(0,len(full_barcode_list)):
	barcodeID = full_barcode_list[i]
	color_hex_value = barcode_color_dict[barcodeID]
	full_barcode_color_list.append(color_hex_value)
	barcode_outfile.write(barcodeID+"\t"+color_hex_value+"\n")
barcode_outfile.close()

if generate_stacked_bar_plots == True:
	print("Creating stacked bar plots")
	titer_dict = {}
	if os.path.isfile(titer_filename) == True:
		titer_list = []
		titer_infile = open(titer_filename,"r")
		for line in titer_infile:
			line = line.strip()
			if len(line)>0:
				if line[0] !='#':
					line = line.split("\t")
					if is_number(line[1]) == True:
						accession = line[0]
						titer = float(line[1])
						titer_dict[accession] = titer
						titer_list.append(titer)
		titer_infile.close()
		titer_list = np.array(titer_list)
		max_obs_titer = np.max(titer_list)
		min_obs_titer = np.min(np.where(titer_list==0,999,titer_list))
		if max_obs_titer/min_obs_titer > 100:
			print("Sample do not appear to be log transformed. Log10 transforming values now.")
			log_titer_dict = {}
			for accession in titer_dict:
				log_titer_dict[accession] = -1*np.log10(titer_dict[accession])
			titer_dict = log_titer_dict
			del log_titer_dict
			del titer_list

	#make the plot
	bars_per_row = len(accession_list)
	width_per_bar = 0.4
	num_col = 1
	num_row = 1

	page_width = max(3.,num_col*(width_per_bar*bars_per_row))
	fig, axs = plt.subplots(num_row, num_col,sharex=True, sharey=True)##num_row, 
	fig.set_size_inches(page_width, num_row*7)

	plot_freq_array = []
	max_mismatch_array = []
	titer_not_found = []
	df_key = ['Sample']
	df_key.extend(full_barcode_list)
	max_titer = 0
	for a in range(0,len(accession_list)):
		accession = accession_list[a]
		try:
			max_mismatch = max_mismatch_dict[accession]
		except:
			max_mismatch = 0
		max_mismatch_array.append(max_mismatch)
		try:
			accession_titer = titer_dict[accession]
		except:
			accession_titer = 1.0
			titer_not_found.append(accession)
		if accession_titer > max_titer:
			max_titer = accession_titer
		barcode_freq_array = barcode_freq_dict[accession]
		### Make scaled bar plots with barcodes scaled to sample titers
		titer_adjusted_freq_list = [accession]
		for i in range(0,len(full_barcode_list)):
			try:
				titer_adjusted_freq = barcode_freq_array[i]*accession_titer
			except:
				titer_adjusted_freq = 0.0
			titer_adjusted_freq_list.append(titer_adjusted_freq)
		plot_freq_array.append(titer_adjusted_freq_list)

	if max_titer == 1.0:
		y_axis_label = 'Proportion of viral population'
		print("\tStacked bar plots will be scaled to 1.0 - add file with viral titers to scale plots to viral titer instead.")
	else:
		y_axis_label = "Viral titer"
		if len(titer_not_found)>0:
			print("\nUnable to pair the following samples with a titer:")
			for t in range(0,len(titer_not_found)):
				print('\t'+titer_not_found[t])
	
	barcode_df = pd.DataFrame(plot_freq_array,columns=df_key)
	print('\tPlotting. Note: this can take a long time to complete.')
	bar_plot = barcode_df.plot(x='Sample', kind='bar', stacked=True,legend=False,color=full_barcode_color_list,ax=axs,width=0.92, ylim=(0, max_titer*1.05),ylabel=y_axis_label)
	if max_titer == 1.0:
		axs.set_yticks([0.0,0.25,0.5,0.75,1.0])
	mismatch_axis = axs.twinx()
	mismatch_axis.plot(max_mismatch_array,color='black',marker='o', markersize=3, mfc='grey',linestyle='None')#linewidth=0.5, 
	mismatch_axis.set_ylim(0.0,max(0.55,round(max(max_mismatch_array),2)))
	if max(max_mismatch_array)<=0.5:
		mismatch_axis.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
	else:
		mismatch_axis.set_yticks(list(np.linspace(0.0, round(max(max_mismatch_array),2), num=11)))
	mismatch_axis.set_ylabel("Max proportion of reads discarded")
	plt.tight_layout()
	if max_titer == 1.0:
		plt.savefig(output_dir+"all_barcode.stacked_bars.pdf")
	else:
		plt.savefig(output_dir+"all_barcode.stacked_bars.log10_freq.pdf")
	plt.close()
print("Processing Complete")
