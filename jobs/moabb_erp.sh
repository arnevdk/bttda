#!/bin/bash

datasets=(
	#"BNCI2014-008" 
	#"BNCI2015-003" 
	#"BrainInvaders2012" 
	#"BrainInvaders2014a" 
	#"BrainInvaders2014b" 
	#"BrainInvaders2015b" 

	#"Cattan2019-VR" 
	#"ErpCore2021-ERN" 
	#"ErpCore2021-LRP" 
	#"ErpCore2021-MMN" 
	#"ErpCore2021-N170" 
	#"ErpCore2021-N2pc" 
	"ErpCore2021-N400" 
	"ErpCore2021-P3" 
	"Huebner2017" 
	#"Huebner2018" 
	#"Lee2019-ERP" 
	#"Sosulski2019" 

	#"EPFLP300" 
	#"BNCI2014-009" 
	#"BrainInvaders2013a" 
	#"BrainInvaders2015a" 

	
	#"DemonsP300" 
)


for dataset in "${datasets[@]}"; do
	echo "$dataset"
	subjects=$(scripts/get_subjects.py "$dataset")
	for subject in $subjects; do
		sbatch jobs/moabb_erp.slurm $dataset $subject;
	done
done

