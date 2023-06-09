/*
 *
 * Task code generated by SAS Studio 3.8 
 *
 * Generated on '4/5/23, 6:44 PM' 
 * Generated by 'u58127406' 
 * Generated on server 'ODAWS03-USW2.ODA.SAS.COM' 
 * Generated on SAS platform 'Linux LIN X64 3.10.0-1062.9.1.el7.x86_64' 
 * Generated on SAS version '9.04.01M7P08062020' 
 * Generated on browser 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36' 
 * Generated on web client 'https://odamid-usw2.oda.sas.com/SASStudio/main?locale=en_US&zone=GMT-04%253A00&ticket=ST-103533-aaXErDfHtUJK42ZHVdsf-cas' 
 *
 */

ods noproctitle;
ods graphics / imagemap=on;

proc sort data=WORK.STUDENTLOANS out=Work.SortTempTableSorted;
	by STUDENTLOANS_BIN;
run;

/* Exploring Data */
proc univariate data=Work.SortTempTableSorted;
	ods select Histogram;
	var SEX_BIN GEN_CLASS MAR_STAT_CLASS IND_ANN_INC_LOW IND_ANN_INC_HI 
		IND_ANN_INC_AVG IND_MON_INC_AVG HSD_GED_BIN BACH_DEG_BIN HIGHER_ED_BIN 
		BACH_HIGHER_ED_BIN CURR_STUDENT_BIN OWN_CAR_BIN HOUSING_CLASS HOUSING_BIN 
		MULTI_GEN_BIN CHILDREN_BIN PETS_BIN VACATION_BIN TRAV_STATE_BIN TRAV_US_BIN 
		MOV_BIN HOB_BIN DINEOUT_BIN ORDERIN_BIN DONATIONS_BIN 'MON_ESS_EXP_MIN ($)'n 
		'MON_ESS_EXP_MAX ($)'n 'MON_ESS_EXP_AVG ($)'n MON_DTIRAT 
		'AVG_ANN_NONESS_EXP ($)'n 'AVG_MON_NONESS_EXP ($)'n 
		'AVG_MON_ESS_NONESS_EXP ($)'n AVG_MON_DTI_RAT AVG_MON_DTI_RAT_PRCNT 
		'TOT_UGG_SLD_BCS ($)'n 'TOT_UGG_SLD_WCS ($)'n 'TOT_UGG_SLD_AVG ($)'n 
		ANN_SLD_INC_DTIRAT MON_SLD_INC_DTIRAT ANN_SLD_INC_DTIRAT_PRCNT;
	histogram SEX_BIN GEN_CLASS MAR_STAT_CLASS IND_ANN_INC_LOW IND_ANN_INC_HI 
		IND_ANN_INC_AVG IND_MON_INC_AVG HSD_GED_BIN BACH_DEG_BIN HIGHER_ED_BIN 
		BACH_HIGHER_ED_BIN CURR_STUDENT_BIN OWN_CAR_BIN HOUSING_CLASS HOUSING_BIN 
		MULTI_GEN_BIN CHILDREN_BIN PETS_BIN VACATION_BIN TRAV_STATE_BIN TRAV_US_BIN 
		MOV_BIN HOB_BIN DINEOUT_BIN ORDERIN_BIN DONATIONS_BIN 'MON_ESS_EXP_MIN ($)'n 
		'MON_ESS_EXP_MAX ($)'n 'MON_ESS_EXP_AVG ($)'n MON_DTIRAT 
		'AVG_ANN_NONESS_EXP ($)'n 'AVG_MON_NONESS_EXP ($)'n 
		'AVG_MON_ESS_NONESS_EXP ($)'n AVG_MON_DTI_RAT AVG_MON_DTI_RAT_PRCNT 
		'TOT_UGG_SLD_BCS ($)'n 'TOT_UGG_SLD_WCS ($)'n 'TOT_UGG_SLD_AVG ($)'n 
		ANN_SLD_INC_DTIRAT MON_SLD_INC_DTIRAT ANN_SLD_INC_DTIRAT_PRCNT / normal;
	by STUDENTLOANS_BIN;
run;

proc delete data=Work.SortTempTableSorted;
run;