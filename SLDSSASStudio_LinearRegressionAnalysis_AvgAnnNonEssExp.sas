/*
 *
 * Task code generated by SAS Studio 3.8 
 *
 * Generated on '4/5/23, 8:20 PM' 
 * Generated by 'u58127406' 
 * Generated on server 'ODAWS03-USW2.ODA.SAS.COM' 
 * Generated on SAS platform 'Linux LIN X64 3.10.0-1062.9.1.el7.x86_64' 
 * Generated on SAS version '9.04.01M7P08062020' 
 * Generated on browser 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36' 
 * Generated on web client 'https://odamid-usw2.oda.sas.com/SASStudio/main?locale=en_US&zone=GMT-04%253A00&ticket=ST-110855-Qi1nRY6Mlfix6YwNVb7R-cas' 
 *
 */

ods noproctitle;
ods graphics / imagemap=on;

proc glmselect data=WORK.STUDENTLOANS outdesign(addinputvars)=Work.reg_design 
		plots=(criterionpanel);
	class SEX MAR_STAT 'IND_ANN_INC ($)'n 'BACH_DEG?'n 'CURR_STUDENT?'n 
		'CHILDREN?'n 'PETS?'n 'TRAV_STATE?'n 'TRAV_US?'n 'MOV?'n 'HOB?'n 'DINEOUT?'n 
		AVG_WEEKLY_ORDERIN ANN_EVENTS_HOST_Q 'MON_ESS_EXP ($)'n GEN 'HSD_GED?'n 
		'OWN_CAR?'n HOUSING AVG_MON_HOB AVG_WEEKLY_DINEOUT ANN_EVENTS_ATT_Q 
		'HIGHER_ED?'n AGE 'MULTI_GEN?'n 'VACATION?'n TRAV_STATE_Q TRAV_US_Q 
		'ORDERIN?'n 'DONATIONS?'n STUDENTLOANS_BIN / param=glm;
	model 'AVG_ANN_NONESS_EXP ($)'n='MON_ESS_EXP_MIN ($)'n 'MON_ESS_EXP_MAX ($)'n 
		MON_DTIRAT 'TOT_UGG_SLD_AVG ($)'n ANN_SLD_INC_DTIRAT IND_ANN_INC_LOW 
		IND_ANN_INC_HI SEX MAR_STAT 'IND_ANN_INC ($)'n 'BACH_DEG?'n 'CURR_STUDENT?'n 
		'CHILDREN?'n 'PETS?'n 'TRAV_STATE?'n 'TRAV_US?'n 'MOV?'n 'HOB?'n 'DINEOUT?'n 
		AVG_WEEKLY_ORDERIN ANN_EVENTS_HOST_Q 'MON_ESS_EXP ($)'n GEN 'HSD_GED?'n 
		'OWN_CAR?'n HOUSING AVG_MON_HOB AVG_WEEKLY_DINEOUT ANN_EVENTS_ATT_Q 
		'HIGHER_ED?'n AGE 'MULTI_GEN?'n 'VACATION?'n TRAV_STATE_Q TRAV_US_Q 
		'ORDERIN?'n 'DONATIONS?'n IND_ANN_INC_AVG IND_MON_INC_AVG MON_SLD_INC_DTIRAT 
		ANN_SLD_INC_DTIRAT_PRCNT 'MON_ESS_EXP_AVG ($)'n STUDENTLOANS_BIN / 
		showpvalues selection=backward
    
   (select=adjrsq stop=adjrsq choose=adjrsq);
run;

proc reg data=Work.reg_design alpha=0.05 plots(only 
		maxpoints=none)=(diagnostics residuals observedbypredicted);
	where SEX is not missing and MAR_STAT is not missing and 'IND_ANN_INC ($)'n is 
		not missing and 'BACH_DEG?'n is not missing and 'CURR_STUDENT?'n is not 
		missing and 'CHILDREN?'n is not missing and 'PETS?'n is not missing and 
		'TRAV_STATE?'n is not missing and 'TRAV_US?'n is not missing and 'MOV?'n is 
		not missing and 'HOB?'n is not missing and 'DINEOUT?'n is not missing and 
		AVG_WEEKLY_ORDERIN is not missing and ANN_EVENTS_HOST_Q is not missing and 
		'MON_ESS_EXP ($)'n is not missing and GEN is not missing and 'HSD_GED?'n is 
		not missing and 'OWN_CAR?'n is not missing and HOUSING is not missing and 
		AVG_MON_HOB is not missing and AVG_WEEKLY_DINEOUT is not missing and 
		ANN_EVENTS_ATT_Q is not missing and 'HIGHER_ED?'n is not missing and AGE is 
		not missing and 'MULTI_GEN?'n is not missing and 'VACATION?'n is not missing 
		and TRAV_STATE_Q is not missing and TRAV_US_Q is not missing and 'ORDERIN?'n 
		is not missing and 'DONATIONS?'n is not missing and STUDENTLOANS_BIN is not 
		missing;
	ods select DiagnosticsPanel ResidualPlot ObservedByPredicted;
	model 'AVG_ANN_NONESS_EXP ($)'n=&_GLSMOD /;
	run;
quit;

proc delete data=Work.reg_design;
run;