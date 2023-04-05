/*
 *
 * Task code generated by SAS Studio 3.8 
 *
 * Generated on '4/5/23, 4:06 PM' 
 * Generated by 'u58127406' 
 * Generated on server 'ODAWS03-USW2.ODA.SAS.COM' 
 * Generated on SAS platform 'Linux LIN X64 3.10.0-1062.9.1.el7.x86_64' 
 * Generated on SAS version '9.04.01M7P08062020' 
 * Generated on browser 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36' 
 * Generated on web client 'https://odamid-usw2.oda.sas.com/SASStudio/main?locale=en_US&zone=GMT-04%253A00&ticket=ST-103533-aaXErDfHtUJK42ZHVdsf-cas' 
 *
 */

ods graphics / reset width=12in height=12in imagemap;

proc sort data=WORK.STUDENTLOANS out=_BarChartTaskData;
	by STUDENTLOANS_BIN;
run;

proc sgplot data=_BarChartTaskData;
	by STUDENTLOANS_BIN;
	title height=14pt "Average Monthly Essential Expenses Distribution in America for People With/Out          Student Loans, By Generation (2023)";
	footnote2 justify=left height=12pt 
		"0 = Do Not Have Student Loans, 1 = Have Student Loans";
	vbar 'MON_ESS_EXP_AVG ($)'n / group=GEN groupdisplay=cluster 
		fillattrs=(transparency=0.5) dataskin=matte;
	xaxis label="Average Monthly Essential Expenses ($)";
	yaxis grid;
run;

ods graphics / reset;
title;
footnote2;

proc datasets library=WORK noprint;
	delete _BarChartTaskData;
	run;