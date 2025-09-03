#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#ifndef ROOTHEADERS
#define ROOTHEADERS
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TH1F.h"
#include "TGraph.h"
#endif
#include "misc.h"
#include "tanh_fit.h"
using namespace std;

//#define NOFIT
int main(int argc,char *argv[])
{
        FILE *stream;   //#文件流
        char filename_input[400] = "NoName";   //#输入文件名
	char filename_base[400] = "NoName";    //#输出文件名
        char run_filename[400];                //#输入文件名
        char root_filename[400];               //#输出文件名
	char misc[200] = " ";                  //#杂项
        unsigned int RUN_Start_NUMBER = 0;     //#运行开始编号
        unsigned int RUN_End_NUMBER = 1;       //#运行结束编号
        if (argc>1) { strcpy(filename_input, argv[1]);}   //#输入文件名
        if (argc>2) { RUN_Start_NUMBER = atoi(argv[2]);}  //#运行开始编号
        if (argc>3) { RUN_End_NUMBER = atoi(argv[3]);}    //#运行结束编号
        if (argc>4) { strcpy(misc, argv[4]);}             //#杂项
	Get_Name(filename_base, filename_input);          //#获取输出文件名
        char date[8];                                     //#日期
        sprintf(date,"%.8s",filename_base);               //#获取日期
        char read_path[200];                              //#读取路径
        sprintf(read_path,"/st0/share/data/raw_data/CEvNS_DZL_test_sanmen/20250516/");
        char save_path[1024];                             //#保存路径
        sprintf(save_path,"/st0/home/wanhf/Data");        //#保存路径

        //# 修改文件夹
        const unsigned int CHANNEL_NUMBER = 16;		//#通道数
        const unsigned int EVENT_NUMBER = 10000;	//#一个bin存的示例
        const unsigned int MAX_WINDOWS = 30000;		//#时间窗  120μs
        const unsigned int PED_RANGE = 1000;		//#取了前1000个点作基线，可以改
        unsigned int ped_range = PED_RANGE;             //#基线范围
        const UInt_t PQNum = MAX_WINDOWS/100;           //#快放拟合的积分范围，100个点一个bin
	const unsigned long int FADCTTT_FULL_COUNT = 0x80000000;//#转码，不管
        const unsigned int THRESHOLD_AC = 1450; 	//#阈值，不管

	// 对于快放作拟合，使用反三角正切。
	const unsigned int FIT_RANGE = 5500;            //#快放拟合的拟合范围
        const unsigned int fit_start = 6000;            //#快放拟合的开始时间
	const unsigned int PED_RANGE_FP = 1000;         //#快放拟合的基线范围
        const unsigned int plastic_timewindow = 200;	//#塑料闪烁体时间窗
	const unsigned int TRY_NUMBER = 10;		//#快放拟合的尝试次数
	// partial Q range(fixed range) 电荷量，需要改
	const unsigned int Qp0_RANDN[TRY_NUMBER] = {3800,3700,3600,3500,4000,4000,4000,3600,3700,3700}; //#快放拟合的开始时间
        const unsigned int Qp0_RANUP[TRY_NUMBER] = {6700,6700,6700,6700,6700,6600,6900,6900,6900,7000}; //#快放拟合的结束时间
        const unsigned int Qp1_RANDN[TRY_NUMBER] = {4300,4400,4500,4600,4600,4600,4700,4800,4900,5000}; //#快放拟合的开始时间
        const unsigned int Qp1_RANUP[TRY_NUMBER] = {8000,8500,8600,8700,8800,8900,9000,9000,9000,9000}; //#快放拟合的结束时间
	// float range 也是积分量
	const unsigned int Qp_RANDN[CHANNEL_NUMBER] = { 900,1800, 500,3000,4990,4990,0,0,0,0,0,0,0,0}; //#快放拟合的开始时间
        const unsigned int Qp_RANUP[CHANNEL_NUMBER] = {1500,3000,4500,5500,5005,5005,0,0,0,0,0,0,0,0}; //#快放拟合的结束时间

        // Run Header
        Double_t pstt = 0.;          //Program Start Time
        UInt_t FiredD = 0;           //Fired Devices
        UInt_t V1725_1_DAC[8] = {0}; //V1725-1 Channel DAC
        UInt_t V1725_1_Tg[8] = {0};  //V1725-1 Trigger Settings
        UInt_t V1725_1_twd = 0;      //V1725-1 Time Window
        UInt_t V1725_1_pretg = 0;    //V1725-1 Pre Trigger
        UInt_t V1725_1_opch = 0;     //V1725-1 Opened Channel
        UInt_t V1725_2_DAC[8] = {0}; //V1725-1 Channel DAC     
        UInt_t V1725_2_Tg[8] = {0};  //V1725-1 Trigger Settings
        UInt_t V1725_2_twd = 0;      //V1725-1 Time Window     
        UInt_t V1725_2_pretg = 0;    //V1725-1 Pre Trigger     
        UInt_t V1725_2_opch = 0;     //V1725-1 Opened Channel  
        UInt_t V1721_DAC[8] = {0};   //V1725-1 Channel DAC     
        UInt_t V1721_Tg[8] = {0};    //V1725-1 Trigger Settings
        UInt_t V1721_twd = 0;        //V1725-1 Time Window     
        UInt_t V1721_pretg = 0;      //V1725-1 Pre Trigger     
        UInt_t V1721_opch = 0;       //V1725-1 Opened Channel  
        UInt_t V1729_th_DAC = 0;     //V1729 Threshold DAC
        UInt_t V1729_posttg = 0;     //V1729 Post Trigger
        UInt_t V1729_tgtype = 0;     //V1729 Trigger Type
        UInt_t V1729_opch = 0;       //V1729 Opened Channel
        Double_t rstt = 0.;          //Run Start Time
        Double_t redt = 0.;          //Run End Time

        // Event Header
        UInt_t Hit_pat = 0;           //#触发模式
        UInt_t V1729_tg_rec = 0;      //#V1729触发记录
        UInt_t Evt_deadtime = 0;      //#事件死时间
        UInt_t Evt_starttime = 0;     //#事件开始时间
        UInt_t Evt_endtime = 0;       //#事件结束时间
        UInt_t V1725_1_tgno = 0;      //#V1725-1触发记录
        UInt_t V1725_2_tgno = 0;      //#V1725-2触发记录
        UInt_t V1721_tgno = 0;        //#V1721触发记录
        UInt_t V1725_1_tag = 0;       //#V1725-1触发记录
	ULong64_t V1725OverFlowCount = 0;//#V1725溢出计数
        ULong64_t TTTV1725 = 0;        //#V1725时间

        //Data Pulses
        UShort_t V1725_1_pulse[CHANNEL_NUMBER][MAX_WINDOWS]; //#V1725-1数据

        //tree variables
        UInt_t idevt;                //#事件编号
        UInt_t trig;                 //#触发记录
        Float_t time;                //#时间
        Float_t deadtime;            //#死时间
        Float_t ped[CHANNEL_NUMBER]; //#基线
        Float_t pedt[CHANNEL_NUMBER];//#后沿基线
        UInt_t q[CHANNEL_NUMBER];    //#全时间窗的积分值
        UInt_t max[CHANNEL_NUMBER];  //#最大值
        UInt_t maxpt[CHANNEL_NUMBER];//#最大值时间
        UInt_t min[CHANNEL_NUMBER];  //#最小值
        UInt_t minpt[CHANNEL_NUMBER];//#最小值时间
	UInt_t time_AC;   // 2013.11.14  timing of NaI cross the Threshold
        Long64_t tb[CHANNEL_NUMBER];//#时间窗
        Float_t rms[CHANNEL_NUMBER];//#均方根

        ULong64_t TimeV1725 = 0;        //#V1725时间，用于触发时间读取
        ULong64_t SystemEndTime = 0;    //#系统结束时间

        UInt_t Qp[CHANNEL_NUMBER];      //#积分值，用于快放拟合
		UInt_t Qp0[TRY_NUMBER]; //#快放拟合的开始时间
        UInt_t Qp1[TRY_NUMBER]; //#快放拟合的结束时间
        UInt_t QPartV1725_1[CHANNEL_NUMBER][PQNum] = {}; //#V1725-1积分值

        Float_t Ped_rms[CHANNEL_NUMBER]; //#基线均方根

        #ifndef NOFIT
        float fit_pulse[MAX_WINDOWS]; //#快放拟合的脉冲
        Float_t chi2; //#快放拟合的chi2
        Float_t famp; //#快放拟合的幅度
        Float_t eamp; //#快放拟合的误差
        Float_t fped; //#快放拟合的基线
        Float_t eped; //#快放拟合的误差
        Float_t fcross; //#快放拟合的交叉
        Float_t ecross; //#快放拟合的误差
        Float_t fslope; //#快放拟合的斜率
        Float_t eslope; //#快放拟合的误差
        Float_t fmid; //#快放拟合的中点
        UInt_t  fit_flag; //#快放拟合的标志
        int fit_mid; //#快放拟合的中点
        TGraph *g_fit; //#快放拟合的图
        TF1 *fit_func; //#快放拟合的函数
        float n_bin[FIT_RANGE]; //#快放拟合的bin

        fstream ofs;                                    //#输出文件流
        ofs.open("20250516_DZL_pulse_RT.txt",ios::out); //#输出文件名
        int numofpulse = 0;                             //#脉冲数

        Float_t fcross_upper;         //#快放拟合的交叉上界
		Float_t fcross_lower; //#快放拟合的交叉下界
        #endif

        for(unsigned int i=RUN_Start_NUMBER;i<RUN_End_NUMBER+1;i++)//Run
           {
                //runtime0 = time(NULL);                //#运行开始时间
                idevt = EVENT_NUMBER*i;                 //#事件编号
                sprintf(run_filename, "%s%sFADC_RAW_Data_%d.bin",read_path,filename_input, i);
                stream=fopen(run_filename,"rb"); //#打开文件
                printf("opening %s\n",run_filename); //#打印文件名
                sprintf(root_filename, "%s/%s_%d.root", save_path, filename_base, i); //#生成root文件名
                TFile *a = new TFile(root_filename, "recreate"); //#创建root文件
                TTree *t1 = new TTree("t1", "tree"); //#创建tree
                t1->Branch("idevt", &idevt, "idevt/i"); //#事件编号
                t1->Branch("trig", &trig, "trig/i"); //#触发记录
                t1->Branch("time", &time, "time/F");				//#板载时间，相对时间，相对于RAND START TIME
                t1->Branch("deadtime", &deadtime, "deadtime/F");                //#死时间
		t1->Branch("TimeV1725", &TimeV1725, "TimeV1725/l");             //#V1725时间
		t1->Branch("TTTV1725", &TTTV1725, "TTTV1725/l");		//#绝对时间
                t1->Branch("ped", ped, Form("ped[%d]/F",CHANNEL_NUMBER));	//前沿基线
                t1->Branch("pedt", pedt, Form("pedt[%d]/F",CHANNEL_NUMBER));	//后沿基线
                t1->Branch("q", q, Form("q[%d]/i",CHANNEL_NUMBER));		//全时间窗的积分值
                t1->Branch("max", max, Form("max[%d]/i",CHANNEL_NUMBER));       //最大值
                t1->Branch("maxpt", maxpt, Form("maxpt[%d]/i",CHANNEL_NUMBER)); //#最大值时间
                t1->Branch("min", min, Form("min[%d]/i",CHANNEL_NUMBER));       //#最小值
                t1->Branch("minpt", minpt, Form("minpt[%d]/i",CHANNEL_NUMBER)); //#最小值时间
		t1->Branch("time_AC", &time_AC, "time_AC/i");                     //先不管
                t1->Branch("tb", tb, Form("tb[%d]/L",CHANNEL_NUMBER));    //#时间窗
                t1->Branch("rms", rms, Form("rms[%d]/F",CHANNEL_NUMBER)); //#均方根
                t1->Branch("Qp", Qp, Form("Qp[%d]/i",CHANNEL_NUMBER));    //#积分值
				t1->Branch("Qp0", Qp0, Form("Qp0[%d]/i",TRY_NUMBER));		//#快放拟合的开始时间
                t1->Branch("Qp1", Qp1, Form("Qp1[%d]/i",TRY_NUMBER));                           //#快放拟合的结束时间
                t1->Branch("QPartV1725_1", QPartV1725_1, Form("QPartV1725_1[%d][%d]/i",CHANNEL_NUMBER, PQNum)); //#V1725-1积分值
                t1->Branch("Ped_rms",Ped_rms, Form("Ped_rms[%d]/F",CHANNEL_NUMBER)); //#基线均方根

		#ifndef NOFIT
		// fitting parameters//快放拟合
                t1->Branch("chi2",&chi2,"chi2/F"); //#快放拟合的chi2
                t1->Branch("famp",&famp,"famp/F"); //#快放拟合的幅度
                t1->Branch("eamp",&eamp,"eamp/F"); //#快放拟合的误差
                t1->Branch("fped",&fped,"fped/F"); //#快放拟合的基线
                t1->Branch("eped",&eped,"eped/F"); //#快放拟合的误差
                t1->Branch("fcross",&fcross,"fcross/F"); //#快放拟合的交叉
                t1->Branch("ecross",&ecross,"ecross/F"); //#快放拟合的误差
                t1->Branch("fslope",&fslope,"fslope/F");	//拟合反三角的
                t1->Branch("eslope",&eslope,"eslope/F");	//#快放拟合的误差
                t1->Branch("fmid",&fmid,"fmid/F");              //#快放拟合的中点
                t1->Branch("fit_flag",&fit_flag,"fit_flag/i");  //#快放拟合的标志
		#endif

                //Run Header
                printf("***************************Run Header**************************************\n");
                fread   (&pstt, sizeof(Double_t),       1,      stream); //#程序开始时间
                printf("* Program Start Time:       %f s.\n",pstt);      //#打印程序开始时间
                /*fread (&FiredD, sizeof(UInt_t), 1,      stream);
                printf("* Fired Devices:            %d ( V1725-1 | V1725-2 | V1729 )\n\n",FiredD);*/ //#打印触发设备
                //V1725-1 Settings
                printf("* V1725-1 Channel DAC:      "); //#打印V1725-1通道DAC
                for(int r=0;r<16;r++)   //#读取V1725-1通道DAC
                   {
                        fread (&V1725_1_DAC[r], sizeof(UInt_t), 1,      stream); //#读取V1725-1通道DAC
                        printf("%d ",V1725_1_DAC[r]); //#打印V1725-1通道DAC
                   }
                printf("\n"); //#换行
                fread (&V1725_1_twd, sizeof(UInt_t),    1,      stream); //#读取V1725-1时间窗
                printf("* V1725-1 Time Window:      %d\n",V1725_1_twd); //#打印V1725-1时间窗
                fread (&V1725_1_pretg, sizeof(UInt_t),  1,      stream); //#读取V1725-1预触发
                printf("* V1725-1 Pre Trigger:      %d\n",V1725_1_pretg); //#打印V1725-1预触发
                fread (&V1725_1_opch, sizeof(UInt_t),   1,      stream); //#读取V1725-1打开通道
                printf("* V1725-1 Opened Channel:   %d\n\n",V1725_1_opch); //#打印V1725-1打开通道

                //V1725-2 Settings
/*                printf("* V1725-2 Channel DAC:      ");
                for(int r=0;r<8;r++)
                   {
                        fread (&V1725_2_DAC[r], sizeof(UInt_t), 1,      stream);
                        printf("%d ",V1725_2_DAC[r]);
                   }
                printf("\n");
                fread (&V1725_2_twd, sizeof(UInt_t),    1,      stream);
                printf("* V1725-2 Time Window:      %d\n",V1725_2_twd);
                fread (&V1725_2_pretg, sizeof(UInt_t),  1,      stream);
                printf("* V1725-2 Pre Trigger:      %d\n",V1725_2_pretg);
                fread (&V1725_2_opch, sizeof(UInt_t),   1,      stream);
                printf("* V1725-2 Opened Channel:   %d\n\n",V1725_2_opch);

                //V1729 Settings
                fread (&V1729_th_DAC, sizeof(UInt_t),   1,      stream);
                printf("* V1729 Threshold DAC:      %d\n",V1729_th_DAC);
                fread (&V1729_posttg, sizeof(UInt_t),   1,      stream);
                printf("* V1729 Post Trigger:       %d\n",V1729_posttg);
                fread (&V1729_tgtype, sizeof(UInt_t),   1,      stream);
                printf("* V1729 Trigger Type:       %d\n",V1729_tgtype);
                fread (&V1729_opch, sizeof(UInt_t),     1,      stream);
                printf("* V1729 Opened Channel:     %d\n\n",V1729_opch);

                //V1721 Settings
                printf("* V1721 Channel DAC:        ");
                for(int r=0;r<8;r++)
                   {
                        fread (&V1721_DAC[r], sizeof(UInt_t),   1,      stream);
                        printf("%d ",V1721_DAC[r]);
                   }
                printf("\n");
                fread (&V1721_twd, sizeof(UInt_t),      1,      stream);
                printf("* V1721 Time Window:        %d\n",V1725_2_twd);
                fread (&V1721_pretg, sizeof(UInt_t),    1,      stream);
                printf("* V1721 Pre Trigger:        %d\n",V1725_2_pretg);
                fread (&V1721_opch, sizeof(UInt_t),     1,      stream);
                printf("* V1721 Opened Channel:     %d\n\n",V1725_1_opch);*/

                fread (&rstt, sizeof(Double_t), 1,      stream); //#读取运行开始时间
                printf("* Run Start Time:           %f s.\n",rstt); //#打印运行开始时间
                printf("***************************************************************************\n"); //#打印运行结束

                for(unsigned int j=0;j<EVENT_NUMBER;j++)//Event while(!feof(stream)) //#事件
                   {
                        //Event Header
                        fread (&Hit_pat, sizeof(UInt_t),        1,      stream); //#读取触发模式
                        fread (&V1729_tg_rec, sizeof(UInt_t),   1,      stream); //#读取V1729触发记录
                        fread (&Evt_endtime, 4,    1,      stream); //#读取事件结束时间
                        fread (&V1725_1_tgno, 4,   1,      stream); //#读取V1725-1触发记录
                        //Evt_endtime &= 0x7FFFFFFF;
                        fread (&V1725_1_tag,  sizeof(UInt_t),   1,      stream); //#读取V1725-1触发记录

                        TTTV1725 = V1725_1_tag&(0x7FFFFFFF); //#读取V1725-1触发记录
						TimeV1725 = 10*( V1725OverFlowCount*FADCTTT_FULL_COUNT + TTTV1725); //#读取V1725-1触发记录
                        //cout<< Evt_endtime << " " <<V1725_1_tgno << " "  << TTTV1725 << endl;
                        //sleep(1);
                        //cout<<Evt_starttime<<" "<<V1725_1_tag<<endl;
                        if(j%1000 == 0)cout<< Evt_endtime << " " <<V1725_1_tgno<<endl; //#打印事件结束时间
                        trig = V1725_1_tgno; //#读取V1725-1触发记录
                        time = Evt_endtime/1000.;//s //#读取事件结束时间
                        deadtime = Evt_deadtime; //#读取事件死时间
			SystemEndTime = Evt_endtime; //#读取事件结束时间
			// modify the V1725 time
			/*if((SystemEndTime*1e6 - TimeV1725) > 20e9){ //#读取事件结束时间
                          V1725OverFlowCount = V1725OverFlowCount + (ULong64_t)(SystemEndTime*1e6 - TimeV1725)/21400000000;
                          TimeV1725 = 10*(V1725OverFlowCount*FADCTTT_FULL_COUNT + TTTV1725);
                        } */

			for(unsigned int Try=0;Try<TRY_NUMBER;Try++) Qp0[Try]=Qp1[Try]=0; //#读取事件结束时间
                        
			//V1725-1 Data
                        for(unsigned int k=0;k<V1725_1_opch;k++)//Channel //#读取V1725-1打开通道
                           {
                                
                                ped[k] = 0.0; 
                                pedt[k] = 0.0; 
                                q[k] = 0; 
                                max[k] = 0; 
                                maxpt[k] = 0; 
                                min[k] = 0xFFFF; 
                                minpt[k] = 0; 
                                tb[k] = 0; 
                                rms[k] = 0; 
				Ped_rms[k] =0.0; 
                                if (k == 6) time_AC = 0; 
				Qp[k] = 0; 
				for(int n = 0;n<PQNum;n++) { QPartV1725_1[k][n]=0; } 
                                for(unsigned int l=0;l<V1725_1_twd;l++)//Time Bin 
                                   {
                                        fread (&V1725_1_pulse[k][l], sizeof(UShort_t),  1,      stream); //#读取V1725-1数据
                                        //if(l%10000 ==0)cout << l << endl; //#打印时间

                                if((k!=2)&&(k!=3)&&(l<ped_range)) { ped[k] += (float)V1725_1_pulse[k][l]; } //#读取V1725-1数据
                                if(((k==2)||(k==3))&&(l<PED_RANGE_FP)) { ped[k] += (float)V1725_1_pulse[k][l]; } //#读取V1725-1数据
                                if((k!=2)&&(k!=3)&&(l>=(V1725_1_twd - ped_range))) { pedt[k] += (float)V1725_1_pulse[k][l]; } //#读取V1725-1数据
                                if(((k==2)||(k==3))&&((l>=fit_start+FIT_RANGE-PED_RANGE_FP)&&(l<fit_start+FIT_RANGE))) {pedt[k] += (float)V1725_1_pulse[k][l];} //#读取V1725-1数据
                                if((k==3)&&(l<2000)) { Ped_rms[k] += (Float_t)((V1725_1_pulse[k][l]-12300.0)*(V1725_1_pulse[k][l]-12300.0));} //#读取V1725-1数据
				//Q & Qp(fixed range for all)
                  		q[k] += V1725_1_pulse[k][l]; //#读取V1725-1数据
				if (l%100==99) {
				  QPartV1725_1[k][l/100] = q[k]; //#读取V1725-1数据
				}
				for(unsigned int Try=0;Try<TRY_NUMBER;Try++)
					{
                               		if(k==0 && l>=Qp0_RANDN[Try] && l<=Qp0_RANUP[Try]) Qp0[Try]+=V1725_1_pulse[k][l];
					if(k==1 && l>=Qp1_RANDN[Try] && l<=Qp1_RANUP[Try]) Qp1[Try]+=V1725_1_pulse[k][l];
					}

				tb[k]+=(Long64_t)(l-4000)*V1725_1_pulse[k][l];
				rms[k]+=(Float_t)(V1725_1_pulse[k][l]*V1725_1_pulse[k][l]);
				//maximum & minimun
				if(V1725_1_pulse[k][l]>max[k])
					{
					max[k] = V1725_1_pulse[k][l];
					maxpt[k] = l;
					}
				if(V1725_1_pulse[k][l]<min[k]) 
					{
                                                if ((l < V1725_1_pretg - 3500 || l > V1725_1_pretg - 1000) && k > 5){continue;}
                                                else{
					                min[k] = V1725_1_pulse[k][l];
					                minpt[k] = l;
                                                }
					}
                                

				} //Time Bin
                                

				/* if(k == 6)
                                {
                                        for(unsigned int l=0;l<V1725_1_twd;l++)
                                        {
                                                if(V1725_1_pulse[k][l]>=THRESHOLD_AC) { time_AC = l; break; }
                                        }
                                }*/

                                if((k!=3)&&(k!=2)) {ped[k] = ped[k]/(float)ped_range; pedt[k] = pedt[k]/(float)ped_range;}
				if((k==3)||(k==2))  {ped[k] = ped[k]/(float)PED_RANGE_FP; pedt[k] = pedt[k]/(float)PED_RANGE_FP;}
                                Ped_rms[k] = (Float_t)(sqrt(Ped_rms[k]/(float)(2000)));
                                rms[k]=(Float_t)(sqrt(rms[k]/(float)(V1725_1_twd)-(float)((q[k]*q[k])/(V1725_1_twd*V1725_1_twd))));

				for(unsigned int l=0;l<V1725_1_twd;l++)//Time Bin
                                	{
					if((k>=2)&& l>=Qp_RANDN[k] && l<=Qp_RANUP[k]) { Qp[k] += V1725_1_pulse[k][l]; }
					if(k<2 && l>maxpt[k]-Qp_RANDN[k] && l<maxpt[k]+Qp_RANUP[k]) Qp[k] += V1725_1_pulse[k][l];
                                }

                           }//channel

		//Fast Pulse Fitting--------------------------------------------------------------------
			#ifndef NOFIT
                        chi2 = 0.0; //#快放拟合的chi2
                        famp = -999.0; //#快放拟合的幅度
                        eamp = 0.0; //#快放拟合的误差
                        fped = -999.0; //#快放拟合的基线
                        eped = 0.0; //#快放拟合的误差
                        fcross = -9999.0; //#快放拟合的交叉
                        ecross = 0.0; //#快放拟合的误差
                        fslope = -99999.0; //#快放拟合的斜率
                        eslope = 0.0; //#快放拟合的误差
                        fmid = 1000.0; //#快放拟合的中点
                        fit_flag = 0; //#快放拟合的标志
			
			fcross_upper = 240*log(max[0]-1200)-max[0]/60+1750; //#快放拟合的交叉上界
			fcross_lower = 300*log(max[0]-1350)-max[0]/50+850; //#快放拟合的交叉下界

//		if( (ped[0]<1700&&ped[0]>900&&min[0]>0&&q[0]>1.58e7) && (ped[1]>900&&ped[1]<1700&&max[1]<16383&&min[1]>0) && (ped[2]>1000) )
//              if( (ped[0]<1700&&ped[0]>900&&min[0]>0&&q[0]>1.58e7) && (ped[1]>900&&ped[1]<1700&&min[1]>0) ) //20160802 adjust scale number of ch1, should delete ped1
		if(ped[0]>900&&min[0]>100&&max[0] > 1100) //#快放拟合的标志
			 {
                                fit_flag = 1; //#快放拟合的标志

/*                                if(q[0]>2e7) { fit_mid = 3400; }//
				else {fit_mid = (int)6000*sqrt(log(q[0]/1.55e7))+1800+5100-q[0]*3.25e-4;}
                               // else  { fit_mid = (int)141.074*log((q[0]-1.8e+07)/8479.12)+2616.86-q[0]/1.313e5; }//
                                if(fit_mid<1500) { fit_mid = 2500; }//
*/
			fit_mid = (fcross_upper + fcross_lower)/2.0; //#快放拟合的中点

			for(unsigned int ii=0;ii<FIT_RANGE;ii++)
                                {
                                        n_bin[ii] = (float)(ii+fit_start); //#快放拟合的开始时间
                                        fit_pulse[ii] = V1725_1_pulse[2][ii+fit_start]; //#快放拟合的结束时间
                                }
                                fit_func = new TF1("fit_func",tanh_fit,fit_start,(fit_start+FIT_RANGE),4); //#快放拟合的函数
                                fit_func->SetParNames("Amp","Ped","Cross","Slope"); //#快放拟合的参数
                                fit_func->SetParLimits(0,100,16000); //#快放拟合的参数
                                fit_func->SetParLimits(1,100,10000); //#快放拟合的参数
                                fit_func->SetParLimits(2,7000,10000); //#快放拟合的参数
                                fit_func->SetParLimits(3,0.00001,100); //#快放拟合的参数

                                fit_func->SetParameter(0,(pedt[2]-ped[2])); //#快放拟合的参数
                                fit_func->SetParameter(1,(0.5*pedt[2]+0.5*ped[2]));
                                fit_func->SetParameter(2,8500);
                                fit_func->SetParameter(3,0.005);
                                fit_func->FixParameter(0,(pedt[2]-ped[2]));
                                fit_func->FixParameter(1,(0.5*pedt[2]+0.5*ped[2]));

                                g_fit = new TGraph(FIT_RANGE,n_bin,fit_pulse); //#快放拟合的图
                                g_fit->Fit("fit_func","Q"); //#快放拟合的图
                                chi2 = (Float_t)fit_func->GetChisquare(); //#快放拟合的chi2
                                famp = (Float_t)fit_func->GetParameter(0); //#快放拟合的幅度
                                eamp = (Float_t)fit_func->GetParError(0); //#快放拟合的误差
                                fped = (Float_t)fit_func->GetParameter(1); //#快放拟合的基线
                                eped = (Float_t)fit_func->GetParError(1); //#快放拟合的误差
                                fcross = (Float_t)fit_func->GetParameter(2); //#快放拟合的交叉
				ecross = (Float_t)fit_func->GetParError(2); //#快放拟合的误差
                                fslope = (Float_t)fit_func->GetParameter(3); //#快放拟合的斜率
                                eslope = (Float_t)fit_func->GetParError(3); //#快放拟合的误差

                                fmid = (Float_t)fit_mid; //#快放拟合的中点
                                g_fit->Delete(); //#快放拟合的图
                                fit_func->Delete(); //#快放拟合的函数
                        } //end of FIT
                        #endif


                        //
                        //if (fit_flag == 1)
                        //{
                                if (max[0] > 1200 && max[0] < 1250 && numofpulse < 20) //#快放拟合的标志
                                //if (idevt > 250100 && numofpulse < 10)
                                {
                                        for (int CH = 0; CH < 16; CH++) //#快放拟合的标志
                                        {
                                                for (int ii = 0; ii < V1725_1_twd; ii++)
                                                {
                                                        ofs << (float)V1725_1_pulse[CH][ii] << " ";
                                                        
                                                }
                                                ofs << famp <<" " << fslope << " " << fcross << " " << fped << " " << idevt << endl;
                                        }
                                        numofpulse = numofpulse + 1;
                                }
                        //}
                        
                        t1->Fill();
                        idevt++;
                   }//Event
		
                //Run Ender
                fread   (&redt, sizeof(Double_t),       1,      stream);
                printf("***************************Run Ending**************************************\n");
                printf("* Run Ending Time:          %f s.\n",redt);
                printf("***************************************************************************\n");

                fclose(stream);
                ofs.close();
                t1->Write("",TObject::kOverwrite);
                TH1F *ST = new TH1F("ST","Run_st",1,rstt,rstt);
                ST->Fill(rstt);
                ST->Write();
                TH1F *ET = new TH1F("ET","Run_et",1,redt,redt);
                ET->Fill(redt);
                ET->Write();
                a->Close();
           }//Run
}


