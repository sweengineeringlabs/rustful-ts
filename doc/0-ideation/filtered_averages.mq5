//------------------------------------------------------------------
#property copyright "© mladen, 2018"
#property link      "mladenfx@gmail.com"
//------------------------------------------------------------------
#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots   1
#property indicator_label1  "Deviation filtered average"
#property indicator_type1   DRAW_COLOR_LINE
#property indicator_color1  clrDarkGray,clrDeepPink,clrMediumSeaGreen
#property indicator_width1  2

//
//--- input parameters
//

input int                inpPeriod = 14;           // Average and filter period
input ENUM_MA_METHOD     inpMethod = MODE_EMA;     // Average method
input ENUM_APPLIED_PRICE inpPrice  = PRICE_MEDIAN; // Price
input double             inpFilter = 2.5;          // Filter size

//
//--- indicator buffers
//
double val[],valc[],avg[];
int  ª_maHandle,ª_maPeriod;

//------------------------------------------------------------------
// Custom indicator initialization function
//------------------------------------------------------------------

int OnInit()
{
   //
   //--- indicator buffers mapping
   //
         SetIndexBuffer(0,val,INDICATOR_DATA);
         SetIndexBuffer(1,valc,INDICATOR_COLOR_INDEX);
         SetIndexBuffer(2,avg ,INDICATOR_CALCULATIONS);
        
         ª_maPeriod   = (inpPeriod>1) ? inpPeriod : 1;
         ª_maHandle   = iMA(_Symbol,0,ª_maPeriod ,0,inpMethod,inpPrice); if (!_checkHandle(ª_maHandle,"Average")) return(INIT_FAILED);
   //        
   //--- indicator short name assignment
   //
   IndicatorSetString(INDICATOR_SHORTNAME,"Deviation filtered average ("+(string)inpPeriod+","+(string)inpFilter+")");
   return (INIT_SUCCEEDED);
}
void OnDeinit(const int reason) {}

//------------------------------------------------------------------
// Custom indicator iteration function
//------------------------------------------------------------------
//
//---
//

int OnCalculate(const int rates_total,const int prev_calculated,const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   int _copyCount = rates_total-prev_calculated+1; if (_copyCount>rates_total) _copyCount=rates_total;
         if (CopyBuffer(ª_maHandle,0,0,_copyCount,avg)!=_copyCount) return(prev_calculated);

   //
   //---
   //
  
   int i= prev_calculated-1; if (i<0) i=0; for (; i<rates_total && !_StopFlag; i++)
   {
      if (avg[i]==EMPTY_VALUE) avg[i] = close[i];
         val[i]  = iFilter(avg[i],inpFilter,inpPeriod,i,rates_total);
         valc[i] = (i>0) ?(val[i]>val[i-1]) ? 2 :(val[i]<val[i-1]) ? 1 : valc[i-1]: 0;
   }
   return(i);
}

//------------------------------------------------------------------
// Custom function(s)
//------------------------------------------------------------------
//
//---
//

double iFilter(double value, double filter, int period, int i, int bars, int instance=0)
{
   #define ¤ instance
   #define _functionInstances 1
      struct sFilterArrayStruct
         {
            double value;
            double change;
            double power;
            double summc;
            double summp;
         };
      static sFilterArrayStruct m_array[][_functionInstances];
      static int m_arraySize = 0;
             if (m_arraySize<bars)
             {
                 int _res = ArrayResize(m_array,bars+500);
                 if (_res<bars) return(0);
                     m_arraySize = _res;
             }

      //
      //---
      //
  
      m_array[i][¤].value  = value;  double _change = (i>0) ? m_array[i][¤].value-m_array[i-1][¤].value : 0;
      m_array[i][¤].change = (_change>0) ? _change : - _change;
         if (i>period)
         {
            #define _power(_val) ((_val)*(_val))
              m_array[i][¤].summc =  m_array[i-1][¤].summc +m_array[i][¤].change-m_array[i-period][¤].change;
              m_array[i][¤].power = _power(m_array[i][¤].change-m_array[i][¤].summc/(double)period);
              m_array[i][¤].summp =  m_array[i-1][¤].summp+m_array[i][¤].power-m_array[i-period][¤].power;
         }              
         else
         {
            m_array[i][¤].summc  =
            m_array[i][¤].summp = 0;
            for(int k=0; k<period && i>=k; k++) m_array[i][¤].summc += m_array[i-k][¤].change;
                                                m_array[i][¤].power  = _power(m_array[i][¤].change-m_array[i][¤].summc/(double)period);
            for(int k=0; k<period && i>=k; k++) m_array[i][¤].summp += m_array[i-k][¤].power;
         }            
         if (i>0 && filter>0 && m_array[i][¤].change<filter*MathSqrt(m_array[i][¤].summp/(double)period)) m_array[i][¤].value=m_array[i-1][¤].value;
   return (m_array[i][¤].value);

   //
   //---
   //
            
   #undef ¤ #undef _power #undef _functionInstances
}

//
//---
//

bool _checkHandle(int _handle, string _description)
{
   static int  _chandles[];
          int  _size   = ArraySize(_chandles);
          bool _answer = (_handle!=INVALID_HANDLE);
          if  (_answer)
               { ArrayResize(_chandles,_size+1); _chandles[_size]=_handle; }
          else { for (int i=_size-1; i>=0; i--) IndicatorRelease(_chandles[i]); ArrayResize(_chandles,0); Alert(_description+" initialization failed"); }
   return(_answer);
}  
//------------------------------------------------------------------
  
