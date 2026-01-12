//------------------------------------------------------------------
#property copyright   "© mladen, 2018"
#property link        "mladenfx@gmail.com"
#property description "Step VHF adaptive VMA"
//------------------------------------------------------------------
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   2
#property indicator_label1  "Shadow"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrGray
#property indicator_width1  6
#property indicator_label2  "Step VMA"
#property indicator_type2   DRAW_COLOR_LINE
#property indicator_color2  clrDarkGray,clrMediumSeaGreen,clrDeepPink
#property indicator_width2  2

//
//--- input parameters
//

input int                inpPeriod   = 14;          // VMA period
input int                inpPeriod2  =  0;          // VHF period (<=1 for same as VMA period)
input ENUM_APPLIED_PRICE inpPrice    = PRICE_CLOSE; // Price
input double             inpStepSize = 1.0;         // Step size (pips)

//
//--- indicator buffers
//

double val[],valc[],vals[],avg[];
double ª_alpha,ª_stepSize;; int ª_adpPeriod;

//------------------------------------------------------------------
//  Custom indicator initialization function
//------------------------------------------------------------------

int OnInit()
{
   //
   //--- indicator buffers mapping
   //
         SetIndexBuffer(0,vals ,INDICATOR_DATA);
         SetIndexBuffer(1,val  ,INDICATOR_DATA);
         SetIndexBuffer(2,valc ,INDICATOR_COLOR_INDEX);
         SetIndexBuffer(3,avg  ,INDICATOR_CALCULATIONS);
            ª_adpPeriod = (inpPeriod2<=1 ? inpPeriod : inpPeriod2);
            ª_alpha     = 2.0/(1.0+inpPeriod);
            ª_stepSize  = (inpStepSize>0?inpStepSize:0)*_Point*MathPow(10,_Digits%2);
   //        
   //--- indicator short name assignment
   //
  
         IndicatorSetString(INDICATOR_SHORTNAME,"VMA ("+(string)inpPeriod+")");
   return (INIT_SUCCEEDED);
}
void OnDeinit(const int reason)
{
}

//------------------------------------------------------------------
//  Custom indicator iteration function
//------------------------------------------------------------------
//
//---
//

#define _setPrice(_priceType,_target,_index) \
   { \
   switch(_priceType) \
   { \
      case PRICE_CLOSE:    _target = close[_index];                                              break; \
      case PRICE_OPEN:     _target = open[_index];                                               break; \
      case PRICE_HIGH:     _target = high[_index];                                               break; \
      case PRICE_LOW:      _target = low[_index];                                                break; \
      case PRICE_MEDIAN:   _target = (high[_index]+low[_index])/2.0;                             break; \
      case PRICE_TYPICAL:  _target = (high[_index]+low[_index]+close[_index])/3.0;               break; \
      case PRICE_WEIGHTED: _target = (high[_index]+low[_index]+close[_index]+close[_index])/4.0; break; \
      default : _target = 0; \
   }}
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
   int i = (prev_calculated>0 ? prev_calculated-1 : 0); for (; i<rates_total && !_StopFlag; i++)
   {
      double _price; _setPrice(inpPrice,_price,i);
      avg[i] = (i>0) ? avg[i-1]+(ª_alpha*iVhf(ª_adpPeriod,close,i,rates_total)*2.0)*(_price-avg[i-1]) : _price;
      
      //
      //---
      //
      
      if (i>0 && ª_stepSize>0)
      {
           double _diff = avg[i]-val[i-1];
           val[i] = val[i-1]+((_diff<ª_stepSize && _diff>-ª_stepSize) ? 0 : (int)(_diff/ª_stepSize)*ª_stepSize);
      }
      else val[i]  = (ª_stepSize>0) ? MathRound(avg[i]/ª_stepSize)*ª_stepSize : avg[i];
           valc[i] = (i>0) ? (val[i]>val[i-1]) ? 1 : (val[i]<val[i-1]) ? 2 : valc[i-1] : 0 ;
           vals[i] = val[i];
   }
   return(i);
}

//------------------------------------------------------------------
//  Custom function(s)
//------------------------------------------------------------------
//
//---
//

#define _checkArrayReserve 500
#define _checkArraySize(_arrayName,_ratesTotal) static bool _arrayError=false; { static int _arrayResizedTo=0; if (_arrayResizedTo<_ratesTotal) { int _res = (_ratesTotal+_checkArrayReserve); _res -= ArrayResize(_arrayName,_ratesTotal+_checkArrayReserve); if (_res) _arrayError=true; else { _arrayResizedTo=_ratesTotal+_checkArrayReserve; }}}

//
//---
//


template <typename T>
double iVhf(int period, T& price[], int i, int bars, int _instance=0)
{
   //
   //---
   //
  
      #define _functionInstancesArraySize 2
      #ifdef  _functionInstances
            static double _workArray[][_functionInstancesArraySize*_functionInstances];
      #else static double _workArray[][_functionInstancesArraySize];
      #endif
         _checkArraySize(_workArray,bars);

   //
   //---
   //
                  
      #ifdef _functionInstances
                int _winst = _instance*_functionInstancesArraySize;
      #else #define _winst   _instance
      #endif
      #define _diff   _winst+1
      #define _noise  _winst

      _workArray[i][_diff] = (i>0) ? (price[i]>price[i-1]) ? price[i]-price[i-1] : price[i-1]-price[i] : 0;
         if (i>period)
                 _workArray[i][_noise] = _workArray[i-1][_noise]-_workArray[i-period][_diff]+_workArray[i][_diff];
         else  { _workArray[i][_noise] = _workArray[i][_diff]; for(int k=1; k<period && i>=k; k++) _workArray[i][_noise] += _workArray[i-k][_diff]; }
  
         //
         //----
         //
        
         int start = i-period+1; if (start<0) start=0;
         double max = price[ArrayMaximum(price,start,period)];
         double min = price[ArrayMinimum(price,start,period)];
   return((_workArray[i][_noise]!=0) ? (max-min)/_workArray[i][_noise] : 0);  

   //
   //---
   //
  
   #undef _diff #undef _noise
   #undef _functionInstances #undef _functionInstancesArraySize
  
}
//------------------------------------------------------------------

  
