import React,{useEffect,useRef} from 'react';
import {useTheme} from '../../app/theme';

export interface WaterfallBackendRow {
  step?:number; time_s?:number; channel?:number; frequency_mhz?:number; dwell_s?:number;
  detected?:boolean; detection_count?:number; reward?:number; score?:number;
  /** Browser-engine visual rows: activity is the observed spectrum occupancy for this dwell. */
  activity?:boolean[]; hit?:boolean; falseAlarm?:boolean; settling?:boolean; band?:number; timeMs?:number;
}
export interface WaterfallViewProps { rows:WaterfallBackendRow[]; accent?:string; height?:number; title?:string; mode?:'live'|'full'; showLegend?:boolean; }

const BASE_FREQ=2000;
const BW=250;
const CHANNELS=32;

export const WaterfallView:React.FC<WaterfallViewProps>=({rows,accent='#c47a18',height=240,title='Spectrum Waterfall',mode='live',showLegend=true})=>{
 const canvasRef=useRef<HTMLCanvasElement|null>(null); const {theme}=useTheme();
 useEffect(()=>{
  const canvas=canvasRef.current;if(!canvas)return;const ctx=canvas.getContext('2d');if(!ctx)return;
  const render=()=>{
   const w=canvas.clientWidth||900,dpr=Math.min(2,window.devicePixelRatio||1);
   canvas.width=Math.floor(w*dpr);canvas.height=Math.floor(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
   const css=getComputedStyle(document.documentElement);
   const bg=css.getPropertyValue('--wf-bg').trim()||'#fff';
   const grid=css.getPropertyValue('--wf-grid').trim()||'rgba(28,40,60,.08)';
   ctx.fillStyle=bg;ctx.fillRect(0,0,w,height);
   if(!rows.length){ctx.fillStyle='#8a93a1';ctx.font='12px Consolas,monospace';ctx.fillText('START RECEIVER TO BEGIN CONTINUOUS WATERFALL…',12,height/2);return;}
   const recent=mode==='live'?rows.slice(-140):rows.slice(-240);
   const rowH=Math.max(2,height/recent.length),col=w/CHANNELS;
   ctx.strokeStyle=grid;ctx.lineWidth=1;
   for(let c=0;c<=CHANNELS;c++){const x=Math.floor(c*col)+.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height);ctx.stroke();}
   recent.forEach((r,i)=>{
    const y=height-(i+1)*rowH;
    const activity=r.activity;
    if(activity?.length){
      for(let c=0;c<Math.min(CHANNELS,activity.length);c++) if(activity[c]){
        const x=c*col;
        const hit=(r.hit||r.detected) && c===(r.band??r.channel);
        ctx.fillStyle=hit?'rgba(30,125,82,.88)':'rgba(80,125,170,.38)';
        ctx.fillRect(x+1,y,col-2,Math.max(1,rowH-1));
        if(hit){ctx.strokeStyle=accent;ctx.lineWidth=1;ctx.strokeRect(x+1,y,col-2,Math.max(1,rowH-1));}
      }
    } else {
      const c=Math.max(0,Math.min(CHANNELS-1,Number(r.band ?? r.channel ?? 0)||0));
      const x=c*col;ctx.fillStyle=r.detected?'rgba(30,125,82,.82)':'rgba(80,100,125,.20)';
      ctx.fillRect(x+1,y,col-2,Math.max(1,rowH-1));
      if(r.detected){ctx.strokeStyle=accent;ctx.strokeRect(x+1,y,col-2,Math.max(1,rowH-1));}
    }
    const cursor=Math.max(0,Math.min(CHANNELS-1,Number(r.band??r.channel)||0));
    ctx.fillStyle='rgba(196,122,24,.20)';ctx.fillRect(cursor*col,y,Math.max(2,col),Math.max(1,rowH-1));
   });
   const last=recent[recent.length-1];const cursor=Math.max(0,Math.min(CHANNELS-1,Number(last.band??last.channel)||0));
   ctx.strokeStyle=accent;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cursor*col+col/2,0);ctx.lineTo(cursor*col+col/2,height);ctx.stroke();
   ctx.fillStyle='rgba(120,120,120,.7)';ctx.font='9px Consolas,monospace';
   for(let c=0;c<CHANNELS;c+=4){ctx.fillText(`${((BASE_FREQ+c*BW)/1000).toFixed(1)}G`,c*col+3,11);}
  };
  const ro=new ResizeObserver(render);ro.observe(canvas);render();return()=>ro.disconnect();
 },[rows,accent,height,mode,theme]);
 return <div className="panel" style={{padding:'10px 12px'}}>{title&&<div className="spread" style={{marginBottom:6}}><span className="panel-title">{title}</span><span className="panel-sub mono">CONTINUOUS RECEIVER VIEW · 2.0–10.0 GHz</span></div>}<canvas ref={canvasRef} style={{width:'100%',height:`${height}px`,display:'block',border:'1px solid var(--border)'}}/>{showLegend&&<div className="spread" style={{marginTop:4,fontSize:10}}><span>blue = observed RF activity · green = receiver hit</span><span>orange = current scan cursor</span></div>}</div>;
};
