import React, {useEffect, useRef} from 'react';

export interface BackendWaterfallRow {
  step:number; time_s:number; channel:number; frequency_mhz:number; dwell_s:number;
  detected:boolean; detection_count:number; reward:number; score:number;
}

export const BackendWaterfall:React.FC<{rows:BackendWaterfallRow[];height?:number;title?:string;channels?:number}> = ({rows,height=300,title='LIVE BACKEND WATERFALL',channels})=>{
  const ref=useRef<HTMLCanvasElement|null>(null);
  useEffect(()=>{
    const canvas=ref.current; if(!canvas)return;
    const ctx=canvas.getContext('2d'); if(!ctx)return;
    const draw=()=>{
      const w=canvas.clientWidth||900,h=height,dpr=Math.min(2,window.devicePixelRatio||1);
      canvas.width=Math.floor(w*dpr); canvas.height=Math.floor(h*dpr); ctx.setTransform(dpr,0,0,dpr,0,0);
      const css=getComputedStyle(document.documentElement);
      const bg=css.getPropertyValue('--wf-bg').trim()||'#fff'; const grid=css.getPropertyValue('--wf-grid').trim()||'rgba(28,40,60,.08)';
      ctx.fillStyle=bg; ctx.fillRect(0,0,w,h);
      if(!rows.length){ctx.fillStyle='#8a93a1';ctx.font='12px Consolas,monospace';ctx.fillText('WAITING FOR LIVE BACKEND TRACE…',12,h/2);return;}
      const recent=rows.slice(-160); const nCh=Math.max(1,channels??Math.max(...recent.map(r=>r.channel),0)+1); const col=w/nCh; const rowH=h/recent.length;
      ctx.strokeStyle=grid;ctx.lineWidth=1; for(let c=0;c<=nCh;c++){const x=Math.floor(c*col)+.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
      recent.forEach((r,i)=>{
        const y=h-(i+1)*rowH; const x=r.channel*col; ctx.fillStyle=r.detected?'rgba(30,120,80,.75)':'rgba(60,90,120,.18)'; ctx.fillRect(x+1,y,col-2,Math.max(1,rowH-1));
        if(r.detected){ctx.strokeStyle='#197a52';ctx.strokeRect(x+1,y,col-2,Math.max(1,rowH-1));}
        ctx.fillStyle='rgba(30,30,30,.5)';ctx.fillRect(0,y,3,Math.max(1,rowH-1));
      });
      const last=recent[recent.length-1];ctx.strokeStyle='#c47a18';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(last.channel*col+col/2,0);ctx.lineTo(last.channel*col+col/2,h);ctx.stroke();
    };
    const ro=new ResizeObserver(draw);ro.observe(canvas);draw();return()=>ro.disconnect();
  },[rows,height,channels]);
  return <div className="panel panel-pad"><div className="spread"><div><div className="panel-title">{title}</div><div className="panel-sub">Every row is emitted by the Python receiver loop · no browser simulation.</div></div><span className="chip">{rows.length} backend decisions</span></div><canvas ref={ref} style={{width:'100%',height,display:'block',marginTop:8,border:'1px solid var(--border)'}}/><div className="spread mono tiny" style={{marginTop:5}}><span>{rows.length?`t=${rows.at(-1)?.time_s.toFixed(4)}s`:''}</span><span>{rows.length?`CH ${rows.at(-1)?.channel} · ${rows.at(-1)?.frequency_mhz.toFixed(1)} MHz`:''}</span></div></div>
};
