import React, {useState} from 'react';
import {getDatasetJob, inspectDataset, startDatasetJob, type DatasetInspect, type Job} from '../app/backendApi';

const Card:React.FC<{label:string;value:React.ReactNode;sub:string}>=({label,value,sub})=><div className="console-card"><div className="console-label">{label}</div><div className="console-value">{value}</div><div className="console-sub">{sub}</div></div>;
const f=(x:any,d=3)=>x===undefined||x===null||Number.isNaN(Number(x))?'—':Number(x).toFixed(d);

export const FinalModelPage:React.FC=()=>{
  const [file,setFile]=useState<File|null>(null);
  const [info,setInfo]=useState<DatasetInspect|null>(null);
  const [job,setJob]=useState<Job|null>(null);
  const [result,setResult]=useState<any>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');

  const inspect=async()=>{
    if(!file)return;
    setBusy(true);setError('');
    try{setInfo(await inspectDataset(file));}
    catch(e:any){setError(e.message||String(e));}
    finally{setBusy(false);}
  };

  const run=async()=>{
    if(!file)return;
    setBusy(true);setError('');setResult(null);
    try{
      const j=await startDatasetJob('final_evaluate',file);setJob(j);
      let polls=0;
      const poll=async()=>{
        try{
          const q=await getDatasetJob(j.job_id);setJob(q);
          if((q.status==='running'||q.status==='queued')&&polls++<720){window.setTimeout(poll,400);return;}
          if(q.status==='completed'&&q.output){setResult(JSON.parse(q.output));}
          else if(q.status==='failed')setError(q.message||'Final Model verification failed.');
          else if(q.status!=='completed')setError('The verification is still running on the backend.');
        }catch(e:any){setError(e.message||String(e));}
        finally{setBusy(false);}
      };
      window.setTimeout(poll,300);
    }catch(e:any){setError(e.message||String(e));setBusy(false);}
  };

  const finalPolicy=result?.replay?.policies?.stare_cognitive;
  const status=result?.replay?.model_status?.stare_cognitive;

  return <div className="page" style={{maxWidth:1120}}>
    <section className="panel panel-pad" style={{marginBottom:14,borderColor:'var(--accent)'}}>
      <div className="hero-kicker">CANONICAL DEMONSTRATION PATH</div>
      <div className="spread" style={{alignItems:'flex-start',gap:18}}>
        <div>
          <h1 className="page-title" style={{marginBottom:6}}>Final Integrated Model</h1>
          <p className="section-note" style={{maxWidth:760,margin:0}}>
            <strong>STARE-Cognitive</strong> — a frozen STARE temporal representation feeding the cognitive belief-driven scan scheduler. This is the single model path intended for the main demonstration.
          </p>
        </div>
        <span className="chip chip-pos">FINAL MODEL · V6.2</span>
      </div>
      <div className="console-grid-2" style={{marginTop:14}}>
        <Card label="TEMPORAL MODEL" value="STARE · GRU" sub="frozen deployment artifact · 64-D embedding"/>
        <Card label="DECISION LAYER" value="Cognitive" sub="belief + prediction + uncertainty + coverage + priority"/>
        <Card label="INPUT BOUNDARY" value="Receiver only" sub="PDW/state + STARE embedding · no ground truth"/>
        <Card label="DEPLOYMENT ARTIFACT" value="stare_inference.pt" sub="models/frozen/stare_inference.pt"/>
      </div>
    </section>

    <div className="note note-neutral" style={{marginBottom:14}}>
      <strong>For a first-time user:</strong> this is the page to open first. Upload a TSRD-compatible HDF5 capture, click <b>Verify Final Model</b>, and the backend runs the frozen STARE encoder and STARE-Cognitive scheduler on receiver-observable windows. The Scenario Lab remains the continuous visual waterfall demonstrator.
    </div>

    <section className="panel panel-pad" style={{marginBottom:14}}>
      <div className="spread"><div><div className="panel-title">1 · Upload a real receiver capture</div><div className="panel-sub">The file is inspected first; the verification path reads bounded windows rather than loading the complete capture into RAM.</div></div><span className="chip chip-blue">HDF5 / HDF5-compatible</span></div>
      <div className="row" style={{gap:8,marginTop:12,flexWrap:'wrap'}}>
        <input type="file" accept=".h5,.hdf5,.he5" onChange={e=>{setFile(e.target.files?.[0]||null);setInfo(null);setResult(null);setJob(null);setError('')}}/>
        <button className="btn" disabled={!file||busy} onClick={inspect}>Inspect capture</button>
        <button className="btn btn-primary" disabled={!file||busy} onClick={run}>{busy?'Verifying…':'Verify Final Model'}</button>
      </div>
      {file&&<div className="mono tiny" style={{marginTop:9}}>{file.name} · {(file.size/1024/1024).toFixed(1)} MB</div>}
      {error&&<div className="console-notice warn" style={{marginTop:10}}>{error}</div>}
    </section>

    {info&&<section className="console-grid-2" style={{marginBottom:14}}>
      <div className="panel panel-pad"><div className="panel-title">Capture validation</div>
        <div className="gate-row"><span>Canonical HDF5 schema</span><b>{info.valid_hdf5_schema?'VALID':'INVALID'}</b><span className={`console-pill ${info.valid_hdf5_schema?'ok':'warn'}`}>{info.valid_hdf5_schema?'PASS':'CHECK'}</span></div>
        <div className="gate-row"><span>Pulse records</span><b>{info.n_pulses.toLocaleString()}</b></div>
        <div className="gate-row"><span>Feature count</span><b>{info.n_features}</b></div>
      </div>
      <div className="panel panel-pad"><div className="panel-title">Receiver features</div><div className="chip-wrap">{info.features.map(x=><span className="artifact-chip" key={x}>{x}</span>)}</div></div>
    </section>}

    {job&&<section className="panel panel-pad" style={{marginBottom:14}}>
      <div className="spread"><div><div className="panel-title">2 · Final Model execution</div><div className="panel-sub">{job.message}</div></div><span className="chip chip-accent">{job.status.toUpperCase()} · {Math.round(job.progress*100)}%</span></div>
      <div className="bar" style={{marginTop:10}}><i style={{width:`${Math.max(0,Math.min(100,job.progress*100))}%`}}/></div>
      <div className="tiny mono" style={{marginTop:7,color:'var(--muted)'}}>STARE encoding → receiver replay → cognitive decisions → evaluation summary</div>
    </section>}

    {result&&<section className="panel panel-pad" style={{marginBottom:14,borderColor:'var(--pos)'}}>
      <div className="spread"><div><div className="panel-title">Final Model verification result</div><div className="panel-sub">{result.note}</div></div><span className="chip chip-pos">VERIFIED · {result.stare?.windows??'—'} STARE WINDOWS</span></div>
      <div className="console-grid-2" style={{marginTop:12}}>
        <Card label="FINAL MODEL" value="STARE-Cognitive" sub="canonical integrated path"/>
        <Card label="MODEL STATUS" value={status?.loaded===false?'CHECK':'READY'} sub={status?.reason||'policy adapter executed'}/>
        <Card label="STARE EMBEDDING" value={`${result.stare?.embedding_dim??64}-D`} sub={`norm μ ${f(result.stare?.embedding_norm_mean,3)}`}/>
        <Card label="ACTIVITY OUTPUT" value={f(result.stare?.activity_mean,3)} sub="mean frozen-STARE activity prediction"/>
      </div>
      <div className="panel panel-pad" style={{marginTop:12,background:'var(--surface-2)'}}>
        <div className="panel-title">Receiver replay · STARE-Cognitive</div>
        <div className="table-scroll" style={{marginTop:8}}><table className="console-table"><thead><tr><th>Status</th><th>Pd*</th><th>Reward*</th><th>Coverage*</th><th>Windows</th><th>Unique channels*</th></tr></thead><tbody><tr><td><b>{finalPolicy?.status||'—'}</b></td><td>{finalPolicy?.pd===undefined?'—':`${(finalPolicy.pd*100).toFixed(1)}%`}</td><td>{f(finalPolicy?.reward,2)}</td><td>{finalPolicy?.coverage===undefined?'—':`${(finalPolicy.coverage*100).toFixed(1)}%`}</td><td>{finalPolicy?.windows??'—'}</td><td>{f(finalPolicy?.unique_channels,1)}</td></tr></tbody></table></div>
        <div className="tiny" style={{marginTop:8,color:'var(--muted)'}}>* Uploaded-file replay metrics are diagnostic receiver-replay measurements. They are not substituted for the locked benchmark.</div>
      </div>
    </section>}

    <section className="panel panel-pad">
      <div className="panel-title">What should I show a judge?</div>
      <div className="console-grid-2" style={{marginTop:10}}>
        <a className="entry" href="#/lab"><div className="eg">LIVE VISUAL</div><div className="et">Scenario Lab → continuous waterfall</div><div className="ed">Shows the receiver scanning the full 2–10 GHz spectrum continuously.</div></a>
        <a className="entry" href="#/method"><div className="eg">EXPLAIN</div><div className="et">Methodology → current architecture</div><div className="ed">Explains STARE, cognitive scheduling, receiver-only information and evaluation boundaries.</div></a>
      </div>
    </section>
  </div>;
};
