export interface BackendHealth { ok: boolean; service: string; root?: string; }
export interface ModelStatus { exists: boolean; path: string; }
export interface ResearchStatus { root: string; files: Record<string, ModelStatus>; rl_models: string[]; policies: string[]; execution: string; }
export interface PolicyMean { pd:number; weighted_poi:number; reward:number; coverage:number; interceptions:number; hp_miss_rate:number; mean_interception_time_s:number; mit_p95_s:number; mit_p99_s:number; staleness_mean_s:number; staleness_max_s:number; }
export interface Evaluation { source:string; execution:string; scenario:{name:string;type:string;seed:number;emitters:number;horizon_s:number;channels:number;truth_events:number;active_frequencies:number[]}; policies:Record<string,{episodes:any[];mean:PolicyMean}>; }
export interface Trace { source:string; execution:string; policy:string; scenario:any; metrics:any; trace:Array<any>; }
export interface DatasetInspect { source:string; file:string; valid_hdf5_schema:boolean; schema_errors:string[]; n_pulses:number; n_features:number; features:string[]; receiver:any; }
export interface Job {job_id:string; stage:string; status:string; progress:number; message:string; output?:string; artifacts?:string[];}

const BASE = (import.meta as ImportMeta & {env?:Record<string,string>}).env?.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

async function request<T>(path:string, init?:RequestInit):Promise<T> {
  const res = await fetch(`${BASE}${path}`, {...init, headers:{Accept:'application/json', ...(init?.headers || {})}});
  if (!res.ok) throw new Error((await res.text()) || `Backend request failed (${res.status})`);
  return res.json() as Promise<T>;
}

export const backendUrl = BASE;
export const backendHealth = () => request<BackendHealth>('/api/health');
export const researchStatus = () => request<ResearchStatus>('/api/research/status');
export const getModels = () => request<any>('/api/models');
export const getPolicies = () => request<{policies:string[]}>('/api/policies');
export const getSyntheticScenarios = () => request<{scenarios:string[]}>('/api/synthetic/scenarios');
export const getLockedResults = () => request<any>('/api/results/locked');

export async function evaluateSynthetic(args:{scenario:string;emitters:number;horizon_s:number;episodes:number;seed:number;policies:string[]}):Promise<Evaluation> {
  return request<Evaluation>('/api/evaluate/synthetic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
}
export async function decisionTrace(args:{scenario:string;emitters:number;horizon_s:number;seed:number;policy:string;max_steps?:number}):Promise<Trace> {
  return request<Trace>('/api/decision/trace',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
}
export async function inspectDataset(file:File):Promise<DatasetInspect> {
  const form = new FormData(); form.append('file',file);
  return request<DatasetInspect>('/api/dataset/inspect',{method:'POST',body:form});
}
export async function startDatasetJob(command:string,file?:File,policies?:string[]):Promise<Job> {
  const form = new FormData(); form.append('command',command); if(file) form.append('file',file);
  if (policies?.length) form.append('policies', policies.join(','));
  return request<Job>('/api/dataset/job',{method:'POST',body:form});
}
export const getDatasetJob = (id:string) => request<Job>(`/api/dataset/job/${encodeURIComponent(id)}`);
export const getDatasetArtifacts = () => request<{source:string;artifacts:string[]}>('/api/dataset/artifacts');

export async function stareInfer(file:File):Promise<any>{ const form=new FormData(); form.append('file',file); return request<any>('/api/stare/infer',{method:'POST',body:form}); }

export interface StreamStatus {
  job_id?: string; stream_id?: string; status:string; progress:number; message:string;
  trace?:any[]; latest?:any; metrics?:any; artifact?:any; steps?:number; detections?:number; total_reward?:number;
  source?:string; scenario_meta?:any; error?:string;
}
export async function startDecisionStream(args:{policy:string;scenario?:string;emitters?:number;horizon_s?:number;seed?:number;max_steps?:number;file?:File}):Promise<{stream_id:string;status:string;policy:string}>{
  const form=new FormData(); form.append('policy',args.policy); form.append('scenario',args.scenario||'mixed'); form.append('emitters',String(args.emitters??8)); form.append('horizon_s',String(args.horizon_s??0.5)); form.append('seed',String(args.seed??26055)); form.append('max_steps',String(args.max_steps??240)); if(args.file) form.append('file',args.file);
  return request('/api/decision/stream',{method:'POST',body:form});
}
export const getDecisionStream = (id:string) => request<StreamStatus>(`/api/decision/stream/${encodeURIComponent(id)}`);
export async function evaluateUploadedDataset(file:File):Promise<Job>{ return startDatasetJob('evaluate',file); }
