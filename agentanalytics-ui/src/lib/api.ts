import axios from "axios";

export const API_BASE =
  (import.meta as any).env?.VITE_AGENTANALYTICS_API_BASE || "http://localhost:8008/api";

export type RunSummary = {
  run_id: string;
  status: string;
  started_ms?: number;
  finished_ms?: number;
  trace_count?: number;
  plugins: string[];
};

export type PluginParam = {
  key: string;
  type: string;
  title: string;
  description?: string;
  default?: any;
  required?: boolean;
  enum?: any[];
  min?: number;
  max?: number;
};

export type PluginMeta = {
  name: string;
  version: string;
  title: string;
  description: string;
  requires: string[];
  defaults: Record<string, any>;
  params: PluginParam[];
};

export async function listRuns(): Promise<RunSummary[]> {
  const r = await axios.get(`${API_BASE}/runs`);
  return r.data;
}

export async function getRun(runId: string): Promise<any> {
  const r = await axios.get(`${API_BASE}/runs/${runId}`);
  return r.data;
}

export async function createRunFromYaml(configYaml: string): Promise<{ run_id: string }> {
  const r = await axios.post(`${API_BASE}/runs`, { config_yaml: configYaml });
  return r.data;
}

export async function listPlugins(): Promise<PluginMeta[]> {
  const r = await axios.get(`${API_BASE}/plugins`);
  return r.data;
}

export function artifactUrl(runId: string, relpath: string): string {
  // relpath already run-dir relative in manifest
  const safe = relpath.replace(/^\/+/, "");
  return `${API_BASE}/runs/${runId}/artifacts/${encodeURIComponent(safe).replace(/%2F/g, "/")}`;
}

export function logsSseUrl(runId: string): string {
  return `${API_BASE}/runs/${runId}/events`;
}