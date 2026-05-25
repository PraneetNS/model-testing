/**
 * Niyantrana API Client
 * Typed fetch wrapper for every backend endpoint used by the dashboard.
 */

export const API_BASE =
  (process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000') +
  '/api/v1';

export const API_KEY =
  process.env.NEXT_PUBLIC_API_KEY || 'mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi';

const BASE_URL = API_BASE;

/** Headers for multipart uploads (do not set Content-Type — browser sets boundary). */
export function apiUploadHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'X-API-Key': API_KEY };
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('niyantrana_token');
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

// ─── Error ────────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

// ─── Core request ────────────────────────────────────────────────────────────

async function safeJson<T>(res: Response): Promise<T> {
  if (res.status === 204) return {} as T;
  const text = await res.text();
  return text ? JSON.parse(text) : ({} as T);
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token =
    typeof window !== 'undefined' ? localStorage.getItem('niyantrana_token') : null;

  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers ?? {}),
    },
  });

  if (!res.ok) {
    const body = await safeJson<{ detail?: string }>(res);
    throw new ApiError(res.status, body.detail ?? `HTTP ${res.status}`);
  }

  return safeJson<T>(res);
}

// ─── Base verbs ───────────────────────────────────────────────────────────────

export const api = {
  get:    <T>(path: string)                  => request<T>(path),
  post:   <T>(path: string, body: unknown)   => request<T>(path, { method: 'POST',   body: JSON.stringify(body) }),
  put:    <T>(path: string, body: unknown)   => request<T>(path, { method: 'PUT',    body: JSON.stringify(body) }),
  patch:  <T>(path: string, body: unknown)   => request<T>(path, { method: 'PATCH',  body: JSON.stringify(body) }),
  delete: <T>(path: string)                  => request<T>(path, { method: 'DELETE' }),
};

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ModelItem {
  model_id: string;
  name: string;
  provider: string;
  version_count: number;
  latest_version: number;
  latest_governance_score: number | null;
  latest_risk_class: string | null;
  created_at: string;
}

export interface ModelDetail {
  id: string;
  name: string;
  provider: string;
  risk_tier: string | null;
  governance_score: number | null;
  deployment_environment: string | null;
  business_owner: string | null;
  technical_owner: string | null;
  version_count: number;
  latest_version: number;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ModelVersion {
  version_id: string;
  version_number: number;
  framework: string | null;
  parameters_count: number | null;
  governance_score: number | null;
  risk_class: string | null;
  artifact_url: string | null;
  deployments: { environment: string; status: string; date: string }[];
  created_at: string;
}

export interface DriftReport {
  id: string;
  model_id: string;
  overall_drift_score: number;
  drift_detected: boolean;
  alert_triggered: boolean;
  method: string;
  sample_count: number;
  feature_results: {
    feature: string;
    method: string;
    score: number;
    severity: string;
    drifted: boolean;
    ref_count: number;
    cur_count: number;
  }[];
  reference_window_start: string | null;
  reference_window_end: string | null;
  current_window_start: string | null;
  current_window_end: string | null;
  created_at: string;
}

export interface DriftHistoryItem {
  id: string;
  overall_drift_score: number;
  drift_detected: boolean;
  method: string;
  sample_count: number;
  created_at: string;
}

export interface AlertRule {
  id: string;
  name: string;
  metric: string;
  condition: Record<string, unknown>;
  severity: string;
  is_active: boolean;
  created_at: string;
}

export interface AlertEvent {
  id: string;
  rule_id: string;
  rule_name?: string;
  severity: string;
  message: string;
  resolved: boolean;
  created_at: string;
}

export interface Contract {
  id: string;
  model_id: string;
  model_name?: string;
  name: string;
  contract_type: string;
  status: string;
  breach_rate?: number;
  last_checked?: string;
  definition: Record<string, unknown>;
}

export interface GovernanceScore {
  model_id: string;
  overall_score: number;
  verdict: string;
  dimension_scores: Record<string, number>;
  computed_at: string;
}

export interface ObservabilityFeed {
  events: {
    id: string;
    event_type: string;
    model_id: string;
    message: string;
    severity: string;
    timestamp: string;
  }[];
  total: number;
}

export interface AuditLog {
  id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  user_id: string;
  timestamp: string;
  details: Record<string, unknown>;
}

export interface RedTeamSession {
  id: string;
  model_id: string;
  session_name: string;
  status: string;
  attack_types: string[];
  vulnerability_count: number;
  created_at: string;
}

export interface DashboardStats {
  total_models: number;
  active_contracts: number;
  alerts_today: number;
  avg_governance_score: number;
  models_by_verdict: Record<string, number>;
  recent_activity: { event: string; model: string; time: string }[];
}

// ─── Models API ───────────────────────────────────────────────────────────────

export const modelsApi = {
  list: (page = 1, perPage = 50) =>
    api.get<{ total: number; page: number; per_page: number; items: ModelItem[] }>(
      `/models?page=${page}&per_page=${perPage}`
    ),
  get: (modelId: string) => api.get<ModelDetail>(`/models/${modelId}`),
  versions: (modelId: string) =>
    api.get<{ model_id: string; model_name: string; versions: ModelVersion[] }>(
      `/models/${modelId}/versions`
    ),
  register: (data: { model_name: string; description?: string; owner?: string }) =>
    api.post<{ model_id: string; model_name: string; status: string }>('/models/register', data),
};

// ─── Drift API ────────────────────────────────────────────────────────────────

export const driftApi = {
  latestReport: (modelId: string) => api.get<DriftReport>(`/drift/${modelId}/report`),
  history: (modelId: string, limit = 30) =>
    api.get<DriftHistoryItem[]>(`/drift/${modelId}/history?limit=${limit}`),
  trigger: (modelId: string, method = 'ks') =>
    api.post<{ status: string; report_id?: string; overall_drift_score?: number; drift_detected?: boolean }>(
      `/drift/${modelId}/trigger?method=${method}`, {}
    ),
};

// ─── Alerts API ───────────────────────────────────────────────────────────────

export const alertsApi = {
  listEvents: (limit = 50, resolved?: boolean) => {
    const qs = resolved !== undefined ? `?limit=${limit}&resolved=${resolved}` : `?limit=${limit}`;
    return api.get<{ items: AlertEvent[]; total: number }>(`/alerts/events${qs}`);
  },
  listRules: () => api.get<{ items: AlertRule[] }>('/alerts/rules'),
  resolve: (eventId: string) => api.post<{ status: string }>(`/alerts/events/${eventId}/resolve`, {}),
  createRule: (data: { name: string; metric: string; condition: Record<string, unknown>; severity: string }) =>
    api.post<AlertRule>('/alerts/rules', data),
};

// ─── Governance / Audit API ───────────────────────────────────────────────────

export const governanceApi = {
  score: (modelId: string) => api.get<GovernanceScore>(`/governance/${modelId}/score`),
  auditLogs: (limit = 50) => api.get<{ items: AuditLog[]; total: number }>(`/governance/audit-logs?limit=${limit}`),
  runAudit: (modelId: string) => api.post<{ task_id: string; status: string }>(`/governance/${modelId}/audit`, {}),
};

// ─── Contracts API ────────────────────────────────────────────────────────────

export const contractsApi = {
  list: (modelId?: string) => {
    const qs = modelId ? `?model_id=${modelId}` : '';
    return api.get<{ items: Contract[]; total: number }>(`/contracts${qs}`);
  },
  get: (contractId: string) => api.get<Contract>(`/contracts/${contractId}`),
  create: (data: { model_id: string; name: string; contract_type: string; definition: Record<string, unknown> }) =>
    api.post<Contract>('/contracts', data),
  evaluate: (contractId: string) => api.post<{ status: string; verdict: string; breach_rate: number }>(
    `/contracts/${contractId}/evaluate`, {}
  ),
};

// ─── Observability API ────────────────────────────────────────────────────────

export const observeApi = {
  feed: (limit = 50, modelId?: string) => {
    const qs = modelId ? `?limit=${limit}&model_id=${modelId}` : `?limit=${limit}`;
    return api.get<ObservabilityFeed>(`/observe/feed${qs}`);
  },
  stats: () => api.get<DashboardStats>('/observe/stats'),
  modelHealth: (modelId: string) =>
    api.get<{ model_id: string; health_score: number; metrics: Record<string, number> }>(
      `/observe/model/${modelId}/health`
    ),
};

// ─── Red Team API ─────────────────────────────────────────────────────────────

export const redTeamApi = {
  list: () => api.get<{ items: RedTeamSession[] }>('/red-team/sessions'),
  get: (sessionId: string) => api.get<RedTeamSession>(`/red-team/sessions/${sessionId}`),
  create: (data: { model_id: string; session_name: string; attack_types: string[] }) =>
    api.post<RedTeamSession>('/red-team/sessions', data),
  run: (sessionId: string) => api.post<{ status: string }>(`/red-team/sessions/${sessionId}/run`, {}),
};

// ─── Inventory / AIBOM ────────────────────────────────────────────────────────

export const inventoryApi = {
  aibom: (modelId: string) =>
    api.get<{ model_id: string; components: { name: string; version: string; type: string; hash: string; cves: number }[] }>(
      `/aibom/${modelId}`
    ),
};

// ─── Reports API ──────────────────────────────────────────────────────────────

export const reportsApi = {
  generatePdf: (modelId: string) =>
    api.post<{ task_id: string; status: string }>(`/reports/${modelId}/pdf`, {}),
  list: (modelId?: string) => {
    const qs = modelId ? `?model_id=${modelId}` : '';
    return api.get<{ items: { id: string; model_id: string; report_type: string; created_at: string; file_url?: string }[] }>(
      `/reports${qs}`
    );
  },
};

// ─── Predictions / Ingest API ─────────────────────────────────────────────────

export const predictionsApi = {
  list: (modelId: string, limit = 50) =>
    api.get<{ items: { id: string; prediction: string; confidence: number | null; latency_ms: number | null; timestamp: string }[]; total: number }>(
      `/predictions/${modelId}?limit=${limit}`
    ),
};
