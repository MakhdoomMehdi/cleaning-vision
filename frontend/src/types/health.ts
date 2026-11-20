export interface HealthResponse {
  status: "ok";
  service: string;
  version: string;
  environment: string;
  timestamp: string;
  openai_configured: boolean;
  openai_model: string;
}
