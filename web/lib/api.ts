const configuredOrigin = process.env.NEXT_PUBLIC_API_URL?.trim();

export const apiOrigin = (configuredOrigin || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiOrigin}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `请求失败（${response.status}）`);
  }
  return response.json() as Promise<T>;
}

export function sseUrl(path: string): string {
  return `${apiOrigin}${path}`;
}
