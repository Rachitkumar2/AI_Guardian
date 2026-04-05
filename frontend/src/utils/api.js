const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || '';

const normalizedBaseUrl = rawBaseUrl.endsWith('/')
  ? rawBaseUrl.slice(0, -1)
  : rawBaseUrl;

export function apiUrl(path) {
  // Keep relative paths in local/dev unless VITE_API_BASE_URL is explicitly set.
  if (!normalizedBaseUrl) return path;
  return `${normalizedBaseUrl}${path}`;
}
