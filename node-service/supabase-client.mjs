// Minimal Supabase/PostgREST access for the Baileys sidecar -- plain
// global fetch (node >= 18), ZERO extra npm deps. Only sandy_config
// key/value rows are touched; same env vars the Python app uses.
//
// Exports getSupabase() with three async helpers:
//   getValue(key) -> parsed jsonb value | null
//   upsertValue(key, value)
//   deleteValue(key)

const REST_URL = `${(process.env.SUPABASE_URL || "").replace(/\/+$/, "")}/rest/v1`;
const API_KEY =
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_KEY || "";

export function getSupabase() {
  return {
    async getValue(key) {
      if (!REST_URL.startsWith("http") || !API_KEY) {
        throw new Error("SUPABASE_URL/SUPABASE_SERVICE_KEY not set");
      }
      const url =
        `${REST_URL}/sandy_config?select=value&key=eq.${encodeURIComponent(key)}`;
      const r = await fetch(url, {
        headers: {
          apikey: API_KEY,
          Authorization: `Bearer ${API_KEY}`,
        },
      });
      if (!r.ok) throw new Error(`select ${key}: HTTP ${r.status}`);
      const rows = await r.json();
      return rows.length ? rows[0].value : null;
    },

    async upsertValue(key, value) {
      if (!REST_URL.startsWith("http") || !API_KEY) {
        throw new Error("SUPABASE_URL/SUPABASE_SERVICE_KEY not set");
      }
      const r = await fetch(`${REST_URL}/sandy_config`, {
        method: "POST",
        headers: {
          apikey: API_KEY,
          Authorization: `Bearer ${API_KEY}`,
          "Content-Type": "application/json",
          Prefer: "resolution=merge-duplicates",
        },
        body: JSON.stringify([{ key, value }]),
      });
      if (!r.ok) {
        throw new Error(`upsert ${key}: HTTP ${r.status} ${await r.text()}`);
      }
    },

    async deleteValue(key) {
      if (!REST_URL.startsWith("http") || !API_KEY) {
        throw new Error("SUPABASE_URL/SUPABASE_SERVICE_KEY not set");
      }
      const r = await fetch(
        `${REST_URL}/sandy_config?key=eq.${encodeURIComponent(key)}`,
        {
          method: "DELETE",
          headers: {
            apikey: API_KEY,
            Authorization: `Bearer ${API_KEY}`,
          },
        }
      );
      if (!r.ok) throw new Error(`delete ${key}: HTTP ${r.status}`);
    },
  };
}
