// Sandy Baileys WhatsApp sidecar (master plan Part 7).
//
// Why a Node sidecar at all: Baileys is the only maintained zero-cost
// WhatsApp client, it's JS-only. It pairs with Ruk's number exactly once
// via QR; session creds persist in Supabase sandy_config['baileys_creds']
// so container rebuilds/restarts reconnect silently instead of asking
// for a new QR every boot.
//
// Endpoints (bound to 127.0.0.1 only -- never exposed publicly):
//   GET  /status -> {connected, hasCreds, qr_needed}
//   POST /send   {text} -> {ok, detail}
//
// Design rules mirroring notify.py: this server never crashes from a
// bad request or a dead socket -- every failure path answers HTTP with
// {ok:false,...} and keeps serving.

import http from "node:http";
import fs from "node:fs";
import {
  default as makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
} from "@whiskeysockets/baileys";
import pino from "pino";
import qrcode from "qrcode-terminal";

import { getSupabase } from "./supabase-client.mjs";

const PORT = 3001;
const CREDS_KEY = "baileys_creds";
const logger = pino({ level: process.env.BAILEYS_LOG_LEVEL || "warn" });

let sock = null;
let connected = false;
let qrNeeded = false;
let starting = false;
let lastConnectError = "";

// ---------------------------------------------------- Supabase creds store

async function loadCreds() {
  // Returns { state, saveCreds } backed by sandy_config[CREDS_KEY].
  const supabase = getSupabase();
  let saved = null;
  try {
    saved = await supabase.getValue(CREDS_KEY);
  } catch (e) {
    log(`creds read failed: ${e}`);
  }
  if (saved && typeof saved === "object") {
    // Rehydrate Baileys' on-disk shape into a temp dir so
    // useMultiFileAuthState can manage it exactly as designed.
    const dir = "/tmp/baileys-creds";
    fs.rmSync(dir, { recursive: true, force: true });
    fs.mkdirSync(dir, { recursive: true });
    for (const [name, content] of Object.entries(saved)) {
      if (typeof content === "string") {
        fs.writeFileSync(`${dir}/${name}`, content);
      } else {
        fs.writeFileSync(`${dir}/${name}`, JSON.stringify(content));
      }
    }
  }
  return await useMultiFileAuthState("/tmp/baileys-creds");
}

async function persistCreds(state) {
  try {
    const dir = "/tmp/baileys-creds";
    const files = fs.readdirSync(dir);
    const blob = {};
    for (const f of files) {
      const raw = fs.readFileSync(`${dir}/${f}`, "utf8");
      // JSON files parse so creds-* keys stay structured in jsonb.
      try {
        blob[f] = JSON.parse(raw);
      } catch {
        blob[f] = raw;
      }
    }
    const supabase = getSupabase();
    await supabase.upsertValue(CREDS_KEY, blob);
    log("creds persisted to Supabase");
  } catch (e) {
    log(`creds persist FAILED: ${e} -- next restart will need a fresh QR`);
  }
}

function log(msg) {
  console.log(`[baileys-sidecar] ${msg}`);
}

// ---------------------------------------------------------------- socket

async function startSocket() {
  if (starting || (sock && connected)) return;
  starting = true;
  qrNeeded = false;
  try {
    const { state, saveCreds } = await loadCreds();
    const hasCreds =
      state.creds && state.creds.registered === true;

    const { version } = await fetchLatestBaileysVersion();
    sock = makeWASocket({
      version,
      auth: state,
      logger,
      printQRInTerminal: false,
      browser: ["Sandy", "Chrome", "1.0"],
      markOnlineOnConnect: false,
    });

    sock.ev.on("creds.update", async () => {
      try {
        await saveCreds();
        await persistCreds(state);
      } catch (e) {
        log(`saveCreds/persist error: ${e}`);
      }
      if (!hasCreds && state.creds && state.creds.registered) {
        log("first pairing complete");
      }
    });

    sock.ev.on("connection.update", async (u) => {
      const { connection, qr } = u;
      if (qr) {
        qrNeeded = true;
        log("QR NEEDED -- scan with WhatsApp (Linked Devices):");
        qrcode.generate(qr, { small: true });
      }
      if (connection === "open") {
        connected = true;
        qrNeeded = false;
        lastConnectError = "";
        log(`connected as ${sock.user?.id || "?"}`);
      }
      if (connection === "close") {
        connected = false;
        const code = u.lastDisconnect?.error?.output?.statusCode;
        const loggedOut = code === DisconnectReason.loggedOut;
        if (loggedOut) {
          log("logged out remotely -- clearing creds, QR required");
          await clearPersistedCreds();
        }
        const shouldReconnect = !loggedOut;
        if (shouldReconnect) {
          const delayMs = 5000 + Math.floor(Math.random() * 5000);
          log(`disconnected (${code}) -- reconnecting in ${delayMs / 1000}s`);
          setTimeout(() => startSocket().catch((e) => log(`reconnect failed: ${e}`)), delayMs);
        } else {
          log("logged out; waiting for fresh QR scan");
        }
      }
    });
  } catch (e) {
    lastConnectError = String(e);
    log(`startSocket failed: ${e} -- retrying in 30s`);
    setTimeout(() => startSocket().catch(() => {}), 30000);
  } finally {
    starting = false;
  }
}

async function clearPersistedCreds() {
  try {
    const supabase = getSupabase();
    await supabase.deleteValue(CREDS_KEY);
    log("persisted creds cleared after remote logout");
  } catch (e) {
    log(`clearing persisted creds failed: ${e}`);
  }
}

async function sendText(text) {
  if (!connected || !sock) {
    return { ok: false, detail: connected ? "" : "not_connected" };
  }
  const to = process.env.RUK_WHATSAPP_NUMBER;
  if (!to) return { ok: false, detail: "RUK_WHATSAPP_NUMBER not set" };
  const jid = `${to.replace(/[^\d]/g, "")}@s.whatsapp.net`;
  try {
    await sock.sendMessage(jid, { text });
    return { ok: true };
  } catch (e) {
    return { ok: false, detail: String(e).slice(0, 160) };
  }
}

// ---------------------------------------------------------------- http

const server = http.createServer(async (req, res) => {
  res.setHeader("Content-Type", "application/json");
  const url = req.url.split("?")[0];
  try {
    if (req.method === "GET" && url === "/status") {
      answer(res, 200, {
        connected,
        hasCreds: Boolean(sock?.user?.id),
        qr_needed: qrNeeded,
        last_error: lastConnectError,
      });
      return;
    }
    if (req.method === "POST" && url === "/send") {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const bodyRaw = Buffer.concat(chunks).toString("utf8").slice(0, 65536);
      let body = {};
      try {
        body = JSON.parse(bodyRaw || "{}");
      } catch {
        answer(res, 400, { ok: false, detail: "bad json" });
        return;
      }
      const text = typeof body.text === "string" ? body.text.slice(0, 4096) : "";
      if (!text) {
        answer(res, 400, { ok: false, detail: "text required" });
        return;
      }
      const result = await sendText(text);
      answer(res, result.ok ? 200 : 503, result);
      return;
    }
    answer(res, 404, { ok: false, detail: "not found" });
  } catch (e) {
    log(`http handler error: ${e}`);
    answer(res, 500, { ok: false, detail: "internal" });
  }
});

function answer(res, code, obj) {
  res.statusCode = code;
  res.end(JSON.stringify(obj));
}

server.listen(PORT, "127.0.0.1", () => {
  log(`listening on 127.0.0.1:${PORT}`);
});
startSocket().catch((e) => log(`initial connect failed: ${e}`));
