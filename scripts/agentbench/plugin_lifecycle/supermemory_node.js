const fs = require("fs");
const path = require("path");

const supermemoryModule = require("supermemory");
const Supermemory = supermemoryModule.default || supermemoryModule.Supermemory || supermemoryModule;

const action = process.argv[2];
const arg1 = process.argv[3];
const arg2 = process.argv[4];
const configPath = process.env.OPENCLAW_CONFIG || path.join(process.env.HOME || "/root", ".openclaw", "openclaw.json");
const pluginName = process.env.SUPERMEMORY_PLUGIN_NAME || "openclaw-supermemory";
const batchSize = Number(process.env.SUPERMEMORY_BATCH_SIZE || 50);

function readConfig() {
  if (!fs.existsSync(configPath)) throw new Error(`OpenClaw config not found: ${configPath}`);
  return JSON.parse(fs.readFileSync(configPath, "utf8"));
}

function writeConfig(cfg) {
  fs.writeFileSync(configPath, JSON.stringify(cfg, null, 2) + "\n");
}

function truthy(value) {
  return ["1", "true", "yes", "y"].includes(String(value || "").toLowerCase());
}

function resolveSettings() {
  const cfg = readConfig();
  const plugins = cfg.plugins || {};
  const entry = ((plugins.entries || {})[pluginName]) || {};
  const pluginCfg = entry.config || {};
  const apiKey = pluginCfg.apiKey || process.env.SUPERMEMORY_OPENCLAW_API_KEY;
  const baseURL = (pluginCfg.baseUrl || process.env.SUPERMEMORY_BASE_URL || "https://api.supermemory.ai").replace(/\/+$/, "");
  const containerTag = process.env.SUPERMEMORY_CONTAINER_TAG || pluginCfg.containerTag;
  if (!apiKey) throw new Error(`${pluginName} apiKey is missing`);
  if (!containerTag) throw new Error(`${pluginName} containerTag is missing`);
  return { cfg, apiKey, baseURL, containerTag };
}

function client() {
  const settings = resolveSettings();
  return {
    ...settings,
    sm: new Supermemory({
      apiKey: settings.apiKey,
      baseURL: settings.baseURL,
      defaultHeaders: { "x-sm-source": "openclaw" },
    }),
  };
}

function asMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const out = {};
  for (const [key, item] of Object.entries(value)) {
    if (["string", "number", "boolean"].includes(typeof item)) out[key] = item;
    else if (Array.isArray(item) && item.every((x) => typeof x === "string")) out[key] = item;
  }
  return out;
}

async function listAll(sm, containerTag, includeContent = true) {
  const docs = [];
  let page = 1;
  while (true) {
    const res = await sm.documents.list({
      containerTags: [containerTag],
      includeContent,
      limit: 100,
      page,
      sort: "createdAt",
      order: "asc",
    });
    docs.push(...(res.memories || []));
    const totalPages = res.pagination && res.pagination.totalPages ? Number(res.pagination.totalPages) : page;
    if (page >= totalPages || !res.memories || res.memories.length === 0) break;
    page += 1;
  }
  return docs;
}

async function wipe(sm, containerTag) {
  const docs = await listAll(sm, containerTag, false);
  const ids = docs.map((doc) => doc.id).filter(Boolean);
  let deleted = 0;
  for (let i = 0; i < ids.length; i += 100) {
    const batch = ids.slice(i, i + 100);
    const res = await sm.documents.deleteBulk({ ids: batch });
    deleted += Number(res.deletedCount || batch.length);
  }
  return { before: ids.length, deleted };
}

async function main() {
  if (action === "configure") {
    const cfg = readConfig();
    cfg.plugins ||= {};
    cfg.plugins.enabled = true;
    cfg.plugins.allow = Array.isArray(cfg.plugins.allow) ? cfg.plugins.allow : [];
    if (!cfg.plugins.allow.includes(pluginName)) cfg.plugins.allow.push(pluginName);
    cfg.plugins.slots ||= {};
    cfg.plugins.slots.memory = pluginName;
    cfg.plugins.entries ||= {};
    cfg.plugins.entries[pluginName] ||= {};
    cfg.plugins.entries[pluginName].enabled = true;
    cfg.plugins.entries[pluginName].hooks ||= {};
    cfg.plugins.entries[pluginName].hooks.allowConversationAccess = true;
    cfg.plugins.entries[pluginName].config ||= {};
    if (process.env.SUPERMEMORY_CONTAINER_TAG) {
      cfg.plugins.entries[pluginName].config.containerTag = process.env.SUPERMEMORY_CONTAINER_TAG;
    }
    cfg.plugins.entries[pluginName].config.autoRecall = true;
    cfg.plugins.entries[pluginName].config.autoCapture = true;
    writeConfig(cfg);
    console.log(JSON.stringify({ configured: pluginName }));
    return;
  }

  if (action === "mode") {
    const mode = arg1;
    if (!["train", "test"].includes(mode)) throw new Error("mode requires train or test");
    const cfg = readConfig();
    const entry = cfg.plugins?.entries?.[pluginName];
    if (!entry) throw new Error(`${pluginName} is not configured`);
    entry.enabled = true;
    entry.config ||= {};
    entry.config.autoRecall = true;
    entry.config.autoCapture = mode === "train" || truthy(process.env.SUPERMEMORY_TEST_AUTOCAPTURE);
    writeConfig(cfg);
    console.log(JSON.stringify({ mode, autoCapture: entry.config.autoCapture }));
    return;
  }

  const { sm, containerTag } = client();

  if (action === "count") {
    const docs = await listAll(sm, containerTag, false);
    console.log(JSON.stringify({ containerTag, total: docs.length }));
    return;
  }

  if (action === "wipe") {
    console.log(JSON.stringify({ containerTag, ...(await wipe(sm, containerTag)) }));
    return;
  }

  if (action === "export") {
    const output = arg1;
    if (!output) throw new Error("export requires output path");
    const docs = await listAll(sm, containerTag, true);
    fs.mkdirSync(path.dirname(output), { recursive: true });
    const stream = fs.createWriteStream(output, { encoding: "utf8" });
    for (const doc of docs) {
      stream.write(JSON.stringify({
        content: doc.content || doc.summary || "",
        customId: doc.customId || undefined,
        filepath: doc.filepath || undefined,
        metadata: asMetadata(doc.metadata),
        taskType: "memory",
        status: doc.status || undefined,
      }) + "\n");
    }
    await new Promise((resolve) => stream.end(resolve));
    console.log(JSON.stringify({ containerTag, exported: docs.length, output }));
    return;
  }

  if (action === "import") {
    const input = arg1;
    const mode = arg2 || "replace";
    if (!input) throw new Error("import requires input path");
    if (mode === "replace") await wipe(sm, containerTag);
    const lines = fs.readFileSync(input, "utf8").split(/\r?\n/).filter(Boolean);
    const docs = lines.map((line) => {
      const item = JSON.parse(line);
      return {
        content: String(item.content || "").trim(),
        customId: item.customId || undefined,
        filepath: item.filepath || undefined,
        metadata: asMetadata(item.metadata),
      };
    }).filter((doc) => doc.content);
    let imported = 0;
    for (let i = 0; i < docs.length; i += batchSize) {
      const batch = docs.slice(i, i + batchSize);
      const res = await sm.documents.batchAdd({ documents: batch, containerTag });
      imported += Number(res.success || batch.length - Number(res.failed || 0));
    }
    console.log(JSON.stringify({ containerTag, imported, input, mode }));
    return;
  }

  throw new Error(`unknown action: ${action}`);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
