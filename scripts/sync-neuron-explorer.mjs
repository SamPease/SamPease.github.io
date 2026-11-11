#!/usr/bin/env node
// Copies the neuron explorer data bundle (index.json + n/<id>.json) into
// public/neuron-explorer/data/, where the explorer page lazy-loads it.
//
// Usage:
//   node scripts/sync-neuron-explorer.mjs <bundle dir>
//   NEURON_BUNDLE=<bundle dir> node scripts/sync-neuron-explorer.mjs
//   node scripts/sync-neuron-explorer.mjs --mock   # local test data (needs scripts/drafts/make-neuron-mock.mjs)
import { cpSync, existsSync, rmSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const dest = join(root, "public/neuron-explorer/data");

let src = process.argv[2] ?? process.env.NEURON_BUNDLE;
if (!src) {
  console.error("Usage: node scripts/sync-neuron-explorer.mjs <bundle dir> (or set NEURON_BUNDLE, or pass --mock)");
  process.exit(1);
}
if (src === "--mock") {
  execFileSync(process.execPath, [join(here, "drafts/make-neuron-mock.mjs")], { stdio: "inherit" });
  src = join(here, "drafts/mock-bundle");
}
src = resolve(src);

if (!existsSync(join(src, "index.json"))) {
  console.error(`No index.json in ${src}. Is the bundle built yet? (Use --mock for test data.)`);
  process.exit(1);
}

rmSync(dest, { recursive: true, force: true });
mkdirSync(dest, { recursive: true });
cpSync(src, dest, { recursive: true, dereference: true });

// Sanity check: every card in index.json has a neuron file.
const index = JSON.parse(readFileSync(join(dest, "index.json"), "utf8"));
const cards = [...(index.part1 ?? []), ...(index.part2 ?? [])];
const files = new Set(existsSync(join(dest, "n")) ? readdirSync(join(dest, "n")) : []);
const missing = cards.filter((c) => !files.has(`${c.id}.json`)).map((c) => c.id);
console.log(`Copied ${src} -> ${dest}`);
console.log(`${cards.length} neuron cards, ${files.size} neuron files.`);
if (missing.length) console.warn(`Missing neuron files for: ${missing.join(", ")}`);
