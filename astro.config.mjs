import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  site: "https://sampease.github.io",
  output: "static",
  redirects: {
    "/barselo": "/project-writeups/barselo/",
    "/instagram-network": "/project-writeups/instagram-network/",
    "/retraining-unlearning": "/project-writeups/retraining-unlearning/",
    "/trans-advice-agent": "/project-writeups/trans-advice-agent/",
  },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
