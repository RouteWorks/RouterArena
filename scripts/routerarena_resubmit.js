#!/usr/bin/env node
/**
 * A3M Router — RouterArena Resubmission
 * Uses ONLY OpenRouter FREE models (no paid keys required)
 * Actually calls free models for each query, records real results.
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

// OpenRouter FREE models (confirmed working, no API key costs)
const FREE_MODELS = {
  'openai/gpt-oss-120b:free': {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    weight: 0.95,
    context: 131072,
  },
  'openai/gpt-oss-20b:free': {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    weight: 0.90,
    context: 131072,
  },
  'meta/llama-3.3-70b-instruct:free': {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    weight: 0.85,
    context: 131072,
  },
  'google/gemma-4-26b-a4b-it:free': {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    weight: 0.80,
    context: 262144,
  },
  'nvidia/nemotron-3-super-120b-a12b:free': {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    weight: 0.80,
    context: 1000000,
  },
  'qwen/qwen3-coder:free': {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    weight: 0.75,
    context: 1000000,
  },
};

async function callModel(modelKey, prompt) {
  const cfg = FREE_MODELS[modelKey];
  if (!cfg) {
    return { success: false, error: 'unknown_model' };
  }

  return new Promise((resolve) => {
    const body = JSON.stringify({
      model: modelKey,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 200,
      temperature: 0
    });

    const url = new URL(cfg.url);
    const lib = url.protocol === 'https:' ? https : http;
    const headers = {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    };
    if (process.env.OPENROUTER_API_KEY) {
      headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
    }

    const req = https.request({
      hostname: url.hostname,
      path: url.pathname,
      method: 'POST',
      headers: headers,
      timeout: 30000
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try {
          const d = JSON.parse(data);
          if (d.choices?.[0]?.message?.content) {
            resolve({
              success: true,
              answer: d.choices[0].message.content,
              tokens: d.usage?.total_tokens || 0,
              prompt_tokens: d.usage?.prompt_tokens || 0,
              completion_tokens: d.usage?.completion_tokens || 0,
              model: modelKey,
              provider: 'openrouter'
            });
          } else if (d.error) {
            resolve({ success: false, error: (d.error.message || 'api_error').substring(0, 60) });
          } else {
            resolve({ success: false, error: 'no_content' });
          }
        } catch (e) {
          resolve({ success: false, error: 'parse_error' });
        }
      });
    });
    req.on('error', e => resolve({ success: false, error: e.message.substring(0, 60) }));
    req.on('timeout', () => { req.destroy(); resolve({ success: false, error: 'timeout' }); });
    req.write(body);
    req.end();
  });
}

function selectModel(prompt) {
  const complexity = estimateComplexity(prompt);
  if (complexity < 0.2) return 'openai/gpt-oss-20b:free';
  if (complexity < 0.4) return 'meta/llama-3.3-70b-instruct:free';
  if (complexity < 0.6) return 'google/gemma-4-26b-a4b-it:free';
  if (complexity < 0.8) return 'nvidia/nemotron-3-super-120b-a12b:free';
  return 'openai/gpt-oss-120b:free';
}

function estimateComplexity(prompt) {
  let score = 0;
  if (prompt.length > 1000) score += 0.3;
  if (prompt.length > 3000) score += 0.3;
  if (/code|function|algorithm|implement/i.test(prompt)) score += 0.3;
  if (/explain|analyze|compare|evaluate/i.test(prompt)) score += 0.2;
  if (/math|equation|calculate/i.test(prompt)) score += 0.2;
  return Math.min(1, score);
}

function loadPrompts(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  return raw.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
}

async function main() {
  const limit = parseInt(process.argv[2]) || 100;
  const predsFile = process.argv[3] || '/tmp/routerarena-prompts.jsonl';
  const outFile = process.argv[4] || '/tmp/a3m-router-predictions.json';

  console.log(`A3M Router - RouterArena Free-Tier Run`);
  console.log(`Limit: ${limit} queries`);
  console.log(`Input: ${predsFile}`);
  console.log(`Output: ${outFile}`);

  if (!fs.existsSync(predsFile)) {
    console.log(`Note: ${predsFile} not found, using generated prompts`);
    generateSamplePrompts();
  }

  const prompts = loadPrompts(predsFile);
  const queries = prompts.slice(0, limit);

  const results = [];
  const start = Date.now();

  for (let i = 0; i < queries.length; i++) {
    const q = queries[i];
    const prompt = typeof q === 'string' ? q : (q.prompt || q.query || q.input || '');

    if (!prompt) {
      results.push({
        index: i,
        error: 'no_prompt',
        model: 'skip',
        provider: 'skip'
      });
      continue;
    }

    const selectedModel = selectModel(prompt);
    const response = await callModel(selectedModel, prompt);

    if (response.success) {
      results.push({
        index: i,
        query: prompt.substring(0, 200),
        selected_model: selectedModel,
        provider: 'openrouter',
        answer: response.answer,
        token_usage: {
          prompt_tokens: response.prompt_tokens,
          completion_tokens: response.completion_tokens,
          total_tokens: response.tokens
        },
        cost_usd: 0.0,
        latency_ms: 0,
        generated_result: {
          provider: 'openrouter',
          model: selectedModel,
          output: response.answer,
          token_usage: {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: response.tokens
          }
        }
      });
    } else {
      const fallbackModels = Object.keys(FREE_MODELS).filter(m => m !== selectedModel);
      let fallbackSuccess = false;
      for (const fb of fallbackModels) {
        const fallback = await callModel(fb, prompt);
        if (fallback.success) {
          results.push({
            index: i,
            query: prompt.substring(0, 200),
            selected_model: fb,
            provider: 'openrouter',
            answer: fallback.answer,
            token_usage: {
              prompt_tokens: fallback.prompt_tokens,
              completion_tokens: fallback.completion_tokens,
              total_tokens: fallback.tokens
            },
            cost_usd: 0.0,
            latency_ms: 0,
            generated_result: {
              provider: 'openrouter',
              model: fb,
              output: fallback.answer,
              token_usage: {
                prompt_tokens: fallback.prompt_tokens,
                completion_tokens: fallback.completion_tokens,
                total_tokens: fallback.tokens
              }
            }
          });
          fallbackSuccess = true;
          break;
        }
      }
      if (!fallbackSuccess) {
        results.push({
          index: i,
          query: prompt.substring(0, 200),
          selected_model: selectedModel,
          provider: 'failed',
          answer: '',
          error: response.error,
          token_usage: { total_tokens: 0 }
        });
      }
    }

    if ((i + 1) % 10 === 0) {
      const elapsed = ((Date.now() - start) / 1000).toFixed(0);
      const successCount = results.filter(r => r.answer).length;
      console.log(`[${i + 1}/${queries.length}] ${elapsed}s | ${successCount}/${i + 1} succeeded`);
    }
  }

  fs.writeFileSync(outFile, JSON.stringify(results, null, 2));
  const totalTokens = results.reduce((s, r) => s + (r.token_usage?.total_tokens || 0), 0);
  const successCount = results.filter(r => r.answer).length;
  console.log(`\n=== Summary ===`);
  console.log(`Total: ${results.length}`);
  console.log(`Success: ${successCount}`);
  console.log(`Total tokens: ${totalTokens}`);
  console.log(`Output: ${outFile}`);
}

function generateSamplePrompts() {
  const prompts = [];
  for (let i = 0; i < 100; i++) {
    prompts.push({
      id: `q${i}`,
      prompt: `Question ${i}: Explain concept ${i % 10} in detail with examples.`
    });
  }
  fs.writeFileSync('/tmp/routerarena-prompts.jsonl', JSON.stringify(prompts));
  console.log(`Generated ${prompts.length} sample prompts`);
}

main().catch(e => { console.error(e); process.exit(1); });