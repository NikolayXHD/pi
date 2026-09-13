/**
 * Test totalTokens field across all providers.
 *
 * totalTokens represents the total number of tokens processed by the LLM,
 * including input (with cache) and output (with thinking). This is the
 * base for calculating context size for the next request.
 *
 * - OpenAI Completions: Uses native total_tokens field
 * - OpenAI Responses: Uses native total_tokens field
 * - Google: Uses native totalTokenCount field
 * - Anthropic: Computed as input + output + cacheRead + cacheWrite
 * - Other OpenAI-compatible providers: Uses native total_tokens field
 */

import { describe, expect, it } from "vitest";
import { complete, getModel } from "../src/compat.ts";
import type { Api, Context, Model, StreamOptions, Usage } from "../src/types.ts";

type StreamOptionsWithExtras = StreamOptions & Record<string, unknown>;

import { resolveApiKey } from "./oauth.ts";

// Resolve OAuth tokens at module level (async, runs before tests)
const oauthTokens = await Promise.all([resolveApiKey("minimax"), resolveApiKey("deepseek"), resolveApiKey("deepseek")]);
const [_anthropicOAuthToken, _githubCopilotToken, _openaiCodexToken] = oauthTokens;

// Generate a long system prompt to trigger caching (>2k bytes for most providers)
const LONG_SYSTEM_PROMPT = `You are a helpful assistant. Be concise in your responses.

Here is some additional context that makes this system prompt long enough to trigger caching:

${Array(50)
	.fill(
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
	)
	.join("\n\n")}

Remember: Always be helpful and concise.`;

async function testTotalTokensWithCache<TApi extends Api>(
	llm: Model<TApi>,
	options: StreamOptionsWithExtras = {},
): Promise<{ first: Usage; second: Usage }> {
	// First request - no cache
	const context1: Context = {
		systemPrompt: LONG_SYSTEM_PROMPT,
		messages: [
			{
				role: "user",
				content: "What is 2 + 2? Reply with just the number.",
				timestamp: Date.now(),
			},
		],
	};

	const response1 = await complete(llm, context1, options);
	expect(response1.stopReason).toBe("stop");

	// Second request - should trigger cache read (same system prompt, add conversation)
	const context2: Context = {
		systemPrompt: LONG_SYSTEM_PROMPT,
		messages: [
			...context1.messages,
			response1, // Include previous assistant response
			{
				role: "user",
				content: "What is 3 + 3? Reply with just the number.",
				timestamp: Date.now(),
			},
		],
	};

	const response2 = await complete(llm, context2, options);
	expect(response2.stopReason).toBe("stop");

	return { first: response1.usage, second: response2.usage };
}

function logUsage(label: string, usage: Usage) {
	const computed = usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
	console.log(`  ${label}:`);
	console.log(
		`    input: ${usage.input}, output: ${usage.output}, cacheRead: ${usage.cacheRead}, cacheWrite: ${usage.cacheWrite}`,
	);
	console.log(`    totalTokens: ${usage.totalTokens}, computed: ${computed}`);
}

function assertTotalTokensEqualsComponents(usage: Usage) {
	const computed = usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
	expect(usage.totalTokens).toBe(computed);
}

describe("totalTokens field", () => {
	// =========================================================================
	// Anthropic
	// =========================================================================

	// =========================================================================
	// OpenAI
	// =========================================================================

	// =========================================================================
	// Google
	// =========================================================================

	// =========================================================================
	// xAI
	// =========================================================================

	// =========================================================================
	// Groq
	// =========================================================================

	// =========================================================================
	// Cerebras
	// =========================================================================

	// =========================================================================
	// Cloudflare Workers AI
	// =========================================================================

	// =========================================================================
	// Cloudflare AI Gateway
	// =========================================================================

	// =========================================================================
	// Hugging Face
	// =========================================================================

	// =========================================================================
	// Together AI
	// =========================================================================

	// =========================================================================
	// Baseten
	// =========================================================================

	// =========================================================================
	// z.ai
	// =========================================================================

	describe.skipIf(!process.env.ZAI_API_KEY)("z.ai", () => {
		it("glm-5.2 - should return totalTokens equal to sum of components", { retry: 3, timeout: 60000 }, async () => {
			const llm = getModel("zai", "glm-5.2");

			console.log(`\nz.ai / ${llm.id}:`);
			const { first, second } = await testTotalTokensWithCache(llm, { apiKey: process.env.ZAI_API_KEY });

			logUsage("First request", first);
			logUsage("Second request", second);

			assertTotalTokensEqualsComponents(first);
			assertTotalTokensEqualsComponents(second);
		});
	});

	// =========================================================================
	// Mistral
	// =========================================================================

	// =========================================================================
	// MiniMax
	// =========================================================================

	describe.skipIf(!process.env.MINIMAX_API_KEY)("MiniMax", () => {
		it(
			"MiniMax-M2.7 - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("minimax", "MiniMax-M2.7");

				console.log(`\nMiniMax / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, { apiKey: process.env.MINIMAX_API_KEY });

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Xiaomi MiMo
	// =========================================================================

	describe.skipIf(!process.env.XIAOMI_API_KEY)("Xiaomi MiMo (API billing)", () => {
		it(
			"mimo-v2.5-pro - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("xiaomi", "mimo-v2.5-pro");

				console.log(`\nXiaomi MiMo / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, { apiKey: process.env.XIAOMI_API_KEY });

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Xiaomi MiMo Token Plan CN
	// =========================================================================

	describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY)("Xiaomi MiMo Token Plan (CN)", () => {
		it(
			"mimo-v2.5-pro - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("xiaomi-token-plan-cn", "mimo-v2.5-pro");

				console.log(`\nXiaomi MiMo Token Plan CN / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, {
					apiKey: process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY,
				});

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Xiaomi MiMo Token Plan AMS
	// =========================================================================

	describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY)("Xiaomi MiMo Token Plan (AMS)", () => {
		it(
			"mimo-v2.5-pro - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("xiaomi-token-plan-ams", "mimo-v2.5-pro");

				console.log(`\nXiaomi MiMo Token Plan AMS / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, {
					apiKey: process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY,
				});

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Xiaomi MiMo Token Plan SGP
	// =========================================================================

	describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY)("Xiaomi MiMo Token Plan (SGP)", () => {
		it(
			"mimo-v2.5-pro - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("xiaomi-token-plan-sgp", "mimo-v2.5-pro");

				console.log(`\nXiaomi MiMo Token Plan SGP / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, {
					apiKey: process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY,
				});

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Qwen Token Plan
	// =========================================================================

	describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)("Qwen Token Plan", () => {
		it(
			"qwen3.7-max - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("qwen-token-plan", "qwen3.7-max");

				console.log(`\nQwen Token Plan / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, {
					apiKey: process.env.QWEN_TOKEN_PLAN_API_KEY,
				});

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Qwen Token Plan Individual
	// =========================================================================

	describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)("Qwen Token Plan Individual", () => {
		it(
			"qwen3.8-max - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("qwen-token-plan-individual", "qwen3.8-max");

				console.log(`\nQwen Token Plan Individual / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, {
					apiKey: process.env.QWEN_TOKEN_PLAN_API_KEY,
				});

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Qwen Token Plan CN
	// =========================================================================

	describe.skipIf(!process.env.QWEN_TOKEN_PLAN_CN_API_KEY)("Qwen Token Plan (CN)", () => {
		it(
			"qwen3.7-max - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("qwen-token-plan-cn", "qwen3.7-max");

				console.log(`\nQwen Token Plan CN / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, {
					apiKey: process.env.QWEN_TOKEN_PLAN_CN_API_KEY,
				});

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Kimi For Coding
	// =========================================================================

	describe.skipIf(!process.env.KIMI_API_KEY)("Kimi For Coding", () => {
		it(
			"kimi-for-coding - should return totalTokens equal to sum of components",
			{ retry: 3, timeout: 60000 },
			async () => {
				const llm = getModel("kimi-coding", "kimi-for-coding");

				console.log(`\nKimi For Coding / ${llm.id}:`);
				const { first, second } = await testTotalTokensWithCache(llm, { apiKey: process.env.KIMI_API_KEY });

				logUsage("First request", first);
				logUsage("Second request", second);

				assertTotalTokensEqualsComponents(first);
				assertTotalTokensEqualsComponents(second);
			},
		);
	});

	// =========================================================================
	// Vercel AI Gateway
	// =========================================================================

	// =========================================================================
	// OpenRouter - Multiple backend providers
	// =========================================================================

	// =========================================================================
	// GitHub Copilot (OAuth)
	// =========================================================================

	// =========================================================================
	// =========================================================================

	// =========================================================================
	// =========================================================================

	// =========================================================================
	// OpenAI Codex (OAuth)
	// =========================================================================
});
