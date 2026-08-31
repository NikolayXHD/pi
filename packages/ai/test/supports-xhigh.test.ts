import { describe, expect, it } from "vitest";
import { getModel, getSupportedThinkingLevels } from "../src/compat.ts";
import type { Model } from "../src/types.ts";

function openRouterModel(
	id: string,
	thinkingLevelMap: Model<"openai-completions">["thinkingLevelMap"],
): Model<"openai-completions"> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200000,
		maxTokens: 8192,
		thinkingLevelMap,
	};
}

describe("getSupportedThinkingLevels", () => {
	it("includes only medium/high/xhigh for OpenRouter GPT-5.5 Pro", () => {
		const model = openRouterModel("openai/gpt-5.5-pro", {
			off: null,
			minimal: null,
			low: null,
			medium: "medium",
			high: "high",
			xhigh: "xhigh",
		});
		expect(getSupportedThinkingLevels(model)).toEqual(["medium", "high", "xhigh"]);
	});

	it("includes low/high/max plus off for DeepSeek V4 Flash on the DeepSeek provider", () => {
		const model = getModel("deepseek", "deepseek-v4-flash");
		expect(model).toBeDefined();
		expect(getSupportedThinkingLevels(model!)).toEqual(["off", "low", "high", "max"]);
	});

	it("excludes thinking off for Moonshot Kimi K2.7 Code models", () => {
		const cases = [getModel("moonshotai", "kimi-k2.7-code"), getModel("moonshotai-cn", "kimi-k2.7-code")];

		for (const model of cases) {
			expect(model).toBeDefined();
			expect(getSupportedThinkingLevels(model!)).toEqual(["minimal", "low", "medium", "high"]);
		}
	});

	it.each(["moonshotai", "moonshotai-cn"] as const)("uses the verified effort options for %s Kimi K3", (provider) => {
		const model = getModel(provider, "kimi-k3");
		expect(model).toBeDefined();
		expect(getSupportedThinkingLevels(model!)).toEqual(["low", "high", "max"]);
	});

	it("includes only low, high, max for Kimi Coding K3", () => {
		const model = getModel("kimi-coding", "k3");
		expect(model).toBeDefined();
		expect(getSupportedThinkingLevels(model!)).toEqual(["low", "high", "max"]);
	});

	it("includes only high/xhigh plus off for DeepSeek V4 Flash on OpenRouter", () => {
		const model = openRouterModel("deepseek/deepseek-v4-flash", {
			off: "off",
			minimal: null,
			low: null,
			medium: null,
			high: "high",
			xhigh: "xhigh",
		});
		expect(getSupportedThinkingLevels(model)).toEqual(["off", "high", "xhigh"]);
	});

	it("includes max but not xhigh for OpenRouter Opus 4.6 (openai-completions API)", () => {
		const model = openRouterModel("anthropic/claude-opus-4.6", {
			off: null,
			minimal: null,
			low: null,
			medium: null,
			high: null,
			max: "max",
		});
		expect(getSupportedThinkingLevels(model)).toContain("max");
		expect(getSupportedThinkingLevels(model)).not.toContain("xhigh");
	});
});
