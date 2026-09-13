import { describe, expect, it } from "vitest";
import { clampThinkingLevel, getSupportedThinkingLevels } from "../src/compat.ts";
import type { Api, Model } from "../src/types.ts";

function _createModel<TApi extends Api>(api: TApi): Model<TApi> {
	return {
		id: "test-model",
		name: "Test Model",
		api,
		provider: "test-provider",
		baseUrl: "https://upstream.test/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 10_000,
		maxTokens: 1_000,
	};
}

function _mockToken(): string {
	const payload = Buffer.from(
		JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "acc_test" } }),
		"utf8",
	).toString("base64");
	return `aaa.${payload}.bbb`;
}

describe("max thinking level", () => {
	it("is opt-in for ordinary reasoning models", () => {
		const model: Model<"openai-completions"> = {
			id: "ordinary-reasoning",
			name: "Ordinary Reasoning",
			api: "openai-completions",
			provider: "test",
			baseUrl: "https://example.com/v1",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 4096,
		};

		expect(getSupportedThinkingLevels(model)).toEqual(["off", "minimal", "low", "medium", "high"]);
		expect(clampThinkingLevel(model, "max")).toBe("high");
	});

	it("supports a hole between high and max", () => {
		const model: Model<"openai-completions"> = {
			id: "high-and-max",
			name: "High and Max",
			api: "openai-completions",
			provider: "test",
			baseUrl: "https://example.com/v1",
			reasoning: true,
			thinkingLevelMap: { xhigh: null, max: "max" },
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 4096,
		};

		expect(getSupportedThinkingLevels(model)).toEqual(["off", "minimal", "low", "medium", "high", "max"]);
		expect(clampThinkingLevel(model, "xhigh")).toBe("max");
	});
});
