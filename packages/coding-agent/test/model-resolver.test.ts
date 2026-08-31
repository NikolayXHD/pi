import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";
import { getModel, streamSimple } from "@earendil-works/pi-ai/compat";
import { getBuiltinModels, getBuiltinProviders } from "@earendil-works/pi-ai/providers/all";
import { afterEach, describe, expect, test, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import {
	defaultModelPerProvider,
	findInitialModel,
	parseModelPattern,
	resolveCliModel,
	resolveModelScope,
	resolveModelScopeWithDiagnostics,
} from "../src/core/model-resolver.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestResourceLoader } from "./utilities.ts";

// Mock models for testing
const mockModels: Model<"anthropic-messages">[] = [
	{
		id: "MiniMax-M2.7",
		name: "Claude Sonnet 4.5",
		api: "anthropic-messages",
		provider: "minimax",
		baseUrl: "https://api.anthropic.com",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200000,
		maxTokens: 8192,
	},
	{
		id: "deepseek-v4-pro",
		name: "GPT-4o",
		api: "anthropic-messages", // Using same type for simplicity
		provider: "deepseek",
		baseUrl: "https://api.openai.com",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 5, output: 15, cacheRead: 0.5, cacheWrite: 5 },
		contextWindow: 128000,
		maxTokens: 4096,
	},
];

// Mock router-style model ids with colons
const mockRouterModels: Model<"anthropic-messages">[] = [
	{
		id: "qwen/qwen3-coder:exacto",
		name: "Qwen3 Coder Exacto",
		api: "anthropic-messages",
		provider: "aggregator",
		baseUrl: "https://aggregator.example.com/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
		contextWindow: 128000,
		maxTokens: 8192,
	},
	{
		id: "openai/gpt-4o:extended",
		name: "GPT-4o Extended",
		api: "anthropic-messages",
		provider: "aggregator",
		baseUrl: "https://aggregator.example.com/v1",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 5, output: 15, cacheRead: 0.5, cacheWrite: 5 },
		contextWindow: 128000,
		maxTokens: 4096,
	},
];

const allModels = [...mockModels, ...mockRouterModels];

describe("parseModelPattern", () => {
	describe("simple patterns without colons", () => {
		test("exact match returns model with undefined thinking level", () => {
			const result = parseModelPattern("MiniMax-M2.7", allModels);
			expect(result.model?.id).toBe("MiniMax-M2.7");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toBeUndefined();
		});

		test("partial match returns best model with undefined thinking level", () => {
			const result = parseModelPattern("sonnet", allModels);
			expect(result.model?.id).toBe("MiniMax-M2.7");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toBeUndefined();
		});

		test("no match returns undefined model and thinking level", () => {
			const result = parseModelPattern("nonexistent", allModels);
			expect(result.model).toBeUndefined();
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toBeUndefined();
		});
	});

	describe("patterns with valid thinking levels", () => {
		test("sonnet:high returns sonnet with high thinking level", () => {
			const result = parseModelPattern("sonnet:high", allModels);
			expect(result.model?.id).toBe("MiniMax-M2.7");
			expect(result.thinkingLevel).toBe("high");
			expect(result.warning).toBeUndefined();
		});

		test("deepseek-v4-pro:medium returns deepseek-v4-pro with medium thinking level", () => {
			const result = parseModelPattern("deepseek-v4-pro:medium", allModels);
			expect(result.model?.id).toBe("deepseek-v4-pro");
			expect(result.thinkingLevel).toBe("medium");
			expect(result.warning).toBeUndefined();
		});

		test("all valid thinking levels work", () => {
			for (const level of ["off", "minimal", "low", "medium", "high", "xhigh", "max"]) {
				const result = parseModelPattern(`sonnet:${level}`, allModels);
				expect(result.model?.id).toBe("MiniMax-M2.7");
				expect(result.thinkingLevel).toBe(level);
				expect(result.warning).toBeUndefined();
			}
		});
	});

	describe("patterns with invalid thinking levels", () => {
		test("sonnet:random returns sonnet with undefined thinking level and warning", () => {
			const result = parseModelPattern("sonnet:random", allModels);
			expect(result.model?.id).toBe("MiniMax-M2.7");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toContain("Invalid thinking level");
			expect(result.warning).toContain("random");
		});

		test("deepseek-v4-pro:invalid returns deepseek-v4-pro with undefined thinking level and warning", () => {
			const result = parseModelPattern("deepseek-v4-pro:invalid", allModels);
			expect(result.model?.id).toBe("deepseek-v4-pro");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toContain("Invalid thinking level");
		});
	});

	describe("router-style model ids with colons", () => {
		test("qwen3-coder:exacto matches the model with undefined thinking level", () => {
			const result = parseModelPattern("qwen/qwen3-coder:exacto", allModels);
			expect(result.model?.id).toBe("qwen/qwen3-coder:exacto");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toBeUndefined();
		});

		test("aggregator/qwen/qwen3-coder:exacto matches with provider prefix", () => {
			const result = parseModelPattern("aggregator/qwen/qwen3-coder:exacto", allModels);
			expect(result.model?.id).toBe("qwen/qwen3-coder:exacto");
			expect(result.model?.provider).toBe("aggregator");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toBeUndefined();
		});

		test("qwen3-coder:exacto:high matches model with high thinking level", () => {
			const result = parseModelPattern("qwen/qwen3-coder:exacto:high", allModels);
			expect(result.model?.id).toBe("qwen/qwen3-coder:exacto");
			expect(result.thinkingLevel).toBe("high");
			expect(result.warning).toBeUndefined();
		});

		test("aggregator/qwen/qwen3-coder:exacto:high matches with provider and thinking level", () => {
			const result = parseModelPattern("aggregator/qwen/qwen3-coder:exacto:high", allModels);
			expect(result.model?.id).toBe("qwen/qwen3-coder:exacto");
			expect(result.model?.provider).toBe("aggregator");
			expect(result.thinkingLevel).toBe("high");
			expect(result.warning).toBeUndefined();
		});

		test("deepseek-v4-pro", () => {
			const result = parseModelPattern("openai/gpt-4o:extended", allModels);
			expect(result.model?.id).toBe("openai/gpt-4o:extended");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toBeUndefined();
		});
	});

	describe("invalid thinking levels with router-style model ids", () => {
		test("qwen3-coder:exacto:random returns model with undefined thinking level and warning", () => {
			const result = parseModelPattern("qwen/qwen3-coder:exacto:random", allModels);
			expect(result.model?.id).toBe("qwen/qwen3-coder:exacto");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toContain("Invalid thinking level");
			expect(result.warning).toContain("random");
		});

		test("qwen3-coder:exacto:high:random returns model with undefined thinking level and warning", () => {
			const result = parseModelPattern("qwen/qwen3-coder:exacto:high:random", allModels);
			expect(result.model?.id).toBe("qwen/qwen3-coder:exacto");
			expect(result.thinkingLevel).toBeUndefined();
			expect(result.warning).toContain("Invalid thinking level");
			expect(result.warning).toContain("random");
		});
	});

	describe("edge cases", () => {
		test("empty pattern matches via partial matching", () => {
			// Empty string is included in all model IDs, so partial matching finds a match
			const result = parseModelPattern("", allModels);
			expect(result.model).not.toBeNull();
			expect(result.thinkingLevel).toBeUndefined();
		});

		test("pattern ending with colon treats empty suffix as invalid", () => {
			const result = parseModelPattern("sonnet:", allModels);
			// Empty string after colon is not a valid thinking level
			// So it tries to match "sonnet:" which won't match, then tries "sonnet"
			expect(result.model?.id).toBe("MiniMax-M2.7");
			expect(result.warning).toContain("Invalid thinking level");
		});
	});
});

describe("resolveModelScopeWithDiagnostics", () => {
	test("returns scoped models and structured diagnostics without writing console warnings", async () => {
		const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
		try {
			const registry = {
				getAvailable: () => allModels,
			} as unknown as Parameters<typeof resolveModelScopeWithDiagnostics>[1];

			const result = await resolveModelScopeWithDiagnostics(
				["sonnet:high", "deepseek-v4-pro:invalid", "missing"],
				registry,
			);

			expect(result.scopedModels.map((scoped) => scoped.model.id)).toEqual(["MiniMax-M2.7", "deepseek-v4-pro"]);
			expect(result.scopedModels[0].thinkingLevel).toBe("high");
			expect(result.scopedModels[1].thinkingLevel).toBeUndefined();
			expect(result.diagnostics).toEqual([
				{
					type: "warning",
					message: 'Invalid thinking level "invalid" in pattern "deepseek-v4-pro:invalid". Using default instead.',
					code: "invalid-thinking-level",
					pattern: "deepseek-v4-pro:invalid",
				},
				{
					type: "warning",
					message: 'No models match pattern "missing"',
					code: "no-match",
					pattern: "missing",
				},
			]);
			expect(warn).not.toHaveBeenCalled();
		} finally {
			warn.mockRestore();
		}
	});

	test("resolveModelScope preserves CLI warning output", async () => {
		const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
		try {
			const registry = {
				getAvailable: () => allModels,
			} as unknown as Parameters<typeof resolveModelScope>[1];

			const scopedModels = await resolveModelScope(["missing"], registry);

			expect(scopedModels).toEqual([]);
			expect(warn).toHaveBeenCalledOnce();
			expect(warn.mock.calls[0][0]).toContain('Warning: No models match pattern "missing"');
		} finally {
			warn.mockRestore();
		}
	});

	test("resolves bracketed model ids as exact references before glob matching", async () => {
		const bracketedModel: Model<"anthropic-messages"> = {
			id: "bracketed-model[1m]",
			name: "Bracketed Model",
			api: "anthropic-messages",
			provider: "custom",
			baseUrl: "https://example.invalid",
			reasoning: true,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};
		const registry = {
			getAvailable: () => [...allModels, bracketedModel],
		} as unknown as Parameters<typeof resolveModelScopeWithDiagnostics>[1];

		const result = await resolveModelScopeWithDiagnostics(["custom/bracketed-model[1m]"], registry);

		expect(result.scopedModels.map((scoped) => scoped.model.id)).toEqual(["bracketed-model[1m]"]);
		expect(result.diagnostics).toEqual([]);
	});

	test("resolves bracketed model ids with thinking levels as exact references before glob matching", async () => {
		const bracketedModel: Model<"anthropic-messages"> = {
			id: "bracketed-model[1m]",
			name: "Bracketed Model",
			api: "anthropic-messages",
			provider: "custom",
			baseUrl: "https://example.invalid",
			reasoning: true,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};
		const registry = {
			getAvailable: () => [...allModels, bracketedModel],
		} as unknown as Parameters<typeof resolveModelScopeWithDiagnostics>[1];

		const result = await resolveModelScopeWithDiagnostics(["custom/bracketed-model[1m]:high"], registry);

		expect(result.scopedModels.map((scoped) => scoped.model.id)).toEqual(["bracketed-model[1m]"]);
		expect(result.scopedModels[0].thinkingLevel).toBe("high");
		expect(result.diagnostics).toEqual([]);
	});
});

describe("resolveCliModel", () => {
	test("resolves --model provider/id without --provider", () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "deepseek/deepseek-v4-pro",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("deepseek");
		expect(result.model?.id).toBe("deepseek-v4-pro");
	});

	test("resolves fuzzy patterns within an explicit provider", () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliProvider: "deepseek",
			cliModel: "v4-pro",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("deepseek");
		expect(result.model?.id).toBe("deepseek-v4-pro");
	});

	test("supports --model <pattern>:<thinking> (without explicit --thinking)", () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "sonnet:high",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.id).toBe("MiniMax-M2.7");
		expect(result.thinkingLevel).toBe("high");
	});

	test("prefers exact model id match over provider inference (router-style ids)", () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "openai/gpt-4o:extended",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("aggregator");
		expect(result.model?.id).toBe("openai/gpt-4o:extended");
	});

	test("does not strip invalid :suffix as thinking level in --model (treat as raw id)", () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliProvider: "deepseek",
			cliModel: "deepseek-v4-pro",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("deepseek");
		expect(result.model?.id).toBe("deepseek-v4-pro");
	});

	test("allows custom model ids for explicit providers without double prefixing", () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliProvider: "aggregator",
			cliModel: "aggregator/openai/ghost-model",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("aggregator");
		expect(result.model?.id).toBe("openai/ghost-model");
	});

	test("returns a clear error when there are no models", () => {
		const registry = {
			getModels: () => [],
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliProvider: "deepseek",
			cliModel: "deepseek-v4-pro",
			modelRuntime: registry,
		});

		expect(result.model).toBeUndefined();
		expect(result.error).toContain("No models available");
	});

	test("prefers the sole authenticated provider for an ambiguous bare exact model id", () => {
		const azureModel: Model<"anthropic-messages"> = {
			...mockModels[1],
			id: "deepseek-v4-pro",
			name: "GPT 5.6 Sol",
			provider: "deepseek",
		};
		const codexModel: Model<"anthropic-messages"> = {
			...mockModels[1],
			id: "deepseek-v4-pro",
			name: "GPT 5.6 Sol",
			provider: "minimax",
		};
		const registry = {
			getModels: () => [azureModel, codexModel],
			hasConfiguredAuth: (provider: string) => provider === "deepseek",
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "deepseek-v4-pro",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("deepseek");
		expect(result.model?.id).toBe("deepseek-v4-pro");
	});

	test("requires an explicit provider for an ambiguous bare exact model id without a unique authenticated provider", () => {
		const azureModel: Model<"anthropic-messages"> = {
			...mockModels[1],
			id: "deepseek-v4-pro",
			name: "GPT 5.6 Sol",
			provider: "deepseek",
		};
		const codexModel: Model<"anthropic-messages"> = {
			...mockModels[1],
			id: "deepseek-v4-pro",
			name: "GPT 5.6 Sol",
			provider: "minimax",
		};
		const registry = {
			getModels: () => [azureModel, codexModel],
			hasConfiguredAuth: () => false,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "deepseek-v4-pro",
			modelRuntime: registry,
		});

		expect(result.model).toBeUndefined();
		expect(result.error).toContain('Model "deepseek-v4-pro" is ambiguous across providers');
		expect(result.error).toContain("deepseek/deepseek-v4-pro");
		expect(result.error).toContain("minimax/deepseek-v4-pro");
		expect(result.error).toContain("Use --provider or provider/model");
	});

	test("prefers provider/model split over gateway model with matching id", () => {
		// When a user writes "zai/glm-5", and both a zai provider model (id: "glm-5")
		// and a gateway model (id: "zai/glm-5") exist, prefer the zai provider model.
		const zaiModel: Model<"anthropic-messages"> = {
			id: "glm-5",
			name: "GLM-5",
			api: "anthropic-messages",
			provider: "zai",
			baseUrl: "https://open.bigmodel.cn/api/paas/v4",
			reasoning: true,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};
		const gatewayModel: Model<"anthropic-messages"> = {
			id: "zai/glm-5",
			name: "GLM-5",
			api: "anthropic-messages",
			provider: "deepseek",
			baseUrl: "https://ai-gateway.vercel.sh",
			reasoning: true,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};
		const registry = {
			getModels: () => [...allModels, zaiModel, gatewayModel],
			hasConfiguredAuth: () => true,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "zai/glm-5",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("zai");
		expect(result.model?.id).toBe("glm-5");
	});

	test("prefers an authenticated exact raw model id over an unauthenticated inferred provider", () => {
		const commandcodeModel: Model<"anthropic-messages"> = {
			id: "xiaomi/mimo-v2.5-pro",
			name: "Xiaomi MiMo via Commandcode",
			api: "anthropic-messages",
			provider: "commandcode",
			baseUrl: "https://example.invalid",
			reasoning: false,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};
		const xiaomiModel: Model<"anthropic-messages"> = {
			id: "mimo-v2.5-pro",
			name: "Xiaomi MiMo",
			api: "anthropic-messages",
			provider: "xiaomi",
			baseUrl: "https://api.xiaomimimo.com",
			reasoning: false,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};
		const registry = {
			getModels: () => [...allModels, commandcodeModel, xiaomiModel],
			hasConfiguredAuth: (provider: string) => provider === "commandcode",
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "xiaomi/mimo-v2.5-pro",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("commandcode");
		expect(result.model?.id).toBe("xiaomi/mimo-v2.5-pro");
	});

	test("resolves provider-prefixed fuzzy patterns (aggregator/qwen -> aggregator model)", () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

		const result = resolveCliModel({
			cliModel: "aggregator/qwen",
			modelRuntime: registry,
		});

		expect(result.error).toBeUndefined();
		expect(result.model?.provider).toBe("aggregator");
		expect(result.model?.id).toBe("qwen/qwen3-coder:exacto");
	});

	describe("custom model fallback with :thinking suffix (#5552)", () => {
		// Models for a provider that has registered models but the specific model ID
		// is not in the registry (triggers buildFallbackModel path).
		const neuralwattModel: Model<"anthropic-messages"> = {
			id: "some-base-model",
			name: "Some Base Model",
			api: "anthropic-messages",
			provider: "neuralwatt",
			baseUrl: "https://api.neuralwatt.com",
			reasoning: false,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};

		const modelsWithNeuralwatt = [...allModels, neuralwattModel];

		test("strips :thinking suffix from custom model id in fallback path", () => {
			const registry = {
				getModels: () => modelsWithNeuralwatt,
			} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

			const result = resolveCliModel({
				cliModel: "neuralwatt/zai-org/GLM-5.1-FP8:high",
				modelRuntime: registry,
			});

			expect(result.error).toBeUndefined();
			expect(result.model?.provider).toBe("neuralwatt");
			// The :high suffix must NOT leak into the model id sent to the API
			expect(result.model?.id).toBe("zai-org/GLM-5.1-FP8");
			expect(result.model?.reasoning).toBe(true);
			expect(result.thinkingLevel).toBe("high");
		});

		test("custom model without thinking suffix works normally in fallback path", () => {
			const registry = {
				getModels: () => modelsWithNeuralwatt,
			} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

			const result = resolveCliModel({
				cliModel: "neuralwatt/zai-org/GLM-5.1-FP8",
				modelRuntime: registry,
			});

			expect(result.error).toBeUndefined();
			expect(result.model?.provider).toBe("neuralwatt");
			expect(result.model?.id).toBe("zai-org/GLM-5.1-FP8");
			expect(result.thinkingLevel).toBeUndefined();
		});

		test("all valid thinking levels work in fallback path", () => {
			const registry = {
				getModels: () => modelsWithNeuralwatt,
			} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

			for (const level of ["off", "minimal", "low", "medium", "high", "xhigh", "max"]) {
				const result = resolveCliModel({
					cliModel: `neuralwatt/zai-org/GLM-5.1-FP8:${level}`,
					modelRuntime: registry,
				});

				expect(result.error).toBeUndefined();
				expect(result.model?.id).toBe("zai-org/GLM-5.1-FP8");
				expect(result.thinkingLevel).toBe(level);
			}
		});

		test("invalid thinking suffix on custom model is treated as part of model id", () => {
			const registry = {
				getModels: () => modelsWithNeuralwatt,
			} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

			const result = resolveCliModel({
				cliModel: "neuralwatt/zai-org/GLM-5.1-FP8:banana",
				modelRuntime: registry,
			});

			expect(result.error).toBeUndefined();
			expect(result.model?.provider).toBe("neuralwatt");
			// Invalid suffix stays in the id (it's not a thinking level)
			expect(result.model?.id).toBe("zai-org/GLM-5.1-FP8:banana");
			expect(result.thinkingLevel).toBeUndefined();
		});

		test("explicit --provider with custom model:thinking strips suffix correctly", () => {
			const registry = {
				getModels: () => modelsWithNeuralwatt,
			} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

			const result = resolveCliModel({
				cliProvider: "neuralwatt",
				cliModel: "zai-org/GLM-5.1-FP8:high",
				modelRuntime: registry,
			});

			expect(result.error).toBeUndefined();
			expect(result.model?.provider).toBe("neuralwatt");
			expect(result.model?.id).toBe("zai-org/GLM-5.1-FP8");
			expect(result.thinkingLevel).toBe("high");
		});

		test("with explicit --thinking, :suffix is kept as part of model id", () => {
			const registry = {
				getModels: () => modelsWithNeuralwatt,
			} as unknown as Parameters<typeof resolveCliModel>[0]["modelRuntime"];

			const result = resolveCliModel({
				cliModel: "neuralwatt/zai-org/GLM-5.1-FP8:high",
				cliThinking: "medium",
				modelRuntime: registry,
			});

			expect(result.error).toBeUndefined();
			expect(result.model?.provider).toBe("neuralwatt");
			// :high is kept as part of the model id since --thinking was explicit
			expect(result.model?.id).toBe("zai-org/GLM-5.1-FP8:high");
			expect(result.thinkingLevel).toBeUndefined();
		});
	});
});

describe("default model selection", () => {
	test("deepseek defaults track current models", () => {
		expect(defaultModelPerProvider.deepseek).toBe("deepseek-v4-pro");
	});

	test("zai, minimax, and ant-ling defaults track current models", () => {
		expect(defaultModelPerProvider.zai).toBe("glm-5.3");
		expect(defaultModelPerProvider["zai-coding-cn"]).toBe("glm-5.3");
		expect(defaultModelPerProvider.minimax).toBe("MiniMax-M2.7");
		expect(defaultModelPerProvider["minimax-cn"]).toBe("MiniMax-M2.7");
		expect(defaultModelPerProvider["ant-ling"]).toBe("Ring-2.6-1T");
	});

	test("built-in defaults exist in generated provider catalogs", () => {
		for (const provider of getBuiltinProviders()) {
			const defaultId = defaultModelPerProvider[provider];
			expect(
				getBuiltinModels(provider).some((model) => model.id === defaultId),
				`${provider} default ${defaultId} should exist in its generated catalog`,
			).toBe(true);
		}
	});

	test("qwen token plan individual default tracks current model", () => {
		expect(defaultModelPerProvider["qwen-token-plan-individual"]).toBe("qwen3.8-max");
	});

	test("findInitialModel accepts explicit provider custom model ids", async () => {
		const registry = {
			getModels: () => allModels,
		} as unknown as Parameters<typeof findInitialModel>[0]["modelRuntime"];

		const result = await findInitialModel({
			cliProvider: "aggregator",
			cliModel: "aggregator/openai/ghost-model",
			scopedModels: [],
			isContinuing: false,
			modelRuntime: registry,
		});

		expect(result.model?.provider).toBe("aggregator");
		expect(result.model?.id).toBe("openai/ghost-model");
	});

	test("findInitialModel selects ai-gateway default when available", async () => {
		const aiGatewayModel: Model<"anthropic-messages"> = {
			id: "anthropic/claude-opus-4-6",
			name: "Claude Opus 4.6",
			api: "anthropic-messages",
			provider: "deepseek",
			baseUrl: "https://ai-gateway.vercel.sh",
			reasoning: true,
			input: ["text", "image"],
			cost: { input: 5, output: 15, cacheRead: 0.5, cacheWrite: 5 },
			contextWindow: 200000,
			maxTokens: 8192,
		};

		const registry = {
			getAvailableSnapshot: () => [aiGatewayModel],
		} as unknown as Parameters<typeof findInitialModel>[0]["modelRuntime"];

		const result = await findInitialModel({
			scopedModels: [],
			isContinuing: false,
			modelRuntime: registry,
		});

		expect(result.model?.provider).toBe("deepseek");
		expect(result.model?.id).toBe("anthropic/claude-opus-4-6");
	});

	test("findInitialModel ignores an unauthenticated saved default", async () => {
		const savedDeepSeekModel: Model<"anthropic-messages"> = {
			id: "deepseek-v4-flash",
			name: "DeepSeek V4 Flash",
			api: "anthropic-messages",
			provider: "deepseek",
			baseUrl: "https://api.deepseek.com",
			reasoning: true,
			input: ["text"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1 },
			contextWindow: 128000,
			maxTokens: 8192,
		};
		const localDeepSeekModel: Model<"anthropic-messages"> = {
			...savedDeepSeekModel,
			provider: "spark-two",
			baseUrl: "http://spark-two:8000/v1",
		};
		const registry = {
			getModel: (provider: string, modelId: string) =>
				provider === savedDeepSeekModel.provider && modelId === savedDeepSeekModel.id
					? savedDeepSeekModel
					: undefined,
			hasConfiguredAuth: (provider: string) => provider === "spark-two",
			getAvailableSnapshot: () => [localDeepSeekModel],
		} as unknown as Parameters<typeof findInitialModel>[0]["modelRuntime"];

		const result = await findInitialModel({
			scopedModels: [],
			isContinuing: false,
			defaultProvider: "deepseek",
			defaultModelId: "deepseek-v4-flash",
			modelRuntime: registry,
		});

		expect(result.model?.provider).toBe("spark-two");
		expect(result.model?.id).toBe("deepseek-v4-flash");
	});

	describe("persisted default model scoping", () => {
		const tempDirs: string[] = [];
		const sonnet = getModel("minimax", "MiniMax-M2.7")!;
		const opus = getModel("kimi-coding", "kimi-for-coding")!;

		afterEach(() => {
			for (const dir of tempDirs.splice(0)) {
				rmSync(dir, { recursive: true, force: true });
			}
		});

		async function createSession(options: { scoped: boolean; persistedScope?: string[] }) {
			const tempDir = join(tmpdir(), `pi-default-scope-${Date.now()}-${Math.random().toString(36).slice(2)}`);
			mkdirSync(tempDir, { recursive: true });
			tempDirs.push(tempDir);

			const settingsManager = SettingsManager.create(tempDir, tempDir);
			if (options.persistedScope) {
				settingsManager.setEnabledModels(options.persistedScope);
			}

			const authStorage = AuthStorage.inMemory({
				minimax: { type: "api_key", key: "test-key" },
				"kimi-coding": { type: "api_key", key: "test-key" },
			});
			const modelRuntime = getModelRuntime(await createModelRegistry(authStorage, join(tempDir, "models.json")));
			const agent = new Agent({
				initialState: {
					model: sonnet,
					systemPrompt: "test",
					tools: [],
				},
				streamFn: streamSimple,
			});
			const session = new AgentSession({
				agent,
				sessionManager: SessionManager.inMemory(tempDir),
				settingsManager,
				cwd: tempDir,
				modelRuntime,
				resourceLoader: createTestResourceLoader(),
				scopedModels: options.scoped ? [{ model: sonnet }] : [],
			});

			return { session, settingsManager };
		}

		test("adds a persisted default to an existing scoped model list", async () => {
			const { session, settingsManager } = await createSession({
				scoped: true,
				persistedScope: [`${sonnet.provider}/${sonnet.id}`],
			});

			await session.setModel(opus, { persist: true });

			expect(settingsManager.getDefaultProvider()).toBe(opus.provider);
			expect(settingsManager.getDefaultModel()).toBe(opus.id);
			expect(session.scopedModels.map((scoped) => `${scoped.model.provider}/${scoped.model.id}`)).toEqual([
				`${sonnet.provider}/${sonnet.id}`,
				`${opus.provider}/${opus.id}`,
			]);
			expect(settingsManager.getEnabledModels()).toEqual([
				`${sonnet.provider}/${sonnet.id}`,
				`${opus.provider}/${opus.id}`,
			]);
		});

		test("does not create a scoped model list when all models are available", async () => {
			const { session, settingsManager } = await createSession({ scoped: false });

			await session.setModel(opus, { persist: true });

			expect(session.scopedModels).toEqual([]);
			expect(settingsManager.getEnabledModels()).toBeUndefined();
		});

		test("keeps session-only model changes out of scope", async () => {
			const { session, settingsManager } = await createSession({
				scoped: true,
				persistedScope: [`${sonnet.provider}/${sonnet.id}`],
			});

			await session.setModel(opus, { persist: false });

			expect(session.scopedModels.map((scoped) => `${scoped.model.provider}/${scoped.model.id}`)).toEqual([
				`${sonnet.provider}/${sonnet.id}`,
			]);
			expect(settingsManager.getEnabledModels()).toEqual([`${sonnet.provider}/${sonnet.id}`]);
		});
	});
});
