// Per-tier provider regression for issues/provider-error-body-passthrough.
//
// Routes a 403-with-body error through the real provider catch path for a
// body-blind text provider (openai-completions). The test asserts the resulting
// errorMessage carries both the HTTP status and the body reason. The
// already-correct happy path (no double body / no duplicated status) is
// asserted via the shared helper in error-body.test.ts.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { Context, Model } from "../src/types.ts";

// openai SDK APIError shape: "<status> status code (no body)" message, the
// parsed body kept on `.error`.
class FakeAPIError extends Error {
	status: number;
	error: unknown;
	constructor(status: number, parsedBody: unknown) {
		super(`${status} status code (no body)`);
		this.name = "PermissionDeniedError";
		this.status = status;
		this.error = parsedBody;
	}
}

const openaiMock = vi.hoisted(() => ({
	// Default parsed body; individual tests may override before invoking.
	parsedBody: { error: "blocked by gateway WAF" } as unknown,
}));

vi.mock("openai", () => {
	function throwingCreate() {
		const promise = Promise.resolve(undefined) as unknown as { withResponse: () => Promise<never> };
		promise.withResponse = async () => {
			throw new FakeAPIError(403, openaiMock.parsedBody);
		};
		return promise;
	}
	class FakeOpenAI {
		chat = { completions: { create: throwingCreate } };
		responses = { create: throwingCreate };
	}
	return { default: FakeOpenAI };
});

const context: Context = {
	systemPrompt: "",
	messages: [{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: 0 }],
	tools: [],
};

const completionsModel: Model<"openai-completions"> = {
	id: "test-model",
	name: "Test Model",
	api: "openai-completions",
	provider: "deepseek",
	baseUrl: "https://api.deepseek.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1000,
	maxTokens: 100,
};

async function drainResult(stream: {
	[Symbol.asyncIterator](): AsyncIterator<unknown>;
	result(): Promise<{ errorMessage?: string; stopReason?: string }>;
}) {
	for await (const _event of stream) {
		void _event;
	}
	return stream.result();
}

describe("provider error body passthrough (per-tier regression)", () => {
	beforeEach(() => {
		openaiMock.parsedBody = { error: "blocked by gateway WAF" };
	});

	it("openai-completions (body-blind text) surfaces status + body", async () => {
		const output = await drainResult(streamOpenAICompletions(completionsModel, context, { apiKey: "test" }));

		expect(output.stopReason).toBe("error");
		expect(output.errorMessage).toContain("403");
		expect(output.errorMessage).toContain("blocked by gateway WAF");
		expect(output.errorMessage).not.toBe("403 status code (no body)");
	});
});
