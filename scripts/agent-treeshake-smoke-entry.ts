import { Agent } from "@earendil-works/pi-agent-core";
import { createModels } from "@earendil-works/pi-ai";
import { minimaxProvider } from "@earendil-works/pi-ai/providers/minimax";

const models = createModels();
models.setProvider(minimaxProvider());
const model = models.getModel("minimax", "MiniMax-M2.7");
if (!model) throw new Error("MiniMax smoke-test model not found");

export const agent = new Agent({
	initialState: { model },
	streamFn: models.streamSimple.bind(models),
});
