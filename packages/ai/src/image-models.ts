import type { ImagesApi, ImagesModel } from "./types.ts";

const imageModelRegistry: Map<string, Map<string, ImagesModel<ImagesApi>>> = new Map();

export function getImageModel(provider: string, modelId: string): ImagesModel<ImagesApi> | undefined {
	return imageModelRegistry.get(provider)?.get(modelId);
}

export function getImageProviders(): string[] {
	return Array.from(imageModelRegistry.keys());
}

export function getImageModels(provider: string): ImagesModel<ImagesApi>[] {
	return Array.from(imageModelRegistry.get(provider)?.values() ?? []);
}
