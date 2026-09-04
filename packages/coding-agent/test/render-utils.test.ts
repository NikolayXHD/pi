import { homedir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { displayPath, renderToolPath } from "../src/core/tools/render-utils.ts";

const cwd = "/home/user/projects/app";

const fakeTheme = {
	fg: (_key: string, text: string) => text,
} as never;

describe("displayPath", () => {
	it("renders absolute paths inside cwd as cwd-relative", () => {
		expect(displayPath("/home/user/projects/app/src/main.ts", cwd)).toBe("src/main.ts");
	});

	it("renders the cwd itself as a dot", () => {
		expect(displayPath("/home/user/projects/app", cwd)).toBe(".");
	});

	it("normalizes relative paths to their cwd-relative form", () => {
		expect(displayPath("./src/../src/main.ts", cwd)).toBe("src/main.ts");
		expect(displayPath("src/main.ts", cwd)).toBe("src/main.ts");
	});

	it("keeps paths outside cwd absolute with a tilde for home", () => {
		expect(displayPath("/opt/tools/bin", cwd)).toBe("/opt/tools/bin");
		expect(displayPath(join(homedir(), "notes", "todo.md"), cwd)).toBe("~/notes/todo.md");
	});
});

describe("renderToolPath", () => {
	it("renders a null path as an invalid argument", () => {
		expect(renderToolPath(null, fakeTheme, cwd)).toBe("[invalid arg]");
	});

	it("renders an empty path with the empty fallback", () => {
		expect(renderToolPath("", fakeTheme, cwd, { emptyFallback: "." })).toBe(".");
	});

	it("uses the cwd-relative display for paths inside cwd", () => {
		expect(renderToolPath("/home/user/projects/app/lib/util.ts", fakeTheme, cwd)).toBe("lib/util.ts");
	});
});
