# Providers

Pi supports subscription-based providers via OAuth and API key providers via environment variables or auth file. Built-in catalogs ship with pi; configured providers may refresh newer catalogs and cache them in `~/.pi/agent/models-store.json` for offline use.

## Table of Contents

- [Subscriptions](#subscriptions)
- [API Keys](#api-keys)
- [Auth File](#auth-file)
- [llama.cpp](#llamacpp)
- [Custom Providers](#custom-providers)
- [Resolution Order](#resolution-order)

## Subscriptions

Use `/login` in interactive mode, then select a provider:

- Kimi For Coding

Use `/logout` to clear credentials. Tokens are stored in `~/.pi/agent/auth.json` and auto-refresh when expired.

### Kimi For Coding

- Run `/login kimi-coding`, then select **Sign in with Kimi Code**
- The device-code flow talks to `auth.kimi.com`; requests to `api.kimi.com/coding` carry the access token as an `Authorization: Bearer` header
- `KIMI_API_KEY` remains available through **Use an API key**

## API Keys

### Environment Variables or Auth File

Use `/login` in interactive mode and select a provider to store an API key in `auth.json`, or set credentials via environment variable:

```bash
export DEEPSEEK_API_KEY=sk-...
pi
```

| Provider | Environment Variable | `auth.json` key |
|----------|----------------------|------------------|
| Ant Ling | `ANT_LING_API_KEY` | `ant-ling` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` |
| Kimi For Coding | `KIMI_API_KEY` | `kimi-coding` |
| MiniMax | `MINIMAX_API_KEY` | `minimax` |
| MiniMax (China) | `MINIMAX_CN_API_KEY` | `minimax-cn` |
| Moonshot AI | `MOONSHOT_API_KEY` | `moonshotai` |
| Moonshot AI (China) | `MOONSHOT_API_KEY` | `moonshotai-cn` |
| Qwen Token Plan (existing catalog) | `QWEN_TOKEN_PLAN_API_KEY` | `qwen-token-plan` |
| Qwen Token Plan (Individual) | `QWEN_TOKEN_PLAN_API_KEY` | `qwen-token-plan-individual` |
| Qwen Token Plan (China) | `QWEN_TOKEN_PLAN_CN_API_KEY` | `qwen-token-plan-cn` |
| Xiaomi MiMo | `XIAOMI_API_KEY` | `xiaomi` |
| Xiaomi MiMo Token Plan (China) | `XIAOMI_TOKEN_PLAN_CN_API_KEY` | `xiaomi-token-plan-cn` |
| Xiaomi MiMo Token Plan (Amsterdam) | `XIAOMI_TOKEN_PLAN_AMS_API_KEY` | `xiaomi-token-plan-ams` |
| Xiaomi MiMo Token Plan (Singapore) | `XIAOMI_TOKEN_PLAN_SGP_API_KEY` | `xiaomi-token-plan-sgp` |
| ZAI Coding Plan (Global) | `ZAI_API_KEY` | `zai` |
| ZAI Coding Plan (China) | `ZAI_CODING_CN_API_KEY` | `zai-coding-cn` |

Reference for environment variables and `auth.json` keys: [`const envMap`](https://github.com/earendil-works/pi-mono/blob/main/packages/ai/src/env-api-keys.ts) in [`packages/ai/src/env-api-keys.ts`](https://github.com/earendil-works/pi-mono/blob/main/packages/ai/src/env-api-keys.ts).

#### Auth File

Store credentials in `~/.pi/agent/auth.json`:

```json
{
  "ant-ling": { "type": "api_key", "key": "..." },
  "deepseek": { "type": "api_key", "key": "sk-..." },
  "kimi-coding": { "type": "oauth", "access": "...", "refresh": "...", "expires": 0 },
  "minimax": { "type": "api_key", "key": "..." },
  "moonshotai": { "type": "api_key", "key": "..." },
  "qwen-token-plan": { "type": "api_key", "key": "sk-sp-..." },
  "qwen-token-plan-individual": { "type": "api_key", "key": "sk-sp-..." },
  "qwen-token-plan-cn": { "type": "api_key", "key": "sk-sp-..." },
  "xiaomi": { "type": "api_key", "key": "..." },
  "xiaomi-token-plan-cn": { "type": "api_key", "key": "..." },
  "xiaomi-token-plan-ams": { "type": "api_key", "key": "..." },
  "xiaomi-token-plan-sgp": { "type": "api_key", "key": "..." },
  "zai": { "type": "api_key", "key": "..." },
  "zai-coding-cn": { "type": "api_key", "key": "..." }
}
```

`qwen-token-plan-individual` uses the same international endpoint and `QWEN_TOKEN_PLAN_API_KEY` as
`qwen-token-plan`, but limits the picker to the models documented for Individual subscriptions. The existing
provider keeps its broader catalog for backward compatibility. When using `auth.json`, store the
credential under the provider you select; an environment variable is shared by both international providers.

The file is created with `0600` permissions (user read/write only). Auth file credentials take priority over environment variables.

API key credentials can also include provider-scoped environment values. These values are used before process environment variables when resolving the credential key, provider/model headers, and settings such as `PI_CACHE_RETENTION` and `HTTP_PROXY`/`HTTPS_PROXY`.

```json
{
  "deepseek": {
    "type": "api_key",
    "key": "$DEEPSEEK_API_KEY",
    "env": {
      "DEEPSEEK_API_KEY": "...",
      "HTTP_PROXY": "http://proxy.internal:3128"
    }
  }
}
```

Use this when pi should use different provider settings than the project shell environment.

### Key Resolution

The `key` field supports command execution, environment interpolation, and literals:

- **Shell command:** `"!command"` at the start executes the whole value as a command and uses stdout (cached for process lifetime)
  ```json
  { "type": "api_key", "key": "!security find-generic-password -ws 'deepseek'" }
  { "type": "api_key", "key": "!op read 'op://vault/item/credential'" }
  ```
- **Environment interpolation:** `"$ENV_VAR"` or `"${ENV_VAR}"` uses the value of the named variable. Interpolation works inside larger literals.
  ```json
  { "type": "api_key", "key": "$MY_DEEPSEEK_KEY" }
  { "type": "api_key", "key": "${KEY_PREFIX}_${KEY_SUFFIX}" }
  ```
  `$FOO_BAR` is the variable `FOO_BAR`; use `${FOO}_BAR` when `BAR` is literal text. Missing environment variables make the value unresolved.
- **Escapes:** `"$$"` emits a literal `"$"`; `"$!"` emits a literal `"!"` without triggering command execution.
  ```json
  { "type": "api_key", "key": "$$literal-dollar-prefix" }
  { "type": "api_key", "key": "$!literal-bang-prefix" }
  ```
- **Literal value:** Used directly. Plain uppercase strings such as `MY_API_KEY` are literals; use `$MY_API_KEY` for environment variables.
  ```json
  { "type": "api_key", "key": "sk-ant-..." }
  { "type": "api_key", "key": "public" }
  ```

OAuth credentials are also stored here after `/login` and managed automatically.

## llama.cpp

Pi supports the llama.cpp router server. Configure it with `/login llama.cpp`, manage loaded models with `/llama`, and select a loaded model with `/model`.

See [llama.cpp](llama-cpp.md) for server setup, model directory layout, environment variables, and command usage.

## Custom Providers

**Via models.json:** Add Ollama, LM Studio, vLLM, or any provider that speaks a supported API (OpenAI Completions, OpenAI Responses, Anthropic Messages, Google Generative AI). See [models.md](models.md).

**Via extensions:** For providers that need custom API implementations or OAuth flows, create an extension. See [custom-provider.md](custom-provider.md) and [examples/extensions/custom-provider-gitlab-duo](../examples/extensions/custom-provider-gitlab-duo/).

## Resolution Order

When resolving credentials for a provider:

1. CLI `--api-key` flag
2. `auth.json` entry (API key or OAuth token)
3. Environment variable
4. Custom provider keys from `models.json`
