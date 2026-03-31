# Local MCP Server

This folder contains a MiniMax-style MCP server with the same two tool names described in the MiniMax MCP guide:

- `web_search(query: string)`
- `understand_image(prompt: string, image_url: string)`

Instead of calling MiniMax, this server uses:

- Exa, Tavily, and Brave behind one unified `web_search` MCP tool
- Your local OpenAI-compatible vision LLM for image understanding

## Files

- `server.py`: MCP server
- `.env_sample`: env template
- `.env`: runtime configuration

## Configure

Create your local env file:

```bash
cp examples/mcp/.env_sample examples/mcp/.env
```

Then edit `examples/mcp/.env`:

- Set any search provider keys you want to use:
  - `EXA_API_KEY`
  - `TAVILY_API_KEY`
  - `BRAVE_API_KEY`
- Set `LOCAL_VISION_MODEL`
- Adjust `LOCAL_VISION_BASE_URL` if your local VLM is not at `http://127.0.0.1:8000/v1`
- Optional: set `SEARCH_PROVIDER_ORDER` and monthly request limits if you want the MCP server to stop using a provider after a configured monthly budget

Your local vision server is expected to support OpenAI-compatible `chat/completions` with image inputs.

## Run

```bash
python examples/mcp/server.py
```

It will start a streamable HTTP MCP server at `http://127.0.0.1:18011/mcp` by default.

## Tool behavior

### `web_search`

- One MCP tool over Exa, Tavily, and Brave
- In `provider="auto"` mode, rotates across configured providers using a persisted local state file
- Returns compact search results plus related suggestions
- Includes the actual `provider` used in the response
- Exposes the most useful optional parameters for agent use:
  - `provider`: `auto`, `exa`, `tavily`, or `brave`
  - `search_type`: `auto`, `fast`, `neural`, `deep`, `deep-reasoning`, or `instant`
  - `num_results`: result count, capped to 25
  - `category`: useful for narrowing to `news`, `research paper`, `company`, `people`, `financial report`, or `personal site`
  - `include_domains`: only search these domains
  - `exclude_domains`: avoid these domains
  - `start_published_date` / `end_published_date`: published-date filtering in ISO format

Provider notes:

- Exa gets the richest native support for `search_type`, `category`, domain filters, and published-date filters
- Tavily maps `search_type` to Tavily `search_depth`, and maps some categories to Tavily topics
- Brave uses Brave Web Search and applies domain filters via `site:` query operators

Monthly rotation:

- Usage state is stored in `.search_provider_usage.json`
- If you set `*_MONTHLY_REQUEST_LIMIT`, a provider is skipped once its configured limit is reached for the current UTC month
- If limits are blank, the server still rotates providers but does not enforce a hard stop

Example:

```json
{
  "query": "Ed Sheeran upcoming concerts",
  "provider": "auto",
  "search_type": "deep",
  "category": "news",
  "include_domains": ["edsheeran.com", "ticketmaster.com"],
  "num_results": 8
}
```

### `understand_image`

- Accepts HTTP/HTTPS image URLs, local file paths, or `data:image/...` URLs
- Local files are converted to a data URL before being sent
- Remote images are fetched and converted to a data URL by default
- Enforces the same basic image constraints the MiniMax guide documents: JPEG, PNG, GIF, or WebP, max 20MB

## Example MCP config

For Gallama/OpenAI-compatible MCP tool definitions:

```json
{
  "type": "mcp",
  "server_label": "local-coding-plan",
  "server_url": "http://127.0.0.1:18011/mcp",
  "allowed_tools": ["web_search", "understand_image"],
  "require_approval": "never"
}
```

## Sources

- MiniMax MCP guide: https://platform.minimax.io/docs/token-plan/mcp-guide
- Exa search API: https://docs.exa.ai/reference/search
- Tavily search API: https://docs.tavily.com/documentation/api-reference/endpoint/search
- Brave Search API: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
