# ITER AI · V2.0

通过对话理解旅行需求，结合真实地点、路线、住宿等信息，生成可以继续调整的逐日行程。

**源码候选版 · 非商业使用许可 · 商用须另行授权**

当前代码可用于研究和自部署体验，不应视为已完成生产验收的稳定服务。第三方服务账号、API 密钥和调用额度不包含在仓库中。

## 能做什么

- 在对话中整理目的地、日期、同行人和旅行偏好。
- 通过景点、饮食和住宿偏好卡逐步确认需求，生成旅行任务书。
- 确认任务书后，LangGraph ReAct 主 Agent 自主选择工具、生成行程，并交由独立 Reviewer 核查和反馈修订。
- Prepare 提前补齐候选和可复用事实；Planner 接入高德 MCP、天气、FlyAI 和可选 Tavily 联网搜索。
- 规划过程以可折叠自然语言短句展示；来源链接归入“本次资料”，登录用户可管理显式长期偏好。
- 展示行程地图，支持历史行程恢复和部分行程调整。

模型负责理解需求、选择工具和修订方案；程序负责时间试算、数据引用、预算、持久化、校验与结果提交。界面使用 React / TypeScript，服务端使用 FastAPI、LangGraph、PostgreSQL 和 Redis。

## V2.0 发布验证

开发版本通过后端 3,045 项、前端 724 项和浏览器确定性回归 11 项。公开分发另行通过类型检查、构建、Python 模块导入与应用装配检查。上述检查不代表重新完成真实模型与 Provider 的整轮旅行验收，也不包含服务器部署。

## 当前限制

- 规划可能返回 `partial`：有可用结果，但仍存在空档、路线不理想或资料不完整等问题；也可能遇到超时或未发布结果。
- 模型和外部查询可能耗时数分钟。服务额度、网络、城市覆盖和可查询日期范围都会影响结果。
- 图片、票价、营业时间、天气等可能缺失。未知价格不能等同于免费，行程和报价也不代表预约或预订成功。
- 出发前请复核营业、预约、交通及价格。不要将未经核验的结果直接用于付费交付。

## 运行准备

以下命令用于 macOS / Linux 的本地开发环境。请准备：

- Python 3.12（项目声明支持 3.11 及以上；建议使用已验证的 3.12）。
- Node.js 22.12 及以上、pnpm 11.19.0。
- 已创建数据库和账号的 PostgreSQL 16，以及可用的 Redis 7；不要使用存有真实用户数据的数据库做初次试运行。
- 可访问的模型服务、高德 Web 服务 Key 和 FlyAI Key。
- 官方 FlyAI CLI，且启动 API 的进程能够在 `PATH` 中找到 `flyai`。

```bash
npm install --global pnpm@11.19.0 @fly-ai/flyai-cli@1.0.16
flyai --help
git clone https://github.com/Qiao-BridgeTAT/ITER-AI.git
cd ITER-AI/travel-agent
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-agent.lock -r requirements-mcp.lock .
cp .env.example .env
```

FlyAI 的安装说明见[官方项目](https://github.com/alibaba-flyai/flyai-skill)。服务授权与费用由相应提供方决定；本项目不会赠送额度。

### 配置自己的环境

编辑 `travel-agent/.env`，不要直接启动尚未填写完成的模板。

| 配置 | 作用 |
| --- | --- |
| `DATABASE_URL`、`REDIS_URL` | 自己的数据库和 Redis 连接；模板中的数据库账号只是示例 |
| `QWEN_BASE_URL`、`QWEN_MODEL`、`QWEN_API_KEY` | 模型服务；地址和名称已提供默认示例，Key 必须自行填写 |
| `AMAP_WEB_SERVICE_KEY`、`FLYAI_API_KEY` | 服务端地点、路线及旅游产品查询 |
| `VITE_AMAP_JS_API_KEY`、`AMAP_JS_SECURITY_CODE` | 浏览器地图 Key 与服务端地图代理安全码；二者与 Web 服务 Key 不是同一种凭据 |
| `SESSION_SECRET`、`PII_ENCRYPTION_KEY`、`EXPORT_SIGNING_SECRET` | 分别设置独立的随机秘密值，建议每项至少 32 个随机字节；不要使用模板值或个人信息 |
| `SMS_PROVIDER`、`SMS_*` | 手机号登录所需的短信服务配置；本地暂不使用登录时可设 `SMS_PROVIDER=unavailable` |
| `WEATHER_API_KEY`、`OPENWEATHER_API_KEY` | 可选天气服务，需要对应产品的有效权限 |
| `VITE_HOME_BACKGROUND_URL` | 可选首页视频地址，只能使用有权使用的素材 |
| `OBJECT_STORAGE_LOCAL_PATH` | 本地文件目录，默认 `.local/artifacts`；本地可将 `OBJECT_STORAGE_ENDPOINT` 留空 |

例如，可在自己的终端运行下面的命令生成一个随机值，重复三次并分别填入上述三个秘密配置。不要把输出发到 Issue 或提交到 Git。

```bash
python3.12 -c "import secrets; print(secrets.token_urlsafe(32))"
```

`AMAP_SEARCH_PROXY_*` 是可选的服务端 POI 查询代理配置；不使用时保持 URL 和 Key 都为空。启用时必须使用 HTTPS，并自行确认代理的可信性和数据处理条款。其余高德接口不经此代理。

`.env` 不是自动配置完成的凭据包。以下命令会执行其中的 shell 赋值，只应加载自己维护的文件；包含空格或 shell 特殊字符的值需要正确引用。

### V2.0 Planner 与 MCP

示例配置为新运行启用 `V4_PLANNER_ENGINE=langgraph-react-2`。Planner 使用 `qwen3.8-flash` 和 JSON Schema，Reviewer 保持独立上下文；已有运行按创建时的引擎恢复。`V4_PLANNER_ENGINE=legacy` 只影响后续新运行。

高德默认连接官方 MCP，使用自己的 `AMAP_WEB_SERVICE_KEY`。如果基础搜索使用单独代理，请给本地搜索 MCP 进程配置自己的 `AMAP_SEARCH_PROXY_URL`、`AMAP_SEARCH_PROXY_KEY` 和匹配网关的参数名，然后在 API 环境配置 `AMAP_SEARCH_MCP_URL=http://127.0.0.1:8766/mcp`：

```bash
.venv/bin/python -m services.amap_search_mcp --env-file .env --port 8766
```

本地服务仅监听回环地址；代理凭据留在该服务环境中，路线工具继续走官方 MCP。联网搜索可设置 `TAVILY_MCP_ENABLED=true` 并填写自己的 `TAVILY_API_KEY`。天气和酒店沿用各自服务配置。

新运行默认总预算 480 秒（最后 20 秒收尾）、24 次主 Agent 决策、初稿后最多 6 次有效方案修改，并限制外部请求与并发。本地验收与生产应保持相同配置。断线和进程恢复沿用原预算；用户主动继续已耗尽的失败运行时，新建有界执行段并保留候选、事实、方案及已完成工具回执。调试可通过配置放宽，但响应时间和外部费用也会增加。无需预约余量即可规划；预约与未知信息作为行程提醒保留。

完整决策的格式错误通过 LangGraph 模型节点返回具体错误字段及被拒方案，修正后继续由 ToolNode 执行，不绕过参数与业务校验。执行超时或重试耗尽的提示依据已保存的停止原因生成，不将未知房态和参考价格误报为酒店接口失败。

### 启动 API

在 `travel-agent/` 下执行：

```bash
set -a
. ./.env
set +a
.venv/bin/python -m alembic upgrade head
.venv/bin/python -m uvicorn services.api.main:create_default_app --factory --host 127.0.0.1 --port 8000
```

迁移会改变指定数据库的结构，已有数据请先备份。API 就绪检查为 `http://127.0.0.1:8000/health/ready`；就绪不代表模型和第三方账号已通过真实规划验证。

### 启动网页

打开另一个终端，在同一 `travel-agent/` 目录重新加载 `.env`，再启动网页：

```bash
set -a
. ./.env
set +a
cd apps/web
pnpm install --frozen-lockfile
pnpm dev --host 127.0.0.1 --port 5173 --strictPort
```

打开 `http://localhost:5173/`。页面与 API 默认通过同源代理连接；如果改变端口或访问域名，需要相应调整公开地址、CORS 和 Cookie 配置。API 端口改变时，还要为前端进程设置 `TRAVEL_AGENT_API_PORT`。

有后台队列任务的部署可在加载相同服务端配置后，从 `travel-agent/` 运行 `.venv/bin/python -m services.worker.main`。

## 构建与部署注意事项

在已加载部署环境变量的终端进入 `travel-agent/apps/web/`：

```bash
pnpm typecheck
pnpm build
```

静态产物位于 `apps/web/dist/`。构建成功不等于生产部署完成：

首页包含“关于 ITER AI”入口。背景视频通过构建时的 `VITE_HOME_BACKGROUND_URL` 配置；未配置时只显示蓝色背景。升级已有站点时必须沿用其背景地址，在重建前加载配置，并实际检查视频播放、关于弹窗、登录入口和开始规划按钮。容器启动后再设置该变量不会改变已生成的静态页面。

- 使用正式 Web 服务器托管静态产物，配置 SPA 路由回退；不要将 Vite 开发服务器作为公网生产服务。
- 将 `/api` 转发到 API，并支持 WebSocket Upgrade；代理读取超时应能覆盖长时间规划。不要直接暴露数据库或 Redis。
- 使用地图时需要部署 `/_AMapService` 代理：地图样式请求转发到 `webapi.amap.com`，其余请求转发到 `restapi.amap.com`，由服务端注入安全码。具体开发代理实现见 `travel-agent/apps/web/vite.config.ts`；静态文件托管不会自动继承它。
- 使用 `APP_ENV=production`、HTTPS、真实站点的 CORS / Cookie 域、生产短信配置和独立密钥。生产启动必填项以 `travel-agent/config/contract.json` 为准，不能直接沿用开发配置。
- 当前存储探针支持本地目录或 MinIO 就绪端点，不应假定配置任意云存储地址就完成了适配。为数据库和持久化文件配置备份与恢复。
- `VITE_*` 会进入浏览器构建，不能放模型 Key、Web 服务 Key、短信密钥或地图安全码。构建产物包含部署相关信息，不要提交到源码仓库。
- 上线前使用自己的真实服务完成从输入、任务书确认、规划、回复到刷新恢复的验证，并检查失败场景、权限、限流与隐私保护。

仓库不包含真实用户数据、运行日志、测试夹具或内部开发记录。运行时仍可能产生数据库记录和本地加密审计日志，部署者需要妥善管理访问权限、加密密钥及保留期限。

## 反馈与联系

- 一般问题和建议：[GitHub Issues](https://github.com/Qiao-BridgeTAT/ITER-AI/issues)。请先阅读简短的[贡献与反馈说明](CONTRIBUTING.md)。
- 安全问题：请按 [SECURITY.md](SECURITY.md) 私下反馈，不要公开敏感细节。
- 联系与商业授权：[Li.qiao02@outlook.com](mailto:Li.qiao02@outlook.com)。

## 许可

原创源码采用 [ITER AI 非商业源码许可](LICENSE)：允许非商业下载、使用、研究和修改，商业使用须事先获得书面授权。它不是 OSI 批准的开源许可证。

字体、依赖库和外部服务遵循各自条款，详见[第三方内容与许可](THIRD_PARTY_NOTICES.md)。


### 高德行政区边界与 JSONP

如果 `DistrictSearch` 接口返回成功，但浏览器回调为 `error` / `[object Event]`，检查地图代理的响应 MIME 类型。高德 JSONP 可能携带 `application/json`，在 `X-Content-Type-Options: nosniff` 下无法作为脚本执行。应只对合法 `callback` 请求返回 `application/javascript`，普通 JSON 保留上游类型；不要关闭 `nosniff`。Nginx 在 location 新增 `add_header` 会覆盖上层继承的响应头，必须同时保留该站点原有的安全响应头。

规划过程视窗最多约六条单行消息，内部自动跟随最新内容；完整历史保留。最终行程页不展示黄色“行程提示”区，服务端校验记录仍保留。
